"""Untrained baseline-shape diagnostics.

The models in this module are freshly initialized. Their rollouts can exercise
evaluation plumbing, but cannot validate a collective-intelligence hypothesis.
Credible comparisons require separately trained controls with matched data,
optimization, parameter, and compute budgets.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Any
from copy import deepcopy

from ..swarm.graph import SwarmGraph, TopologyType


@dataclass
class BaselineResult:
    """Result of baseline comparison."""
    baseline_name: str
    baseline_score: float
    swarm_score: float

    # Comparison
    delta: float  # swarm - baseline
    relative_improvement: float  # delta / baseline
    wins_out_of: tuple  # (wins, total_trials)

    # Statistics
    baseline_std: float = 0.0
    swarm_std: float = 0.0
    p_value: float = 1.0


class SingleAgentBaseline(nn.Module):
    """
    Single large agent with same parameter count as swarm.

    This is the primary baseline: does distributing computation
    across agents help vs concentrating it in one network?
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        total_params: int,
        num_layers: int = 4,
    ):
        super().__init__()

        # Calculate hidden dim to match parameters
        # Params ≈ input*hidden + (layers-1)*hidden^2 + hidden*output
        a = num_layers - 1
        b = input_dim + output_dim
        c = -total_params

        if a > 0:
            hidden = int((-b + np.sqrt(b**2 - 4*a*c + 1e-10)) / (2*a))
        else:
            hidden = total_params // (input_dim + output_dim + 1)

        hidden = max(32, min(hidden, 2048))

        layers = []
        prev = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(0.1))
            prev = hidden

        layers.append(nn.Linear(hidden, output_dim))
        self.network = nn.Sequential(*layers)

        actual = sum(p.numel() for p in self.parameters())
        print(f"SingleAgentBaseline: target={total_params:,}, actual={actual:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class EnsembleBaseline(nn.Module):
    """
    Ensemble of independent agents with majority vote.

    No interaction between agents - pure ensemble learning.
    """

    def __init__(
        self,
        num_agents: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        aggregation: str = 'mean',
    ):
        super().__init__()

        self.num_agents = num_agents
        self.aggregation = aggregation

        self.agents = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
            for _ in range(num_agents)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [agent(x) for agent in self.agents]
        stacked = torch.stack(outputs, dim=0)

        if self.aggregation == 'mean':
            return stacked.mean(dim=0)
        elif self.aggregation == 'max':
            return stacked.max(dim=0)[0]
        elif self.aggregation == 'vote':
            # Majority vote on argmax
            votes = torch.stack([out.argmax(dim=-1) for out in outputs], dim=0)
            return torch.mode(votes, dim=0)[0]
        else:
            return stacked.mean(dim=0)


class CentralizedBaseline(nn.Module):
    """
    Centralized controller that gets all information.

    Single network that processes concatenated observations.
    """

    def __init__(
        self,
        num_agents: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Central controller gets all observations concatenated
        total_input = num_agents * input_dim

        self.encoder = nn.Sequential(
            nn.Linear(total_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Per-agent output heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim)
            for _ in range(num_agents)
        ])

        self.num_agents = num_agents

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: Either (batch, num_agents, input_dim) or (batch, input_dim)
        """
        if observations.dim() == 2:
            # Assume same observation for all agents
            batch_size = observations.shape[0]
            observations = observations.unsqueeze(1).expand(-1, self.num_agents, -1)

        # Flatten all observations
        batch_size = observations.shape[0]
        flat = observations.reshape(batch_size, -1)

        # Central processing
        hidden = self.encoder(flat)

        # Per-agent outputs
        outputs = [head(hidden) for head in self.heads]

        return torch.stack(outputs, dim=1).mean(dim=1)


class IndependentAgentsBaseline(nn.Module):
    """
    Independent agents (same as swarm but no message passing).

    Tests whether the communication structure matters.
    """

    def __init__(
        self,
        num_agents: int,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.agents = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
            for _ in range(num_agents)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [agent(x) for agent in self.agents]
        return torch.stack(outputs, dim=0).mean(dim=0)


class RandomBaseline(nn.Module):
    """Random output baseline for sanity check."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return torch.randn(batch_size, self.output_dim, device=x.device)


class BaselineComparison:
    """Framework for comparing swarm against baselines."""

    def __init__(
        self,
        swarm: SwarmGraph,
        evaluation_fn: Callable[[nn.Module, int], Dict[str, float]],
        device: str = "cpu",
    ):
        """
        Args:
            swarm: The swarm to evaluate
            evaluation_fn: Function that evaluates a model, returns metrics dict
            device: Device to run on
        """
        # Handle SwarmGraph (no .to()) and nn.Module (has .to())
        if hasattr(swarm, 'to'):
            self.swarm = swarm.to(device)
        else:
            self.swarm = swarm
        self.evaluation_fn = evaluation_fn
        self.device = device

        # Extract swarm parameters - handle SwarmGraph (config) and nn.Module (direct attrs)
        self.num_agents = len(swarm.agents)
        if hasattr(swarm, 'config'):
            self.input_dim = swarm.config.input_dim
            self.output_dim = swarm.config.output_dim
            self.hidden_dim = swarm.config.hidden_dim
        else:
            self.input_dim = getattr(swarm, 'input_dim', 137)
            self.output_dim = getattr(swarm, 'output_dim', 5)
            self.hidden_dim = getattr(swarm, 'hidden_dim', 128)

        # Get total params - SwarmGraph uses total_parameters property
        if hasattr(swarm, 'total_parameters'):
            self.total_params = swarm.total_parameters
        elif hasattr(swarm, 'parameters'):
            self.total_params = sum(p.numel() for p in swarm.parameters())
        else:
            self.total_params = 100000  # default fallback

        self.results: Dict[str, BaselineResult] = {}

    def create_baselines(self) -> Dict[str, nn.Module]:
        """Create freshly initialized diagnostic baseline models."""
        baselines = {}

        # Single large agent (same params)
        baselines['single_agent'] = SingleAgentBaseline(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            total_params=self.total_params,
        )

        # Ensemble (same structure, no communication)
        baselines['ensemble'] = EnsembleBaseline(
            num_agents=self.num_agents,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
        )

        # Centralized controller
        baselines['centralized'] = CentralizedBaseline(
            num_agents=self.num_agents,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim * 2,  # Give it more capacity
        )

        # Independent agents
        baselines['independent'] = IndependentAgentsBaseline(
            num_agents=self.num_agents,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
        )

        # Random baseline
        baselines['random'] = RandomBaseline(self.output_dim)

        return baselines

    def compare(
        self,
        num_seeds: int = 10,
        metric: str = 'reward',
        verbose: bool = True,
    ) -> Dict[str, BaselineResult]:
        """
        Compare a swarm against fresh, untrained diagnostic baselines.

        These descriptive smoke-test outputs are not a trained performance
        benchmark or evidence of architectural superiority.

        Args:
            num_seeds: Number of random seeds
            metric: Which metric to compare
            verbose: Print progress

        Returns:
            Dictionary of baseline name -> comparison result.
        """
        from scipy import stats

        baselines = self.create_baselines()

        # Evaluate swarm
        if verbose:
            print("Evaluating swarm...")

        swarm_scores = []
        for seed in range(num_seeds):
            metrics = self.evaluation_fn(self.swarm, seed)
            swarm_scores.append(metrics.get(metric, 0.0))

        swarm_scores = np.array(swarm_scores)

        # Evaluate each baseline
        for name, baseline in baselines.items():
            if verbose:
                print(f"Evaluating baseline: {name}...")

            baseline = baseline.to(self.device)
            baseline_scores = []

            for seed in range(num_seeds):
                torch.manual_seed(seed)
                metrics = self.evaluation_fn(baseline, seed)
                baseline_scores.append(metrics.get(metric, 0.0))

            baseline_scores = np.array(baseline_scores)

            # Compute comparison
            delta = swarm_scores.mean() - baseline_scores.mean()
            relative = delta / (abs(baseline_scores.mean()) + 1e-10)

            # Count wins
            wins = np.sum(swarm_scores > baseline_scores)

            # Statistical test
            _, p_value = stats.ttest_rel(swarm_scores, baseline_scores)

            result = BaselineResult(
                baseline_name=name,
                baseline_score=baseline_scores.mean(),
                swarm_score=swarm_scores.mean(),
                delta=delta,
                relative_improvement=relative,
                wins_out_of=(int(wins), num_seeds),
                baseline_std=baseline_scores.std(),
                swarm_std=swarm_scores.std(),
                p_value=p_value,
            )

            self.results[name] = result

            if verbose:
                print(f"  {name}: swarm wins {wins}/{num_seeds}, delta={delta:+.4f}")

        return self.results

    def summary(self) -> str:
        """Generate summary report."""
        lines = [
            "\n" + "=" * 70,
            "UNTRAINED BASELINE DIAGNOSTIC SUMMARY",
            "=" * 70,
            "",
            f"Swarm: {self.num_agents} agents, {self.total_params:,} parameters",
            "",
            f"{'Baseline':<20} {'Score':>10} {'Swarm':>10} {'Delta':>10} {'Wins':>10} {'p-value':>10}",
            "-" * 70,
        ]

        for name, result in self.results.items():
            sig = "***" if result.p_value < 0.001 else "**" if result.p_value < 0.01 else "*" if result.p_value < 0.05 else ""
            lines.append(
                f"{name:<20} {result.baseline_score:>10.4f} {result.swarm_score:>10.4f} "
                f"{result.delta:>+10.4f} {result.wins_out_of[0]}/{result.wins_out_of[1]:>5} "
                f"{result.p_value:>9.4f}{sig}"
            )

        # Descriptive count only; these controls have not been trained.
        lines.append("-" * 70)

        must_beat = ['single_agent', 'ensemble', 'independent', 'centralized']
        beaten = sum(1 for name in must_beat if name in self.results and self.results[name].delta > 0)

        verdict = (
            f"Descriptive only: swarm scored higher than {beaten}/{len(must_beat)} "
            "freshly initialized controls; no superiority claim is supported"
        )

        lines.extend(["", verdict, "=" * 70])

        return "\n".join(lines)


def quick_baseline_check(
    swarm: SwarmGraph,
    test_input: torch.Tensor,
    test_target: torch.Tensor,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Quick baseline comparison on single batch.

    Returns MSE for each model.
    """
    results = {}

    swarm = swarm.to(device)
    test_input = test_input.to(device)
    test_target = test_target.to(device)

    # Swarm
    with torch.no_grad():
        out = swarm(test_input)
        results['swarm'] = ((out - test_target) ** 2).mean().item()

    # Create baselines
    total_params = sum(p.numel() for p in swarm.parameters())

    single = SingleAgentBaseline(
        input_dim=test_input.shape[-1],
        output_dim=test_target.shape[-1],
        total_params=total_params,
    ).to(device)

    with torch.no_grad():
        out = single(test_input)
        results['single_agent'] = ((out - test_target) ** 2).mean().item()

    ensemble = EnsembleBaseline(
        num_agents=len(swarm.agents),
        input_dim=test_input.shape[-1],
        output_dim=test_target.shape[-1],
    ).to(device)

    with torch.no_grad():
        out = ensemble(test_input)
        results['ensemble'] = ((out - test_target) ** 2).mean().item()

    return results
