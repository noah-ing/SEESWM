"""
Ablation Study Framework.

Systematically test which components of SEESWM actually matter for performance.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum, auto
import numpy as np
from copy import deepcopy

from ..agents.micro_agent import MicroAgent, AgentConfig
from ..swarm.graph import SwarmGraph, SwarmConfig, TopologyType
from ..world_model.jepa import WorldModel


class AblationType(Enum):
    """Types of ablations to test."""
    NO_SWARM = auto()  # Single large agent with same params
    NO_WORLD_MODEL = auto()  # Remove curiosity signal
    NO_NEUROMOD = auto()  # Fixed learning rate
    NO_MESSAGE_PASSING = auto()  # Isolated agents
    RANDOM_TOPOLOGY = auto()  # Random vs structured topology
    NO_SPECIALIZATION = auto()  # All general agents
    REDUCED_AGENTS = auto()  # Fewer agents, same total params
    NO_MEMORY = auto()  # Remove agent memory/state
    FULL_CONNECTIVITY = auto()  # All-to-all vs sparse
    ABLATE_AGENT_TYPE = auto()  # Remove specific agent types


@dataclass
class AblationConfig:
    """Configuration for an ablation study."""
    name: str
    ablation_type: AblationType
    description: str

    # Parameters for the ablation
    params: Dict[str, Any] = field(default_factory=dict)

    # Metrics to track
    metrics: List[str] = field(default_factory=lambda: [
        'reward', 'synergy', 'coverage', 'survival_time'
    ])

    # Number of seeds for statistical significance
    num_seeds: int = 10


@dataclass
class AblationResult:
    """Results from an ablation experiment."""
    config: AblationConfig

    # Performance metrics
    baseline_scores: np.ndarray  # Shape: (num_seeds, num_metrics)
    ablated_scores: np.ndarray

    # Derived statistics
    mean_delta: np.ndarray
    std_delta: np.ndarray
    p_values: np.ndarray
    effect_sizes: np.ndarray  # Cohen's d

    # Additional info
    training_curves: Optional[Dict[str, List[float]]] = None


class SingleLargeAgent(nn.Module):
    """A single large agent with equivalent parameter count to swarm."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        total_params: int,
        hidden_layers: int = 4,
    ):
        super().__init__()

        # Calculate hidden dim to match parameter count
        # For MLP: params ≈ input_dim * hidden + (layers-1) * hidden^2 + hidden * output
        # Solve quadratic: (layers-1) * h^2 + (input + output) * h - total_params = 0
        a = hidden_layers - 1
        b = input_dim + output_dim
        c = -total_params

        if a > 0:
            hidden_dim = int((-b + np.sqrt(b**2 - 4*a*c)) / (2*a))
        else:
            hidden_dim = total_params // (input_dim + output_dim)

        hidden_dim = max(32, min(hidden_dim, 1024))  # Clamp to reasonable range

        layers = []
        prev_dim = input_dim

        for i in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.hidden_dim = hidden_dim

        actual_params = sum(p.numel() for p in self.parameters())
        print(f"SingleLargeAgent: target={total_params}, actual={actual_params}, hidden={hidden_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class IsolatedSwarm(nn.Module):
    """Swarm with no message passing between agents.

    This wrapper works with both nn.Module swarms and SwarmGraph.
    For SwarmGraph: uses step/reset pattern
    For nn.Module: acts as callable wrapper
    """

    def __init__(self, base_swarm):
        super().__init__()
        self.swarm = deepcopy(base_swarm)
        self.agents = self.swarm.agents
        self._is_swarm_graph = hasattr(base_swarm, 'step') and hasattr(base_swarm, 'reset')

    def reset(self, batch_size: int = 1) -> None:
        """Reset the swarm (for SwarmGraph compatibility)."""
        if self._is_swarm_graph:
            self.swarm.reset(batch_size)

    def step(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward for SwarmGraph pattern."""
        self.swarm.reset(batch_size=observations.shape[0])
        return self.swarm.step(observations)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward for nn.Module pattern."""
        if self._is_swarm_graph:
            return self.step(x)
        # For nn.Module swarms, just call the wrapped swarm
        return self.swarm(x, **kwargs)


class NoMemorySwarm(nn.Module):
    """Swarm with agent memory/state disabled.

    This wrapper works with both nn.Module swarms and SwarmGraph.
    """

    def __init__(self, base_swarm):
        super().__init__()
        self.swarm = deepcopy(base_swarm)
        self.agents = self.swarm.agents
        self._is_swarm_graph = hasattr(base_swarm, 'step') and hasattr(base_swarm, 'reset')

    def reset(self, batch_size: int = 1) -> None:
        """Reset the swarm (for SwarmGraph compatibility)."""
        if self._is_swarm_graph:
            self.swarm.reset(batch_size)

    def step(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward with memory cleared each step (for SwarmGraph)."""
        batch_size = observations.shape[0]
        agents = self.swarm.agents
        if isinstance(agents, dict):
            for agent in agents.values():
                if hasattr(agent, 'reset_state'):
                    agent.reset_state(batch_size)
        return self.swarm.step(observations)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward for nn.Module pattern."""
        if self._is_swarm_graph:
            return self.step(x)
        return self.swarm(x, **kwargs)


class AblationStudy:
    """Framework for running systematic ablation studies."""

    def __init__(
        self,
        base_config: Dict[str, Any],
        evaluation_fn: Callable,
        device: str = "cpu",
    ):
        """
        Args:
            base_config: Configuration for creating base swarm
            evaluation_fn: Function that evaluates a model and returns metrics
            device: Device to run experiments on
        """
        self.base_config = base_config
        self.evaluation_fn = evaluation_fn
        self.device = device

        self.results: Dict[str, AblationResult] = {}

    def create_base_swarm(self, seed: int) -> SwarmGraph:
        """Create the baseline swarm."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        cfg = self.base_config
        num_agents = cfg.get('num_agents', 20)

        swarm_config = SwarmConfig(
            num_agents=num_agents,
            input_dim=cfg.get('input_dim', 137),
            hidden_dim=cfg.get('hidden_dim', 128),
            output_dim=cfg.get('output_dim', 5),
            message_dim=cfg.get('hidden_dim', 128),
            topology=cfg.get('topology', TopologyType.SMALL_WORLD),
            num_perception=num_agents // 4,
            num_reasoning=num_agents // 4,
            num_memory=num_agents // 4,
            num_planning=num_agents - 3 * (num_agents // 4),
        )
        return SwarmGraph(swarm_config, device=self.device)

    def _create_swarm_config(
        self,
        num_agents: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        topology: TopologyType,
    ) -> SwarmConfig:
        """Helper to create SwarmConfig."""
        return SwarmConfig(
            num_agents=num_agents,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            message_dim=hidden_dim,
            topology=topology,
            num_perception=num_agents // 4,
            num_reasoning=num_agents // 4,
            num_memory=num_agents // 4,
            num_planning=num_agents - 3 * (num_agents // 4),
        )

    def create_ablated_model(
        self,
        ablation: AblationConfig,
        seed: int,
    ) -> nn.Module:
        """Create an ablated version of the model."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        cfg = self.base_config
        ablation_type = ablation.ablation_type
        num_agents = cfg.get('num_agents', 20)

        if ablation_type == AblationType.NO_SWARM:
            # Single large agent
            base_swarm = self.create_base_swarm(seed)
            total_params = base_swarm.total_parameters

            return SingleLargeAgent(
                input_dim=cfg.get('input_dim', 137),
                output_dim=cfg.get('output_dim', 5),
                total_params=total_params,
            )

        elif ablation_type == AblationType.NO_MESSAGE_PASSING:
            base_swarm = self.create_base_swarm(seed)
            return IsolatedSwarm(base_swarm)

        elif ablation_type == AblationType.NO_MEMORY:
            base_swarm = self.create_base_swarm(seed)
            return NoMemorySwarm(base_swarm)

        elif ablation_type == AblationType.RANDOM_TOPOLOGY:
            swarm_cfg = self._create_swarm_config(
                num_agents=num_agents,
                input_dim=cfg.get('input_dim', 137),
                hidden_dim=cfg.get('hidden_dim', 128),
                output_dim=cfg.get('output_dim', 5),
                topology=TopologyType.RANDOM,
            )
            return SwarmGraph(swarm_cfg, device=self.device)

        elif ablation_type == AblationType.FULL_CONNECTIVITY:
            swarm_cfg = self._create_swarm_config(
                num_agents=num_agents,
                input_dim=cfg.get('input_dim', 137),
                hidden_dim=cfg.get('hidden_dim', 128),
                output_dim=cfg.get('output_dim', 5),
                topology=TopologyType.FULLY_CONNECTED,
            )
            return SwarmGraph(swarm_cfg, device=self.device)

        elif ablation_type == AblationType.REDUCED_AGENTS:
            # Half the agents, larger hidden dim
            reduced_agents = num_agents // 2
            swarm_cfg = self._create_swarm_config(
                num_agents=reduced_agents,
                input_dim=cfg.get('input_dim', 137),
                hidden_dim=int(cfg.get('hidden_dim', 128) * 1.4),
                output_dim=cfg.get('output_dim', 5),
                topology=cfg.get('topology', TopologyType.SMALL_WORLD),
            )
            return SwarmGraph(swarm_cfg, device=self.device)

        else:
            # Default: return base swarm
            return self.create_base_swarm(seed)

    def run_ablation(
        self,
        ablation: AblationConfig,
        verbose: bool = True,
    ) -> AblationResult:
        """Run a single ablation study."""
        if verbose:
            print(f"\n{'='*60}")
            print(f"Running ablation: {ablation.name}")
            print(f"Type: {ablation.ablation_type.name}")
            print(f"Description: {ablation.description}")
            print(f"{'='*60}")

        num_metrics = len(ablation.metrics)
        baseline_scores = np.zeros((ablation.num_seeds, num_metrics))
        ablated_scores = np.zeros((ablation.num_seeds, num_metrics))

        for seed in range(ablation.num_seeds):
            if verbose:
                print(f"\n  Seed {seed + 1}/{ablation.num_seeds}")

            # Evaluate baseline
            base_model = self.create_base_swarm(seed)
            if hasattr(base_model, 'to'):
                base_model = base_model.to(self.device)
            base_metrics = self.evaluation_fn(base_model, seed)

            for i, metric in enumerate(ablation.metrics):
                baseline_scores[seed, i] = base_metrics.get(metric, 0.0)

            # Evaluate ablated
            ablated_model = self.create_ablated_model(ablation, seed)
            if hasattr(ablated_model, 'to'):
                ablated_model = ablated_model.to(self.device)
            ablated_metrics = self.evaluation_fn(ablated_model, seed)

            for i, metric in enumerate(ablation.metrics):
                ablated_scores[seed, i] = ablated_metrics.get(metric, 0.0)

            if verbose:
                print(f"    Baseline: {base_metrics}")
                print(f"    Ablated:  {ablated_metrics}")

        # Compute statistics
        mean_delta = np.mean(ablated_scores - baseline_scores, axis=0)
        std_delta = np.std(ablated_scores - baseline_scores, axis=0)

        # Paired t-test
        from scipy import stats
        p_values = np.zeros(num_metrics)
        effect_sizes = np.zeros(num_metrics)

        for i in range(num_metrics):
            _, p_values[i] = stats.ttest_rel(
                baseline_scores[:, i],
                ablated_scores[:, i]
            )

            # Cohen's d
            diff = baseline_scores[:, i] - ablated_scores[:, i]
            effect_sizes[i] = np.mean(diff) / (np.std(diff) + 1e-8)

        result = AblationResult(
            config=ablation,
            baseline_scores=baseline_scores,
            ablated_scores=ablated_scores,
            mean_delta=mean_delta,
            std_delta=std_delta,
            p_values=p_values,
            effect_sizes=effect_sizes,
        )

        self.results[ablation.name] = result
        return result

    def summarize_results(self) -> str:
        """Generate summary of all ablation results."""
        lines = [
            "\n" + "=" * 80,
            "ABLATION STUDY SUMMARY",
            "=" * 80,
        ]

        for name, result in self.results.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Description: {result.config.description}")

            for i, metric in enumerate(result.config.metrics):
                delta = result.mean_delta[i]
                std = result.std_delta[i]
                p = result.p_values[i]
                d = result.effect_sizes[i]

                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

                lines.append(
                    f"  {metric}: Δ = {delta:+.4f} ± {std:.4f} "
                    f"(p={p:.4f}{sig}, d={d:.2f})"
                )

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)


def run_ablation_suite(
    base_config: Dict[str, Any],
    evaluation_fn: Callable,
    device: str = "cpu",
    num_seeds: int = 10,
) -> Dict[str, AblationResult]:
    """
    Run the complete ablation suite.

    Returns:
        Dictionary mapping ablation names to results.
    """
    study = AblationStudy(base_config, evaluation_fn, device)

    ablations = [
        AblationConfig(
            name="no_swarm",
            ablation_type=AblationType.NO_SWARM,
            description="Single large agent with same total parameters",
            num_seeds=num_seeds,
        ),
        AblationConfig(
            name="no_message_passing",
            ablation_type=AblationType.NO_MESSAGE_PASSING,
            description="Agents process independently, no communication",
            num_seeds=num_seeds,
        ),
        AblationConfig(
            name="no_memory",
            ablation_type=AblationType.NO_MEMORY,
            description="Agent state/memory cleared each step",
            num_seeds=num_seeds,
        ),
        AblationConfig(
            name="random_topology",
            ablation_type=AblationType.RANDOM_TOPOLOGY,
            description="Random graph instead of small-world",
            num_seeds=num_seeds,
        ),
        AblationConfig(
            name="full_connectivity",
            ablation_type=AblationType.FULL_CONNECTIVITY,
            description="All-to-all connectivity instead of sparse",
            num_seeds=num_seeds,
        ),
        AblationConfig(
            name="reduced_agents",
            ablation_type=AblationType.REDUCED_AGENTS,
            description="Half agents, larger each (same total params)",
            num_seeds=num_seeds,
        ),
    ]

    for ablation in ablations:
        study.run_ablation(ablation)

    print(study.summarize_results())

    return study.results
