"""
Scaling Law Experiments.

Test whether collective intelligence emerges at scale, following
Chinchilla-style analysis of compute vs performance.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from scipy import stats
from scipy.optimize import curve_fit

from ..swarm.graph import SwarmGraph, SwarmConfig, TopologyType


@dataclass
class ScalingPoint:
    """Data point for scaling analysis."""
    num_agents: int
    hidden_dim: int
    total_params: int
    message_rounds: int

    # Metrics
    performance: float
    synergy: float
    training_steps: int

    # Costs
    compute_flops: float
    memory_bytes: int


@dataclass
class ScalingLaw:
    """Fitted scaling law parameters."""
    # Power law: performance = A * N^alpha + B
    coefficient_A: float
    exponent_alpha: float
    constant_B: float

    # Goodness of fit
    r_squared: float
    residual_std: float

    # Phase transition detection
    phase_transition_point: Optional[float] = None
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


class ScalingExperiment:
    """Framework for scaling law experiments."""

    def __init__(
        self,
        base_input_dim: int = 137,
        base_output_dim: int = 5,
        evaluation_fn: Optional[Callable] = None,
        device: str = "cpu",
    ):
        self.base_input_dim = base_input_dim
        self.base_output_dim = base_output_dim
        self.evaluation_fn = evaluation_fn
        self.device = device

        self.data_points: List[ScalingPoint] = []

    def estimate_flops(
        self,
        num_agents: int,
        hidden_dim: int,
        message_rounds: int,
    ) -> float:
        """Estimate FLOPs for one forward pass."""
        input_dim = self.base_input_dim
        output_dim = self.base_output_dim

        # Per-agent: 2 * input * hidden + 2 * hidden * hidden + 2 * hidden * output
        agent_flops = (
            2 * input_dim * hidden_dim +  # Input layer
            2 * hidden_dim * hidden_dim +  # Hidden layer
            2 * hidden_dim * output_dim    # Output layer
        )

        # Message passing: each round, each agent aggregates from neighbors
        # Assume average degree ~4 for small-world
        avg_degree = 4
        message_flops = num_agents * avg_degree * hidden_dim * 2  # aggregate + transform

        total = num_agents * agent_flops + message_rounds * message_flops

        return float(total)

    def run_scaling_sweep(
        self,
        agent_counts: List[int] = [5, 10, 20, 50, 100, 200, 500],
        hidden_dims: List[int] = [64, 128, 256],
        message_rounds: List[int] = [1, 3, 5],
        num_seeds: int = 5,
        training_steps: int = 1000,
    ) -> List[ScalingPoint]:
        """Run comprehensive scaling sweep."""
        print("Running scaling sweep...")
        print(f"  Agent counts: {agent_counts}")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Message rounds: {message_rounds}")
        print(f"  Seeds per config: {num_seeds}")

        for n_agents in agent_counts:
            for h_dim in hidden_dims:
                for m_rounds in message_rounds:
                    print(f"\n  Testing: {n_agents} agents, dim={h_dim}, rounds={m_rounds}")

                    performances = []
                    synergies = []

                    for seed in range(num_seeds):
                        torch.manual_seed(seed)
                        np.random.seed(seed)

                        # Create swarm
                        swarm_config = SwarmConfig(
                            num_agents=n_agents,
                            input_dim=self.base_input_dim,
                            hidden_dim=h_dim,
                            output_dim=self.base_output_dim,
                            message_dim=h_dim,
                            topology=TopologyType.SMALL_WORLD,
                            num_perception=n_agents // 4,
                            num_reasoning=n_agents // 4,
                            num_memory=n_agents // 4,
                            num_planning=n_agents - 3 * (n_agents // 4),
                        )
                        swarm = SwarmGraph(swarm_config, device=self.device)

                        # Evaluate
                        if self.evaluation_fn:
                            metrics = self.evaluation_fn(swarm, seed)
                            performances.append(metrics.get('performance', 0.0))
                            synergies.append(metrics.get('synergy', 0.0))
                        else:
                            # Default: random task performance
                            x = torch.randn(1, self.base_input_dim, device=self.device)
                            swarm.reset(batch_size=1)
                            with torch.no_grad():
                                out = swarm.step(x)
                            performances.append(out.mean().item())
                            synergies.append(0.0)

                    # Compute statistics
                    total_params = swarm.total_parameters
                    flops = self.estimate_flops(n_agents, h_dim, m_rounds)
                    memory = total_params * 4  # float32

                    point = ScalingPoint(
                        num_agents=n_agents,
                        hidden_dim=h_dim,
                        total_params=total_params,
                        message_rounds=m_rounds,
                        performance=np.mean(performances),
                        synergy=np.mean(synergies),
                        training_steps=training_steps,
                        compute_flops=flops,
                        memory_bytes=memory,
                    )

                    self.data_points.append(point)
                    print(f"    Params: {total_params:,}, Perf: {point.performance:.4f}")

        return self.data_points

    def fit_scaling_law(
        self,
        metric: str = 'performance',
        scale_var: str = 'num_agents',
    ) -> ScalingLaw:
        """Fit power law to scaling data."""

        # Extract data
        x_data = np.array([getattr(p, scale_var) for p in self.data_points])
        y_data = np.array([getattr(p, metric) for p in self.data_points])

        # Power law: y = A * x^alpha + B
        def power_law(x, A, alpha, B):
            return A * np.power(x, alpha) + B

        # Initial guess
        p0 = [1.0, 0.5, 0.0]

        try:
            popt, pcov = curve_fit(
                power_law, x_data, y_data, p0=p0,
                maxfev=10000, bounds=([-np.inf, -2, -np.inf], [np.inf, 2, np.inf])
            )

            # Compute R²
            y_pred = power_law(x_data, *popt)
            ss_res = np.sum((y_data - y_pred) ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # Confidence interval for exponent
            perr = np.sqrt(np.diag(pcov))

            scaling_law = ScalingLaw(
                coefficient_A=popt[0],
                exponent_alpha=popt[1],
                constant_B=popt[2],
                r_squared=r_squared,
                residual_std=np.std(y_data - y_pred),
                confidence_interval=(popt[1] - 1.96*perr[1], popt[1] + 1.96*perr[1]),
            )

        except Exception as e:
            print(f"Warning: Could not fit scaling law: {e}")
            scaling_law = ScalingLaw(
                coefficient_A=0, exponent_alpha=0, constant_B=np.mean(y_data),
                r_squared=0, residual_std=np.std(y_data),
            )

        return scaling_law


def find_phase_transitions(
    data_points: List[ScalingPoint],
    metric: str = 'synergy',
    window_size: int = 3,
) -> List[Tuple[int, float]]:
    """
    Detect phase transitions in scaling behavior.

    Returns list of (scale_point, transition_magnitude) tuples.
    """
    # Sort by num_agents
    sorted_points = sorted(data_points, key=lambda p: p.num_agents)

    values = [getattr(p, metric) for p in sorted_points]
    scales = [p.num_agents for p in sorted_points]

    if len(values) < 2 * window_size:
        return []

    transitions = []

    # Compute moving average of gradient
    gradients = np.diff(values) / np.diff(scales)

    for i in range(window_size, len(gradients) - window_size):
        # Compare gradient before and after
        before = np.mean(gradients[i-window_size:i])
        after = np.mean(gradients[i:i+window_size])

        # Large change in gradient indicates phase transition
        change = abs(after - before)
        threshold = 2 * np.std(gradients)

        if change > threshold:
            transitions.append((scales[i], change))

    return transitions


def plot_scaling_laws(
    data_points: List[ScalingPoint],
    scaling_law: Optional[ScalingLaw] = None,
    save_path: Optional[str] = None,
) -> None:
    """Generate scaling law plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Performance vs agents
    ax = axes[0, 0]
    agents = [p.num_agents for p in data_points]
    perfs = [p.performance for p in data_points]
    ax.scatter(agents, perfs, alpha=0.7)
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Performance')
    ax.set_xscale('log')
    ax.set_title('Performance Scaling')

    if scaling_law:
        x_fit = np.linspace(min(agents), max(agents), 100)
        y_fit = (scaling_law.coefficient_A *
                 np.power(x_fit, scaling_law.exponent_alpha) +
                 scaling_law.constant_B)
        ax.plot(x_fit, y_fit, 'r--',
                label=f'Fit: y = {scaling_law.coefficient_A:.2f}x^{scaling_law.exponent_alpha:.2f}')
        ax.legend()

    # Synergy vs agents
    ax = axes[0, 1]
    synergies = [p.synergy for p in data_points]
    ax.scatter(agents, synergies, alpha=0.7, c='green')
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Synergy')
    ax.set_xscale('log')
    ax.set_title('Synergy Scaling')

    # Performance vs compute
    ax = axes[1, 0]
    flops = [p.compute_flops for p in data_points]
    ax.scatter(flops, perfs, alpha=0.7, c='orange')
    ax.set_xlabel('FLOPs')
    ax.set_ylabel('Performance')
    ax.set_xscale('log')
    ax.set_title('Compute Efficiency')

    # Performance vs params
    ax = axes[1, 1]
    params = [p.total_params for p in data_points]
    ax.scatter(params, perfs, alpha=0.7, c='purple')
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Performance')
    ax.set_xscale('log')
    ax.set_title('Parameter Efficiency')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved scaling plot to {save_path}")
    else:
        plt.show()


def compute_optimal_allocation(
    compute_budget: float,
    data_points: List[ScalingPoint],
) -> Dict[str, int]:
    """
    Given compute budget, find optimal agent count and size.

    Follows Chinchilla-style analysis.
    """
    # Group by compute level
    within_budget = [p for p in data_points if p.compute_flops <= compute_budget]

    if not within_budget:
        # Use smallest configuration
        return {
            'num_agents': min(p.num_agents for p in data_points),
            'hidden_dim': min(p.hidden_dim for p in data_points),
            'message_rounds': 1,
        }

    # Find best performing within budget
    best = max(within_budget, key=lambda p: p.performance)

    return {
        'num_agents': best.num_agents,
        'hidden_dim': best.hidden_dim,
        'message_rounds': best.message_rounds,
    }
