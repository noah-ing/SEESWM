#!/usr/bin/env python3
"""
Hypothesis Validation Experiments for SEESWM.

Core hypothesis: "Collective intelligence from many small specialized agents,
grounded in world simulation, will exhibit emergent capabilities that
equivalent-parameter monolithic models cannot."

This script runs controlled experiments to test:
1. Synergy: Does the swarm outperform individual agents?
2. Scaling: Does synergy increase with more agents?
3. Topology: Which graph structure produces best emergent behavior?
4. Specialization: Do typed agents outperform generic ones?
5. Curiosity: Does world-model curiosity improve exploration?

Results are saved to JSON for analysis.
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.swarm.graph import SwarmGraph, SwarmConfig, TopologyType
from src.agents.micro_agent import MicroAgent, AgentConfig, AgentType
from src.environment.cosmos import CosmosEnvironment, EnvironmentConfig


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_name: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    duration_seconds: float
    timestamp: str


def create_baseline_mlp(input_dim: int, hidden_dim: int, output_dim: int,
                        total_params: int) -> nn.Module:
    """
    Create a monolithic MLP with approximately the same parameter count as a swarm.
    This is our baseline to compare against.
    """
    # Calculate layers needed to match param count
    # params = input_dim * h + h + h * h + h + ... + h * output_dim + output_dim
    # Simplified: aim for similar capacity

    layers = []
    current_dim = input_dim

    # Estimate depth needed
    params_per_layer = hidden_dim * hidden_dim + hidden_dim
    num_hidden = max(1, total_params // params_per_layer - 1)
    num_hidden = min(num_hidden, 10)  # Cap at 10 layers

    # Input layer
    layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])

    # Hidden layers
    for _ in range(num_hidden):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

    # Output layer
    layers.append(nn.Linear(hidden_dim, output_dim))

    model = nn.Sequential(*layers)
    actual_params = sum(p.numel() for p in model.parameters())
    print(f"  Baseline MLP: {actual_params:,} params ({num_hidden} hidden layers)")

    return model


def experiment_synergy_vs_baseline(
    num_trials: int = 10,
    num_agents: int = 20,
    device: str = "cpu",
) -> ExperimentResult:
    """
    Experiment 1: Compare swarm performance to equivalent-parameter MLP.

    Tests whether collective intelligence emerges from the swarm architecture.
    """
    print("\n" + "="*60)
    print("Experiment 1: Synergy vs Baseline MLP")
    print("="*60)

    start_time = time.time()

    input_dim, hidden_dim, output_dim = 32, 64, 16

    # Create swarm
    config = SwarmConfig(
        num_agents=num_agents,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        topology=TopologyType.SMALL_WORLD,
    )
    swarm = SwarmGraph(config, device=device)
    swarm_params = swarm.total_parameters
    print(f"  Swarm: {swarm_params:,} params, {num_agents} agents")

    # Create baseline MLP with similar param count
    baseline = create_baseline_mlp(input_dim, hidden_dim, output_dim, swarm_params)
    baseline = baseline.to(device)

    # Run trials
    swarm_errors = []
    baseline_errors = []
    synergies = []

    for trial in range(num_trials):
        # Random regression task
        X = torch.randn(32, input_dim, device=device)
        # Target: non-linear function
        y = torch.sin(X[:, :output_dim]) + 0.1 * torch.randn(32, output_dim, device=device)

        with torch.no_grad():
            # Swarm prediction
            swarm_pred = swarm.step(X)
            swarm_error = F.mse_loss(swarm_pred, y).item()
            swarm_errors.append(swarm_error)

            # Baseline prediction
            baseline_pred = baseline(X)
            baseline_error = F.mse_loss(baseline_pred, y).item()
            baseline_errors.append(baseline_error)

            # Synergy computation
            synergy_result = swarm.compute_synergy(X, y)
            synergies.append(synergy_result['synergy'])

    metrics = {
        "swarm_error_mean": np.mean(swarm_errors),
        "swarm_error_std": np.std(swarm_errors),
        "baseline_error_mean": np.mean(baseline_errors),
        "baseline_error_std": np.std(baseline_errors),
        "synergy_mean": np.mean(synergies),
        "synergy_std": np.std(synergies),
        "swarm_params": swarm_params,
        "baseline_params": sum(p.numel() for p in baseline.parameters()),
        "swarm_wins": sum(1 for s, b in zip(swarm_errors, baseline_errors) if s < b),
    }

    print(f"\nResults ({num_trials} trials):")
    print(f"  Swarm MSE:    {metrics['swarm_error_mean']:.4f} ± {metrics['swarm_error_std']:.4f}")
    print(f"  Baseline MSE: {metrics['baseline_error_mean']:.4f} ± {metrics['baseline_error_std']:.4f}")
    print(f"  Synergy:      {metrics['synergy_mean']:.4f} ± {metrics['synergy_std']:.4f}")
    print(f"  Swarm wins:   {metrics['swarm_wins']}/{num_trials}")

    return ExperimentResult(
        experiment_name="synergy_vs_baseline",
        config={"num_agents": num_agents, "num_trials": num_trials},
        metrics=metrics,
        duration_seconds=time.time() - start_time,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def experiment_scaling_synergy(
    agent_counts: List[int] = [5, 10, 20, 50, 100],
    num_trials: int = 5,
    device: str = "cpu",
) -> ExperimentResult:
    """
    Experiment 2: How does synergy scale with number of agents?

    Tests whether more agents lead to more emergent capability.
    """
    print("\n" + "="*60)
    print("Experiment 2: Scaling Synergy with Agent Count")
    print("="*60)

    start_time = time.time()

    input_dim, hidden_dim, output_dim = 32, 64, 16
    results_by_count = {}

    for num_agents in agent_counts:
        print(f"\n  Testing {num_agents} agents...")

        config = SwarmConfig(
            num_agents=num_agents,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            topology=TopologyType.SMALL_WORLD,
        )
        swarm = SwarmGraph(config, device=device)

        synergies = []
        collective_perfs = []
        individual_perfs = []

        for _ in range(num_trials):
            X = torch.randn(32, input_dim, device=device)
            y = torch.sin(X[:, :output_dim]) + 0.1 * torch.randn(32, output_dim, device=device)

            with torch.no_grad():
                result = swarm.compute_synergy(X, y)
                synergies.append(result['synergy'])
                collective_perfs.append(result['collective_performance'])
                individual_perfs.append(result['avg_individual_performance'])

        results_by_count[num_agents] = {
            "synergy_mean": np.mean(synergies),
            "synergy_std": np.std(synergies),
            "collective_perf_mean": np.mean(collective_perfs),
            "individual_perf_mean": np.mean(individual_perfs),
            "params": swarm.total_parameters,
        }

        print(f"    Synergy: {results_by_count[num_agents]['synergy_mean']:.4f}")
        print(f"    Params:  {swarm.total_parameters:,}")

    # Compute scaling efficiency
    base_synergy = results_by_count[agent_counts[0]]['synergy_mean']
    scaling_efficiency = []
    for n in agent_counts[1:]:
        ratio = results_by_count[n]['synergy_mean'] / (base_synergy + 1e-8)
        agent_ratio = n / agent_counts[0]
        efficiency = ratio / agent_ratio  # >1 means superlinear scaling
        scaling_efficiency.append(efficiency)

    metrics = {
        "results_by_count": results_by_count,
        "agent_counts": agent_counts,
        "scaling_efficiency": scaling_efficiency,
        "best_agent_count": max(results_by_count.keys(),
                                 key=lambda k: results_by_count[k]['synergy_mean']),
    }

    print(f"\nScaling Summary:")
    print(f"  Best agent count: {metrics['best_agent_count']}")
    print(f"  Scaling efficiency (vs {agent_counts[0]} agents): {scaling_efficiency}")

    return ExperimentResult(
        experiment_name="scaling_synergy",
        config={"agent_counts": agent_counts, "num_trials": num_trials},
        metrics=metrics,
        duration_seconds=time.time() - start_time,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def experiment_topology_comparison(
    num_agents: int = 30,
    num_trials: int = 10,
    device: str = "cpu",
) -> ExperimentResult:
    """
    Experiment 3: Which topology produces best emergent behavior?

    Compares different graph structures for their synergy.
    """
    print("\n" + "="*60)
    print("Experiment 3: Topology Comparison")
    print("="*60)

    start_time = time.time()

    input_dim, hidden_dim, output_dim = 32, 64, 16
    topologies = [
        TopologyType.RANDOM,
        TopologyType.SMALL_WORLD,
        TopologyType.SCALE_FREE,
        TopologyType.MODULAR,
        TopologyType.HIERARCHICAL,
        TopologyType.FULLY_CONNECTED,
    ]

    results_by_topology = {}

    for topo in topologies:
        print(f"\n  Testing {topo.name}...")

        config = SwarmConfig(
            num_agents=num_agents,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            topology=topo,
        )
        swarm = SwarmGraph(config, device=device)

        synergies = []
        errors = []

        for _ in range(num_trials):
            X = torch.randn(32, input_dim, device=device)
            y = torch.sin(X[:, :output_dim]) + 0.1 * torch.randn(32, output_dim, device=device)

            with torch.no_grad():
                pred = swarm.step(X)
                error = F.mse_loss(pred, y).item()
                errors.append(error)

                result = swarm.compute_synergy(X, y)
                synergies.append(result['synergy'])

        topo_stats = swarm.get_topology_stats()

        results_by_topology[topo.name] = {
            "synergy_mean": np.mean(synergies),
            "synergy_std": np.std(synergies),
            "error_mean": np.mean(errors),
            "clustering": topo_stats.get('clustering_coefficient', 0),
            "diameter": topo_stats.get('diameter', 0),
            "avg_degree": topo_stats.get('avg_degree', 0),
        }

        print(f"    Synergy: {results_by_topology[topo.name]['synergy_mean']:.4f}")
        print(f"    Error:   {results_by_topology[topo.name]['error_mean']:.4f}")

    # Find best topology
    best_topo = max(results_by_topology.keys(),
                    key=lambda k: results_by_topology[k]['synergy_mean'])

    metrics = {
        "results_by_topology": results_by_topology,
        "best_topology": best_topo,
        "best_synergy": results_by_topology[best_topo]['synergy_mean'],
    }

    print(f"\nBest Topology: {best_topo}")
    print(f"  Synergy: {metrics['best_synergy']:.4f}")

    return ExperimentResult(
        experiment_name="topology_comparison",
        config={"num_agents": num_agents, "num_trials": num_trials},
        metrics=metrics,
        duration_seconds=time.time() - start_time,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def experiment_environment_performance(
    num_episodes: int = 10,
    episode_length: int = 100,
    num_agents: int = 20,
    device: str = "cpu",
) -> ExperimentResult:
    """
    Experiment 4: Test swarm in actual environment.

    Measures exploration, survival, and resource collection.
    """
    print("\n" + "="*60)
    print("Experiment 4: Environment Performance")
    print("="*60)

    start_time = time.time()

    # Create environment
    env = CosmosEnvironment(
        grid_size=16,
        num_resources=20,
        num_hazards=5,
        vision_radius=3,
    )

    obs_dim = env.observation_dim
    action_dim = 5  # up, down, left, right, stay

    # Create swarm
    swarm_config = SwarmConfig(
        num_agents=num_agents,
        input_dim=obs_dim,
        hidden_dim=64,
        output_dim=action_dim,
        topology=TopologyType.SMALL_WORLD,
    )
    swarm = SwarmGraph(swarm_config, device=device)

    # Run episodes
    episode_rewards = []
    episode_survivals = []
    episode_explorations = []

    for ep in range(num_episodes):
        obs_list = env.reset()
        obs_tensor = obs_list[0].to_tensor(device).unsqueeze(0)

        total_reward = 0
        cells_visited = set()

        for step in range(episode_length):
            with torch.no_grad():
                action_logits = swarm.step(obs_tensor)
                action = action_logits.argmax(dim=-1).item()

            obs_list, rewards, dones = env.step([action])
            obs_tensor = obs_list[0].to_tensor(device).unsqueeze(0)
            reward = rewards[0]
            done = dones[0]
            info = {'position': obs_list[0].position}

            total_reward += reward
            cells_visited.add((info['position'][0], info['position'][1]))

            if done:
                break

        episode_rewards.append(total_reward)
        episode_survivals.append(step + 1)
        episode_explorations.append(len(cells_visited) / (16 ** 2))  # grid_size=16

    metrics = {
        "reward_mean": np.mean(episode_rewards),
        "reward_std": np.std(episode_rewards),
        "survival_mean": np.mean(episode_survivals),
        "survival_std": np.std(episode_survivals),
        "exploration_mean": np.mean(episode_explorations),
        "exploration_std": np.std(episode_explorations),
        "num_episodes": num_episodes,
        "episode_length": episode_length,
    }

    print(f"\nResults ({num_episodes} episodes):")
    print(f"  Reward:      {metrics['reward_mean']:.2f} ± {metrics['reward_std']:.2f}")
    print(f"  Survival:    {metrics['survival_mean']:.1f} ± {metrics['survival_std']:.1f} steps")
    print(f"  Exploration: {metrics['exploration_mean']*100:.1f}% ± {metrics['exploration_std']*100:.1f}%")

    return ExperimentResult(
        experiment_name="environment_performance",
        config={"num_agents": num_agents, "num_episodes": num_episodes},
        metrics=metrics,
        duration_seconds=time.time() - start_time,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )


def run_all_experiments(
    output_path: str = "results/hypothesis_validation.json",
    device: str = "cpu",
) -> Dict[str, ExperimentResult]:
    """Run all experiments and save results."""

    print("\n" + "="*60)
    print("SEESWM Hypothesis Validation")
    print("="*60)
    print(f"Device: {device}")
    print(f"Output: {output_path}")

    results = {}

    # Run experiments
    results['synergy_vs_baseline'] = experiment_synergy_vs_baseline(device=device)
    results['scaling_synergy'] = experiment_scaling_synergy(device=device)
    results['topology_comparison'] = experiment_topology_comparison(device=device)
    results['environment_performance'] = experiment_environment_performance(device=device)

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        name: asdict(result) for name, result in results.items()
    }

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    # Print key findings
    synergy_result = results['synergy_vs_baseline']
    print(f"\n1. Synergy vs Baseline:")
    print(f"   Swarm wins {synergy_result.metrics['swarm_wins']}/10 trials")
    print(f"   Mean synergy: {synergy_result.metrics['synergy_mean']:.4f}")

    scaling_result = results['scaling_synergy']
    print(f"\n2. Scaling:")
    print(f"   Best agent count: {scaling_result.metrics['best_agent_count']}")

    topo_result = results['topology_comparison']
    print(f"\n3. Best Topology: {topo_result.metrics['best_topology']}")

    env_result = results['environment_performance']
    print(f"\n4. Environment:")
    print(f"   Exploration: {env_result.metrics['exploration_mean']*100:.1f}%")
    print(f"   Survival: {env_result.metrics['survival_mean']:.1f} steps")

    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SEESWM Hypothesis Validation")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--output", type=str, default="results/hypothesis_validation.json")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "synergy", "scaling", "topology", "environment"])
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"

    if args.experiment == "all":
        run_all_experiments(args.output, args.device)
    elif args.experiment == "synergy":
        result = experiment_synergy_vs_baseline(device=args.device)
        print(f"\n{result}")
    elif args.experiment == "scaling":
        result = experiment_scaling_synergy(device=args.device)
        print(f"\n{result}")
    elif args.experiment == "topology":
        result = experiment_topology_comparison(device=args.device)
        print(f"\n{result}")
    elif args.experiment == "environment":
        result = experiment_environment_performance(device=args.device)
        print(f"\n{result}")


if __name__ == "__main__":
    main()
