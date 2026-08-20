#!/usr/bin/env python3
"""
Exploratory fresh-network experiments for SEESWM.

This script creates new randomly initialized networks and runs diagnostics for:
1. aggregate-versus-individual output error;
2. the internal aggregation score across agent counts;
3. descriptive differences across graph topologies; and
4. rollouts of a fresh random policy in the grid world.

It does not train a model or test emergence, specialization, or architectural
superiority.

Results are saved to JSON for analysis.
"""

import argparse
import json
import math
import platform
import random
import subprocess
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from collections.abc import Sequence
from typing import Any, Dict, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.swarm.graph import SwarmGraph, SwarmConfig, TopologyType
from src.environment.cosmos import CosmosEnvironment


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _role_counts(num_agents: int) -> tuple[int, int, int, int]:
    """Split all agents across four declared roles without hidden extras."""
    if num_agents < 1:
        raise ValueError("num_agents must be positive")
    base, remainder = divmod(num_agents, 4)
    counts = [base + (index < remainder) for index in range(4)]
    return tuple(counts)


def _set_swarm_eval(swarm: SwarmGraph) -> None:
    for agent in swarm.agents.values():
        agent.network.eval()


def _source_provenance() -> Dict[str, Any]:
    repository = Path(__file__).resolve().parent.parent
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return {"revision": None, "dirty": None}
    return {
        "revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _dependency_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for distribution in ("torch", "numpy", "networkx"):
        try:
            versions[distribution] = str(package_version(distribution))
        except PackageNotFoundError:
            versions[distribution] = None
    return versions


def _to_json_native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_native(item) for item in value]
    return value


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_name: str
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    duration_seconds: float
    timestamp: str


def create_baseline_mlp(
    input_dim: int,
    output_dim: int,
    total_params: int,
) -> nn.Module:
    """
    Create a two-hidden-layer MLP close to a requested parameter budget.

    The prior implementation capped a narrow network at ten layers and could
    miss the requested budget by more than an order of magnitude.
    """
    # For width w, the parameter count is:
    #   w^2 + (input_dim + output_dim + 2) * w + output_dim
    linear = input_dim + output_dim + 2
    discriminant = linear * linear - 4 * (output_dim - total_params)
    estimated_width = max(1, round((-linear + math.sqrt(discriminant)) / 2))

    def parameter_count(width: int) -> int:
        return (
            input_dim * width + width
            + width * width + width
            + width * output_dim + output_dim
        )

    candidates = range(max(1, estimated_width - 3), estimated_width + 4)
    matched_width = min(candidates, key=lambda width: abs(parameter_count(width) - total_params))

    model = nn.Sequential(
        nn.Linear(input_dim, matched_width),
        nn.ReLU(),
        nn.Linear(matched_width, matched_width),
        nn.ReLU(),
        nn.Linear(matched_width, output_dim),
    )
    actual_params = sum(p.numel() for p in model.parameters())
    print(
        f"  Baseline MLP: {actual_params:,} params "
        f"(target {total_params:,}; width {matched_width})"
    )

    return model


def experiment_synergy_vs_baseline(
    num_trials: int = 10,
    num_agents: int = 20,
    device: str = "cpu",
    seed: int = 0,
) -> ExperimentResult:
    """
    Compare fresh swarm outputs to a parameter-matched, fresh MLP.

    This probes random-initialization behavior; it does not test learned
    coordination or emergent intelligence.
    """
    print("\n" + "="*60)
    if num_trials < 1:
        raise ValueError("num_trials must be positive")

    print("Experiment 1: Fresh-output comparison")
    print("="*60)

    start_time = time.time()
    _seed_everything(seed)

    input_dim, hidden_dim, output_dim = 32, 64, 16

    # Create swarm
    role_counts = _role_counts(num_agents)
    config = SwarmConfig(
        num_agents=num_agents,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        topology=TopologyType.SMALL_WORLD,
        num_perception=role_counts[0],
        num_reasoning=role_counts[1],
        num_memory=role_counts[2],
        num_planning=role_counts[3],
    )
    swarm = SwarmGraph(config, device=device)
    _set_swarm_eval(swarm)
    swarm_params = swarm.total_parameters
    print(f"  Swarm: {swarm_params:,} params, {num_agents} agents")

    # Create baseline MLP with similar param count
    baseline = create_baseline_mlp(input_dim, output_dim, swarm_params)
    baseline = baseline.to(device).eval()

    # Run trials
    swarm_errors = []
    baseline_errors = []
    aggregation_scores = []

    for trial in range(num_trials):
        # Random regression task
        X = torch.randn(32, input_dim, device=device)
        # Target: non-linear function
        y = torch.sin(X[:, :output_dim]) + 0.1 * torch.randn(32, output_dim, device=device)

        with torch.no_grad():
            # Swarm prediction
            swarm.reset(batch_size=X.shape[0])
            swarm_pred = swarm.step(X)
            swarm_error = F.mse_loss(swarm_pred, y).item()
            swarm_errors.append(swarm_error)

            # Baseline prediction
            baseline_pred = baseline(X)
            baseline_error = F.mse_loss(baseline_pred, y).item()
            baseline_errors.append(baseline_error)

            # Legacy API name; this is an internal aggregation diagnostic.
            swarm.reset(batch_size=X.shape[0])
            synergy_result = swarm.compute_synergy(X, y)
            aggregation_scores.append(synergy_result['synergy'])

    metrics = {
        "swarm_error_mean": np.mean(swarm_errors),
        "swarm_error_std": np.std(swarm_errors),
        "baseline_error_mean": np.mean(baseline_errors),
        "baseline_error_std": np.std(baseline_errors),
        "internal_aggregation_score_mean": np.mean(aggregation_scores),
        "internal_aggregation_score_std": np.std(aggregation_scores),
        "swarm_params": swarm_params,
        "baseline_params": sum(p.numel() for p in baseline.parameters()),
        "baseline_parameter_gap_fraction": abs(
            sum(p.numel() for p in baseline.parameters()) - swarm_params
        ) / swarm_params,
        "swarm_wins": sum(1 for s, b in zip(swarm_errors, baseline_errors) if s < b),
    }

    print(f"\nResults ({num_trials} trials):")
    print(
        f"  Swarm MSE:    {metrics['swarm_error_mean']:.4f} "
        f"± {metrics['swarm_error_std']:.4f}"
    )
    print(
        f"  Baseline MSE: {metrics['baseline_error_mean']:.4f} "
        f"± {metrics['baseline_error_std']:.4f}"
    )
    print(
        "  Internal aggregation score: "
        f"{metrics['internal_aggregation_score_mean']:.4f} "
        f"± {metrics['internal_aggregation_score_std']:.4f}"
    )
    print(f"  Swarm wins:   {metrics['swarm_wins']}/{num_trials}")

    return ExperimentResult(
        experiment_name="synergy_vs_baseline",
        config={"num_agents": num_agents, "num_trials": num_trials, "seed": seed},
        metrics=metrics,
        duration_seconds=time.time() - start_time,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def experiment_scaling_synergy(
    agent_counts: Sequence[int] = (5, 10, 20, 50, 100),
    num_trials: int = 5,
    device: str = "cpu",
    seed: int = 0,
) -> ExperimentResult:
    """
    Describe the fresh-network internal aggregation score across agent counts.
    """
    print("\n" + "="*60)
    if num_trials < 1:
        raise ValueError("num_trials must be positive")
    if not agent_counts or any(count < 1 for count in agent_counts):
        raise ValueError("agent_counts must contain positive values")

    agent_counts = list(agent_counts)
    print("Experiment 2: Internal aggregation score by agent count")
    print("="*60)

    start_time = time.time()

    input_dim, hidden_dim, output_dim = 32, 64, 16
    results_by_count = {}

    for num_agents in agent_counts:
        _seed_everything(seed)
        print(f"\n  Testing {num_agents} agents...")

        role_counts = _role_counts(num_agents)
        config = SwarmConfig(
            num_agents=num_agents,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            topology=TopologyType.SMALL_WORLD,
            num_perception=role_counts[0],
            num_reasoning=role_counts[1],
            num_memory=role_counts[2],
            num_planning=role_counts[3],
        )
        swarm = SwarmGraph(config, device=device)
        _set_swarm_eval(swarm)

        aggregation_scores = []
        collective_perfs = []
        individual_perfs = []

        for _ in range(num_trials):
            X = torch.randn(32, input_dim, device=device)
            y = torch.sin(X[:, :output_dim]) + 0.1 * torch.randn(32, output_dim, device=device)

            with torch.no_grad():
                swarm.reset(batch_size=X.shape[0])
                result = swarm.compute_synergy(X, y)
                aggregation_scores.append(result['synergy'])
                collective_perfs.append(result['collective_performance'])
                individual_perfs.append(result['avg_individual_performance'])

        results_by_count[num_agents] = {
            "internal_aggregation_score_mean": np.mean(aggregation_scores),
            "internal_aggregation_score_std": np.std(aggregation_scores),
            "collective_perf_mean": np.mean(collective_perfs),
            "individual_perf_mean": np.mean(individual_perfs),
            "params": swarm.total_parameters,
        }

        print(
            "    Internal aggregation score: "
            f"{results_by_count[num_agents]['internal_aggregation_score_mean']:.4f}"
        )
        print(f"    Params:  {swarm.total_parameters:,}")

    # Report raw score changes rather than interpreting them as scaling laws.
    base_score = results_by_count[agent_counts[0]][
        'internal_aggregation_score_mean'
    ]
    score_deltas = []
    for n in agent_counts[1:]:
        score_deltas.append(
            results_by_count[n]['internal_aggregation_score_mean'] - base_score
        )

    highest_score_count = max(
        results_by_count,
        key=lambda count: results_by_count[count][
            'internal_aggregation_score_mean'
        ],
    )

    metrics = {
        "results_by_count": results_by_count,
        "agent_counts": agent_counts,
        "score_delta_from_smallest_count": score_deltas,
        "highest_score_agent_count": highest_score_count,
    }

    print("\nAgent-count summary:")
    print(f"  Highest observed score at: {highest_score_count} agents")
    print(f"  Score deltas vs {agent_counts[0]} agents: {score_deltas}")

    return ExperimentResult(
        experiment_name="scaling_synergy",
        config={"agent_counts": agent_counts, "num_trials": num_trials, "seed": seed},
        metrics=metrics,
        duration_seconds=time.time() - start_time,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def experiment_topology_comparison(
    num_agents: int = 30,
    num_trials: int = 10,
    device: str = "cpu",
    seed: int = 0,
) -> ExperimentResult:
    """
    Compare fresh-network diagnostics across graph topologies.
    """
    print("\n" + "="*60)
    if num_trials < 1 or num_agents < 1:
        raise ValueError("num_trials and num_agents must be positive")

    print("Experiment 3: Fresh topology comparison")
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
        _seed_everything(seed)
        print(f"\n  Testing {topo.name}...")

        role_counts = _role_counts(num_agents)
        config = SwarmConfig(
            num_agents=num_agents,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            topology=topo,
            num_perception=role_counts[0],
            num_reasoning=role_counts[1],
            num_memory=role_counts[2],
            num_planning=role_counts[3],
        )
        swarm = SwarmGraph(config, device=device)
        _set_swarm_eval(swarm)

        aggregation_scores = []
        errors = []

        for _ in range(num_trials):
            X = torch.randn(32, input_dim, device=device)
            y = torch.sin(X[:, :output_dim]) + 0.1 * torch.randn(32, output_dim, device=device)

            with torch.no_grad():
                swarm.reset(batch_size=X.shape[0])
                pred = swarm.step(X)
                error = F.mse_loss(pred, y).item()
                errors.append(error)

                swarm.reset(batch_size=X.shape[0])
                result = swarm.compute_synergy(X, y)
                aggregation_scores.append(result['synergy'])

        topo_stats = swarm.get_topology_stats()

        results_by_topology[topo.name] = {
            "internal_aggregation_score_mean": np.mean(aggregation_scores),
            "internal_aggregation_score_std": np.std(aggregation_scores),
            "error_mean": np.mean(errors),
            "clustering": topo_stats.get('clustering_coefficient', 0),
            "diameter": topo_stats.get('diameter', 0),
            "avg_degree": topo_stats.get('avg_degree', 0),
        }

        print(
            "    Internal aggregation score: "
            f"{results_by_topology[topo.name]['internal_aggregation_score_mean']:.4f}"
        )
        print(f"    Error:   {results_by_topology[topo.name]['error_mean']:.4f}")

    highest_score_topology = max(
        results_by_topology,
        key=lambda topology: results_by_topology[topology][
            'internal_aggregation_score_mean'
        ],
    )

    metrics = {
        "results_by_topology": results_by_topology,
        "highest_score_topology": highest_score_topology,
        "highest_internal_aggregation_score": results_by_topology[
            highest_score_topology
        ]["internal_aggregation_score_mean"],
    }

    print(f"\nHighest observed score topology: {highest_score_topology}")
    print(
        "  Internal aggregation score: "
        f"{metrics['highest_internal_aggregation_score']:.4f}"
    )

    return ExperimentResult(
        experiment_name="topology_comparison",
        config={"num_agents": num_agents, "num_trials": num_trials, "seed": seed},
        metrics=metrics,
        duration_seconds=time.time() - start_time,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def experiment_environment_performance(
    num_episodes: int = 10,
    episode_length: int = 100,
    num_agents: int = 20,
    device: str = "cpu",
    seed: int = 0,
) -> ExperimentResult:
    """
    Roll out a fresh random swarm in the grid environment.

    Measures exploration, survival, and resource collection.
    """
    print("\n" + "="*60)
    if num_episodes < 1 or episode_length < 1 or num_agents < 1:
        raise ValueError(
            "num_episodes, episode_length, and num_agents must be positive"
        )

    print("Experiment 4: Fresh-policy environment rollout")
    print("="*60)

    start_time = time.time()
    _seed_everything(seed)

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
    role_counts = _role_counts(num_agents)
    swarm_config = SwarmConfig(
        num_agents=num_agents,
        input_dim=obs_dim,
        hidden_dim=64,
        output_dim=action_dim,
        topology=TopologyType.SMALL_WORLD,
        num_perception=role_counts[0],
        num_reasoning=role_counts[1],
        num_memory=role_counts[2],
        num_planning=role_counts[3],
    )
    swarm = SwarmGraph(swarm_config, device=device)
    _set_swarm_eval(swarm)

    # Run episodes
    episode_rewards = []
    episode_survivals = []
    episode_explorations = []

    for ep in range(num_episodes):
        obs_list = env.reset()
        obs_tensor = obs_list[0].to_tensor(device).unsqueeze(0)
        swarm.reset(batch_size=1)

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
        config={"num_agents": num_agents, "num_episodes": num_episodes, "seed": seed},
        metrics=metrics,
        duration_seconds=time.time() - start_time,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_all_experiments(
    output_path: str = "results/hypothesis_validation.local.json",
    device: str = "cpu",
    seed: int = 0,
) -> Dict[str, ExperimentResult]:
    """Run all experiments and save results."""

    print("\n" + "="*60)
    print("SEESWM Fresh-Network Experiments")
    print("="*60)
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print(f"Output: {output_path}")

    results = {}

    # Run experiments
    results['synergy_vs_baseline'] = experiment_synergy_vs_baseline(device=device, seed=seed)
    results['scaling_synergy'] = experiment_scaling_synergy(device=device, seed=seed)
    results['topology_comparison'] = experiment_topology_comparison(device=device, seed=seed)
    results['environment_performance'] = experiment_environment_performance(
        device=device,
        seed=seed,
    )

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    serializable: Dict[str, Any] = {
        name: asdict(result) for name, result in results.items()
    }
    serializable["_provenance"] = {
        "scope": "fresh randomly initialized diagnostics; no training",
        "seed": seed,
        "source": _source_provenance(),
        "dependencies": _dependency_versions(),
    }
    serializable = _to_json_native(serializable)

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    # Print key findings
    synergy_result = results['synergy_vs_baseline']
    print(f"\n1. Fresh-network regression diagnostic:")
    comparison_trials = synergy_result.config['num_trials']
    print(
        f"   Lower swarm MSE in {synergy_result.metrics['swarm_wins']}"
        f"/{comparison_trials} trials"
    )
    print(
        "   Mean internal aggregation score: "
        f"{synergy_result.metrics['internal_aggregation_score_mean']:.4f}"
    )

    scaling_result = results['scaling_synergy']
    print(f"\n2. Fresh-network agent-count sweep:")
    print(
        "   Highest observed internal score at: "
        f"{scaling_result.metrics['highest_score_agent_count']} agents"
    )

    topo_result = results['topology_comparison']
    print(
        "\n3. Highest-scoring fresh topology in this diagnostic: "
        f"{topo_result.metrics['highest_score_topology']}"
    )

    env_result = results['environment_performance']
    print(f"\n4. Environment:")
    print(f"   Exploration: {env_result.metrics['exploration_mean']*100:.1f}%")
    print(f"   Survival: {env_result.metrics['survival_mean']:.1f} steps")

    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="SEESWM fresh-network experiments")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument(
        "--output",
        type=str,
        default="results/hypothesis_validation.local.json",
    )
    parser.add_argument("--seed", type=int, default=0, help="Non-negative random seed")
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "synergy", "scaling", "topology", "environment"])
    args = parser.parse_args()

    if args.seed < 0:
        parser.error("--seed must be non-negative")

    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    if args.experiment == "all":
        run_all_experiments(args.output, args.device, args.seed)
    elif args.experiment == "synergy":
        result = experiment_synergy_vs_baseline(device=args.device, seed=args.seed)
        print(f"\n{result}")
    elif args.experiment == "scaling":
        result = experiment_scaling_synergy(device=args.device, seed=args.seed)
        print(f"\n{result}")
    elif args.experiment == "topology":
        result = experiment_topology_comparison(device=args.device, seed=args.seed)
        print(f"\n{result}")
    elif args.experiment == "environment":
        result = experiment_environment_performance(device=args.device, seed=args.seed)
        print(f"\n{result}")


if __name__ == "__main__":
    main()
