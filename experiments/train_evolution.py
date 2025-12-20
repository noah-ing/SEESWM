#!/usr/bin/env python3
"""
Training script for evolution and scaling experiments.

Demonstrates:
1. Evolutionary optimization of swarm topology
2. Neural Architecture Search for agents
3. CMA-ES for continuous parameters
4. Hierarchical swarms for scaling
5. Large-scale swarm training
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evolution import (
    # Genetic
    EvolutionConfig,
    SwarmGenome,
    AgentGene,
    ConnectionGene,
    GeneticOptimizer,
    CMAES,
    # NAS
    SearchSpace,
    OperationType,
    DARTSAgent,
    DARTSSearcher,
    ENASController,
    ENASSharedNetwork,
    ENASSearcher,
    RandomSearchNAS,
    # Scaling
    ScalingConfig,
    AgentPool,
    HierarchicalSwarm,
    SparseMessageGraph,
    estimate_memory_usage,
)


def evolve_swarm_topology(
    num_generations: int = 50,
    population_size: int = 30,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Evolve swarm topology using genetic algorithm.

    Optimizes number of agents, connections, and agent types.
    """
    print("\n=== Evolving Swarm Topology ===")
    print(f"Generations: {num_generations}, Population: {population_size}")

    # Evolution config
    config = EvolutionConfig(
        population_size=population_size,
        elite_size=3,
        tournament_size=3,
        weight_mutation_rate=0.8,
        add_agent_rate=0.1,
        remove_agent_rate=0.05,
        add_connection_rate=0.15,
        remove_connection_rate=0.1,
        max_agents=50,
        min_agents=5,
    )

    # Fitness function: rewards performance and penalizes complexity
    def fitness_fn(genome: SwarmGenome) -> float:
        # Simulate swarm performance
        num_agents = len(genome.agents)
        num_connections = len(genome.connections)

        # Connectivity score
        if num_agents > 0:
            connectivity = num_connections / (num_agents * (num_agents - 1) / 2 + 1)
        else:
            connectivity = 0

        # Type diversity score
        if genome.agents:
            types = [a.agent_type for a in genome.agents.values()]
            type_diversity = len(set(types)) / 4.0  # 4 agent types
        else:
            type_diversity = 0

        # Simulate task performance (random for demo)
        task_performance = random.gauss(0.5 + 0.1 * type_diversity, 0.1)
        task_performance = max(0, min(1, task_performance))

        # Penalize very large or very small swarms
        size_penalty = abs(num_agents - 20) / 50

        fitness = (
            0.5 * task_performance +
            0.2 * connectivity +
            0.2 * type_diversity -
            0.1 * size_penalty
        )

        return max(0, fitness)

    # Create optimizer
    optimizer = GeneticOptimizer(config, fitness_fn)

    # Create initial genome
    initial = SwarmGenome(input_dim=64, output_dim=32, hidden_dim=64)
    for i in range(10):
        initial.agents[i] = AgentGene(
            innovation_number=i,
            agent_id=i,
            agent_type=i % 4,
        )

    # Add some connections
    for i in range(10):
        for j in range(i + 1, min(i + 3, 10)):
            innovation = 100 + i * 10 + j
            initial.connections[innovation] = ConnectionGene(
                innovation_number=innovation,
                from_agent=i,
                to_agent=j,
                weight=random.gauss(0, 1),
            )

    # Initialize population
    optimizer.initialize_population(initial)

    # Evolve
    for gen in range(num_generations):
        optimizer.evolve()

        if (gen + 1) % 10 == 0:
            stats = optimizer.get_statistics()
            print(f"Generation {gen + 1}/{num_generations}")
            print(f"  Best Fitness: {stats['best_fitness']:.4f}")
            print(f"  Best Agents: {stats['best_num_agents']}, Connections: {stats['best_num_connections']}")
            print(f"  Species: {stats['num_species']}")

    # Final results
    best = optimizer.get_best()
    final_stats = optimizer.get_statistics()

    print(f"\nFinal Best Genome:")
    print(f"  Agents: {len(best.agents)}")
    print(f"  Connections: {len(best.connections)}")
    print(f"  Fitness: {best.fitness:.4f}")

    return final_stats


def optimize_parameters_cmaes(
    num_iterations: int = 100,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Optimize continuous swarm parameters with CMA-ES.
    """
    print("\n=== CMA-ES Parameter Optimization ===")
    print(f"Iterations: {num_iterations}")

    # Parameters to optimize:
    # [learning_rate, hidden_dim_scale, message_rounds, sparsity]
    dim = 4

    cmaes = CMAES(dim=dim, sigma=0.3)

    # Fitness function
    def fitness_fn(params):
        lr = 10 ** (params[0] * 3 - 5)  # Log scale: 1e-5 to 1e-2
        hidden_scale = params[1] * 2 + 0.5  # 0.5 to 2.5
        msg_rounds = int(params[2] * 5) + 1  # 1 to 6
        sparsity = params[3] * 0.5 + 0.1  # 0.1 to 0.6

        # Simulate training with these parameters
        # (In real usage, this would actually train and evaluate)
        performance = (
            0.3 * (1 - abs(lr - 0.001) / 0.01) +  # Optimal LR around 0.001
            0.2 * (1 - abs(hidden_scale - 1.5) / 2) +  # Optimal scale around 1.5
            0.2 * (1 - abs(msg_rounds - 3) / 5) +  # Optimal rounds around 3
            0.3 * (1 - abs(sparsity - 0.3) / 0.5)  # Optimal sparsity around 0.3
        )

        return performance + random.gauss(0, 0.05)

    # Optimize
    best_params, best_fitness = cmaes.optimize(fitness_fn, num_iterations)

    print(f"\nBest Parameters Found:")
    print(f"  Learning Rate: {10 ** (best_params[0] * 3 - 5):.6f}")
    print(f"  Hidden Dim Scale: {best_params[1] * 2 + 0.5:.2f}")
    print(f"  Message Rounds: {int(best_params[2] * 5) + 1}")
    print(f"  Sparsity: {best_params[3] * 0.5 + 0.1:.2f}")
    print(f"  Fitness: {best_fitness:.4f}")

    return {
        "best_params": best_params.tolist(),
        "best_fitness": best_fitness,
        "generations": cmaes.generation,
    }


def search_agent_architecture_darts(
    num_epochs: int = 50,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Search for optimal agent architecture using DARTS.
    """
    print("\n=== DARTS Architecture Search ===")
    print(f"Epochs: {num_epochs}")

    # Create DARTS model
    search_space = SearchSpace(
        operations=[
            OperationType.SKIP,
            OperationType.LINEAR,
            OperationType.RELU_LINEAR,
            OperationType.GELU_LINEAR,
        ],
    )

    model = DARTSAgent(
        input_dim=32,
        hidden_dim=64,
        output_dim=16,
        num_cells=2,
        nodes_per_cell=3,
        search_space=search_space,
    ).to(device)

    # Searcher
    searcher = DARTSSearcher(model, arch_lr=3e-4, weight_lr=0.01)

    # Generate data
    def generate_data(batch_size=32):
        x = torch.randn(batch_size, 32, device=device)
        y = torch.randn(batch_size, 16, device=device)
        return x, y

    # Search
    history = []
    for epoch in range(num_epochs):
        train_data = generate_data()
        val_data = generate_data()

        metrics = searcher.search_step(
            train_data,
            val_data,
            loss_fn=F.mse_loss,
        )

        history.append(metrics)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Arch Loss: {metrics['arch_loss']:.4f}, Weight Loss: {metrics['weight_loss']:.4f}")

    # Get final architecture
    architecture = searcher.derive_architecture()

    print(f"\nDiscovered Architecture:")
    for i, cell_arch in enumerate(architecture):
        print(f"  Cell {i}:")
        for edge, op in cell_arch.items():
            print(f"    {edge}: {op.name}")

    return {
        "architecture": [[{k: v.name for k, v in cell.items()} for cell in architecture]],
        "final_loss": history[-1]["weight_loss"],
    }


def search_agent_architecture_enas(
    num_iterations: int = 100,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Search for optimal agent architecture using ENAS.
    """
    print("\n=== ENAS Architecture Search ===")
    print(f"Iterations: {num_iterations}")

    search_space = SearchSpace()

    # Controller and shared network
    controller = ENASController(search_space, num_layers=4, hidden_dim=32).to(device)
    shared_network = ENASSharedNetwork(
        input_dim=32,
        hidden_dim=64,
        output_dim=16,
        num_layers=4,
        search_space=search_space,
    ).to(device)

    searcher = ENASSearcher(controller, shared_network)

    # Generate data
    def generate_data(batch_size=32):
        x = torch.randn(batch_size, 32, device=device)
        y = torch.randn(batch_size, 16, device=device)
        return x, y

    # Search
    for iteration in range(num_iterations):
        # Train shared weights
        train_data = generate_data()
        shared_loss = searcher.train_shared(train_data, F.mse_loss, num_steps=5)

        # Train controller
        val_data = generate_data()
        controller_metrics = searcher.train_controller(val_data, F.mse_loss, num_samples=5)

        if (iteration + 1) % 20 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"  Shared Loss: {shared_loss:.4f}")
            print(f"  Avg Reward: {controller_metrics['avg_reward']:.4f}")

    # Sample best architecture
    best_arch = controller.sample_architecture()

    print(f"\nBest Architecture:")
    print(f"  Operations: {[op.name for op in best_arch['operations']]}")
    print(f"  Skip Connections: {best_arch['skip_connections']}")

    return best_arch


def demo_hierarchical_swarm(
    num_agents: int = 500,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Demonstrate hierarchical swarm for large-scale computation.
    """
    print("\n=== Hierarchical Swarm Demo ===")
    print(f"Agents: {num_agents}")

    # Memory estimation
    memory_est = estimate_memory_usage(num_agents, hidden_dim=64, num_layers=2)
    print(f"Estimated Memory: {memory_est['total_params_mb']:.2f} MB for parameters")

    # Agent factory
    def agent_factory(agent_id: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    # Config
    config = ScalingConfig(
        use_hierarchy=True,
        num_levels=3,
        agents_per_group=10,
        groups_per_supergroup=10,
        sparse_messaging=True,
        message_sparsity=0.2,
    )

    # Create hierarchical swarm
    swarm = HierarchicalSwarm(
        num_agents=num_agents,
        agent_factory=agent_factory,
        config=config,
        hidden_dim=64,
        device=device,
    )

    # Test forward pass
    inputs = {i: torch.randn(1, 32, device=device) for i in range(min(100, num_agents))}
    outputs = swarm(inputs)

    stats = swarm.get_statistics()

    print(f"\nSwarm Statistics:")
    print(f"  Total Agents: {stats['num_agents']}")
    print(f"  Groups: {stats['num_groups']}")
    print(f"  Agents Cached: {stats['agents_cached']}")
    print(f"  Hierarchy Levels: {stats['hierarchy_levels']}")

    return stats


def demo_sparse_messaging(
    num_agents: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Demonstrate sparse message passing for efficiency.
    """
    print("\n=== Sparse Message Passing Demo ===")
    print(f"Agents: {num_agents}")

    config = ScalingConfig(
        sparse_messaging=True,
        message_sparsity=0.1,
    )

    # Random positions
    positions = torch.rand(num_agents, 2)

    # Create sparse graph
    graph = SparseMessageGraph(num_agents, config, positions)

    # Count connections
    total_connections = sum(len(neighbors) for neighbors in graph.adjacency.values())
    max_connections = num_agents * (num_agents - 1)
    sparsity = 1 - total_connections / max_connections

    print(f"  Total Connections: {total_connections}")
    print(f"  Max Possible: {max_connections}")
    print(f"  Actual Sparsity: {sparsity:.2%}")

    # Sample message targets
    sample_agent = 0
    targets = graph.get_message_targets(sample_agent, stochastic=True)
    print(f"  Agent 0 sends to {len(targets)} targets: {targets[:5]}...")

    return {
        "total_connections": total_connections,
        "sparsity": sparsity,
        "avg_neighbors": total_connections / num_agents,
    }


def benchmark_scaling(
    max_agents: int = 1000,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """
    Benchmark scaling performance.
    """
    print("\n=== Scaling Benchmark ===")

    agent_counts = [10, 50, 100, 200, 500, max_agents]
    results = {"agents": [], "time_ms": [], "memory_mb": []}

    def agent_factory(agent_id: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    config = ScalingConfig(
        use_hierarchy=True,
        agents_per_group=20,
        groups_per_supergroup=10,
    )

    import time

    for num_agents in agent_counts:
        if num_agents > max_agents:
            break

        # Estimate memory
        memory = estimate_memory_usage(num_agents, 64, 2)

        # Create swarm
        start = time.time()
        swarm = HierarchicalSwarm(
            num_agents=num_agents,
            agent_factory=agent_factory,
            config=config,
            hidden_dim=64,
            device=device,
        )

        # Forward pass
        inputs = {i: torch.randn(1, 32, device=device) for i in range(min(50, num_agents))}
        _ = swarm(inputs)

        elapsed = (time.time() - start) * 1000

        results["agents"].append(num_agents)
        results["time_ms"].append(elapsed)
        results["memory_mb"].append(memory["total_params_mb"])

        print(f"  {num_agents} agents: {elapsed:.1f}ms, {memory['total_params_mb']:.1f}MB")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evolution and scaling experiments")
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["evolve", "cmaes", "darts", "enas", "hierarchy", "sparse", "benchmark", "all"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--agents", type=int, default=500)
    args = parser.parse_args()

    device = args.device

    if args.experiment in ["evolve", "all"]:
        evolve_swarm_topology(num_generations=args.generations, device=device)

    if args.experiment in ["cmaes", "all"]:
        optimize_parameters_cmaes(num_iterations=100, device=device)

    if args.experiment in ["darts", "all"]:
        search_agent_architecture_darts(num_epochs=50, device=device)

    if args.experiment in ["enas", "all"]:
        search_agent_architecture_enas(num_iterations=100, device=device)

    if args.experiment in ["hierarchy", "all"]:
        demo_hierarchical_swarm(num_agents=args.agents, device=device)

    if args.experiment in ["sparse", "all"]:
        demo_sparse_messaging(num_agents=100, device=device)

    if args.experiment in ["benchmark", "all"]:
        benchmark_scaling(max_agents=args.agents, device=device)

    print("\n=== Evolution & Scaling Experiments Complete ===")


if __name__ == "__main__":
    main()
