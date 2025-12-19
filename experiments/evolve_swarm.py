"""
Evolutionary optimization of swarm architecture.

Phase 6: Use genetic algorithms to find optimal configurations.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.swarm.evolution import SwarmEvolution, Genome
from src.swarm.graph import SwarmGraph


def simple_benchmark(swarm: SwarmGraph) -> dict:
    """Simple benchmark for evolution testing."""
    # Random task: predict output from input
    inputs = torch.randn(32, swarm.config.input_dim)
    targets = torch.randn(32, swarm.config.output_dim)

    swarm.reset(batch_size=32)
    outputs = swarm.step(inputs)

    # MSE as negative accuracy (lower is better, invert for fitness)
    mse = ((outputs - targets) ** 2).mean().item()
    accuracy = 1.0 / (1.0 + mse)

    synergy = swarm.compute_synergy(inputs, targets)

    return {
        "accuracy": accuracy,
        "synergy": synergy["synergy"],
    }


def main():
    print("Swarm Evolution Demo")
    print("=" * 50)

    # Initialize evolution
    evolution = SwarmEvolution(population_size=10, mutation_rate=0.2)

    print(f"Initial population: {len(evolution.population)} genomes")
    print(f"Sample genome: {evolution.population[0].to_dict()}")
    print()

    # Run a few generations
    for gen in range(3):
        stats = evolution.evolve_generation(simple_benchmark)
        print(f"Generation {stats['generation']}:")
        print(f"  Best fitness: {stats['best_fitness']:.4f}")
        print(f"  Avg fitness: {stats['avg_fitness']:.4f}")
        print(f"  Best synergy: {stats['best_synergy']:.4f}")
        print()

    print("Best genome found:")
    print(evolution.best_genome.to_dict())


if __name__ == "__main__":
    main()
