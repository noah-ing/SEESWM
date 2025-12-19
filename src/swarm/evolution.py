"""
Genetic algorithm for evolving swarm architectures.

Evolves:
- Number of agents per type
- Hidden dimensions
- Graph topology
- Learning rates
- Neuromodulation sensitivity
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional
import copy

from .graph import SwarmGraph, SwarmConfig, TopologyType


@dataclass
class Genome:
    """Genome encoding swarm hyperparameters."""

    num_perception: int = 25
    num_reasoning: int = 25
    num_memory: int = 25
    num_planning: int = 25
    hidden_dim: int = 128
    topology: str = "small_world"
    base_lr: float = 1e-4
    dopamine_sensitivity: float = 1.0
    curiosity_weight: float = 0.5
    message_passing_rounds: int = 3

    def to_dict(self) -> dict:
        return {
            "num_perception": self.num_perception,
            "num_reasoning": self.num_reasoning,
            "num_memory": self.num_memory,
            "num_planning": self.num_planning,
            "hidden_dim": self.hidden_dim,
            "topology": self.topology,
            "base_lr": self.base_lr,
            "dopamine_sensitivity": self.dopamine_sensitivity,
            "curiosity_weight": self.curiosity_weight,
            "message_passing_rounds": self.message_passing_rounds,
        }

    @classmethod
    def random(cls) -> Genome:
        """Create a random genome."""
        return cls(
            num_perception=random.randint(10, 50),
            num_reasoning=random.randint(10, 50),
            num_memory=random.randint(10, 50),
            num_planning=random.randint(10, 50),
            hidden_dim=random.choice([64, 128, 256]),
            topology=random.choice(["random", "small_world", "modular", "hierarchical"]),
            base_lr=10 ** random.uniform(-5, -3),
            dopamine_sensitivity=random.uniform(0.1, 2.0),
            curiosity_weight=random.uniform(0.0, 1.0),
            message_passing_rounds=random.randint(2, 5),
        )


@dataclass
class FitnessResult:
    """Result of evaluating a genome."""

    performance: float
    synergy: float
    efficiency: float
    fitness: float  # Combined score


class SwarmEvolution:
    """
    Evolve swarm architectures using genetic algorithms.

    Multi-objective optimization balancing:
    - Task performance
    - Synergy (collective > individuals)
    - Efficiency (performance per parameter)
    """

    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        elitism: int = 2,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism

        # Initialize population
        self.population: list[Genome] = [
            Genome.random() for _ in range(population_size)
        ]
        self.generation = 0
        self.best_genome: Optional[Genome] = None
        self.best_fitness: float = float("-inf")
        self.history: list[dict] = []

    def evaluate_fitness(
        self,
        genome: Genome,
        benchmark_fn,
        device: str = "cpu",
    ) -> FitnessResult:
        """
        Evaluate fitness of a genome.

        Args:
            genome: The genome to evaluate
            benchmark_fn: Function that takes a SwarmGraph and returns
                         dict with 'accuracy' and 'synergy' keys
            device: Device to run evaluation on

        Returns:
            FitnessResult with performance metrics
        """
        # Build swarm from genome
        swarm = SwarmGraph.from_genome(genome.to_dict(), device=device)

        # Run benchmark
        results = benchmark_fn(swarm)

        performance = results.get("accuracy", 0.0)
        synergy = results.get("synergy", 0.0)

        # Efficiency: performance per million parameters
        params_millions = swarm.total_parameters / 1e6
        efficiency = performance / max(0.1, params_millions)

        # Combined fitness (weighted sum)
        fitness = performance + 0.3 * synergy + 0.1 * efficiency

        return FitnessResult(
            performance=performance,
            synergy=synergy,
            efficiency=efficiency,
            fitness=fitness,
        )

    def select_parents(
        self,
        fitness_scores: list[tuple[Genome, FitnessResult]],
    ) -> list[Genome]:
        """Tournament selection for parents."""
        parents = []
        tournament_size = 3

        for _ in range(self.population_size - self.elitism):
            # Random tournament
            tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
            winner = max(tournament, key=lambda x: x[1].fitness)
            parents.append(copy.deepcopy(winner[0]))

        return parents

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """Single-point crossover between two genomes."""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1)

        child = Genome()
        fields = list(child.to_dict().keys())
        crossover_point = random.randint(1, len(fields) - 1)

        for i, field in enumerate(fields):
            if i < crossover_point:
                setattr(child, field, getattr(parent1, field))
            else:
                setattr(child, field, getattr(parent2, field))

        return child

    def mutate(self, genome: Genome) -> Genome:
        """Apply random mutations to genome."""
        genome = copy.deepcopy(genome)

        if random.random() < self.mutation_rate:
            genome.num_perception = max(5, genome.num_perception + random.randint(-10, 10))
        if random.random() < self.mutation_rate:
            genome.num_reasoning = max(5, genome.num_reasoning + random.randint(-10, 10))
        if random.random() < self.mutation_rate:
            genome.num_memory = max(5, genome.num_memory + random.randint(-10, 10))
        if random.random() < self.mutation_rate:
            genome.num_planning = max(5, genome.num_planning + random.randint(-10, 10))
        if random.random() < self.mutation_rate:
            genome.hidden_dim = random.choice([64, 128, 256])
        if random.random() < self.mutation_rate:
            genome.topology = random.choice(["random", "small_world", "modular", "hierarchical"])
        if random.random() < self.mutation_rate:
            genome.base_lr *= 10 ** random.uniform(-0.5, 0.5)
        if random.random() < self.mutation_rate:
            genome.dopamine_sensitivity = max(0.1, min(3.0, genome.dopamine_sensitivity + random.uniform(-0.3, 0.3)))
        if random.random() < self.mutation_rate:
            genome.curiosity_weight = max(0.0, min(1.0, genome.curiosity_weight + random.uniform(-0.2, 0.2)))

        return genome

    def evolve_generation(self, benchmark_fn, device: str = "cpu") -> dict:
        """
        Run one generation of evolution.

        Returns:
            Dictionary with generation statistics
        """
        # Evaluate all genomes
        fitness_scores = [
            (genome, self.evaluate_fitness(genome, benchmark_fn, device))
            for genome in self.population
        ]

        # Sort by fitness
        fitness_scores.sort(key=lambda x: x[1].fitness, reverse=True)

        # Track best
        if fitness_scores[0][1].fitness > self.best_fitness:
            self.best_fitness = fitness_scores[0][1].fitness
            self.best_genome = copy.deepcopy(fitness_scores[0][0])

        # Elitism: keep top performers
        new_population = [copy.deepcopy(g) for g, _ in fitness_scores[: self.elitism]]

        # Select parents and create offspring
        parents = self.select_parents(fitness_scores)

        while len(new_population) < self.population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

        # Record history
        stats = {
            "generation": self.generation,
            "best_fitness": fitness_scores[0][1].fitness,
            "avg_fitness": sum(f.fitness for _, f in fitness_scores) / len(fitness_scores),
            "best_performance": fitness_scores[0][1].performance,
            "best_synergy": fitness_scores[0][1].synergy,
            "best_genome": fitness_scores[0][0].to_dict(),
        }
        self.history.append(stats)

        return stats
