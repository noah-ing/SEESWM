"""
Evolutionary optimization for swarm topology and parameters.

Implements:
1. Genetic algorithms for topology evolution
2. NEAT-style structural mutation
3. CMA-ES for continuous parameter optimization
4. Multi-objective evolution (performance + efficiency)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable, Any
from enum import Enum
import random
import math
import copy

import torch
import torch.nn as nn
import numpy as np


class MutationType(Enum):
    """Types of structural mutations."""
    ADD_AGENT = "add_agent"
    REMOVE_AGENT = "remove_agent"
    ADD_CONNECTION = "add_connection"
    REMOVE_CONNECTION = "remove_connection"
    CHANGE_AGENT_TYPE = "change_type"
    MODIFY_WEIGHTS = "modify_weights"


@dataclass
class Gene:
    """Base gene representation."""
    innovation_number: int
    enabled: bool = True


@dataclass
class AgentGene:
    """Gene representing an agent in the swarm."""
    innovation_number: int
    agent_id: int
    agent_type: int  # 0-3 for different types
    hidden_dim: int = 64
    position: Tuple[float, float] = (0.0, 0.0)  # For visualization
    enabled: bool = True


@dataclass
class ConnectionGene:
    """Gene representing a connection between agents."""
    innovation_number: int
    from_agent: int
    to_agent: int
    weight: float = 1.0
    enabled: bool = True


@dataclass
class SwarmGenome:
    """
    Complete genome representing a swarm configuration.

    Encodes topology, agent types, and connection weights.
    """
    agents: Dict[int, AgentGene] = field(default_factory=dict)
    connections: Dict[int, ConnectionGene] = field(default_factory=dict)

    # Configuration
    input_dim: int = 64
    output_dim: int = 32
    hidden_dim: int = 64

    # Fitness (set after evaluation)
    fitness: float = 0.0
    secondary_fitness: Dict[str, float] = field(default_factory=dict)

    # Tracking
    generation: int = 0
    species_id: int = 0

    def copy(self) -> SwarmGenome:
        """Deep copy of genome."""
        new_genome = SwarmGenome(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            hidden_dim=self.hidden_dim,
            generation=self.generation,
            species_id=self.species_id,
        )
        new_genome.agents = {k: copy.copy(v) for k, v in self.agents.items()}
        new_genome.connections = {k: copy.copy(v) for k, v in self.connections.items()}
        return new_genome

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "agents": {k: vars(v) for k, v in self.agents.items()},
            "connections": {k: vars(v) for k, v in self.connections.items()},
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "fitness": self.fitness,
        }


@dataclass
class EvolutionConfig:
    """Configuration for evolutionary optimization."""

    # Population
    population_size: int = 50
    elite_size: int = 5
    tournament_size: int = 3

    # Mutation rates
    weight_mutation_rate: float = 0.8
    weight_mutation_power: float = 0.5
    add_agent_rate: float = 0.05
    remove_agent_rate: float = 0.02
    add_connection_rate: float = 0.1
    remove_connection_rate: float = 0.05
    change_type_rate: float = 0.05

    # Crossover
    crossover_rate: float = 0.75

    # Speciation
    species_threshold: float = 3.0
    excess_coefficient: float = 1.0
    disjoint_coefficient: float = 1.0
    weight_coefficient: float = 0.4

    # Limits
    max_agents: int = 100
    min_agents: int = 5
    max_connections_per_agent: int = 10

    # Multi-objective
    use_pareto: bool = True
    objectives: List[str] = field(default_factory=lambda: ["fitness", "efficiency"])


class InnovationTracker:
    """
    Tracks innovation numbers for structural mutations.

    Ensures consistent gene alignment across the population.
    """

    def __init__(self):
        self.agent_innovations: Dict[int, int] = {}
        self.connection_innovations: Dict[Tuple[int, int], int] = {}
        self.next_innovation = 0

    def get_agent_innovation(self, agent_id: int) -> int:
        """Get innovation number for an agent."""
        if agent_id not in self.agent_innovations:
            self.agent_innovations[agent_id] = self.next_innovation
            self.next_innovation += 1
        return self.agent_innovations[agent_id]

    def get_connection_innovation(self, from_id: int, to_id: int) -> int:
        """Get innovation number for a connection."""
        key = (from_id, to_id)
        if key not in self.connection_innovations:
            self.connection_innovations[key] = self.next_innovation
            self.next_innovation += 1
        return self.connection_innovations[key]


class SwarmMutator:
    """
    Mutates swarm genomes.

    Implements structural and weight mutations.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        innovation_tracker: InnovationTracker,
    ):
        self.config = config
        self.tracker = innovation_tracker

    def mutate(self, genome: SwarmGenome) -> SwarmGenome:
        """Apply mutations to genome."""
        mutated = genome.copy()

        # Weight mutations
        if random.random() < self.config.weight_mutation_rate:
            self._mutate_weights(mutated)

        # Structural mutations
        if random.random() < self.config.add_agent_rate:
            self._add_agent(mutated)

        if random.random() < self.config.remove_agent_rate:
            self._remove_agent(mutated)

        if random.random() < self.config.add_connection_rate:
            self._add_connection(mutated)

        if random.random() < self.config.remove_connection_rate:
            self._remove_connection(mutated)

        if random.random() < self.config.change_type_rate:
            self._change_agent_type(mutated)

        return mutated

    def _mutate_weights(self, genome: SwarmGenome) -> None:
        """Mutate connection weights."""
        for conn in genome.connections.values():
            if random.random() < 0.9:
                # Perturb
                conn.weight += random.gauss(0, self.config.weight_mutation_power)
            else:
                # Random new weight
                conn.weight = random.gauss(0, 1)

    def _add_agent(self, genome: SwarmGenome) -> None:
        """Add a new agent."""
        if len(genome.agents) >= self.config.max_agents:
            return

        # New agent ID
        new_id = max(genome.agents.keys(), default=-1) + 1

        # Random type
        agent_type = random.randint(0, 3)

        # Create gene
        innovation = self.tracker.get_agent_innovation(new_id)
        gene = AgentGene(
            innovation_number=innovation,
            agent_id=new_id,
            agent_type=agent_type,
            hidden_dim=genome.hidden_dim,
            position=(random.random(), random.random()),
        )

        genome.agents[new_id] = gene

        # Connect to random existing agent
        if genome.agents:
            other_id = random.choice([
                aid for aid in genome.agents.keys() if aid != new_id
            ])
            self._add_specific_connection(genome, other_id, new_id)
            self._add_specific_connection(genome, new_id, other_id)

    def _remove_agent(self, genome: SwarmGenome) -> None:
        """Remove an agent."""
        if len(genome.agents) <= self.config.min_agents:
            return

        # Remove random agent
        agent_id = random.choice(list(genome.agents.keys()))
        del genome.agents[agent_id]

        # Remove associated connections
        to_remove = []
        for conn_id, conn in genome.connections.items():
            if conn.from_agent == agent_id or conn.to_agent == agent_id:
                to_remove.append(conn_id)

        for conn_id in to_remove:
            del genome.connections[conn_id]

    def _add_connection(self, genome: SwarmGenome) -> None:
        """Add a new connection."""
        if len(genome.agents) < 2:
            return

        agent_ids = list(genome.agents.keys())
        from_id = random.choice(agent_ids)
        to_id = random.choice([a for a in agent_ids if a != from_id])

        self._add_specific_connection(genome, from_id, to_id)

    def _add_specific_connection(
        self,
        genome: SwarmGenome,
        from_id: int,
        to_id: int,
    ) -> None:
        """Add a specific connection."""
        # Check if exists
        for conn in genome.connections.values():
            if conn.from_agent == from_id and conn.to_agent == to_id:
                return

        # Check connection limit
        from_connections = sum(
            1 for c in genome.connections.values()
            if c.from_agent == from_id
        )
        if from_connections >= self.config.max_connections_per_agent:
            return

        innovation = self.tracker.get_connection_innovation(from_id, to_id)
        conn = ConnectionGene(
            innovation_number=innovation,
            from_agent=from_id,
            to_agent=to_id,
            weight=random.gauss(0, 1),
        )

        genome.connections[innovation] = conn

    def _remove_connection(self, genome: SwarmGenome) -> None:
        """Remove a random connection."""
        if not genome.connections:
            return

        conn_id = random.choice(list(genome.connections.keys()))
        del genome.connections[conn_id]

    def _change_agent_type(self, genome: SwarmGenome) -> None:
        """Change an agent's type."""
        if not genome.agents:
            return

        agent_id = random.choice(list(genome.agents.keys()))
        genome.agents[agent_id].agent_type = random.randint(0, 3)


class SwarmCrossover:
    """
    Crossover operator for swarm genomes.

    Implements NEAT-style crossover with gene alignment.
    """

    def crossover(
        self,
        parent1: SwarmGenome,
        parent2: SwarmGenome,
    ) -> SwarmGenome:
        """
        Create offspring from two parents.

        More fit parent contributes disjoint/excess genes.
        """
        # Determine fitter parent
        if parent1.fitness >= parent2.fitness:
            fitter, less_fit = parent1, parent2
        else:
            fitter, less_fit = parent2, parent1

        offspring = SwarmGenome(
            input_dim=fitter.input_dim,
            output_dim=fitter.output_dim,
            hidden_dim=fitter.hidden_dim,
            generation=max(parent1.generation, parent2.generation) + 1,
        )

        # Crossover agents
        all_agent_innovations = set(parent1.agents.keys()) | set(parent2.agents.keys())

        for innovation in all_agent_innovations:
            in_p1 = innovation in parent1.agents
            in_p2 = innovation in parent2.agents

            if in_p1 and in_p2:
                # Matching: random choice
                if random.random() < 0.5:
                    offspring.agents[innovation] = copy.copy(parent1.agents[innovation])
                else:
                    offspring.agents[innovation] = copy.copy(parent2.agents[innovation])
            elif in_p1:
                # Disjoint/excess from parent1
                if fitter == parent1:
                    offspring.agents[innovation] = copy.copy(parent1.agents[innovation])
            else:
                # Disjoint/excess from parent2
                if fitter == parent2:
                    offspring.agents[innovation] = copy.copy(parent2.agents[innovation])

        # Crossover connections (similar logic)
        all_conn_innovations = set(parent1.connections.keys()) | set(parent2.connections.keys())

        for innovation in all_conn_innovations:
            in_p1 = innovation in parent1.connections
            in_p2 = innovation in parent2.connections

            # Check if connection agents exist in offspring
            def conn_valid(conn):
                return (
                    conn.from_agent in offspring.agents and
                    conn.to_agent in offspring.agents
                )

            if in_p1 and in_p2:
                conn = copy.copy(
                    parent1.connections[innovation] if random.random() < 0.5
                    else parent2.connections[innovation]
                )
                if conn_valid(conn):
                    offspring.connections[innovation] = conn
            elif in_p1 and fitter == parent1:
                conn = copy.copy(parent1.connections[innovation])
                if conn_valid(conn):
                    offspring.connections[innovation] = conn
            elif in_p2 and fitter == parent2:
                conn = copy.copy(parent2.connections[innovation])
                if conn_valid(conn):
                    offspring.connections[innovation] = conn

        return offspring


class Species:
    """
    Species for speciation-based evolution.

    Groups similar genomes together.
    """

    def __init__(self, species_id: int, representative: SwarmGenome):
        self.species_id = species_id
        self.representative = representative
        self.members: List[SwarmGenome] = [representative]
        self.best_fitness = representative.fitness
        self.generations_without_improvement = 0

    def add_member(self, genome: SwarmGenome) -> None:
        """Add genome to species."""
        genome.species_id = self.species_id
        self.members.append(genome)

        if genome.fitness > self.best_fitness:
            self.best_fitness = genome.fitness
            self.generations_without_improvement = 0

    def clear_members(self) -> None:
        """Clear members but keep representative."""
        self.members = [self.representative]
        self.generations_without_improvement += 1

    def update_representative(self) -> None:
        """Update representative to best member."""
        if self.members:
            self.representative = max(self.members, key=lambda g: g.fitness)


def genome_distance(
    g1: SwarmGenome,
    g2: SwarmGenome,
    config: EvolutionConfig,
) -> float:
    """
    Compute distance between two genomes.

    Uses NEAT-style distance with excess, disjoint, and weight difference.
    """
    # Agent genes
    agent_innovations_1 = set(g1.agents.keys())
    agent_innovations_2 = set(g2.agents.keys())

    max_innovation = max(
        max(agent_innovations_1, default=0),
        max(agent_innovations_2, default=0),
    )

    matching = agent_innovations_1 & agent_innovations_2
    disjoint = len((agent_innovations_1 | agent_innovations_2) - matching)
    excess = 0  # Simplified: treat all non-matching as disjoint

    # Connection genes
    conn_innovations_1 = set(g1.connections.keys())
    conn_innovations_2 = set(g2.connections.keys())

    conn_matching = conn_innovations_1 & conn_innovations_2
    conn_disjoint = len((conn_innovations_1 | conn_innovations_2) - conn_matching)

    # Weight difference for matching connections
    weight_diff = 0.0
    if conn_matching:
        for innovation in conn_matching:
            weight_diff += abs(
                g1.connections[innovation].weight - g2.connections[innovation].weight
            )
        weight_diff /= len(conn_matching)

    # Normalize
    n = max(len(g1.agents), len(g2.agents), 1)

    distance = (
        config.excess_coefficient * excess / n +
        config.disjoint_coefficient * (disjoint + conn_disjoint) / n +
        config.weight_coefficient * weight_diff
    )

    return distance


class GeneticOptimizer:
    """
    Genetic algorithm optimizer for swarm evolution.

    Manages population, selection, and evolution.
    """

    def __init__(
        self,
        config: EvolutionConfig,
        fitness_fn: Callable[[SwarmGenome], float],
    ):
        self.config = config
        self.fitness_fn = fitness_fn

        self.population: List[SwarmGenome] = []
        self.species: List[Species] = []
        self.generation = 0

        self.tracker = InnovationTracker()
        self.mutator = SwarmMutator(config, self.tracker)
        self.crossover = SwarmCrossover()

        self.best_genome: Optional[SwarmGenome] = None
        self.history: List[Dict[str, float]] = []

    def initialize_population(
        self,
        initial_genome: Optional[SwarmGenome] = None,
    ) -> None:
        """Initialize population with random or seed genome."""
        if initial_genome:
            # Create variations of seed
            self.population = [initial_genome.copy()]
            for _ in range(self.config.population_size - 1):
                mutated = self.mutator.mutate(initial_genome.copy())
                self.population.append(mutated)
        else:
            # Create random population
            for _ in range(self.config.population_size):
                genome = self._create_random_genome()
                self.population.append(genome)

    def _create_random_genome(self) -> SwarmGenome:
        """Create a random genome."""
        genome = SwarmGenome(hidden_dim=64)

        # Random number of agents
        num_agents = random.randint(
            self.config.min_agents,
            min(self.config.min_agents + 10, self.config.max_agents)
        )

        for i in range(num_agents):
            innovation = self.tracker.get_agent_innovation(i)
            agent = AgentGene(
                innovation_number=innovation,
                agent_id=i,
                agent_type=random.randint(0, 3),
                position=(random.random(), random.random()),
            )
            genome.agents[i] = agent

        # Random connections (sparse)
        for i in range(num_agents):
            num_connections = random.randint(1, 3)
            targets = random.sample(
                [j for j in range(num_agents) if j != i],
                min(num_connections, num_agents - 1)
            )
            for target in targets:
                innovation = self.tracker.get_connection_innovation(i, target)
                conn = ConnectionGene(
                    innovation_number=innovation,
                    from_agent=i,
                    to_agent=target,
                    weight=random.gauss(0, 1),
                )
                genome.connections[innovation] = conn

        return genome

    def evaluate_population(self) -> None:
        """Evaluate fitness of all genomes."""
        for genome in self.population:
            genome.fitness = self.fitness_fn(genome)

        # Track best
        current_best = max(self.population, key=lambda g: g.fitness)
        if self.best_genome is None or current_best.fitness > self.best_genome.fitness:
            self.best_genome = current_best.copy()

    def speciate(self) -> None:
        """Assign genomes to species."""
        # Clear existing species
        for species in self.species:
            species.clear_members()

        # Assign each genome
        for genome in self.population:
            placed = False

            for species in self.species:
                distance = genome_distance(genome, species.representative, self.config)
                if distance < self.config.species_threshold:
                    species.add_member(genome)
                    placed = True
                    break

            if not placed:
                # New species
                new_species = Species(len(self.species), genome)
                self.species.append(new_species)

        # Remove empty species
        self.species = [s for s in self.species if len(s.members) > 0]

        # Update representatives
        for species in self.species:
            species.update_representative()

    def select_parents(self) -> List[Tuple[SwarmGenome, SwarmGenome]]:
        """Select parent pairs for reproduction."""
        parents = []

        # Calculate offspring per species
        total_fitness = sum(
            sum(g.fitness for g in s.members) / len(s.members)
            for s in self.species
        )

        for species in self.species:
            species_fitness = sum(g.fitness for g in species.members) / len(species.members)
            num_offspring = int(
                (species_fitness / max(total_fitness, 1e-8))
                * self.config.population_size
            )

            for _ in range(num_offspring):
                # Tournament selection within species
                p1 = self._tournament_select(species.members)
                p2 = self._tournament_select(species.members)
                parents.append((p1, p2))

        return parents

    def _tournament_select(self, pool: List[SwarmGenome]) -> SwarmGenome:
        """Tournament selection."""
        contestants = random.sample(
            pool,
            min(self.config.tournament_size, len(pool))
        )
        return max(contestants, key=lambda g: g.fitness)

    def evolve(self) -> None:
        """Perform one generation of evolution."""
        self.generation += 1

        # Evaluate
        self.evaluate_population()

        # Speciate
        self.speciate()

        # Record history
        self.history.append({
            "generation": self.generation,
            "best_fitness": self.best_genome.fitness if self.best_genome else 0,
            "avg_fitness": sum(g.fitness for g in self.population) / len(self.population),
            "num_species": len(self.species),
            "population_size": len(self.population),
        })

        # Select parents
        parents = self.select_parents()

        # Create next generation
        new_population = []

        # Elitism
        sorted_pop = sorted(self.population, key=lambda g: g.fitness, reverse=True)
        for i in range(self.config.elite_size):
            new_population.append(sorted_pop[i].copy())

        # Reproduction
        while len(new_population) < self.config.population_size:
            if not parents:
                # Fallback to mutation only
                parent = random.choice(self.population)
                child = self.mutator.mutate(parent.copy())
            elif random.random() < self.config.crossover_rate:
                p1, p2 = random.choice(parents)
                child = self.crossover.crossover(p1, p2)
                child = self.mutator.mutate(child)
            else:
                parent = random.choice([p for pair in parents for p in pair])
                child = self.mutator.mutate(parent.copy())

            child.generation = self.generation
            new_population.append(child)

        self.population = new_population

    def get_best(self) -> SwarmGenome:
        """Get best genome found."""
        return self.best_genome

    def get_statistics(self) -> Dict[str, Any]:
        """Get evolution statistics."""
        if not self.history:
            return {}

        return {
            "generations": self.generation,
            "best_fitness": self.best_genome.fitness if self.best_genome else 0,
            "best_num_agents": len(self.best_genome.agents) if self.best_genome else 0,
            "best_num_connections": len(self.best_genome.connections) if self.best_genome else 0,
            "num_species": len(self.species),
            "history": self.history,
        }


class CMAES:
    """
    CMA-ES optimizer for continuous parameter optimization.

    Covariance Matrix Adaptation Evolution Strategy.
    """

    def __init__(
        self,
        dim: int,
        population_size: Optional[int] = None,
        sigma: float = 0.5,
    ):
        self.dim = dim
        self.sigma = sigma

        # Population size
        self.lambda_ = population_size or 4 + int(3 * math.log(dim))
        self.mu = self.lambda_ // 2

        # Weights
        weights = [math.log(self.mu + 0.5) - math.log(i + 1) for i in range(self.mu)]
        self.weights = np.array(weights) / sum(weights)
        self.mu_eff = 1.0 / sum(self.weights ** 2)

        # Adaptation parameters
        self.cc = (4 + self.mu_eff / dim) / (dim + 4 + 2 * self.mu_eff / dim)
        self.cs = (self.mu_eff + 2) / (dim + self.mu_eff + 5)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mu_eff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((dim + 2) ** 2 + self.mu_eff)
        )
        self.damps = 1 + 2 * max(0, math.sqrt((self.mu_eff - 1) / (dim + 1)) - 1) + self.cs

        # State
        self.mean = np.zeros(dim)
        self.C = np.eye(dim)
        self.ps = np.zeros(dim)
        self.pc = np.zeros(dim)

        self.generation = 0
        self.best_solution = None
        self.best_fitness = float('-inf')

    def ask(self) -> np.ndarray:
        """Sample new solutions."""
        # Eigendecomposition of covariance
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        # Sample
        samples = np.zeros((self.lambda_, self.dim))
        for i in range(self.lambda_):
            z = np.random.randn(self.dim)
            samples[i] = self.mean + self.sigma * eigenvectors @ (np.sqrt(eigenvalues) * z)

        return samples

    def tell(self, solutions: np.ndarray, fitnesses: np.ndarray) -> None:
        """Update based on evaluated solutions."""
        # Sort by fitness (descending)
        indices = np.argsort(-fitnesses)
        solutions = solutions[indices]
        fitnesses = fitnesses[indices]

        # Update best
        if fitnesses[0] > self.best_fitness:
            self.best_fitness = fitnesses[0]
            self.best_solution = solutions[0].copy()

        # Selected solutions
        selected = solutions[:self.mu]

        # Update mean
        old_mean = self.mean.copy()
        self.mean = np.sum(self.weights[:, None] * selected, axis=0)

        # Update evolution paths
        eigenvalues, eigenvectors = np.linalg.eigh(self.C)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        invsqrt_C = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        self.ps = (1 - self.cs) * self.ps + \
                  math.sqrt(self.cs * (2 - self.cs) * self.mu_eff) * \
                  invsqrt_C @ (self.mean - old_mean) / self.sigma

        hsig = np.linalg.norm(self.ps) / \
               math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1))) / \
               (1.4 + 2 / (self.dim + 1)) < 1.0

        self.pc = (1 - self.cc) * self.pc + \
                  hsig * math.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * \
                  (self.mean - old_mean) / self.sigma

        # Update covariance
        artmp = (selected - old_mean) / self.sigma
        self.C = (1 - self.c1 - self.cmu) * self.C + \
                 self.c1 * np.outer(self.pc, self.pc) + \
                 self.cmu * artmp.T @ np.diag(self.weights) @ artmp

        # Update sigma
        self.sigma *= math.exp(
            (self.cs / self.damps) *
            (np.linalg.norm(self.ps) / math.sqrt(self.dim) - 1)
        )

        self.generation += 1

    def optimize(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        max_generations: int = 100,
    ) -> Tuple[np.ndarray, float]:
        """Run optimization."""
        for _ in range(max_generations):
            solutions = self.ask()
            fitnesses = np.array([fitness_fn(s) for s in solutions])
            self.tell(solutions, fitnesses)

        return self.best_solution, self.best_fitness
