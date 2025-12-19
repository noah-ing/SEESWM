"""
SwarmGraph - Manages agent connectivity and collective computation.

The swarm is organized as a graph where nodes are agents and edges
define which agents can communicate. Different topologies lead to
different emergent behaviors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import random

import networkx as nx
import torch
import torch.nn.functional as F

from ..agents.micro_agent import MicroAgent, AgentConfig, AgentType
from .messaging import MessageBus, Message


class TopologyType(Enum):
    """Available graph topologies for the swarm."""

    RANDOM = auto()  # Erdos-Renyi random graph
    SMALL_WORLD = auto()  # Watts-Strogatz (efficient + clustered)
    SCALE_FREE = auto()  # Barabasi-Albert (hub-and-spoke)
    MODULAR = auto()  # Clusters for each specialization
    HIERARCHICAL = auto()  # Layers: perception -> reasoning -> planning
    FULLY_CONNECTED = auto()  # All-to-all (baseline)


@dataclass
class SwarmConfig:
    """Configuration for the swarm."""

    num_agents: int = 100
    topology: TopologyType = TopologyType.SMALL_WORLD
    message_passing_rounds: int = 3
    input_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 64
    message_dim: int = 64

    # Topology-specific parameters
    random_edge_prob: float = 0.1  # For RANDOM
    small_world_k: int = 4  # Neighbors in ring for SMALL_WORLD
    small_world_p: float = 0.1  # Rewiring probability
    scale_free_m: int = 3  # Edges per new node

    # Agent type distribution (should sum to num_agents)
    num_perception: int = 25
    num_reasoning: int = 25
    num_memory: int = 25
    num_planning: int = 25


class SwarmGraph:
    """
    Manages the swarm of micro-agents and their interactions.

    Key responsibilities:
    - Build and maintain graph topology
    - Orchestrate message passing rounds
    - Distribute inputs and collect outputs
    - Compute synergy metrics
    """

    def __init__(
        self,
        config: Optional[SwarmConfig] = None,
        device: str = "cpu",
    ):
        self.config = config or SwarmConfig()
        self.device = device

        # Build the graph topology
        self.graph = self._build_topology()

        # Create agents
        self.agents: dict[int, MicroAgent] = {}
        self._create_agents()

        # Message bus for inter-agent communication
        self.message_bus = MessageBus(
            num_agents=self.config.num_agents,
            message_dim=self.config.message_dim,
            device=device,
        )

        # Track which agents receive input and which produce output
        self.input_agents: list[int] = []
        self.output_agents: list[int] = []
        self._assign_io_roles()

    def _build_topology(self) -> nx.Graph:
        """Build graph based on configured topology."""
        n = self.config.num_agents
        topology = self.config.topology

        if topology == TopologyType.RANDOM:
            graph = nx.erdos_renyi_graph(n, self.config.random_edge_prob)
        elif topology == TopologyType.SMALL_WORLD:
            graph = nx.watts_strogatz_graph(
                n, self.config.small_world_k, self.config.small_world_p
            )
        elif topology == TopologyType.SCALE_FREE:
            graph = nx.barabasi_albert_graph(n, self.config.scale_free_m)
        elif topology == TopologyType.MODULAR:
            graph = self._build_modular_topology()
        elif topology == TopologyType.HIERARCHICAL:
            graph = self._build_hierarchical_topology()
        elif topology == TopologyType.FULLY_CONNECTED:
            graph = nx.complete_graph(n)
        else:
            raise ValueError(f"Unknown topology: {topology}")

        # Ensure graph is connected
        if not nx.is_connected(graph):
            # Connect components with minimal edges
            components = list(nx.connected_components(graph))
            for i in range(len(components) - 1):
                node1 = random.choice(list(components[i]))
                node2 = random.choice(list(components[i + 1]))
                graph.add_edge(node1, node2)

        return graph

    def _build_modular_topology(self) -> nx.Graph:
        """Build modular topology with clusters for each agent type."""
        graph = nx.Graph()
        n = self.config.num_agents

        # Create clusters
        cluster_sizes = [
            self.config.num_perception,
            self.config.num_reasoning,
            self.config.num_memory,
            self.config.num_planning,
        ]

        # Dense connections within clusters
        node_id = 0
        cluster_nodes: list[list[int]] = []
        for size in cluster_sizes:
            nodes = list(range(node_id, node_id + size))
            cluster_nodes.append(nodes)
            # Create dense subgraph within cluster
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i + 1 :]:
                    if random.random() < 0.3:  # 30% internal connectivity
                        graph.add_edge(n1, n2)
            node_id += size

        # Sparse connections between clusters
        for i, cluster1 in enumerate(cluster_nodes):
            for cluster2 in cluster_nodes[i + 1 :]:
                # Add a few inter-cluster edges
                num_bridges = max(1, len(cluster1) // 5)
                for _ in range(num_bridges):
                    n1 = random.choice(cluster1)
                    n2 = random.choice(cluster2)
                    graph.add_edge(n1, n2)

        return graph

    def _build_hierarchical_topology(self) -> nx.Graph:
        """Build hierarchical topology: perception -> reasoning -> planning."""
        graph = nx.Graph()

        # Layer assignments
        perception_ids = list(range(self.config.num_perception))
        reasoning_start = self.config.num_perception
        reasoning_ids = list(
            range(reasoning_start, reasoning_start + self.config.num_reasoning)
        )
        memory_start = reasoning_start + self.config.num_reasoning
        memory_ids = list(range(memory_start, memory_start + self.config.num_memory))
        planning_start = memory_start + self.config.num_memory
        planning_ids = list(
            range(planning_start, planning_start + self.config.num_planning)
        )

        # Within-layer connections (sparse)
        for layer in [perception_ids, reasoning_ids, memory_ids, planning_ids]:
            for i, n1 in enumerate(layer):
                for n2 in layer[i + 1 :]:
                    if random.random() < 0.1:
                        graph.add_edge(n1, n2)

        # Between-layer connections (feed-forward)
        layers = [perception_ids, reasoning_ids, memory_ids, planning_ids]
        for i in range(len(layers) - 1):
            for n1 in layers[i]:
                # Connect to ~20% of next layer
                for n2 in layers[i + 1]:
                    if random.random() < 0.2:
                        graph.add_edge(n1, n2)

        # Memory connects to all layers (skip connections)
        for n1 in memory_ids:
            for layer in [perception_ids, planning_ids]:
                for n2 in layer:
                    if random.random() < 0.1:
                        graph.add_edge(n1, n2)

        return graph

    def _assign_agent_type(self, agent_id: int) -> AgentType:
        """Assign agent type based on ID and configuration."""
        if agent_id < self.config.num_perception:
            return AgentType.PERCEPTION
        elif agent_id < self.config.num_perception + self.config.num_reasoning:
            return AgentType.REASONING
        elif agent_id < (
            self.config.num_perception
            + self.config.num_reasoning
            + self.config.num_memory
        ):
            return AgentType.MEMORY
        else:
            return AgentType.PLANNING

    def _create_agents(self) -> None:
        """Create all agents in the swarm."""
        for agent_id in range(self.config.num_agents):
            agent_type = self._assign_agent_type(agent_id)
            agent_config = AgentConfig(
                agent_type=agent_type,
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
                message_dim=self.config.message_dim,
            )
            self.agents[agent_id] = MicroAgent(
                agent_id=agent_id,
                config=agent_config,
                device=self.device,
            )

    def _assign_io_roles(self) -> None:
        """Assign which agents receive input and produce output."""
        # Perception agents receive input
        self.input_agents = [
            i for i, a in self.agents.items() if a.agent_type == AgentType.PERCEPTION
        ]
        # Planning agents produce output
        self.output_agents = [
            i for i, a in self.agents.items() if a.agent_type == AgentType.PLANNING
        ]

        # Fallback if no specialized agents
        if not self.input_agents:
            self.input_agents = list(range(min(10, self.config.num_agents)))
        if not self.output_agents:
            self.output_agents = list(
                range(
                    max(0, self.config.num_agents - 10), self.config.num_agents
                )
            )

    def reset(self, batch_size: int = 1) -> None:
        """Reset all agents and message bus for new episode."""
        for agent in self.agents.values():
            agent.reset_state(batch_size)
        self.message_bus.reset()

    def step(self, global_input: torch.Tensor) -> torch.Tensor:
        """
        Run one forward pass through the swarm.

        1. Distribute input to perception agents
        2. Run message passing for K rounds
        3. Collect and aggregate outputs from planning agents

        Args:
            global_input: Input tensor [batch, input_dim]

        Returns:
            Collective output [batch, output_dim]
        """
        batch_size = global_input.shape[0]

        # Ensure agents are initialized
        for agent in self.agents.values():
            if agent.local_state is None:
                agent.reset_state(batch_size)

        # Initialize messages from input
        # Input agents receive the global input; others get zeros
        current_outputs: dict[int, torch.Tensor] = {}

        # First, run input agents with the global input
        for agent_id in self.input_agents:
            agent = self.agents[agent_id]
            # Input agents get the global input, no neighbor messages yet
            output = agent.forward(global_input, [])
            current_outputs[agent_id] = output

        # Non-input agents start with zero input
        zero_input = torch.zeros(batch_size, self.config.input_dim, device=self.device)
        for agent_id in self.agents:
            if agent_id not in current_outputs:
                current_outputs[agent_id] = zero_input.clone()

        # Message passing rounds
        for round_idx in range(self.config.message_passing_rounds):
            new_outputs: dict[int, torch.Tensor] = {}

            for agent_id, agent in self.agents.items():
                # Gather messages from neighbors
                neighbor_ids = list(self.graph.neighbors(agent_id))
                neighbor_messages = [
                    current_outputs[n_id] for n_id in neighbor_ids if n_id in current_outputs
                ]

                # Determine input for this agent
                if agent_id in self.input_agents:
                    agent_input = global_input
                else:
                    agent_input = zero_input

                # Process
                output = agent.forward(agent_input, neighbor_messages)
                new_outputs[agent_id] = output

            current_outputs = new_outputs

        # Aggregate outputs from planning agents
        output_tensors = [current_outputs[i] for i in self.output_agents]
        if output_tensors:
            # Mean pooling over output agents
            stacked = torch.stack(output_tensors, dim=0)
            collective_output = stacked.mean(dim=0)
        else:
            collective_output = torch.zeros(
                batch_size, self.config.output_dim, device=self.device
            )

        return collective_output

    def compute_synergy(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute synergy metrics for the swarm.

        Synergy = collective performance - sum of individual performances
        Positive synergy means the whole is greater than sum of parts.

        Args:
            inputs: Test inputs [num_samples, input_dim]
            labels: Ground truth [num_samples, output_dim]

        Returns:
            Dictionary with synergy metrics
        """
        # Get collective prediction
        collective_output = self.step(inputs)
        collective_error = F.mse_loss(collective_output, labels).item()

        # Get individual predictions (each agent in isolation)
        individual_errors = []
        zero_input = torch.zeros_like(inputs)

        for agent_id in self.output_agents:
            agent = self.agents[agent_id]
            agent.reset_state(inputs.shape[0])
            # Run agent alone with input
            if agent_id in self.input_agents:
                output = agent.forward(inputs, [])
            else:
                output = agent.forward(zero_input, [])
            error = F.mse_loss(output, labels).item()
            individual_errors.append(error)

        avg_individual_error = sum(individual_errors) / max(1, len(individual_errors))

        # Synergy: if collective error is lower, we have positive synergy
        # Convert to "performance" (inverse of error) for intuitive interpretation
        collective_perf = 1.0 / (1.0 + collective_error)
        avg_individual_perf = 1.0 / (1.0 + avg_individual_error)

        synergy = collective_perf - avg_individual_perf

        return {
            "synergy": synergy,
            "collective_error": collective_error,
            "avg_individual_error": avg_individual_error,
            "collective_performance": collective_perf,
            "avg_individual_performance": avg_individual_perf,
        }

    def get_topology_stats(self) -> dict:
        """Get statistics about the graph topology."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            "clustering_coefficient": nx.average_clustering(self.graph),
            "is_connected": nx.is_connected(self.graph),
            "diameter": nx.diameter(self.graph) if nx.is_connected(self.graph) else float("inf"),
        }

    def get_agent_stats(self) -> list[dict]:
        """Get statistics for all agents."""
        return [agent.get_stats() for agent in self.agents.values()]

    @property
    def total_parameters(self) -> int:
        """Total parameters across all agents."""
        return sum(agent.num_parameters for agent in self.agents.values())

    def state_dict(self) -> dict:
        """Get state for saving."""
        return {
            "config": self.config,
            "agents": {i: a.state_dict() for i, a in self.agents.items()},
            "graph_edges": list(self.graph.edges()),
        }

    def load_state_dict(self, state: dict) -> None:
        """Load saved state."""
        self.config = state["config"]
        for i, agent_state in state["agents"].items():
            self.agents[int(i)].load_state_dict(agent_state)
        # Rebuild graph from edges
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.config.num_agents))
        self.graph.add_edges_from(state["graph_edges"])

    @classmethod
    def from_genome(cls, genome: dict, device: str = "cpu") -> SwarmGraph:
        """Create a SwarmGraph from an evolutionary genome."""
        topology_map = {
            "random": TopologyType.RANDOM,
            "small_world": TopologyType.SMALL_WORLD,
            "scale_free": TopologyType.SCALE_FREE,
            "modular": TopologyType.MODULAR,
            "hierarchical": TopologyType.HIERARCHICAL,
        }

        config = SwarmConfig(
            num_agents=sum([
                genome.get("num_perception", 25),
                genome.get("num_reasoning", 25),
                genome.get("num_memory", 25),
                genome.get("num_planning", 25),
            ]),
            topology=topology_map.get(genome.get("topology", "small_world"), TopologyType.SMALL_WORLD),
            hidden_dim=genome.get("hidden_dim", 128),
            num_perception=genome.get("num_perception", 25),
            num_reasoning=genome.get("num_reasoning", 25),
            num_memory=genome.get("num_memory", 25),
            num_planning=genome.get("num_planning", 25),
        )

        return cls(config=config, device=device)
