"""
SpecializedSwarmGraph - Swarm with specialized agents and typed messaging.

Combines:
- Specialized agent architectures (Perception, Reasoning, Memory, Planning)
- Typed message passing with semantic routing
- Role emergence tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import random

import networkx as nx
import torch
import torch.nn as nn

from ..agents.micro_agent import AgentConfig, AgentType
from ..agents.specializations import (
    SpecializedAgent,
    PerceptionNetwork,
    ReasoningNetwork,
    MemoryNetwork,
    PlanningNetwork,
)
from .graph import SwarmConfig, TopologyType
from .typed_messaging import (
    TypedMessageBus,
    MessageType,
    TypedMessage,
    MessageEncoder,
    TypeAwareAggregator,
    AGENT_MESSAGE_TYPES,
)


@dataclass
class SpecializedSwarmConfig(SwarmConfig):
    """Extended configuration for specialized swarms."""

    # Enable/disable typed messaging
    use_typed_messaging: bool = True

    # Message type embedding
    encode_message_types: bool = True

    # Role emergence tracking
    track_role_emergence: bool = True

    # Specialization training parameters
    specialization_bonus: float = 0.1  # Bonus for using specialized capabilities


class SpecializedSwarmGraph:
    """
    Swarm with specialized agent architectures and typed messaging.

    Key features:
    - Each agent type has distinct architectural biases
    - Messages carry semantic type information
    - Role emergence metrics track specialization development
    """

    def __init__(
        self,
        config: Optional[SpecializedSwarmConfig] = None,
        device: str = "cpu",
    ):
        self.config = config or SpecializedSwarmConfig()
        self.device = device

        # Build graph topology
        self.graph = self._build_topology()

        # Create specialized agents
        self.agents: Dict[int, SpecializedAgent] = {}
        self._create_agents()

        # Typed message bus (use output_dim since that's message content size)
        self.message_bus = TypedMessageBus(
            num_agents=self.config.num_agents,
            message_dim=self.config.output_dim,
            device=device,
        )
        self._register_agents_with_bus()

        # Message encoder (optional)
        if self.config.encode_message_types:
            self.message_encoder = MessageEncoder(
                self.config.message_dim
            ).to(device)
        else:
            self.message_encoder = None

        # Type-aware aggregator for each agent
        self.aggregators: Dict[int, TypeAwareAggregator] = {}
        self._create_aggregators()

        # I/O roles
        self.input_agents: List[int] = []
        self.output_agents: List[int] = []
        self._assign_io_roles()

        # Role emergence tracking
        if self.config.track_role_emergence:
            self.role_metrics = RoleEmergenceTracker(self.agents)

    def _build_topology(self) -> nx.Graph:
        """Build graph topology (same as SwarmGraph)."""
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

        # Ensure connected
        if not nx.is_connected(graph):
            components = list(nx.connected_components(graph))
            for i in range(len(components) - 1):
                node1 = random.choice(list(components[i]))
                node2 = random.choice(list(components[i + 1]))
                graph.add_edge(node1, node2)

        return graph

    def _build_modular_topology(self) -> nx.Graph:
        """Build modular topology with clusters per agent type."""
        graph = nx.Graph()
        n = self.config.num_agents
        graph.add_nodes_from(range(n))

        cluster_sizes = [
            self.config.num_perception,
            self.config.num_reasoning,
            self.config.num_memory,
            self.config.num_planning,
        ]

        # Dense intra-cluster connections
        node_id = 0
        cluster_nodes: List[List[int]] = []
        for size in cluster_sizes:
            nodes = list(range(node_id, node_id + size))
            cluster_nodes.append(nodes)
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i + 1:]:
                    if random.random() < 0.3:
                        graph.add_edge(n1, n2)
            node_id += size

        # Sparse inter-cluster connections
        for i, cluster1 in enumerate(cluster_nodes):
            for cluster2 in cluster_nodes[i + 1:]:
                num_bridges = max(1, len(cluster1) // 5)
                for _ in range(num_bridges):
                    n1 = random.choice(cluster1)
                    n2 = random.choice(cluster2)
                    graph.add_edge(n1, n2)

        return graph

    def _build_hierarchical_topology(self) -> nx.Graph:
        """Build hierarchical: perception -> reasoning -> planning."""
        graph = nx.Graph()
        graph.add_nodes_from(range(self.config.num_agents))

        # Layer assignments
        perception_ids = list(range(self.config.num_perception))
        reasoning_start = self.config.num_perception
        reasoning_ids = list(
            range(reasoning_start, reasoning_start + self.config.num_reasoning)
        )
        memory_start = reasoning_start + self.config.num_reasoning
        memory_ids = list(
            range(memory_start, memory_start + self.config.num_memory)
        )
        planning_start = memory_start + self.config.num_memory
        planning_ids = list(
            range(planning_start, planning_start + self.config.num_planning)
        )

        # Within-layer (sparse)
        for layer in [perception_ids, reasoning_ids, memory_ids, planning_ids]:
            for i, n1 in enumerate(layer):
                for n2 in layer[i + 1:]:
                    if random.random() < 0.1:
                        graph.add_edge(n1, n2)

        # Between-layer (feed-forward)
        layers = [perception_ids, reasoning_ids, memory_ids, planning_ids]
        for i in range(len(layers) - 1):
            for n1 in layers[i]:
                for n2 in layers[i + 1]:
                    if random.random() < 0.2:
                        graph.add_edge(n1, n2)

        # Memory skip connections
        for n1 in memory_ids:
            for layer in [perception_ids, planning_ids]:
                for n2 in layer:
                    if random.random() < 0.1:
                        graph.add_edge(n1, n2)

        return graph

    def _assign_agent_type(self, agent_id: int) -> AgentType:
        """Assign type based on ID."""
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
        """Create specialized agents."""
        for agent_id in range(self.config.num_agents):
            agent_type = self._assign_agent_type(agent_id)
            agent_config = AgentConfig(
                agent_type=agent_type,
                input_dim=self.config.input_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
                message_dim=self.config.message_dim,
            )
            self.agents[agent_id] = SpecializedAgent(
                agent_id=agent_id,
                config=agent_config,
                device=self.device,
            )

    def _register_agents_with_bus(self) -> None:
        """Register agents with the typed message bus."""
        for agent_id, agent in self.agents.items():
            self.message_bus.register_agent(agent_id, agent.config.agent_type)

    def _create_aggregators(self) -> None:
        """Create type-aware aggregators for each agent."""
        # Use output_dim since that's what agents output as message content
        for agent_id in self.agents:
            self.aggregators[agent_id] = TypeAwareAggregator(
                self.config.output_dim,
                self.config.message_dim,
            ).to(self.device)

    def _assign_io_roles(self) -> None:
        """Assign input/output roles."""
        self.input_agents = [
            i for i, a in self.agents.items()
            if a.config.agent_type == AgentType.PERCEPTION
        ]
        self.output_agents = [
            i for i, a in self.agents.items()
            if a.config.agent_type == AgentType.PLANNING
        ]

        if not self.input_agents:
            self.input_agents = list(range(min(10, self.config.num_agents)))
        if not self.output_agents:
            self.output_agents = list(
                range(max(0, self.config.num_agents - 10), self.config.num_agents)
            )

    def reset(self, batch_size: int = 1) -> None:
        """Reset all agents and message bus."""
        for agent in self.agents.values():
            agent.reset_state(batch_size)
        self.message_bus.reset()

    def step(self, global_input: torch.Tensor) -> torch.Tensor:
        """
        Run one forward pass with typed messaging.

        1. Input agents receive global input
        2. Message passing with typed messages
        3. Aggregate output from planning agents
        """
        batch_size = global_input.shape[0]

        # Ensure agents are initialized
        for agent in self.agents.values():
            if agent.local_state is None:
                agent.reset_state(batch_size)

        # Run input agents first
        current_outputs: Dict[int, torch.Tensor] = {}
        zero_input = torch.zeros(
            batch_size, self.config.input_dim, device=self.device
        )

        # First pass: input agents
        for agent_id in self.input_agents:
            agent = self.agents[agent_id]
            neighbor_msg = torch.zeros(
                batch_size, self.config.message_dim, device=self.device
            )
            output, new_state = agent.network(
                global_input, neighbor_msg, agent.local_state
            )
            agent.local_state = new_state
            current_outputs[agent_id] = output

            # Send typed message
            msg_types = AGENT_MESSAGE_TYPES.get(agent.config.agent_type, [MessageType.GENERIC])
            self.message_bus.send(
                content=output.detach().mean(dim=0),  # Batch-averaged
                source_id=agent_id,
                message_type=msg_types[0] if msg_types else MessageType.GENERIC,
            )

        # Initialize non-input agents
        for agent_id in self.agents:
            if agent_id not in current_outputs:
                current_outputs[agent_id] = zero_input.clone()

        # Message passing rounds
        for round_idx in range(self.config.message_passing_rounds):
            self.message_bus.step()
            new_outputs: Dict[int, torch.Tensor] = {}

            for agent_id, agent in self.agents.items():
                # Get typed messages
                messages = self.message_bus.receive(agent_id)

                # Aggregate messages
                if messages:
                    msg_agg = self.aggregators[agent_id](messages, self.device)
                    msg_agg = msg_agg.expand(batch_size, -1)
                else:
                    msg_agg = torch.zeros(
                        batch_size, self.config.message_dim, device=self.device
                    )

                # Determine input
                if agent_id in self.input_agents:
                    agent_input = global_input
                else:
                    agent_input = zero_input

                # Forward pass
                output, new_state = agent.network(
                    agent_input, msg_agg, agent.local_state
                )
                agent.local_state = new_state
                new_outputs[agent_id] = output

                # Send output as typed message
                msg_types = AGENT_MESSAGE_TYPES.get(
                    agent.config.agent_type, [MessageType.GENERIC]
                )
                self.message_bus.send(
                    content=output.detach().mean(dim=0),
                    source_id=agent_id,
                    message_type=msg_types[0] if msg_types else MessageType.GENERIC,
                )

            current_outputs = new_outputs

        # Aggregate planning outputs
        output_tensors = [current_outputs[i] for i in self.output_agents]
        if output_tensors:
            stacked = torch.stack(output_tensors, dim=0)
            collective_output = stacked.mean(dim=0)
        else:
            collective_output = torch.zeros(
                batch_size, self.config.output_dim, device=self.device
            )

        # Update role metrics if tracking
        if self.config.track_role_emergence and hasattr(self, 'role_metrics'):
            self.role_metrics.update(self.message_bus.get_stats())

        return collective_output

    def get_topology_stats(self) -> Dict:
        """Get graph statistics."""
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "avg_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes(),
            "clustering_coefficient": nx.average_clustering(self.graph),
            "is_connected": nx.is_connected(self.graph),
            "diameter": nx.diameter(self.graph) if nx.is_connected(self.graph) else float("inf"),
        }

    def get_message_stats(self) -> Dict:
        """Get typed message statistics."""
        return self.message_bus.get_stats()

    def get_role_emergence_metrics(self) -> Dict:
        """Get role emergence metrics."""
        if hasattr(self, 'role_metrics'):
            return self.role_metrics.get_metrics()
        return {}

    @property
    def total_parameters(self) -> int:
        """Total parameters across all agents and aggregators."""
        agent_params = sum(
            sum(p.numel() for p in agent.network.parameters())
            for agent in self.agents.values()
        )
        agg_params = sum(
            sum(p.numel() for p in agg.parameters())
            for agg in self.aggregators.values()
        )
        encoder_params = (
            sum(p.numel() for p in self.message_encoder.parameters())
            if self.message_encoder else 0
        )
        return agent_params + agg_params + encoder_params

    def parameters(self):
        """Get all trainable parameters."""
        params = []
        for agent in self.agents.values():
            params.extend(agent.network.parameters())
        for agg in self.aggregators.values():
            params.extend(agg.parameters())
        if self.message_encoder:
            params.extend(self.message_encoder.parameters())
        return params

    def state_dict(self) -> Dict:
        """Get state for saving."""
        return {
            "config": self.config,
            "agents": {i: a.network.state_dict() for i, a in self.agents.items()},
            "aggregators": {i: a.state_dict() for i, a in self.aggregators.items()},
            "graph_edges": list(self.graph.edges()),
            "message_encoder": self.message_encoder.state_dict() if self.message_encoder else None,
        }

    def load_state_dict(self, state: Dict) -> None:
        """Load saved state."""
        for i, agent_state in state["agents"].items():
            self.agents[int(i)].network.load_state_dict(agent_state)
        for i, agg_state in state["aggregators"].items():
            self.aggregators[int(i)].load_state_dict(agg_state)
        if state["message_encoder"] and self.message_encoder:
            self.message_encoder.load_state_dict(state["message_encoder"])


class RoleEmergenceTracker:
    """
    Track metrics related to agent role emergence and specialization.

    Measures:
    - Message type distribution per agent
    - Communication patterns
    - Specialization entropy
    """

    def __init__(self, agents: Dict[int, SpecializedAgent]):
        self.agents = agents
        self.message_counts: Dict[int, Dict[MessageType, int]] = {
            i: {t: 0 for t in MessageType} for i in agents
        }
        self.activation_history: Dict[int, List[float]] = {
            i: [] for i in agents
        }
        self.total_updates = 0

    def update(self, message_stats: Dict) -> None:
        """Update metrics with new message statistics."""
        self.total_updates += 1

        # Track message type distribution
        for msg_type, count in message_stats.get("by_type", {}).items():
            try:
                mt = MessageType[msg_type]
                # Distribute across relevant agents (simplified)
                for agent_id, agent in self.agents.items():
                    if mt in AGENT_MESSAGE_TYPES.get(agent.config.agent_type, []):
                        self.message_counts[agent_id][mt] += count
            except KeyError:
                pass

    def get_metrics(self) -> Dict:
        """Compute role emergence metrics."""
        import math

        metrics = {}

        # Specialization entropy per agent
        entropies = []
        for agent_id, counts in self.message_counts.items():
            total = sum(counts.values())
            if total > 0:
                probs = [c / total for c in counts.values() if c > 0]
                entropy = -sum(p * math.log(p) for p in probs)
                entropies.append(entropy)

        metrics["avg_specialization_entropy"] = (
            sum(entropies) / len(entropies) if entropies else 0.0
        )

        # Message type concentration by agent type
        type_concentrations = {t: 0.0 for t in AgentType}
        for agent_id, counts in self.message_counts.items():
            agent_type = self.agents[agent_id].config.agent_type
            expected_types = AGENT_MESSAGE_TYPES.get(agent_type, [])
            total = sum(counts.values())
            if total > 0 and expected_types:
                expected_count = sum(counts[t] for t in expected_types)
                type_concentrations[agent_type] += expected_count / total

        metrics["type_concentrations"] = {
            t.name: type_concentrations[t] / max(1, sum(
                1 for a in self.agents.values() if a.config.agent_type == t
            ))
            for t in AgentType
        }

        metrics["total_updates"] = self.total_updates

        return metrics
