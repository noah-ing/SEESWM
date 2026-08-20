"""
SpecializedSwarmGraph - Swarm with specialized agents and typed messaging.

Combines:
- Specialized agent architectures (Perception, Reasoning, Memory, Planning)
- Typed message passing with semantic routing
- Descriptive role-pattern tracking
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx
import torch

from ..agents.micro_agent import AgentConfig, AgentType
from ..agents.specializations import (
    SpecializedAgent,
)
from .graph import (
    SWARM_STATE_SCHEMA_VERSION,
    SwarmConfig,
    TopologyType,
    _graph_edges_to_list,
    _require_exact_keys,
    _require_float,
    _require_int,
    _validate_exact_int_ids,
    _validate_graph_edges,
    _validate_tensor_state_dict,
    swarm_config_from_dict,
    swarm_config_to_dict,
)
from .typed_messaging import (
    TypedMessageBus,
    MessageType,
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


_SPECIALIZED_CONFIG_KEYS = set(swarm_config_to_dict(SwarmConfig())) | {
    "use_typed_messaging",
    "encode_message_types",
    "track_role_emergence",
    "specialization_bonus",
}


def specialized_swarm_config_to_dict(config: SpecializedSwarmConfig) -> dict:
    """Convert a specialized swarm config to weights-only-safe primitives."""
    if not isinstance(config, SpecializedSwarmConfig):
        raise TypeError("config must be a SpecializedSwarmConfig")
    for name in (
        "use_typed_messaging",
        "encode_message_types",
        "track_role_emergence",
    ):
        if type(getattr(config, name)) is not bool:
            raise TypeError(f"specialized swarm config {name} must be a native bool")
    if not math.isfinite(config.specialization_bonus) or config.specialization_bonus < 0:
        raise ValueError("specialization_bonus must be a finite non-negative number")
    result = swarm_config_to_dict(config)
    result.update(
        {
            "use_typed_messaging": bool(config.use_typed_messaging),
            "encode_message_types": bool(config.encode_message_types),
            "track_role_emergence": bool(config.track_role_emergence),
            "specialization_bonus": float(config.specialization_bonus),
        }
    )
    return result


def specialized_swarm_config_from_dict(data: object) -> SpecializedSwarmConfig:
    """Strictly reconstruct a specialized config from native primitives."""
    if not isinstance(data, dict):
        raise TypeError("specialized swarm config must be a native dict")
    _require_exact_keys(data, _SPECIALIZED_CONFIG_KEYS, "specialized swarm config")

    base_keys = set(swarm_config_to_dict(SwarmConfig()))
    base = swarm_config_from_dict({key: data[key] for key in base_keys})

    bool_values = {}
    for key in (
        "use_typed_messaging",
        "encode_message_types",
        "track_role_emergence",
    ):
        value = data[key]
        if type(value) is not bool:
            raise TypeError(f"specialized swarm config {key} must be a native bool")
        bool_values[key] = value

    config = SpecializedSwarmConfig(
        num_agents=base.num_agents,
        topology=base.topology,
        message_passing_rounds=base.message_passing_rounds,
        input_dim=base.input_dim,
        hidden_dim=base.hidden_dim,
        output_dim=base.output_dim,
        message_dim=base.message_dim,
        random_edge_prob=base.random_edge_prob,
        small_world_k=base.small_world_k,
        small_world_p=base.small_world_p,
        scale_free_m=base.scale_free_m,
        num_perception=base.num_perception,
        num_reasoning=base.num_reasoning,
        num_memory=base.num_memory,
        num_planning=base.num_planning,
        use_typed_messaging=bool_values["use_typed_messaging"],
        encode_message_types=bool_values["encode_message_types"],
        track_role_emergence=bool_values["track_role_emergence"],
        specialization_bonus=_require_float(
            data["specialization_bonus"],
            "specialized swarm config specialization_bonus",
        ),
    )
    if not math.isfinite(config.specialization_bonus) or config.specialization_bonus < 0:
        raise ValueError("specialization_bonus must be a finite non-negative number")
    return config


class SpecializedSwarmGraph:
    """
    Swarm with specialized agent architectures and typed messaging.

    Key features:
    - Each agent type has distinct architectural biases
    - Messages carry semantic type information
    - Role-pattern metrics summarize observed message distributions
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
        nonempty_clusters = [cluster for cluster in cluster_nodes if cluster]
        for i, cluster1 in enumerate(nonempty_clusters):
            for cluster2 in nonempty_clusters[i + 1:]:
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
            agent = SpecializedAgent(
                agent_id=agent_id,
                config=agent_config,
                device=self.device,
            )
            agent.last_output = None
            self.agents[agent_id] = agent

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
            agent.last_output = output.detach()
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
                agent.last_output = output.detach()
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
        """Get descriptive role-pattern metrics (legacy API name)."""
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
        """Get a versioned, weights-only-safe persistent state."""
        expected_ids = set(range(self.config.num_agents))
        _validate_exact_int_ids(self.agents, expected_ids, "specialized swarm agents")
        _validate_exact_int_ids(
            self.aggregators,
            expected_ids,
            "specialized swarm aggregators",
        )
        return {
            "schema_version": SWARM_STATE_SCHEMA_VERSION,
            "config": specialized_swarm_config_to_dict(self.config),
            "agents": {i: a.state_dict() for i, a in self.agents.items()},
            "aggregators": {
                i: dict(aggregator.state_dict())
                for i, aggregator in self.aggregators.items()
            },
            "graph_edges": _graph_edges_to_list(
                self.graph, self.config.num_agents
            ),
            "message_encoder": (
                dict(self.message_encoder.state_dict())
                if self.message_encoder is not None
                else None
            ),
        }

    def load_state_dict(self, state: Dict) -> None:
        """Strictly load a schema-v2 state into a matching specialized swarm."""
        if not isinstance(state, dict):
            raise TypeError("specialized swarm state must be a native dict")
        _require_exact_keys(
            state,
            {
                "schema_version",
                "config",
                "agents",
                "aggregators",
                "graph_edges",
                "message_encoder",
            },
            "specialized swarm state",
        )

        schema_version = _require_int(
            state["schema_version"], "specialized swarm state schema_version"
        )
        if schema_version != SWARM_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported specialized swarm state schema_version {schema_version}; "
                f"expected {SWARM_STATE_SCHEMA_VERSION}"
            )

        saved_config = specialized_swarm_config_from_dict(state["config"])
        if saved_config != self.config:
            raise ValueError(
                f"Specialized swarm config mismatch: checkpoint={saved_config!r}, "
                f"instance={self.config!r}"
            )

        expected_ids = set(self.agents.keys())
        agent_states = _validate_exact_int_ids(
            state["agents"], expected_ids, "specialized swarm agent states"
        )
        aggregator_states = _validate_exact_int_ids(
            state["aggregators"], expected_ids, "specialized swarm aggregator states"
        )
        validated_aggregators = {
            agent_id: _validate_tensor_state_dict(
                aggregator_states[agent_id],
                f"aggregator {agent_id} state",
            )
            for agent_id in expected_ids
        }
        edges = _validate_graph_edges(state["graph_edges"], self.config.num_agents)

        encoder_state = state["message_encoder"]
        if self.message_encoder is None:
            if encoder_state is not None:
                raise ValueError(
                    "Checkpoint has a message encoder but this swarm disables it"
                )
            validated_encoder = None
        else:
            if encoder_state is None:
                raise ValueError(
                    "Checkpoint omits the message encoder required by this swarm"
                )
            validated_encoder = _validate_tensor_state_dict(
                encoder_state, "message encoder state"
            )

        for agent_id in sorted(expected_ids):
            self.agents[agent_id].load_state_dict(agent_states[agent_id])
            self.aggregators[agent_id].load_state_dict(
                validated_aggregators[agent_id], strict=True
            )
        if self.message_encoder is not None and validated_encoder is not None:
            self.message_encoder.load_state_dict(validated_encoder, strict=True)

        graph = nx.Graph()
        graph.add_nodes_from(range(self.config.num_agents))
        graph.add_edges_from(edges)
        self.graph = graph


class RoleEmergenceTracker:
    """
    Track descriptive message-distribution and specialization proxies.

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
        """Compute descriptive role-pattern metrics."""
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
