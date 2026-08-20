"""
SwarmGraph - Manages agent connectivity and collective computation.

The swarm is organized as a graph where nodes are agents and edges
define which agents can communicate. Topology changes the available message
paths; behavioral consequences require controlled evaluation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
import math
import random

import networkx as nx
import torch
import torch.nn.functional as F

from ..agents.micro_agent import MicroAgent, AgentConfig, AgentType
from .messaging import MessageBus


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


SWARM_STATE_SCHEMA_VERSION = 2
MAX_PERSISTED_AGENTS = 512
MAX_PERSISTED_DIMENSION = 4096
MAX_PERSISTED_MESSAGE_ROUNDS = 32
MAX_PERSISTED_MODEL_COMPLEXITY = 50_000_000

_SWARM_CONFIG_KEYS = {
    "num_agents",
    "topology",
    "message_passing_rounds",
    "input_dim",
    "hidden_dim",
    "output_dim",
    "message_dim",
    "random_edge_prob",
    "small_world_k",
    "small_world_p",
    "scale_free_m",
    "num_perception",
    "num_reasoning",
    "num_memory",
    "num_planning",
}


def _require_exact_keys(
    value: Mapping,
    expected: set[str],
    context: str,
) -> None:
    actual = set(value.keys())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted((repr(key) for key in actual - expected))
        raise ValueError(
            f"Invalid {context} keys; missing={missing}, extra={extra}"
        )


def _require_int(value: object, context: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be a native int")
    return value


def _require_float(value: object, context: str) -> float:
    if type(value) not in (int, float):
        raise TypeError(f"{context} must be a native int or float")
    return float(value)


def validate_persisted_swarm_config(config: SwarmConfig) -> None:
    """Validate resource bounds and structural invariants for checkpoints.

    Runtime experiments may construct bespoke configurations directly. A
    persisted configuration has a stronger trust boundary: the evaluator uses
    it to allocate graphs and neural-network layers, so malformed or enormous
    values must be rejected before model construction.
    """
    integer_fields = {
        "num_agents": config.num_agents,
        "message_passing_rounds": config.message_passing_rounds,
        "input_dim": config.input_dim,
        "hidden_dim": config.hidden_dim,
        "output_dim": config.output_dim,
        "message_dim": config.message_dim,
        "small_world_k": config.small_world_k,
        "scale_free_m": config.scale_free_m,
        "num_perception": config.num_perception,
        "num_reasoning": config.num_reasoning,
        "num_memory": config.num_memory,
        "num_planning": config.num_planning,
    }
    for name, value in integer_fields.items():
        if type(value) is not int:
            raise TypeError(f"swarm config {name} must be a native int")

    if not 1 <= config.num_agents <= MAX_PERSISTED_AGENTS:
        raise ValueError(
            f"swarm config num_agents must be in [1, {MAX_PERSISTED_AGENTS}]"
        )
    if not 0 <= config.message_passing_rounds <= MAX_PERSISTED_MESSAGE_ROUNDS:
        raise ValueError(
            "swarm config message_passing_rounds must be in "
            f"[0, {MAX_PERSISTED_MESSAGE_ROUNDS}]"
        )
    for name in ("input_dim", "hidden_dim", "output_dim", "message_dim"):
        value = integer_fields[name]
        if not 1 <= value <= MAX_PERSISTED_DIMENSION:
            raise ValueError(
                f"swarm config {name} must be in [1, {MAX_PERSISTED_DIMENSION}]"
            )

    # Conservative allocation proxy covering dense projections used by both
    # generic and specialized agents. Individual field bounds are insufficient:
    # a tiny file could otherwise request hundreds of enormous networks before
    # strict state loading begins.
    per_agent_complexity = (
        (config.input_dim + config.message_dim + 64) * config.hidden_dim
        + 8 * config.hidden_dim * config.hidden_dim
        + 2 * config.hidden_dim * config.output_dim
    )
    total_complexity = config.num_agents * per_agent_complexity
    if total_complexity > MAX_PERSISTED_MODEL_COMPLEXITY:
        raise ValueError(
            "swarm config exceeds the persisted-model allocation budget: "
            f"{total_complexity} > {MAX_PERSISTED_MODEL_COMPLEXITY}"
        )

    role_counts = (
        config.num_perception,
        config.num_reasoning,
        config.num_memory,
        config.num_planning,
    )
    if any(value < 0 for value in role_counts):
        raise ValueError("swarm config role counts must be non-negative")
    if sum(role_counts) != config.num_agents:
        raise ValueError(
            "swarm config role counts must sum exactly to num_agents"
        )

    probability_fields = {
        "random_edge_prob": config.random_edge_prob,
        "small_world_p": config.small_world_p,
    }
    for name, value in probability_fields.items():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise TypeError(f"swarm config {name} must be a finite native number")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"swarm config {name} must be in [0, 1]")

    if config.small_world_k < 0:
        raise ValueError("swarm config small_world_k must be non-negative")
    if (
        config.topology == TopologyType.SMALL_WORLD
        and config.small_world_k > config.num_agents
    ):
        raise ValueError("small_world_k must not exceed num_agents")
    if config.scale_free_m < 1:
        raise ValueError("swarm config scale_free_m must be positive")
    if config.topology == TopologyType.SCALE_FREE and config.scale_free_m >= config.num_agents:
        raise ValueError("scale_free_m must be smaller than num_agents")


def swarm_config_to_dict(config: SwarmConfig) -> dict:
    """Convert a SwarmConfig to a weights-only-safe primitive mapping."""
    if not isinstance(config, SwarmConfig):
        raise TypeError("config must be a SwarmConfig")
    if not isinstance(config.topology, TopologyType):
        raise TypeError("config topology must be a TopologyType")
    validate_persisted_swarm_config(config)
    return {
        "num_agents": int(config.num_agents),
        "topology": config.topology.name,
        "message_passing_rounds": int(config.message_passing_rounds),
        "input_dim": int(config.input_dim),
        "hidden_dim": int(config.hidden_dim),
        "output_dim": int(config.output_dim),
        "message_dim": int(config.message_dim),
        "random_edge_prob": float(config.random_edge_prob),
        "small_world_k": int(config.small_world_k),
        "small_world_p": float(config.small_world_p),
        "scale_free_m": int(config.scale_free_m),
        "num_perception": int(config.num_perception),
        "num_reasoning": int(config.num_reasoning),
        "num_memory": int(config.num_memory),
        "num_planning": int(config.num_planning),
    }


def swarm_config_from_dict(data: object) -> SwarmConfig:
    """Strictly reconstruct a SwarmConfig from native primitives."""
    if not isinstance(data, Mapping):
        raise TypeError("swarm config must be a mapping")
    _require_exact_keys(data, _SWARM_CONFIG_KEYS, "swarm config")

    topology_name = data["topology"]
    if type(topology_name) is not str:
        raise TypeError("swarm config topology must be an enum name string")
    try:
        topology = TopologyType[topology_name]
    except KeyError as exc:
        raise ValueError(f"Unknown topology name: {topology_name!r}") from exc

    config = SwarmConfig(
        num_agents=_require_int(data["num_agents"], "swarm config num_agents"),
        topology=topology,
        message_passing_rounds=_require_int(
            data["message_passing_rounds"],
            "swarm config message_passing_rounds",
        ),
        input_dim=_require_int(data["input_dim"], "swarm config input_dim"),
        hidden_dim=_require_int(data["hidden_dim"], "swarm config hidden_dim"),
        output_dim=_require_int(data["output_dim"], "swarm config output_dim"),
        message_dim=_require_int(data["message_dim"], "swarm config message_dim"),
        random_edge_prob=_require_float(
            data["random_edge_prob"], "swarm config random_edge_prob"
        ),
        small_world_k=_require_int(
            data["small_world_k"], "swarm config small_world_k"
        ),
        small_world_p=_require_float(
            data["small_world_p"], "swarm config small_world_p"
        ),
        scale_free_m=_require_int(
            data["scale_free_m"], "swarm config scale_free_m"
        ),
        num_perception=_require_int(
            data["num_perception"], "swarm config num_perception"
        ),
        num_reasoning=_require_int(
            data["num_reasoning"], "swarm config num_reasoning"
        ),
        num_memory=_require_int(
            data["num_memory"], "swarm config num_memory"
        ),
        num_planning=_require_int(
            data["num_planning"], "swarm config num_planning"
        ),
    )
    validate_persisted_swarm_config(config)
    return config


def _validate_tensor_state_dict(data: object, context: str) -> dict[str, torch.Tensor]:
    if not isinstance(data, Mapping):
        raise TypeError(f"{context} must be a mapping")
    result: dict[str, torch.Tensor] = {}
    for key, value in data.items():
        if type(key) is not str:
            raise TypeError(f"{context} keys must be native strings")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{context}[{key!r}] must be a tensor")
        result[key] = value
    return result


def _validate_exact_int_ids(
    data: object,
    expected_ids: set[int],
    context: str,
) -> Mapping:
    if not isinstance(data, Mapping):
        raise TypeError(f"{context} must be a mapping")
    for key in data:
        if type(key) is not int:
            raise TypeError(f"{context} keys must be native integer IDs")
    actual_ids = set(data.keys())
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            f"{context} IDs do not match; missing={missing}, extra={extra}"
        )
    return data


def _validate_graph_edges(data: object, num_agents: int) -> list[tuple[int, int]]:
    """Validate and normalize a primitive undirected edge list."""
    if type(data) is not list:
        raise TypeError("graph_edges must be a native list")

    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for index, edge in enumerate(data):
        if type(edge) is not list or len(edge) != 2:
            raise TypeError(f"graph_edges[{index}] must be a two-item native list")
        source = _require_int(edge[0], f"graph_edges[{index}][0]")
        target = _require_int(edge[1], f"graph_edges[{index}][1]")
        if not 0 <= source < num_agents or not 0 <= target < num_agents:
            raise ValueError(
                f"graph_edges[{index}] endpoint outside [0, {num_agents}): "
                f"{source}, {target}"
            )
        if source == target:
            raise ValueError(f"graph_edges[{index}] is a self-loop")
        normalized = (min(source, target), max(source, target))
        if normalized in seen:
            raise ValueError(f"graph_edges[{index}] duplicates edge {normalized}")
        seen.add(normalized)
        edges.append(normalized)

    graph = nx.Graph()
    graph.add_nodes_from(range(num_agents))
    graph.add_edges_from(edges)
    if num_agents > 0 and not nx.is_connected(graph):
        raise ValueError("graph_edges must describe a connected graph")
    return edges


def _graph_edges_to_list(graph: nx.Graph, num_agents: int) -> list[list[int]]:
    expected_nodes = set(range(num_agents))
    for node in graph.nodes:
        if type(node) is not int:
            raise TypeError("graph node IDs must be native integers")
    actual_nodes = set(graph.nodes)
    if actual_nodes != expected_nodes:
        missing = sorted(expected_nodes - actual_nodes)
        extra = sorted(actual_nodes - expected_nodes)
        raise ValueError(
            f"Graph node IDs do not match config; missing={missing}, extra={extra}"
        )

    edge_list = [
        [int(source), int(target)]
        for source, target in sorted(
            (min(source, target), max(source, target))
            for source, target in graph.edges()
        )
    ]
    _validate_graph_edges(edge_list, num_agents)
    return edge_list


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

        # Add all nodes first to ensure they exist
        graph.add_nodes_from(range(n))

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
        nonempty_clusters = [cluster for cluster in cluster_nodes if cluster]
        for i, cluster1 in enumerate(nonempty_clusters):
            for cluster2 in nonempty_clusters[i + 1 :]:
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

        # Add all nodes first to ensure they exist
        graph.add_nodes_from(range(self.config.num_agents))

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
        Compute a legacy-named internal aggregation diagnostic.

        The score is collective inverse-MSE minus average individual inverse-MSE.
        It is descriptive and does not establish causal coordination or
        emergent collective behavior.

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

        # Convert error to a bounded inverse-error score for comparison.
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
        """Get a versioned, weights-only-safe persistent state."""
        expected_ids = set(range(self.config.num_agents))
        _validate_exact_int_ids(self.agents, expected_ids, "swarm agents")
        return {
            "schema_version": SWARM_STATE_SCHEMA_VERSION,
            "config": swarm_config_to_dict(self.config),
            "agents": {i: a.state_dict() for i, a in self.agents.items()},
            "graph_edges": _graph_edges_to_list(
                self.graph, self.config.num_agents
            ),
        }

    def load_state_dict(self, state: dict) -> None:
        """Strictly load a schema-v2 state into a matching swarm."""
        if not isinstance(state, Mapping):
            raise TypeError("swarm state must be a mapping")
        _require_exact_keys(
            state,
            {"schema_version", "config", "agents", "graph_edges"},
            "swarm state",
        )

        schema_version = _require_int(
            state["schema_version"], "swarm state schema_version"
        )
        if schema_version != SWARM_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported swarm state schema_version {schema_version}; "
                f"expected {SWARM_STATE_SCHEMA_VERSION}"
            )

        saved_config = swarm_config_from_dict(state["config"])
        if saved_config != self.config:
            raise ValueError(
                f"Swarm config mismatch: checkpoint={saved_config!r}, "
                f"instance={self.config!r}"
            )

        expected_ids = set(self.agents.keys())
        agent_states = _validate_exact_int_ids(
            state["agents"], expected_ids, "swarm agent states"
        )
        edges = _validate_graph_edges(state["graph_edges"], self.config.num_agents)

        for agent_id in sorted(expected_ids):
            self.agents[agent_id].load_state_dict(agent_states[agent_id])

        graph = nx.Graph()
        graph.add_nodes_from(range(self.config.num_agents))
        graph.add_edges_from(edges)
        self.graph = graph

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
