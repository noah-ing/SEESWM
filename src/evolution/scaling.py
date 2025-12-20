"""
Scaling utilities for large swarms (1000+ agents).

Implements:
1. Hierarchical swarm organization
2. Sparse message passing
3. Agent grouping/clustering
4. Memory-efficient training
5. Lazy agent initialization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable, Any, Iterator
from enum import Enum
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ScalingConfig:
    """Configuration for large-scale swarms."""

    # Hierarchical organization
    use_hierarchy: bool = True
    num_levels: int = 3  # Levels in hierarchy
    agents_per_group: int = 10  # Agents per lowest-level group
    groups_per_supergroup: int = 10

    # Message passing
    sparse_messaging: bool = True
    message_sparsity: float = 0.1  # Fraction of messages sent
    local_radius: int = 5  # Local communication radius

    # Memory optimization
    use_checkpointing: bool = True
    lazy_init: bool = True
    offload_to_cpu: bool = False

    # Batching
    agent_batch_size: int = 32  # Process agents in batches
    async_messaging: bool = False


class AgentPool:
    """
    Pool of agents with lazy initialization.

    Creates agents on-demand to save memory.
    """

    def __init__(
        self,
        num_agents: int,
        agent_factory: Callable[[int], nn.Module],
        device: str = "cpu",
        cache_size: int = 100,
    ):
        self.num_agents = num_agents
        self.agent_factory = agent_factory
        self.device = device
        self.cache_size = cache_size

        # LRU cache of active agents
        self.cache: Dict[int, nn.Module] = {}
        self.access_order: List[int] = []

        # Stored states for all agents
        self.states: Dict[int, Dict[str, torch.Tensor]] = {}

    def get(self, agent_id: int) -> nn.Module:
        """Get or create an agent."""
        if agent_id in self.cache:
            # Update access order
            self.access_order.remove(agent_id)
            self.access_order.append(agent_id)
            return self.cache[agent_id]

        # Create new agent
        agent = self.agent_factory(agent_id).to(self.device)

        # Restore state if exists
        if agent_id in self.states:
            agent.load_state_dict(self.states[agent_id])

        # Add to cache
        self.cache[agent_id] = agent
        self.access_order.append(agent_id)

        # Evict if over capacity
        while len(self.cache) > self.cache_size:
            evict_id = self.access_order.pop(0)
            evicted = self.cache.pop(evict_id)
            # Save state
            self.states[evict_id] = evicted.state_dict()
            del evicted

        return agent

    def save_state(self, agent_id: int) -> None:
        """Save agent state."""
        if agent_id in self.cache:
            self.states[agent_id] = self.cache[agent_id].state_dict()

    def save_all_states(self) -> None:
        """Save all cached agent states."""
        for agent_id, agent in self.cache.items():
            self.states[agent_id] = agent.state_dict()

    def clear_cache(self) -> None:
        """Clear agent cache."""
        self.save_all_states()
        self.cache.clear()
        self.access_order.clear()

    def __len__(self) -> int:
        return self.num_agents


class AgentGroup(nn.Module):
    """
    Group of agents that communicate locally.

    Forms the building block of hierarchical swarms.
    """

    def __init__(
        self,
        group_id: int,
        agent_ids: List[int],
        agent_pool: AgentPool,
        hidden_dim: int,
    ):
        super().__init__()
        self.group_id = group_id
        self.agent_ids = agent_ids
        self.agent_pool = agent_pool
        self.hidden_dim = hidden_dim

        # Group-level aggregator
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message buffer
        self.message_buffer: Dict[int, torch.Tensor] = {}

    def forward(
        self,
        inputs: Dict[int, torch.Tensor],
        external_messages: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """
        Process group with local message passing.

        Returns:
            (agent_outputs, group_representation)
        """
        outputs = {}

        # Process each agent
        for agent_id in self.agent_ids:
            agent = self.agent_pool.get(agent_id)

            # Get input
            agent_input = inputs.get(agent_id)
            if agent_input is None:
                continue

            # Add local messages
            local_messages = self._get_local_messages(agent_id)
            if local_messages is not None and external_messages is not None:
                combined = local_messages + external_messages
            elif local_messages is not None:
                combined = local_messages
            else:
                combined = external_messages

            # Forward - try with messages first, fall back to input only
            if combined is not None:
                try:
                    output = agent(agent_input, combined)
                except TypeError:
                    # Agent doesn't accept messages, just use input
                    output = agent(agent_input)
            else:
                output = agent(agent_input)

            outputs[agent_id] = output
            self.message_buffer[agent_id] = output.detach()

        # Compute group representation
        if outputs:
            stacked = torch.stack(list(outputs.values()), dim=0)
            mean_output = stacked.mean(dim=0)
            # Ensure consistent shape - add batch dim if needed
            if mean_output.dim() == 1:
                mean_output = mean_output.unsqueeze(0)
            group_repr = self.aggregator(mean_output).squeeze(0)
        else:
            group_repr = torch.zeros(self.hidden_dim)

        return outputs, group_repr

    def _get_local_messages(self, agent_id: int) -> Optional[torch.Tensor]:
        """Get messages from other agents in group."""
        messages = []
        for other_id in self.agent_ids:
            if other_id != agent_id and other_id in self.message_buffer:
                messages.append(self.message_buffer[other_id])

        if messages:
            return torch.stack(messages, dim=0).mean(dim=0)
        return None


class HierarchicalSwarm(nn.Module):
    """
    Hierarchical organization of agents.

    Enables efficient scaling to thousands of agents.
    """

    def __init__(
        self,
        num_agents: int,
        agent_factory: Callable[[int], nn.Module],
        config: ScalingConfig,
        hidden_dim: int = 64,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_agents = num_agents
        self.config = config
        self.hidden_dim = hidden_dim
        self.device = device

        # Agent pool
        self.agent_pool = AgentPool(
            num_agents,
            agent_factory,
            device,
            cache_size=config.agents_per_group * config.groups_per_supergroup,
        )

        # Build hierarchy
        self.groups, self.hierarchy = self._build_hierarchy()

        # Inter-group communication
        self.inter_group_comm = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Group representations
        self.group_representations: Dict[int, torch.Tensor] = {}

    def _build_hierarchy(self) -> Tuple[Dict[int, AgentGroup], Dict[int, List[int]]]:
        """Build hierarchical grouping of agents."""
        groups = {}
        hierarchy = {}  # group_id -> list of child group/agent ids

        # Create lowest-level groups
        agent_ids = list(range(self.num_agents))
        num_groups = math.ceil(self.num_agents / self.config.agents_per_group)

        group_id = 0
        for i in range(0, self.num_agents, self.config.agents_per_group):
            group_agents = agent_ids[i:i + self.config.agents_per_group]
            groups[group_id] = AgentGroup(
                group_id,
                group_agents,
                self.agent_pool,
                self.hidden_dim,
            )
            hierarchy[group_id] = group_agents
            group_id += 1

        # Create higher levels if needed
        current_level_groups = list(range(group_id))
        level = 1

        while len(current_level_groups) > 1 and level < self.config.num_levels:
            next_level_groups = []

            for i in range(0, len(current_level_groups), self.config.groups_per_supergroup):
                child_groups = current_level_groups[i:i + self.config.groups_per_supergroup]
                hierarchy[group_id] = child_groups
                next_level_groups.append(group_id)
                group_id += 1

            current_level_groups = next_level_groups
            level += 1

        return groups, hierarchy

    def forward(
        self,
        inputs: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """Forward pass through hierarchical swarm."""
        all_outputs = {}

        # Process groups bottom-up
        for group_id, group in self.groups.items():
            # Get external messages from neighboring groups
            external = self._get_inter_group_messages(group_id)

            # Process group
            outputs, group_repr = group(inputs, external)

            all_outputs.update(outputs)
            self.group_representations[group_id] = group_repr

        return all_outputs

    def _get_inter_group_messages(self, group_id: int) -> Optional[torch.Tensor]:
        """Get messages from neighboring groups."""
        # Find parent in hierarchy
        parent_id = None
        siblings = []

        for parent, children in self.hierarchy.items():
            if group_id in children:
                parent_id = parent
                siblings = [c for c in children if c != group_id and c in self.group_representations]
                break

        if not siblings:
            return None

        # Aggregate sibling representations
        sibling_reprs = [self.group_representations[s] for s in siblings]
        aggregated = torch.stack(sibling_reprs, dim=0).mean(dim=0)

        return self.inter_group_comm(aggregated)

    def get_statistics(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        return {
            "num_agents": self.num_agents,
            "num_groups": len(self.groups),
            "agents_cached": len(self.agent_pool.cache),
            "hierarchy_levels": self.config.num_levels,
        }


class SparseMessageGraph:
    """
    Sparse message passing graph.

    Only connects nearby agents to reduce communication.
    """

    def __init__(
        self,
        num_agents: int,
        config: ScalingConfig,
        positions: Optional[torch.Tensor] = None,
    ):
        self.num_agents = num_agents
        self.config = config

        # Agent positions (for spatial locality)
        if positions is None:
            # Random positions in unit square
            self.positions = torch.rand(num_agents, 2)
        else:
            self.positions = positions

        # Build sparse adjacency
        self.adjacency = self._build_sparse_adjacency()

    def _build_sparse_adjacency(self) -> Dict[int, List[int]]:
        """Build sparse adjacency based on locality."""
        adjacency = {i: [] for i in range(self.num_agents)}

        # Compute pairwise distances
        for i in range(self.num_agents):
            distances = torch.norm(self.positions - self.positions[i], dim=1)

            # Connect to k-nearest neighbors
            k = int(self.num_agents * self.config.message_sparsity)
            k = max(1, min(k, self.num_agents - 1))

            _, indices = torch.topk(distances, k + 1, largest=False)
            neighbors = indices[1:].tolist()  # Exclude self

            adjacency[i] = neighbors

        return adjacency

    def get_neighbors(self, agent_id: int) -> List[int]:
        """Get neighbors of an agent."""
        return self.adjacency.get(agent_id, [])

    def get_message_targets(
        self,
        agent_id: int,
        stochastic: bool = True,
    ) -> List[int]:
        """Get agents to send messages to."""
        neighbors = self.adjacency.get(agent_id, [])

        if stochastic and len(neighbors) > 0:
            # Randomly sample subset
            num_targets = max(1, int(len(neighbors) * self.config.message_sparsity))
            return random.sample(neighbors, num_targets)

        return neighbors


class CheckpointedSwarm(nn.Module):
    """
    Swarm with gradient checkpointing for memory efficiency.

    Trades compute for memory by recomputing activations during backward.
    """

    def __init__(
        self,
        agents: nn.ModuleDict,
        checkpoint_every: int = 10,
    ):
        super().__init__()
        self.agents = agents
        self.checkpoint_every = checkpoint_every

    def _agent_forward(
        self,
        agent_id: str,
        x: torch.Tensor,
        messages: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward for single agent (for checkpointing)."""
        agent = self.agents[agent_id]
        if messages is not None:
            try:
                return agent(x, messages)
            except TypeError:
                # Agent doesn't accept messages
                return agent(x)
        return agent(x)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        messages: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward with checkpointing."""
        outputs = {}

        for i, (agent_id, x) in enumerate(inputs.items()):
            msg = messages.get(agent_id)

            # Use checkpointing every N agents
            if self.training and i % self.checkpoint_every == 0:
                output = checkpoint(
                    lambda aid, inp, m: self._agent_forward(aid, inp, m),
                    agent_id,
                    x,
                    msg,
                    use_reentrant=False,
                )
            else:
                output = self._agent_forward(agent_id, x, msg)

            outputs[agent_id] = output

        return outputs


class BatchedAgentProcessor:
    """
    Process agents in batches for efficiency.

    Useful when agents have similar architectures.
    """

    def __init__(
        self,
        agent_template: nn.Module,
        num_agents: int,
        batch_size: int = 32,
        hidden_dim: Optional[int] = None,
    ):
        self.agent_template = agent_template
        self.num_agents = num_agents
        self.batch_size = batch_size

        # Infer hidden_dim if not provided
        if hidden_dim is None:
            hidden_dim = getattr(agent_template, 'hidden_dim', 64)

        # Per-agent parameters (if needed for differentiation)
        self.agent_biases = nn.Parameter(
            torch.zeros(num_agents, hidden_dim)
        )

    def process_batch(
        self,
        agent_ids: List[int],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Process a batch of agents together."""
        # Stack inputs
        batch_size = len(agent_ids)

        # Forward through shared template
        outputs = self.agent_template(inputs)

        # Add per-agent biases
        biases = self.agent_biases[agent_ids]
        outputs = outputs + biases

        return outputs

    def process_all(
        self,
        inputs: Dict[int, torch.Tensor],
    ) -> Dict[int, torch.Tensor]:
        """Process all agents in batches."""
        outputs = {}

        # Sort by ID for consistent batching
        sorted_ids = sorted(inputs.keys())

        for i in range(0, len(sorted_ids), self.batch_size):
            batch_ids = sorted_ids[i:i + self.batch_size]
            batch_inputs = torch.stack([inputs[aid] for aid in batch_ids])

            batch_outputs = self.process_batch(batch_ids, batch_inputs)

            for j, aid in enumerate(batch_ids):
                outputs[aid] = batch_outputs[j]

        return outputs


class OffloadedSwarm(nn.Module):
    """
    Swarm that offloads inactive agents to CPU.

    Saves GPU memory for very large swarms.
    """

    def __init__(
        self,
        agents: nn.ModuleDict,
        gpu_capacity: int = 100,
        device: str = "cuda",
    ):
        super().__init__()
        self.agents = agents
        self.gpu_capacity = gpu_capacity
        self.device = device
        self.cpu_device = torch.device("cpu")

        # Track which agents are on GPU
        self.on_gpu: List[str] = []

        # Initially all on CPU
        for agent in self.agents.values():
            agent.to(self.cpu_device)

    def _ensure_on_gpu(self, agent_id: str) -> None:
        """Ensure agent is on GPU."""
        if agent_id in self.on_gpu:
            return

        # Evict if at capacity
        while len(self.on_gpu) >= self.gpu_capacity:
            evict_id = self.on_gpu.pop(0)
            self.agents[evict_id].to(self.cpu_device)

        # Move to GPU
        self.agents[agent_id].to(self.device)
        self.on_gpu.append(agent_id)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward with automatic offloading."""
        outputs = {}

        for agent_id, x in inputs.items():
            # Ensure on GPU
            self._ensure_on_gpu(agent_id)

            # Move input to GPU
            x_gpu = x.to(self.device)

            # Forward
            output = self.agents[agent_id](x_gpu)
            outputs[agent_id] = output

        return outputs


class StreamingSwarm:
    """
    Streaming swarm for very large scale.

    Processes agents in a streaming fashion without
    loading all into memory.
    """

    def __init__(
        self,
        agent_factory: Callable[[int], nn.Module],
        num_agents: int,
        stream_batch_size: int = 100,
        device: str = "cpu",
    ):
        self.agent_factory = agent_factory
        self.num_agents = num_agents
        self.stream_batch_size = stream_batch_size
        self.device = device

    def stream_forward(
        self,
        input_generator: Iterator[Tuple[int, torch.Tensor]],
    ) -> Iterator[Tuple[int, torch.Tensor]]:
        """
        Stream processing of agents.

        Yields (agent_id, output) pairs.
        """
        batch_ids = []
        batch_inputs = []

        for agent_id, x in input_generator:
            batch_ids.append(agent_id)
            batch_inputs.append(x)

            if len(batch_ids) >= self.stream_batch_size:
                # Process batch
                yield from self._process_batch(batch_ids, batch_inputs)
                batch_ids = []
                batch_inputs = []

        # Process remaining
        if batch_ids:
            yield from self._process_batch(batch_ids, batch_inputs)

    def _process_batch(
        self,
        agent_ids: List[int],
        inputs: List[torch.Tensor],
    ) -> Iterator[Tuple[int, torch.Tensor]]:
        """Process a batch of agents."""
        for agent_id, x in zip(agent_ids, inputs):
            # Create agent
            agent = self.agent_factory(agent_id).to(self.device)

            # Forward
            output = agent(x.to(self.device))

            yield agent_id, output

            # Free memory
            del agent


def estimate_memory_usage(
    num_agents: int,
    hidden_dim: int,
    num_layers: int = 2,
) -> Dict[str, float]:
    """
    Estimate memory usage for a swarm.

    Returns memory in MB.
    """
    # Parameters per agent
    params_per_agent = (
        hidden_dim * hidden_dim * num_layers +  # Weights
        hidden_dim * num_layers  # Biases
    )

    # Activations per agent (per batch element)
    activations_per_agent = hidden_dim * num_layers

    # Total
    total_params = params_per_agent * num_agents
    total_params_mb = total_params * 4 / 1e6  # 4 bytes per float32

    return {
        "params_per_agent": params_per_agent,
        "total_params": total_params,
        "total_params_mb": total_params_mb,
        "estimated_activations_mb_per_batch": activations_per_agent * num_agents * 4 / 1e6,
    }
