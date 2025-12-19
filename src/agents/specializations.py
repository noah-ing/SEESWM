"""
Specialized agent architectures for different roles.

Each agent type has a different architectural inductive bias:
- Perception: Attention-based feature extraction
- Reasoning: Transformer-style self-attention for composition
- Memory: Key-value memory with read/write operations
- Planning: Hierarchical with goal conditioning

These specialized architectures enable emergent division of labor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .micro_agent import MicroAgent, AgentConfig, AgentType, PlasticityParams


# =============================================================================
# Perception Agent - Attention-based feature extraction
# =============================================================================

class SpatialAttention(nn.Module):
    """Spatial attention for focusing on relevant parts of input."""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head self-attention.

        Returns: (output, attention_weights)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x, attn


class PerceptionNetwork(nn.Module):
    """
    Perception agent architecture.

    Features:
    - Spatial attention for focusing on relevant features
    - Feature pyramid for multi-scale processing
    - Saliency detection for important regions
    """

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # Multi-scale feature extraction
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            for _ in range(3)  # 3 scales
        ])

        # Spatial attention
        self.attention = SpatialAttention(config.hidden_dim, num_heads=4)

        # Saliency head (what's important?)
        self.saliency = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)

        # State update
        self.state_dim = config.state_dim
        self.state_gate = nn.Linear(config.hidden_dim + config.state_dim, config.state_dim)
        self.state_update = nn.Linear(config.hidden_dim + config.state_dim, config.state_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        neighbor_messages: torch.Tensor,
        local_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]

        # Project input
        x = self.input_proj(inputs)

        # Reshape for attention (treat as sequence of features)
        x = x.unsqueeze(1)  # [B, 1, H]

        # Multi-scale processing
        scales = []
        for conv in self.scale_convs:
            scales.append(conv(x))

        # Concatenate scales
        x = torch.cat(scales, dim=1)  # [B, 3, H]

        # Apply attention
        x, attn_weights = self.attention(x)

        # Compute saliency
        saliency = self.saliency(x)  # [B, 3, 1]

        # Weight by saliency and pool
        x = (x * saliency).sum(dim=1)  # [B, H]

        # Incorporate neighbor messages
        if neighbor_messages.shape[-1] > 0:
            x = x + 0.1 * neighbor_messages.mean(dim=-1, keepdim=True).expand_as(x)

        # Output
        output = self.output_proj(x)

        # Update state
        state_input = torch.cat([x, local_state], dim=-1)
        gate = torch.sigmoid(self.state_gate(state_input))
        candidate = torch.tanh(self.state_update(state_input))
        new_state = gate * local_state + (1 - gate) * candidate

        return output, new_state


# =============================================================================
# Reasoning Agent - Transformer-style composition
# =============================================================================

class ReasoningBlock(nn.Module):
    """Single reasoning block with self-attention and FFN."""

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(self.norm1(x))
        x = x + attn_out

        # FFN with residual
        x = x + self.mlp(self.norm2(x))

        return x


class ReasoningNetwork(nn.Module):
    """
    Reasoning agent architecture.

    Features:
    - Transformer-style blocks for relational reasoning
    - Working memory slots for intermediate computations
    - Composition operations for combining concepts
    """

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        self.num_slots = 4  # Working memory slots

        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)

        # Message projection
        self.msg_proj = nn.Linear(config.message_dim, config.hidden_dim)

        # Working memory initialization
        self.slot_init = nn.Parameter(torch.randn(1, self.num_slots, config.hidden_dim) * 0.02)

        # Reasoning blocks
        self.blocks = nn.ModuleList([
            ReasoningBlock(config.hidden_dim, num_heads=4)
            for _ in range(2)
        ])

        # Composition head (combine slots)
        self.compose = nn.Sequential(
            nn.Linear(config.hidden_dim * self.num_slots, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        # State
        self.state_dim = config.state_dim
        self.state_gate = nn.Linear(config.hidden_dim + config.state_dim, config.state_dim)
        self.state_update = nn.Linear(config.hidden_dim + config.state_dim, config.state_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        neighbor_messages: torch.Tensor,
        local_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]

        # Project inputs
        x = self.input_proj(inputs).unsqueeze(1)  # [B, 1, H]

        # Initialize working memory slots
        slots = self.slot_init.expand(batch_size, -1, -1)  # [B, num_slots, H]

        # Add input to first slot
        slots = torch.cat([x, slots[:, 1:]], dim=1)

        # Process neighbor messages if available
        if neighbor_messages.numel() > 0 and neighbor_messages.shape[-1] > 0:
            msg = self.msg_proj(neighbor_messages).unsqueeze(1)
            # Add message to second slot
            slots = torch.cat([slots[:, :1], msg, slots[:, 2:]], dim=1)

        # Apply reasoning blocks
        for block in self.blocks:
            slots = block(slots)

        # Compose final output
        composed = slots.reshape(batch_size, -1)
        output = self.compose(composed)

        # Pool for state update
        pooled = slots.mean(dim=1)

        # Update state
        state_input = torch.cat([pooled, local_state], dim=-1)
        gate = torch.sigmoid(self.state_gate(state_input))
        candidate = torch.tanh(self.state_update(state_input))
        new_state = gate * local_state + (1 - gate) * candidate

        return output, new_state


# =============================================================================
# Memory Agent - Key-value memory with read/write
# =============================================================================

class MemoryNetwork(nn.Module):
    """
    Memory agent architecture.

    Features:
    - Episodic memory bank with key-value storage
    - Content-based addressing for retrieval
    - Write operations for storing new information
    """

    def __init__(self, config: AgentConfig, memory_size: int = 32):
        super().__init__()
        self.config = config
        self.memory_size = memory_size

        # Memory bank (learnable initialization)
        self.memory_keys = nn.Parameter(torch.randn(1, memory_size, config.hidden_dim) * 0.02)
        self.memory_values = nn.Parameter(torch.randn(1, memory_size, config.hidden_dim) * 0.02)

        # Input projections
        self.query_proj = nn.Linear(config.input_dim + config.message_dim, config.hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        # Write gate (should we store this?)
        self.write_gate = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Write content
        self.write_content = nn.Linear(config.input_dim, config.hidden_dim)

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        # State (stores memory updates across steps)
        self.state_dim = config.state_dim
        self.state_proj = nn.Linear(config.hidden_dim, config.state_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        neighbor_messages: torch.Tensor,
        local_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]

        # Expand memory for batch
        keys = self.memory_keys.expand(batch_size, -1, -1)
        values = self.memory_values.expand(batch_size, -1, -1)

        # Create query from input + messages
        if neighbor_messages.numel() > 0 and neighbor_messages.shape[-1] == self.config.message_dim:
            query_input = torch.cat([inputs, neighbor_messages], dim=-1)
        else:
            query_input = torch.cat([inputs, torch.zeros(batch_size, self.config.message_dim, device=inputs.device)], dim=-1)

        query = self.query_proj(query_input)  # [B, H]

        # Read from memory (content-based addressing)
        proj_keys = self.key_proj(keys)  # [B, M, H]
        attn_scores = torch.einsum('bh,bmh->bm', query, proj_keys) / math.sqrt(self.config.hidden_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)

        proj_values = self.value_proj(values)
        read_content = torch.einsum('bm,bmh->bh', attn_weights, proj_values)

        # Write decision
        write_strength = self.write_gate(inputs)  # [B, 1]
        write_content = self.write_content(inputs)  # [B, H]

        # Combine read and write
        combined = torch.cat([read_content, write_content * write_strength], dim=-1)
        output = self.output_proj(combined)

        # Update state with memory trace
        new_state = self.state_proj(read_content)

        return output, new_state


# =============================================================================
# Planning Agent - Goal-conditioned hierarchical
# =============================================================================

class PlanningNetwork(nn.Module):
    """
    Planning agent architecture.

    Features:
    - Goal representation and conditioning
    - Hierarchical action abstraction
    - Value estimation for planning
    """

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config

        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Message integrator
        self.msg_integrator = nn.Sequential(
            nn.Linear(config.message_dim, config.hidden_dim),
            nn.GELU(),
        )

        # Goal-conditioned policy
        self.policy = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

        # Value head (for planning)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

        # Subgoal generator (hierarchical)
        self.subgoal_gen = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.state_dim),
        )

        # State
        self.state_dim = config.state_dim

    def forward(
        self,
        inputs: torch.Tensor,
        neighbor_messages: torch.Tensor,
        local_state: torch.Tensor,  # This is the current goal
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = inputs.shape[0]

        # Encode current state
        state_enc = self.state_encoder(inputs)

        # Encode goal (from local state)
        goal_enc = self.goal_encoder(local_state)

        # Process messages
        if neighbor_messages.numel() > 0 and neighbor_messages.shape[-1] == self.config.message_dim:
            msg_enc = self.msg_integrator(neighbor_messages)
        else:
            msg_enc = torch.zeros(batch_size, self.config.hidden_dim, device=inputs.device)

        # Goal-conditioned policy
        combined = torch.cat([state_enc, goal_enc, msg_enc], dim=-1)
        output = self.policy(combined)

        # Generate subgoal for next step
        subgoal_input = torch.cat([state_enc, goal_enc], dim=-1)
        new_goal = self.subgoal_gen(subgoal_input)

        # Value estimation (could be used for planning)
        value_input = torch.cat([state_enc, goal_enc], dim=-1)
        self._last_value = self.value_head(value_input)

        return output, new_goal


# =============================================================================
# Specialized Agent Factory
# =============================================================================

class SpecializedAgent(MicroAgent):
    """
    MicroAgent with specialized architecture based on type.

    Automatically selects the appropriate network architecture
    based on the agent's specialization type.
    """

    def __init__(
        self,
        agent_id: int,
        config: Optional[AgentConfig] = None,
        device: str = "cpu",
    ):
        # Don't call parent __init__ yet - we'll set up network ourselves
        self.agent_id = agent_id
        self.config = config or AgentConfig()
        self.device = device

        # Build specialized network based on type
        self.network = self._build_specialized_network().to(device)

        # Initialize local state
        self.local_state: Optional[torch.Tensor] = None
        self._batch_size: Optional[int] = None

        # Plasticity parameters
        self.plasticity = PlasticityParams()

        # Statistics
        self.activation_count = 0
        self.total_output_magnitude = 0.0

    def _build_specialized_network(self) -> nn.Module:
        """Build network based on agent type."""
        agent_type = self.config.agent_type

        if agent_type == AgentType.PERCEPTION:
            return PerceptionNetwork(self.config)
        elif agent_type == AgentType.REASONING:
            return ReasoningNetwork(self.config)
        elif agent_type == AgentType.MEMORY:
            return MemoryNetwork(self.config)
        elif agent_type == AgentType.PLANNING:
            return PlanningNetwork(self.config)
        else:
            # Fall back to generic MLP for GENERAL type
            from .micro_agent import AgentNetwork
            return AgentNetwork(self.config)


def create_specialized_swarm(
    num_perception: int = 5,
    num_reasoning: int = 5,
    num_memory: int = 5,
    num_planning: int = 5,
    hidden_dim: int = 64,
    device: str = "cpu",
) -> List[SpecializedAgent]:
    """
    Create a swarm of specialized agents.

    Returns list of agents with appropriate specializations.
    """
    agents = []
    agent_id = 0

    # Perception agents
    for _ in range(num_perception):
        config = AgentConfig(agent_type=AgentType.PERCEPTION, hidden_dim=hidden_dim)
        agents.append(SpecializedAgent(agent_id, config, device))
        agent_id += 1

    # Reasoning agents
    for _ in range(num_reasoning):
        config = AgentConfig(agent_type=AgentType.REASONING, hidden_dim=hidden_dim)
        agents.append(SpecializedAgent(agent_id, config, device))
        agent_id += 1

    # Memory agents
    for _ in range(num_memory):
        config = AgentConfig(agent_type=AgentType.MEMORY, hidden_dim=hidden_dim)
        agents.append(SpecializedAgent(agent_id, config, device))
        agent_id += 1

    # Planning agents
    for _ in range(num_planning):
        config = AgentConfig(agent_type=AgentType.PLANNING, hidden_dim=hidden_dim)
        agents.append(SpecializedAgent(agent_id, config, device))
        agent_id += 1

    return agents
