"""
MicroAgent - The fundamental unit of the swarm.

Each agent is a small neural network (10k-1M parameters) that processes
inputs and messages from neighbors, maintains local state, and produces
outputs for the swarm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AgentType(Enum):
    """Specialization types for micro-agents."""

    PERCEPTION = auto()  # Feature extraction, attention
    REASONING = auto()  # Logic, composition, CoT
    MEMORY = auto()  # Episodic/semantic storage & retrieval
    PLANNING = auto()  # Goal-directed action selection
    GENERAL = auto()  # Unspecialized (for ablation studies)


@dataclass
class PlasticityParams:
    """Parameters controlling agent learning dynamics."""

    base_lr: float = 1e-4
    dopamine_multiplier: float = 1.0
    curiosity_bias: float = 0.0
    fear_dampening: float = 0.0


@dataclass
class AgentConfig:
    """Configuration for a MicroAgent."""

    agent_type: AgentType = AgentType.GENERAL
    input_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 64
    message_dim: int = 64
    num_layers: int = 2
    use_residual: bool = True
    dropout: float = 0.1
    state_dim: int = 32  # Local state dimension


class AgentNetwork(nn.Module):
    """
    Neural network backbone for a MicroAgent.

    Architecture: MLP with residual connections and optional dropout.
    Fuses agent inputs with neighbor messages before processing.
    """

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config

        # Input fusion: combine direct input with aggregated neighbor messages
        fusion_dim = config.input_dim + config.message_dim + config.state_dim

        # Build MLP layers
        layers = []
        prev_dim = fusion_dim
        for i in range(config.num_layers):
            layers.append(nn.Linear(prev_dim, config.hidden_dim))
            layers.append(nn.LayerNorm(config.hidden_dim))
            layers.append(nn.GELU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            prev_dim = config.hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Output head
        self.output_head = nn.Linear(config.hidden_dim, config.output_dim)

        # State update (GRU-style gating for local state)
        self.state_gate = nn.Linear(config.hidden_dim + config.state_dim, config.state_dim)
        self.state_update = nn.Linear(config.hidden_dim + config.state_dim, config.state_dim)

        # Residual projection if dimensions differ
        self.use_residual = config.use_residual
        if self.use_residual and fusion_dim != config.hidden_dim:
            self.residual_proj = nn.Linear(fusion_dim, config.hidden_dim)
        else:
            self.residual_proj = None

    def forward(
        self,
        inputs: torch.Tensor,
        neighbor_messages: torch.Tensor,
        local_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process inputs and produce output + updated state.

        Args:
            inputs: Direct observations [batch, input_dim]
            neighbor_messages: Aggregated messages from neighbors [batch, message_dim]
            local_state: Agent's local state [batch, state_dim]

        Returns:
            output: Message to send [batch, output_dim]
            new_state: Updated local state [batch, state_dim]
        """
        # Fuse all inputs
        fused = torch.cat([inputs, neighbor_messages, local_state], dim=-1)

        # Process through backbone
        hidden = self.backbone(fused)

        # Residual connection
        if self.use_residual:
            if self.residual_proj is not None:
                hidden = hidden + self.residual_proj(fused)
            else:
                hidden = hidden + fused[..., : hidden.shape[-1]]

        # Compute output
        output = self.output_head(hidden)

        # Update local state with GRU-style gating
        state_input = torch.cat([hidden, local_state], dim=-1)
        gate = torch.sigmoid(self.state_gate(state_input))
        candidate = torch.tanh(self.state_update(state_input))
        new_state = gate * local_state + (1 - gate) * candidate

        return output, new_state


class MicroAgent:
    """
    A single micro-agent in the swarm.

    Each agent:
    - Has a specialization type (perception, reasoning, memory, planning)
    - Maintains local state across timesteps
    - Receives messages from neighbors and produces outputs
    - Has plasticity parameters modulated by neuromodulatory signals
    """

    def __init__(
        self,
        agent_id: int,
        config: Optional[AgentConfig] = None,
        device: str = "cpu",
    ):
        self.agent_id = agent_id
        self.config = config or AgentConfig()
        self.device = device

        # Build network
        self.network = AgentNetwork(self.config).to(device)

        # Initialize local state
        self.local_state: Optional[torch.Tensor] = None
        self._batch_size: Optional[int] = None

        # Plasticity parameters (modulated by neuromodulatory system)
        self.plasticity = PlasticityParams()

        # Statistics for monitoring
        self.activation_count = 0
        self.total_output_magnitude = 0.0

        # Track last output for emergence analysis
        self.last_output: Optional[torch.Tensor] = None

    @property
    def agent_type(self) -> AgentType:
        return self.config.agent_type

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.network.parameters())

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset local state for new episode."""
        self._batch_size = batch_size
        self.local_state = torch.zeros(
            batch_size, self.config.state_dim, device=self.device
        )

    def forward(
        self,
        inputs: torch.Tensor,
        neighbor_messages: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Process inputs and neighbor messages, return output message.

        Args:
            inputs: Direct observations [batch, input_dim]
            neighbor_messages: List of messages from neighbors, each [batch, output_dim]

        Returns:
            output: Message to broadcast [batch, output_dim]
        """
        # Initialize state if needed
        batch_size = inputs.shape[0]
        if self.local_state is None or self._batch_size != batch_size:
            self.reset_state(batch_size)

        # Aggregate neighbor messages (mean pooling)
        if neighbor_messages:
            # Pad messages to message_dim if needed
            padded_messages = []
            for msg in neighbor_messages:
                if msg.shape[-1] < self.config.message_dim:
                    padding = torch.zeros(
                        *msg.shape[:-1],
                        self.config.message_dim - msg.shape[-1],
                        device=self.device,
                    )
                    msg = torch.cat([msg, padding], dim=-1)
                elif msg.shape[-1] > self.config.message_dim:
                    msg = msg[..., : self.config.message_dim]
                padded_messages.append(msg)

            stacked = torch.stack(padded_messages, dim=0)
            aggregated = stacked.mean(dim=0)
        else:
            aggregated = torch.zeros(
                batch_size, self.config.message_dim, device=self.device
            )

        # Forward pass
        output, self.local_state = self.network(inputs, aggregated, self.local_state)

        # Store for emergence analysis
        self.last_output = output.detach()

        # Update statistics
        self.activation_count += 1
        self.total_output_magnitude += output.abs().mean().item()

        return output

    def modulate(self, signal_type: str, magnitude: float) -> None:
        """
        Apply neuromodulatory signal to plasticity parameters.

        Args:
            signal_type: One of 'dopamine', 'curiosity', 'fear', 'uncertainty'
            magnitude: Signal strength (typically 0-1, can exceed)
        """
        if signal_type == "dopamine":
            # Boost learning and strengthen active connections
            self.plasticity.dopamine_multiplier = 1.0 + magnitude
        elif signal_type == "curiosity":
            # Drive exploration
            self.plasticity.curiosity_bias = magnitude
        elif signal_type == "fear":
            # Increase caution, dampen learning
            self.plasticity.fear_dampening = magnitude
        elif signal_type == "uncertainty":
            # Flag high uncertainty (handled by self-model)
            pass

    def get_effective_lr(self) -> float:
        """Get learning rate after neuromodulation."""
        lr = self.plasticity.base_lr
        lr *= self.plasticity.dopamine_multiplier
        lr *= 1.0 - 0.5 * self.plasticity.fear_dampening
        return lr

    def get_stats(self) -> dict:
        """Get agent statistics for monitoring."""
        avg_magnitude = (
            self.total_output_magnitude / max(1, self.activation_count)
        )
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.name,
            "activation_count": self.activation_count,
            "avg_output_magnitude": avg_magnitude,
            "num_parameters": self.num_parameters,
            "effective_lr": self.get_effective_lr(),
        }

    def state_dict(self) -> dict:
        """Get state for saving."""
        return {
            "network": self.network.state_dict(),
            "config": self.config,
            "plasticity": self.plasticity,
            "local_state": self.local_state,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load saved state."""
        self.network.load_state_dict(state["network"])
        self.config = state["config"]
        self.plasticity = state["plasticity"]
        self.local_state = state["local_state"]
