"""
MicroAgent - The fundamental unit of the swarm.

Each agent is a small neural network (10k-1M parameters) that processes
inputs and messages from neighbors, maintains local state, and produces
outputs for the swarm.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch
import torch.nn as nn


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


AGENT_STATE_SCHEMA_VERSION = 2

_AGENT_CONFIG_KEYS = {
    "agent_type",
    "input_dim",
    "hidden_dim",
    "output_dim",
    "message_dim",
    "num_layers",
    "use_residual",
    "dropout",
    "state_dim",
}
_PLASTICITY_KEYS = {
    "base_lr",
    "dopamine_multiplier",
    "curiosity_bias",
    "fear_dampening",
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


def _agent_config_to_dict(config: AgentConfig) -> dict:
    """Convert an agent configuration to weights-only-safe primitives."""
    return {
        "agent_type": config.agent_type.name,
        "input_dim": int(config.input_dim),
        "hidden_dim": int(config.hidden_dim),
        "output_dim": int(config.output_dim),
        "message_dim": int(config.message_dim),
        "num_layers": int(config.num_layers),
        "use_residual": bool(config.use_residual),
        "dropout": float(config.dropout),
        "state_dim": int(config.state_dim),
    }


def _agent_config_from_dict(data: object) -> AgentConfig:
    """Reconstruct an AgentConfig from a strict primitive mapping."""
    if not isinstance(data, Mapping):
        raise TypeError("agent config must be a mapping")
    _require_exact_keys(data, _AGENT_CONFIG_KEYS, "agent config")

    agent_type_name = data["agent_type"]
    if type(agent_type_name) is not str:
        raise TypeError("agent config agent_type must be an enum name string")
    try:
        agent_type = AgentType[agent_type_name]
    except KeyError as exc:
        raise ValueError(f"Unknown agent type name: {agent_type_name!r}") from exc

    use_residual = data["use_residual"]
    if type(use_residual) is not bool:
        raise TypeError("agent config use_residual must be a native bool")

    return AgentConfig(
        agent_type=agent_type,
        input_dim=_require_int(data["input_dim"], "agent config input_dim"),
        hidden_dim=_require_int(data["hidden_dim"], "agent config hidden_dim"),
        output_dim=_require_int(data["output_dim"], "agent config output_dim"),
        message_dim=_require_int(data["message_dim"], "agent config message_dim"),
        num_layers=_require_int(data["num_layers"], "agent config num_layers"),
        use_residual=use_residual,
        dropout=_require_float(data["dropout"], "agent config dropout"),
        state_dim=_require_int(data["state_dim"], "agent config state_dim"),
    )


def _plasticity_to_dict(plasticity: PlasticityParams) -> dict:
    return {
        "base_lr": float(plasticity.base_lr),
        "dopamine_multiplier": float(plasticity.dopamine_multiplier),
        "curiosity_bias": float(plasticity.curiosity_bias),
        "fear_dampening": float(plasticity.fear_dampening),
    }


def _plasticity_from_dict(data: object) -> PlasticityParams:
    if not isinstance(data, Mapping):
        raise TypeError("agent plasticity must be a mapping")
    _require_exact_keys(data, _PLASTICITY_KEYS, "agent plasticity")
    plasticity = PlasticityParams(
        base_lr=_require_float(data["base_lr"], "plasticity base_lr"),
        dopamine_multiplier=_require_float(
            data["dopamine_multiplier"], "plasticity dopamine_multiplier"
        ),
        curiosity_bias=_require_float(
            data["curiosity_bias"], "plasticity curiosity_bias"
        ),
        fear_dampening=_require_float(
            data["fear_dampening"], "plasticity fear_dampening"
        ),
    )
    values = (
        plasticity.base_lr,
        plasticity.dopamine_multiplier,
        plasticity.curiosity_bias,
        plasticity.fear_dampening,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("agent plasticity values must be finite")
    return plasticity


def _tensor_state_dict(data: object, context: str) -> dict[str, torch.Tensor]:
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
        """Get a versioned, weights-only-safe persistent state."""
        return {
            "schema_version": AGENT_STATE_SCHEMA_VERSION,
            "agent_id": int(self.agent_id),
            "config": _agent_config_to_dict(self.config),
            "network": dict(self.network.state_dict()),
            "plasticity": _plasticity_to_dict(self.plasticity),
        }

    def load_state_dict(self, state: dict) -> None:
        """Strictly load persistent state and reset runtime-only state."""
        if not isinstance(state, Mapping):
            raise TypeError("agent state must be a mapping")
        _require_exact_keys(
            state,
            {"schema_version", "agent_id", "config", "network", "plasticity"},
            "agent state",
        )

        schema_version = _require_int(
            state["schema_version"], "agent state schema_version"
        )
        if schema_version != AGENT_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported agent state schema_version {schema_version}; "
                f"expected {AGENT_STATE_SCHEMA_VERSION}"
            )

        agent_id = _require_int(state["agent_id"], "agent state agent_id")
        if agent_id != self.agent_id:
            raise ValueError(
                f"Agent ID mismatch: checkpoint has {agent_id}, instance has {self.agent_id}"
            )

        saved_config = _agent_config_from_dict(state["config"])
        if saved_config != self.config:
            raise ValueError(
                f"Agent config mismatch for agent {self.agent_id}: "
                f"checkpoint={saved_config!r}, instance={self.config!r}"
            )

        network_state = _tensor_state_dict(state["network"], "agent network state")
        plasticity = _plasticity_from_dict(state["plasticity"])
        self.network.load_state_dict(network_state, strict=True)
        self.plasticity = plasticity

        # Recurrent activations and monitoring outputs are episode-local, not model state.
        self.local_state = None
        self._batch_size = None
        self.last_output = None
        self.activation_count = 0
        self.total_output_magnitude = 0.0
