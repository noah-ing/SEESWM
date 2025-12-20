"""
Leaky Integrate-and-Fire (LIF) neurons with temporal dynamics.

LIF neurons are the foundation of spiking neural networks:
- Membrane potential accumulates input over time
- Leak term causes exponential decay toward resting potential
- Spike emitted when threshold is exceeded
- Reset after spiking (with optional refractory period)

This module provides:
1. Basic LIF neuron
2. Adaptive LIF (with spike-frequency adaptation)
3. LIF with refractory period
4. Batched LIF layer for efficient computation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResetMode(Enum):
    """How to reset membrane potential after spike."""
    HARD = "hard"  # Reset to resting potential
    SOFT = "soft"  # Subtract threshold (preserves excess)
    NONE = "none"  # No reset (for analysis)


@dataclass
class LIFConfig:
    """Configuration for LIF neurons."""

    # Membrane dynamics
    tau_mem: float = 20.0  # Membrane time constant (ms)
    tau_syn: float = 5.0   # Synaptic time constant (ms)

    # Threshold and reset
    threshold: float = 1.0  # Spike threshold
    v_rest: float = 0.0     # Resting potential
    v_reset: float = 0.0    # Reset potential
    reset_mode: ResetMode = ResetMode.SOFT

    # Refractory period
    refractory_steps: int = 0  # Steps of absolute refractory period

    # Adaptation
    adapt: bool = False
    tau_adapt: float = 100.0  # Adaptation time constant
    adapt_increment: float = 0.1  # Threshold increase per spike

    # Surrogate gradient for backprop
    surrogate_slope: float = 25.0  # Slope of surrogate gradient


@dataclass
class LIFState:
    """State of a LIF neuron or layer."""

    membrane: torch.Tensor  # Membrane potential
    synaptic: Optional[torch.Tensor] = None  # Synaptic current
    refractory: Optional[torch.Tensor] = None  # Refractory counter
    adaptation: Optional[torch.Tensor] = None  # Adaptation variable

    # Statistics
    spike_count: int = 0
    total_steps: int = 0


class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient for spike function.

    Forward: Hard threshold (Heaviside)
    Backward: Smooth surrogate (fast sigmoid)

    This allows gradient-based training of spiking networks.
    """

    @staticmethod
    def forward(ctx, membrane: torch.Tensor, threshold: float, slope: float) -> torch.Tensor:
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        ctx.slope = slope
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        membrane, = ctx.saved_tensors

        # Fast sigmoid surrogate gradient
        x = (membrane - ctx.threshold) * ctx.slope
        surrogate_grad = ctx.slope / (1 + x.abs()) ** 2

        return grad_output * surrogate_grad, None, None


def spike_fn(membrane: torch.Tensor, threshold: float, slope: float = 25.0) -> torch.Tensor:
    """Apply spike function with surrogate gradient."""
    return SurrogateSpike.apply(membrane, threshold, slope)


class LIFNeuron(nn.Module):
    """
    Single LIF neuron with full dynamics.

    Implements the differential equation:
        tau_mem * dV/dt = -(V - V_rest) + I_syn
        tau_syn * dI/dt = -I_syn + I_input

    Discretized with exponential Euler method for stability.
    """

    def __init__(self, config: Optional[LIFConfig] = None):
        super().__init__()
        self.config = config or LIFConfig()

        # Compute decay factors (for 1ms timestep)
        self.register_buffer(
            "alpha",
            torch.tensor(1.0 - 1.0 / self.config.tau_mem)
        )
        self.register_buffer(
            "beta",
            torch.tensor(1.0 - 1.0 / self.config.tau_syn)
        )

        # State
        self.state: Optional[LIFState] = None

    def init_state(self, batch_size: int, device: str = "cpu") -> LIFState:
        """Initialize neuron state."""
        membrane = torch.full((batch_size,), self.config.v_rest, device=device)
        synaptic = torch.zeros(batch_size, device=device)

        refractory = None
        if self.config.refractory_steps > 0:
            refractory = torch.zeros(batch_size, dtype=torch.long, device=device)

        adaptation = None
        if self.config.adapt:
            adaptation = torch.zeros(batch_size, device=device)

        return LIFState(
            membrane=membrane,
            synaptic=synaptic,
            refractory=refractory,
            adaptation=adaptation,
        )

    def forward(
        self,
        input_current: torch.Tensor,
        state: Optional[LIFState] = None,
    ) -> Tuple[torch.Tensor, LIFState]:
        """
        Single timestep update.

        Args:
            input_current: Input current [batch_size]
            state: Current state (or None to use internal state)

        Returns:
            (spikes, new_state)
        """
        if state is None:
            if self.state is None:
                self.state = self.init_state(input_current.shape[0], input_current.device)
            state = self.state

        # Update synaptic current
        new_syn = self.beta * state.synaptic + input_current

        # Check refractory
        if state.refractory is not None:
            # Neurons in refractory period don't integrate
            not_refractory = (state.refractory == 0).float()
        else:
            not_refractory = 1.0

        # Get effective threshold (with adaptation)
        if state.adaptation is not None:
            effective_threshold = self.config.threshold + state.adaptation
        else:
            effective_threshold = self.config.threshold

        # Update membrane potential
        leak = self.alpha * (state.membrane - self.config.v_rest)
        new_mem = state.membrane + not_refractory * (-leak + new_syn)

        # Generate spikes
        spikes = spike_fn(new_mem, effective_threshold, self.config.surrogate_slope)

        # Reset
        if self.config.reset_mode == ResetMode.HARD:
            new_mem = new_mem * (1 - spikes) + self.config.v_reset * spikes
        elif self.config.reset_mode == ResetMode.SOFT:
            new_mem = new_mem - effective_threshold * spikes

        # Update refractory
        new_refractory = None
        if state.refractory is not None:
            # Decrement counter, set to max on spike
            new_refractory = torch.clamp(state.refractory - 1, min=0)
            new_refractory = torch.where(
                spikes > 0,
                torch.full_like(new_refractory, self.config.refractory_steps),
                new_refractory,
            )

        # Update adaptation
        new_adapt = None
        if state.adaptation is not None:
            adapt_decay = 1.0 - 1.0 / self.config.tau_adapt
            new_adapt = adapt_decay * state.adaptation + self.config.adapt_increment * spikes

        new_state = LIFState(
            membrane=new_mem,
            synaptic=new_syn,
            refractory=new_refractory,
            adaptation=new_adapt,
            spike_count=state.spike_count + int(spikes.sum().item()),
            total_steps=state.total_steps + 1,
        )

        self.state = new_state
        return spikes, new_state

    def reset(self) -> None:
        """Reset internal state."""
        self.state = None


class LIFLayer(nn.Module):
    """
    Layer of LIF neurons with learnable weights.

    Combines linear transformation with LIF dynamics.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: Optional[LIFConfig] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or LIFConfig()

        # Learnable weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        # Decay factors
        self.register_buffer(
            "alpha",
            torch.tensor(1.0 - 1.0 / self.config.tau_mem)
        )
        self.register_buffer(
            "beta",
            torch.tensor(1.0 - 1.0 / self.config.tau_syn)
        )

        # State
        self.membrane: Optional[torch.Tensor] = None
        self.synaptic: Optional[torch.Tensor] = None

    def init_state(self, batch_size: int, device: str = "cpu") -> None:
        """Initialize layer state."""
        self.membrane = torch.full(
            (batch_size, self.out_features),
            self.config.v_rest,
            device=device
        )
        self.synaptic = torch.zeros(batch_size, self.out_features, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LIF layer.

        Args:
            x: Input spikes or currents [batch, in_features]

        Returns:
            Output spikes [batch, out_features]
        """
        batch_size = x.shape[0]

        # Initialize state if needed
        if self.membrane is None or self.membrane.shape[0] != batch_size:
            self.init_state(batch_size, x.device)

        # Linear transformation
        current = F.linear(x, self.weight, self.bias)

        # Synaptic dynamics
        self.synaptic = self.beta * self.synaptic + current

        # Membrane dynamics
        leak = self.alpha * (self.membrane - self.config.v_rest)
        self.membrane = self.membrane - leak + self.synaptic

        # Spike generation
        spikes = spike_fn(self.membrane, self.config.threshold, self.config.surrogate_slope)

        # Reset (soft)
        self.membrane = self.membrane - self.config.threshold * spikes

        return spikes

    def reset(self) -> None:
        """Reset layer state."""
        self.membrane = None
        self.synaptic = None


class RecurrentLIFLayer(nn.Module):
    """
    LIF layer with recurrent connections.

    Implements lateral connections between neurons in the same layer.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        config: Optional[LIFConfig] = None,
    ):
        super().__init__()
        self.hidden_features = hidden_features
        self.config = config or LIFConfig()

        # Input weights
        self.input_weight = nn.Parameter(torch.empty(hidden_features, in_features))

        # Recurrent weights (no self-connections on diagonal)
        self.recurrent_weight = nn.Parameter(torch.empty(hidden_features, hidden_features))

        # Initialize
        nn.init.kaiming_uniform_(self.input_weight, a=5**0.5)
        nn.init.orthogonal_(self.recurrent_weight, gain=0.5)

        # Mask out self-connections
        self.register_buffer(
            "recurrent_mask",
            1 - torch.eye(hidden_features)
        )

        # Decay factors
        self.register_buffer("alpha", torch.tensor(1.0 - 1.0 / self.config.tau_mem))
        self.register_buffer("beta", torch.tensor(1.0 - 1.0 / self.config.tau_syn))

        # State
        self.membrane: Optional[torch.Tensor] = None
        self.synaptic: Optional[torch.Tensor] = None
        self.prev_spikes: Optional[torch.Tensor] = None

    def init_state(self, batch_size: int, device: str = "cpu") -> None:
        """Initialize layer state."""
        self.membrane = torch.zeros(batch_size, self.hidden_features, device=device)
        self.synaptic = torch.zeros(batch_size, self.hidden_features, device=device)
        self.prev_spikes = torch.zeros(batch_size, self.hidden_features, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with recurrent dynamics."""
        batch_size = x.shape[0]

        if self.membrane is None or self.membrane.shape[0] != batch_size:
            self.init_state(batch_size, x.device)

        # Input current
        input_current = F.linear(x, self.input_weight)

        # Recurrent current (masked to remove self-connections)
        masked_weights = self.recurrent_weight * self.recurrent_mask
        recurrent_current = F.linear(self.prev_spikes, masked_weights)

        # Total current
        current = input_current + recurrent_current

        # Synaptic dynamics
        self.synaptic = self.beta * self.synaptic + current

        # Membrane dynamics
        leak = self.alpha * self.membrane
        self.membrane = self.membrane - leak + self.synaptic

        # Spike
        spikes = spike_fn(self.membrane, self.config.threshold, self.config.surrogate_slope)

        # Reset
        self.membrane = self.membrane - self.config.threshold * spikes

        # Store for next step
        self.prev_spikes = spikes.detach()

        return spikes

    def reset(self) -> None:
        """Reset layer state."""
        self.membrane = None
        self.synaptic = None
        self.prev_spikes = None


class AdaptiveLIFLayer(nn.Module):
    """
    LIF layer with spike-frequency adaptation.

    Neurons become less excitable after sustained firing,
    implementing a form of homeostatic plasticity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_adapt: float = 100.0,
        adapt_increment: float = 0.1,
        config: Optional[LIFConfig] = None,
    ):
        super().__init__()
        self.out_features = out_features
        self.tau_adapt = tau_adapt
        self.adapt_increment = adapt_increment

        config = config or LIFConfig()
        config.adapt = True
        config.tau_adapt = tau_adapt
        config.adapt_increment = adapt_increment
        self.config = config

        # Weights
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        # Decay factors
        self.register_buffer("alpha", torch.tensor(1.0 - 1.0 / config.tau_mem))
        self.register_buffer("beta", torch.tensor(1.0 - 1.0 / config.tau_syn))
        self.register_buffer("gamma", torch.tensor(1.0 - 1.0 / tau_adapt))

        # State
        self.membrane: Optional[torch.Tensor] = None
        self.synaptic: Optional[torch.Tensor] = None
        self.adaptation: Optional[torch.Tensor] = None

    def init_state(self, batch_size: int, device: str = "cpu") -> None:
        """Initialize layer state."""
        self.membrane = torch.zeros(batch_size, self.out_features, device=device)
        self.synaptic = torch.zeros(batch_size, self.out_features, device=device)
        self.adaptation = torch.zeros(batch_size, self.out_features, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with adaptation."""
        batch_size = x.shape[0]

        if self.membrane is None or self.membrane.shape[0] != batch_size:
            self.init_state(batch_size, x.device)

        # Linear
        current = F.linear(x, self.weight, self.bias)

        # Synaptic
        self.synaptic = self.beta * self.synaptic + current

        # Membrane
        leak = self.alpha * self.membrane
        self.membrane = self.membrane - leak + self.synaptic

        # Adaptive threshold
        effective_threshold = self.config.threshold + self.adaptation

        # Spike
        spikes = spike_fn(self.membrane, self.config.threshold, self.config.surrogate_slope)
        # Use effective threshold for actual spiking decision
        spikes = (self.membrane >= effective_threshold).float()

        # Reset
        self.membrane = self.membrane - effective_threshold * spikes

        # Update adaptation
        self.adaptation = self.gamma * self.adaptation + self.adapt_increment * spikes

        return spikes

    def reset(self) -> None:
        """Reset layer state."""
        self.membrane = None
        self.synaptic = None
        self.adaptation = None

    def get_adaptation_stats(self) -> Dict[str, float]:
        """Get adaptation statistics."""
        if self.adaptation is None:
            return {}
        return {
            "mean_adaptation": self.adaptation.mean().item(),
            "max_adaptation": self.adaptation.max().item(),
            "min_adaptation": self.adaptation.min().item(),
        }


class PopulationLIF(nn.Module):
    """
    Population of LIF neurons for population coding.

    Encodes continuous values as spike patterns across a population,
    enabling rate coding or temporal coding schemes.
    """

    def __init__(
        self,
        num_neurons: int,
        input_dim: int = 1,
        encoding: str = "rate",  # "rate" or "latency"
        config: Optional[LIFConfig] = None,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.encoding = encoding
        self.config = config or LIFConfig()

        # Tuning curves (Gaussian centers and widths)
        self.register_buffer(
            "centers",
            torch.linspace(0, 1, num_neurons).unsqueeze(0)  # [1, num_neurons]
        )
        self.width = nn.Parameter(torch.tensor(0.2))

        # Gain for rate coding
        self.gain = nn.Parameter(torch.tensor(1.0))

        # LIF dynamics
        self.register_buffer("alpha", torch.tensor(1.0 - 1.0 / self.config.tau_mem))

        # State
        self.membrane: Optional[torch.Tensor] = None

    def init_state(self, batch_size: int, device: str = "cpu") -> None:
        """Initialize population state."""
        self.membrane = torch.zeros(batch_size, self.num_neurons, device=device)

    def forward(self, x: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        Encode input as spike train.

        Args:
            x: Input values [batch, input_dim] in range [0, 1]
            num_steps: Number of timesteps to simulate

        Returns:
            Spike train [batch, num_steps, num_neurons]
        """
        batch_size = x.shape[0]
        device = x.device

        # Normalize input to [0, 1]
        x_norm = torch.sigmoid(x)  # [batch, input_dim]

        # Compute tuning curve response (Gaussian)
        # For simplicity, average across input dimensions
        x_mean = x_norm.mean(dim=-1, keepdim=True)  # [batch, 1]

        # Distance to each neuron's center
        distance = (x_mean - self.centers) ** 2  # [batch, num_neurons]
        response = torch.exp(-distance / (2 * self.width ** 2))  # Gaussian response

        # Generate spikes over time
        self.init_state(batch_size, device)
        spikes_list = []

        for t in range(num_steps):
            if self.encoding == "rate":
                # Rate coding: current proportional to response
                current = self.gain * response
            else:
                # Latency coding: stronger response = earlier spike
                # Decaying current based on response
                current = self.gain * response * torch.exp(-t / (response + 0.1))

            # LIF dynamics
            self.membrane = self.alpha * self.membrane + current
            spikes = spike_fn(self.membrane, self.config.threshold, self.config.surrogate_slope)
            self.membrane = self.membrane - self.config.threshold * spikes

            spikes_list.append(spikes)

        return torch.stack(spikes_list, dim=1)  # [batch, num_steps, num_neurons]

    def decode(self, spike_train: torch.Tensor) -> torch.Tensor:
        """
        Decode spike train back to continuous value.

        Args:
            spike_train: [batch, num_steps, num_neurons]

        Returns:
            Decoded value [batch, 1]
        """
        # Rate-based decoding: weighted average of centers by spike counts
        spike_counts = spike_train.sum(dim=1)  # [batch, num_neurons]
        total_spikes = spike_counts.sum(dim=-1, keepdim=True) + 1e-8

        # Weighted average of centers
        decoded = (spike_counts * self.centers).sum(dim=-1, keepdim=True) / total_spikes

        return decoded

    def reset(self) -> None:
        """Reset population state."""
        self.membrane = None
