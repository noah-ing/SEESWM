"""
Spiking Neural Network layers and architectures.

Combines LIF neurons with STDP learning into complete network modules:
1. SpikingLinear: Dense layer with spiking dynamics
2. SpikingConv2d: Convolutional spiking layer
3. SpikingRNN: Recurrent spiking network
4. SpikingNetwork: Complete SNN with multiple layers
5. LiquidStateMachine: Reservoir computing with spikes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lif import LIFConfig, LIFLayer, RecurrentLIFLayer, AdaptiveLIFLayer, spike_fn
from .stdp import STDPConfig, STDPLayer, ClassicSTDP, RewardModulatedSTDP


@dataclass
class SNNConfig:
    """Configuration for spiking neural networks."""

    # LIF parameters
    lif: LIFConfig = None

    # STDP parameters (if using online learning)
    stdp: Optional[STDPConfig] = None

    # Network parameters
    num_steps: int = 25       # Timesteps per forward pass
    dt: float = 1.0           # Timestep in ms

    # Training mode
    use_stdp: bool = False    # Online STDP learning
    use_surrogate: bool = True  # Surrogate gradients for backprop

    # Output decoding
    output_mode: str = "rate"  # "rate", "last", "max", "all"

    def __post_init__(self):
        if self.lif is None:
            self.lif = LIFConfig()


class SpikingLinear(nn.Module):
    """
    Fully-connected spiking layer.

    Combines linear transformation with LIF dynamics.
    Supports both surrogate gradient training and STDP.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: Optional[SNNConfig] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config or SNNConfig()

        # LIF layer
        self.lif = LIFLayer(in_features, out_features, self.config.lif, bias)

        # Optional STDP
        if self.config.use_stdp and self.config.stdp:
            self.stdp_layer = STDPLayer(in_features, out_features, self.config.stdp)
        else:
            self.stdp_layer = None

        # Spike traces for STDP
        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None
        self.trace_decay = 0.95

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass over multiple timesteps.

        Args:
            x: Input tensor [batch, in_features] or [batch, time, in_features]
            num_steps: Number of timesteps (uses config if not specified)

        Returns:
            Output spikes [batch, time, out_features] or decoded output
        """
        num_steps = num_steps or self.config.num_steps

        # Handle temporal input
        if x.dim() == 3:
            # Already has time dimension
            batch_size, T, _ = x.shape
            num_steps = T
            temporal_input = True
        else:
            batch_size = x.shape[0]
            temporal_input = False

        # Reset state
        self.lif.reset()

        # Collect spikes over time
        spike_train = []

        for t in range(num_steps):
            # Get input for this timestep
            if temporal_input:
                x_t = x[:, t]
            else:
                x_t = x  # Same input every step (rate coding)

            # LIF forward
            spikes = self.lif(x_t)
            spike_train.append(spikes)

            # STDP update (if enabled)
            if self.stdp_layer is not None and self.training:
                self.stdp_layer(x_t, spikes, learn=True)

        spike_train = torch.stack(spike_train, dim=1)  # [batch, time, out_features]

        # Decode output
        return self._decode_output(spike_train)

    def _decode_output(self, spike_train: torch.Tensor) -> torch.Tensor:
        """Decode spike train to output format."""
        if self.config.output_mode == "rate":
            return spike_train.mean(dim=1)  # [batch, out_features]
        elif self.config.output_mode == "last":
            return spike_train[:, -1]
        elif self.config.output_mode == "max":
            return spike_train.max(dim=1)[0]
        elif self.config.output_mode == "all":
            return spike_train
        else:
            return spike_train.mean(dim=1)

    def reset(self) -> None:
        """Reset layer state."""
        self.lif.reset()
        if self.stdp_layer:
            self.stdp_layer.reset()


class SpikingConv2d(nn.Module):
    """
    Convolutional spiking layer.

    Applies convolution followed by LIF dynamics.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        padding: int = 0,
        config: Optional[SNNConfig] = None,
    ):
        super().__init__()
        self.config = config or SNNConfig()

        # Convolution
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=True
        )

        # LIF parameters
        self.register_buffer("alpha", torch.tensor(1.0 - 1.0 / self.config.lif.tau_mem))
        self.register_buffer("beta", torch.tensor(1.0 - 1.0 / self.config.lif.tau_syn))
        self.threshold = self.config.lif.threshold
        self.surrogate_slope = self.config.lif.surrogate_slope

        # State
        self.membrane: Optional[torch.Tensor] = None
        self.synaptic: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, num_steps: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, channels, height, width] or [batch, time, channels, height, width]
            num_steps: Number of timesteps

        Returns:
            Output spikes
        """
        num_steps = num_steps or self.config.num_steps

        if x.dim() == 5:
            batch_size, T, C, H, W = x.shape
            temporal_input = True
        else:
            batch_size, C, H, W = x.shape
            temporal_input = False
            T = num_steps

        # Get output shape from convolution
        with torch.no_grad():
            test_out = self.conv(x[:1, :T] if temporal_input else x[:1])
            out_shape = test_out.shape[1:]

        # Initialize state
        self.membrane = torch.zeros(batch_size, *out_shape, device=x.device)
        self.synaptic = torch.zeros(batch_size, *out_shape, device=x.device)

        spike_train = []

        for t in range(T):
            if temporal_input:
                x_t = x[:, t]
            else:
                x_t = x

            # Convolution
            current = self.conv(x_t)

            # Synaptic dynamics
            self.synaptic = self.beta * self.synaptic + current

            # Membrane dynamics
            leak = self.alpha * self.membrane
            self.membrane = self.membrane - leak + self.synaptic

            # Spike
            spikes = spike_fn(self.membrane, self.threshold, self.surrogate_slope)
            self.membrane = self.membrane - self.threshold * spikes

            spike_train.append(spikes)

        spike_train = torch.stack(spike_train, dim=1)

        if self.config.output_mode == "rate":
            return spike_train.mean(dim=1)
        return spike_train

    def reset(self) -> None:
        """Reset state."""
        self.membrane = None
        self.synaptic = None


class SpikingRNN(nn.Module):
    """
    Recurrent spiking network.

    Uses LIF neurons with recurrent connections for temporal processing.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        config: Optional[SNNConfig] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config or SNNConfig()

        # Recurrent LIF layer
        self.rnn = RecurrentLIFLayer(input_size, hidden_size, self.config.lif)

        # Output projection (spiking)
        self.output = LIFLayer(hidden_size, output_size, self.config.lif)

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input [batch, input_size] or [batch, time, input_size]

        Returns:
            (output, hidden_spikes)
        """
        num_steps = num_steps or self.config.num_steps

        if x.dim() == 3:
            temporal_input = True
            num_steps = x.shape[1]
        else:
            temporal_input = False

        self.rnn.reset()
        self.output.reset()

        hidden_spikes = []
        output_spikes = []

        for t in range(num_steps):
            x_t = x[:, t] if temporal_input else x

            # Recurrent layer
            h_spikes = self.rnn(x_t)
            hidden_spikes.append(h_spikes)

            # Output layer
            o_spikes = self.output(h_spikes)
            output_spikes.append(o_spikes)

        hidden_spikes = torch.stack(hidden_spikes, dim=1)
        output_spikes = torch.stack(output_spikes, dim=1)

        # Decode
        if self.config.output_mode == "rate":
            output = output_spikes.mean(dim=1)
        else:
            output = output_spikes

        return output, hidden_spikes

    def reset(self) -> None:
        """Reset state."""
        self.rnn.reset()
        self.output.reset()


class SpikingNetwork(nn.Module):
    """
    Multi-layer spiking neural network.

    Provides a simple API for building SNNs.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        config: Optional[SNNConfig] = None,
        use_adaptive: bool = False,
    ):
        super().__init__()
        self.config = config or SNNConfig()
        self.layer_sizes = layer_sizes

        # Build layers
        layers = []
        for i in range(len(layer_sizes) - 1):
            if use_adaptive and i < len(layer_sizes) - 2:
                layer = AdaptiveLIFLayer(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    config=self.config.lif,
                )
            else:
                layer = LIFLayer(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                    self.config.lif,
                )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
        return_all_layers: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through all layers.

        Args:
            x: Input tensor [batch, input_size]
            num_steps: Number of timesteps
            return_all_layers: Return spike trains from all layers

        Returns:
            Output (rate-coded) or list of spike trains
        """
        num_steps = num_steps or self.config.num_steps

        # Reset all layers
        self.reset()

        # Collect outputs
        all_outputs = [[] for _ in self.layers]

        for t in range(num_steps):
            current = x

            for i, layer in enumerate(self.layers):
                spikes = layer(current)
                all_outputs[i].append(spikes)
                current = spikes

        # Stack time dimension
        all_outputs = [torch.stack(out, dim=1) for out in all_outputs]

        if return_all_layers:
            return all_outputs

        # Return rate-coded output of last layer
        return all_outputs[-1].mean(dim=1)

    def reset(self) -> None:
        """Reset all layers."""
        for layer in self.layers:
            layer.reset()

    def get_spike_rates(self) -> List[float]:
        """Get average spike rate per layer (after forward pass)."""
        # This would need spike counting in layers
        return [0.0] * len(self.layers)


class LiquidStateMachine(nn.Module):
    """
    Liquid State Machine (LSM) for reservoir computing.

    A randomly connected recurrent spiking network (the "liquid")
    projects temporal patterns into a high-dimensional space.
    A simple readout layer learns from these representations.

    Reference: "Real-Time Computing Without Stable States" (Maass, 2002)
    """

    def __init__(
        self,
        input_size: int,
        reservoir_size: int = 256,
        output_size: int = 10,
        spectral_radius: float = 0.9,
        input_scaling: float = 0.1,
        sparsity: float = 0.1,
        config: Optional[SNNConfig] = None,
    ):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.config = config or SNNConfig()

        # Input weights (sparse, fixed)
        self.register_buffer(
            "W_in",
            self._create_sparse_weights(input_size, reservoir_size, sparsity) * input_scaling
        )

        # Recurrent weights (sparse, fixed, scaled by spectral radius)
        W_res = self._create_sparse_weights(reservoir_size, reservoir_size, sparsity)
        W_res = self._scale_spectral_radius(W_res, spectral_radius)
        self.register_buffer("W_res", W_res)

        # LIF dynamics
        self.register_buffer("alpha", torch.tensor(1.0 - 1.0 / self.config.lif.tau_mem))
        self.threshold = self.config.lif.threshold
        self.surrogate_slope = self.config.lif.surrogate_slope

        # Trainable readout
        self.readout = nn.Linear(reservoir_size, output_size)

        # State
        self.membrane: Optional[torch.Tensor] = None
        self.prev_spikes: Optional[torch.Tensor] = None

    def _create_sparse_weights(
        self,
        rows: int,
        cols: int,
        sparsity: float,
    ) -> torch.Tensor:
        """Create sparse random weights."""
        W = torch.randn(rows, cols)
        mask = torch.rand(rows, cols) < sparsity
        return W * mask.float()

    def _scale_spectral_radius(
        self,
        W: torch.Tensor,
        target_radius: float,
    ) -> torch.Tensor:
        """Scale weights to target spectral radius."""
        eigenvalues = torch.linalg.eigvals(W)
        current_radius = eigenvalues.abs().max().item()
        if current_radius > 0:
            W = W * (target_radius / current_radius)
        return W

    def init_state(self, batch_size: int, device: str = "cpu") -> None:
        """Initialize reservoir state."""
        self.membrane = torch.zeros(batch_size, self.reservoir_size, device=device)
        self.prev_spikes = torch.zeros(batch_size, self.reservoir_size, device=device)

    def forward(
        self,
        x: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through LSM.

        Args:
            x: Input [batch, input_size] or [batch, time, input_size]

        Returns:
            (output, reservoir_states) where reservoir_states is [batch, time, reservoir_size]
        """
        num_steps = num_steps or self.config.num_steps

        if x.dim() == 3:
            temporal_input = True
            num_steps = x.shape[1]
        else:
            temporal_input = False

        batch_size = x.shape[0]
        device = x.device

        self.init_state(batch_size, device)

        reservoir_states = []

        for t in range(num_steps):
            x_t = x[:, t] if temporal_input else x

            # Input current
            input_current = F.linear(x_t, self.W_in.t())

            # Recurrent current
            recurrent_current = F.linear(self.prev_spikes, self.W_res.t())

            # Total current
            current = input_current + recurrent_current

            # LIF dynamics
            leak = self.alpha * self.membrane
            self.membrane = self.membrane - leak + current

            # Spike
            spikes = spike_fn(self.membrane, self.threshold, self.surrogate_slope)
            self.membrane = self.membrane - self.threshold * spikes

            self.prev_spikes = spikes.detach()
            reservoir_states.append(spikes)

        reservoir_states = torch.stack(reservoir_states, dim=1)

        # Rate-coded reservoir output
        reservoir_rate = reservoir_states.mean(dim=1)

        # Readout
        output = self.readout(reservoir_rate)

        return output, reservoir_states

    def reset(self) -> None:
        """Reset reservoir state."""
        self.membrane = None
        self.prev_spikes = None


class TemporalCoding(nn.Module):
    """
    Temporal coding layer.

    Converts rate-coded inputs to precisely timed spike patterns.
    Used as input layer for SNNs.
    """

    def __init__(
        self,
        input_size: int,
        num_steps: int = 25,
        coding_type: str = "latency",  # "latency", "phase", "burst"
    ):
        super().__init__()
        self.input_size = input_size
        self.num_steps = num_steps
        self.coding_type = coding_type

        # Learnable timing parameters
        self.time_scale = nn.Parameter(torch.ones(input_size))
        self.time_offset = nn.Parameter(torch.zeros(input_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous values to spike train.

        Args:
            x: Input values [batch, input_size] in range [0, 1]

        Returns:
            Spike train [batch, num_steps, input_size]
        """
        batch_size = x.shape[0]
        device = x.device

        # Normalize input
        x_norm = torch.sigmoid(x)

        if self.coding_type == "latency":
            # First-spike latency coding
            # Higher values → earlier spikes
            spike_time = ((1 - x_norm) * self.num_steps * self.time_scale + self.time_offset).long()
            spike_time = torch.clamp(spike_time, 0, self.num_steps - 1)

            # Create spike train
            spikes = torch.zeros(batch_size, self.num_steps, self.input_size, device=device)
            for t in range(self.num_steps):
                spikes[:, t] = (spike_time == t).float()

        elif self.coding_type == "phase":
            # Phase coding (periodic)
            t = torch.arange(self.num_steps, device=device).float()
            phase = 2 * math.pi * x_norm.unsqueeze(1) * t.unsqueeze(0).unsqueeze(-1) / self.num_steps
            spikes = (torch.sin(phase) > 0.9).float()

        elif self.coding_type == "burst":
            # Burst coding (number of spikes encodes value)
            num_spikes = (x_norm * self.num_steps).long()
            spikes = torch.zeros(batch_size, self.num_steps, self.input_size, device=device)
            for i in range(batch_size):
                for j in range(self.input_size):
                    n = min(num_spikes[i, j].item(), self.num_steps)
                    spikes[i, :n, j] = 1.0

        else:
            raise ValueError(f"Unknown coding type: {self.coding_type}")

        return spikes


class SpikeDecoder(nn.Module):
    """
    Decode spike trains to continuous values.

    Provides multiple decoding strategies.
    """

    def __init__(
        self,
        output_size: int,
        decoding_type: str = "rate",  # "rate", "first_spike", "weighted"
        num_steps: int = 25,
    ):
        super().__init__()
        self.output_size = output_size
        self.decoding_type = decoding_type
        self.num_steps = num_steps

        if decoding_type == "weighted":
            # Learnable time weights
            self.time_weights = nn.Parameter(torch.ones(num_steps))

    def forward(self, spike_train: torch.Tensor) -> torch.Tensor:
        """
        Decode spike train.

        Args:
            spike_train: [batch, time, output_size]

        Returns:
            Decoded values [batch, output_size]
        """
        if self.decoding_type == "rate":
            return spike_train.mean(dim=1)

        elif self.decoding_type == "first_spike":
            # Time of first spike (normalized)
            first_spike_time = torch.argmax(spike_train, dim=1).float()
            has_spike = spike_train.sum(dim=1) > 0
            first_spike_time = torch.where(
                has_spike,
                first_spike_time / self.num_steps,
                torch.ones_like(first_spike_time)
            )
            return 1 - first_spike_time  # Earlier spike = higher value

        elif self.decoding_type == "weighted":
            # Weighted sum over time
            weights = F.softmax(self.time_weights, dim=0)
            return torch.einsum("bto,t->bo", spike_train, weights)

        else:
            return spike_train.mean(dim=1)
