"""
Spike-Timing Dependent Plasticity (STDP) learning rules.

STDP is a Hebbian learning rule based on the relative timing of spikes:
- Pre before post → Long-Term Potentiation (LTP, strengthen)
- Post before pre → Long-Term Depression (LTD, weaken)

This module provides:
1. Classic STDP with exponential windows
2. Triplet STDP (more biologically accurate)
3. Reward-modulated STDP for reinforcement learning
4. Symmetric STDP for unsupervised feature learning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class STDPMode(Enum):
    """Type of STDP learning rule."""
    CLASSIC = "classic"       # Standard asymmetric STDP
    TRIPLET = "triplet"       # Triplet-based STDP
    SYMMETRIC = "symmetric"   # Symmetric (Hebbian) STDP
    ANTI_HEBBIAN = "anti"     # Anti-Hebbian STDP


@dataclass
class STDPConfig:
    """Configuration for STDP learning."""

    # Learning rates
    lr_plus: float = 0.01    # LTP learning rate
    lr_minus: float = 0.01   # LTD learning rate

    # Time constants (in timesteps)
    tau_plus: float = 20.0   # LTP time constant
    tau_minus: float = 20.0  # LTD time constant

    # Weight bounds
    w_min: float = 0.0       # Minimum weight
    w_max: float = 1.0       # Maximum weight

    # Triplet STDP parameters
    tau_x: float = 20.0      # Presynaptic trace slow
    tau_y: float = 20.0      # Postsynaptic trace slow
    triplet_ltp: float = 0.01  # Triplet LTP factor
    triplet_ltd: float = 0.01  # Triplet LTD factor

    # Reward modulation
    reward_tau: float = 100.0  # Eligibility trace decay

    # Mode
    mode: STDPMode = STDPMode.CLASSIC


class STDPTrace:
    """
    Spike trace for STDP computation.

    Maintains exponentially decaying trace of recent spikes.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        tau: float = 20.0,
        device: str = "cpu",
    ):
        self.tau = tau
        self.decay = 1.0 - 1.0 / tau
        self.trace = torch.zeros(shape, device=device)

    def update(self, spikes: torch.Tensor) -> torch.Tensor:
        """Update trace with new spikes."""
        self.trace = self.decay * self.trace + spikes
        return self.trace

    def reset(self) -> None:
        """Reset trace to zero."""
        self.trace.zero_()


class ClassicSTDP(nn.Module):
    """
    Classic pair-based STDP.

    Weight update:
        Δw = A_+ * exp(-Δt/τ_+) if Δt > 0 (pre before post, LTP)
        Δw = -A_- * exp(Δt/τ_-) if Δt < 0 (post before pre, LTD)

    where Δt = t_post - t_pre
    """

    def __init__(self, config: Optional[STDPConfig] = None):
        super().__init__()
        self.config = config or STDPConfig()

        # Decay factors
        self.register_buffer(
            "decay_plus",
            torch.tensor(1.0 - 1.0 / self.config.tau_plus)
        )
        self.register_buffer(
            "decay_minus",
            torch.tensor(1.0 - 1.0 / self.config.tau_minus)
        )

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        pre_trace: torch.Tensor,
        post_trace: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weight update from spikes and traces.

        Args:
            weights: Current weights [post, pre]
            pre_spikes: Presynaptic spikes [batch, pre]
            post_spikes: Postsynaptic spikes [batch, post]
            pre_trace: Presynaptic trace [batch, pre]
            post_trace: Postsynaptic trace [batch, post]

        Returns:
            Weight update [post, pre]
        """
        batch_size = pre_spikes.shape[0]

        # LTP: post spike with pre trace
        # When post fires, strengthen connections from recently active pre
        ltp = torch.einsum("bp,bo->op", pre_trace, post_spikes)  # [post, pre]
        ltp = self.config.lr_plus * ltp / batch_size

        # LTD: pre spike with post trace
        # When pre fires, weaken connections to recently active post
        ltd = torch.einsum("bp,bo->op", pre_spikes, post_trace)  # [post, pre]
        ltd = self.config.lr_minus * ltd / batch_size

        # Total update
        dw = ltp - ltd

        # Weight dependence (soft bounds)
        # LTP is stronger for weak synapses
        ltp_factor = (self.config.w_max - weights) / self.config.w_max
        # LTD is stronger for strong synapses
        ltd_factor = (weights - self.config.w_min) / self.config.w_max

        dw_bounded = ltp * ltp_factor - ltd * ltd_factor

        return dw_bounded

    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        pre_trace: torch.Tensor,
        post_trace: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply STDP update.

        Returns:
            (new_weights, weight_delta)
        """
        dw = self.compute_update(weights, pre_spikes, post_spikes, pre_trace, post_trace)
        new_weights = torch.clamp(weights + dw, self.config.w_min, self.config.w_max)
        return new_weights, dw


class TripletSTDP(nn.Module):
    """
    Triplet STDP rule.

    More biologically accurate than pair-based STDP.
    Considers triplets of spikes (pre-post-pre or post-pre-post).

    Reference: "Triplet model of STDP" (Pfister & Gerstner, 2006)
    """

    def __init__(self, config: Optional[STDPConfig] = None):
        super().__init__()
        self.config = config or STDPConfig(mode=STDPMode.TRIPLET)

        # Fast traces (for pair interactions)
        self.register_buffer("decay_plus", torch.tensor(1.0 - 1.0 / self.config.tau_plus))
        self.register_buffer("decay_minus", torch.tensor(1.0 - 1.0 / self.config.tau_minus))

        # Slow traces (for triplet interactions)
        self.register_buffer("decay_x", torch.tensor(1.0 - 1.0 / self.config.tau_x))
        self.register_buffer("decay_y", torch.tensor(1.0 - 1.0 / self.config.tau_y))

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        pre_trace_fast: torch.Tensor,
        post_trace_fast: torch.Tensor,
        pre_trace_slow: torch.Tensor,
        post_trace_slow: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute triplet STDP update.

        LTP = post * pre_fast * (1 + post_slow)  # Pair + triplet
        LTD = pre * post_fast * (1 + pre_slow)   # Pair + triplet
        """
        batch_size = pre_spikes.shape[0]

        # Pair-based LTP
        pair_ltp = torch.einsum("bp,bo->op", pre_trace_fast, post_spikes)

        # Triplet LTP (scaled by slow post trace)
        triplet_ltp = torch.einsum("bp,bo,bo->op", pre_trace_fast, post_spikes, post_trace_slow)

        # Total LTP
        ltp = self.config.lr_plus * pair_ltp + self.config.triplet_ltp * triplet_ltp
        ltp = ltp / batch_size

        # Pair-based LTD
        pair_ltd = torch.einsum("bp,bo->op", pre_spikes, post_trace_fast)

        # Triplet LTD (scaled by slow pre trace)
        triplet_ltd = torch.einsum("bp,bo,bp->op", pre_spikes, post_trace_fast, pre_trace_slow)

        # Total LTD
        ltd = self.config.lr_minus * pair_ltd + self.config.triplet_ltd * triplet_ltd
        ltd = ltd / batch_size

        # Weight-dependent bounds
        ltp_factor = (self.config.w_max - weights) / self.config.w_max
        ltd_factor = (weights - self.config.w_min) / self.config.w_max

        dw = ltp * ltp_factor - ltd * ltd_factor

        return dw

    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        traces: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply triplet STDP."""
        dw = self.compute_update(
            weights,
            pre_spikes,
            post_spikes,
            traces["pre_fast"],
            traces["post_fast"],
            traces["pre_slow"],
            traces["post_slow"],
        )
        new_weights = torch.clamp(weights + dw, self.config.w_min, self.config.w_max)
        return new_weights, dw


class RewardModulatedSTDP(nn.Module):
    """
    Reward-modulated STDP for reinforcement learning.

    STDP creates eligibility traces that are converted to weight
    changes only when reward signal arrives.

    Δw = reward * eligibility_trace

    Reference: "Solving the distal reward problem" (Izhikevich, 2007)
    """

    def __init__(self, config: Optional[STDPConfig] = None):
        super().__init__()
        self.config = config or STDPConfig()

        # STDP component
        self.stdp = ClassicSTDP(config)

        # Eligibility trace decay
        self.register_buffer(
            "eligibility_decay",
            torch.tensor(1.0 - 1.0 / self.config.reward_tau)
        )

        # Eligibility trace storage
        self.eligibility: Optional[torch.Tensor] = None

    def compute_eligibility(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        pre_trace: torch.Tensor,
        post_trace: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute and update eligibility trace.

        Eligibility = decayed old eligibility + new STDP update
        """
        # STDP-style update (but stored as eligibility)
        stdp_update = self.stdp.compute_update(
            weights, pre_spikes, post_spikes, pre_trace, post_trace
        )

        # Initialize eligibility if needed
        if self.eligibility is None or self.eligibility.shape != weights.shape:
            self.eligibility = torch.zeros_like(weights)

        # Decay and accumulate
        self.eligibility = self.eligibility_decay * self.eligibility + stdp_update

        return self.eligibility

    def apply_reward(
        self,
        weights: torch.Tensor,
        reward: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert eligibility to weight change with reward signal.

        Args:
            weights: Current weights
            reward: Scalar reward signal

        Returns:
            (new_weights, weight_delta)
        """
        if self.eligibility is None:
            return weights, torch.zeros_like(weights)

        # Weight change proportional to reward * eligibility
        dw = reward * self.eligibility

        # Apply bounds
        new_weights = torch.clamp(
            weights + dw,
            self.config.w_min,
            self.config.w_max
        )

        return new_weights, dw

    def reset(self) -> None:
        """Reset eligibility trace."""
        self.eligibility = None


class SymmetricSTDP(nn.Module):
    """
    Symmetric (Hebbian) STDP.

    Strengthens connections for any correlated activity,
    regardless of spike order.

    Useful for unsupervised feature learning.
    """

    def __init__(self, config: Optional[STDPConfig] = None):
        super().__init__()
        config = config or STDPConfig()
        config.mode = STDPMode.SYMMETRIC
        self.config = config

        self.register_buffer(
            "decay",
            torch.tensor(1.0 - 1.0 / self.config.tau_plus)
        )

    def compute_update(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        pre_trace: torch.Tensor,
        post_trace: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute symmetric Hebbian update.

        Strengthens connections when both pre and post are active
        (regardless of order).
        """
        batch_size = pre_spikes.shape[0]

        # Correlation: pre trace * post spike + post trace * pre spike
        correlation = (
            torch.einsum("bp,bo->op", pre_trace, post_spikes) +
            torch.einsum("bp,bo->op", pre_spikes, post_trace)
        ) / 2.0

        # Homeostatic term (prevent runaway)
        # Reduce weights that are too strong
        homeostatic = -self.config.lr_minus * (weights - 0.5 * self.config.w_max)

        dw = self.config.lr_plus * correlation / batch_size + 0.01 * homeostatic

        return dw

    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        pre_trace: torch.Tensor,
        post_trace: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply symmetric STDP."""
        dw = self.compute_update(weights, pre_spikes, post_spikes, pre_trace, post_trace)
        new_weights = torch.clamp(weights + dw, self.config.w_min, self.config.w_max)
        return new_weights, dw


class STDPLayer(nn.Module):
    """
    Linear layer with STDP learning.

    Combines LIF-like dynamics with online STDP weight updates.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        stdp_config: Optional[STDPConfig] = None,
        initial_weight: float = 0.5,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = stdp_config or STDPConfig()

        # Weights (initialized to middle of range)
        self.weight = nn.Parameter(
            torch.full((out_features, in_features), initial_weight)
        )

        # STDP rule
        if self.config.mode == STDPMode.TRIPLET:
            self.stdp = TripletSTDP(self.config)
        elif self.config.mode == STDPMode.SYMMETRIC:
            self.stdp = SymmetricSTDP(self.config)
        else:
            self.stdp = ClassicSTDP(self.config)

        # Spike traces
        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None

        # Trace decay
        self.register_buffer("trace_decay", torch.tensor(1.0 - 1.0 / self.config.tau_plus))

    def init_traces(self, batch_size: int, device: str = "cpu") -> None:
        """Initialize spike traces."""
        self.pre_trace = torch.zeros(batch_size, self.in_features, device=device)
        self.post_trace = torch.zeros(batch_size, self.out_features, device=device)

    def forward(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        learn: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with optional STDP learning.

        Args:
            pre_spikes: Input spikes [batch, in_features]
            post_spikes: Output spikes [batch, out_features]
            learn: Whether to apply STDP update

        Returns:
            Weight updates (for monitoring)
        """
        batch_size = pre_spikes.shape[0]
        device = pre_spikes.device

        # Initialize traces if needed
        if self.pre_trace is None or self.pre_trace.shape[0] != batch_size:
            self.init_traces(batch_size, device)

        # Update traces
        self.pre_trace = self.trace_decay * self.pre_trace + pre_spikes
        self.post_trace = self.trace_decay * self.post_trace + post_spikes

        if learn:
            # Apply STDP
            with torch.no_grad():
                new_weight, dw = self.stdp(
                    self.weight,
                    pre_spikes,
                    post_spikes,
                    self.pre_trace,
                    self.post_trace,
                )
                self.weight.data = new_weight
                return dw

        return torch.zeros_like(self.weight)

    def compute_current(self, pre_spikes: torch.Tensor) -> torch.Tensor:
        """Compute synaptic current from presynaptic spikes."""
        return F.linear(pre_spikes, self.weight)

    def reset(self) -> None:
        """Reset traces."""
        self.pre_trace = None
        self.post_trace = None

    def get_weight_stats(self) -> Dict[str, float]:
        """Get weight statistics."""
        return {
            "mean": self.weight.mean().item(),
            "std": self.weight.std().item(),
            "min": self.weight.min().item(),
            "max": self.weight.max().item(),
            "sparsity": (self.weight < 0.1).float().mean().item(),
        }


class HomeostaticSTDP(nn.Module):
    """
    STDP with homeostatic plasticity.

    Maintains target firing rates through adaptive thresholds
    and weight scaling.
    """

    def __init__(
        self,
        num_neurons: int,
        target_rate: float = 0.1,  # Target spike rate
        homeostatic_tau: float = 1000.0,  # Slow timescale
        config: Optional[STDPConfig] = None,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.target_rate = target_rate
        self.config = config or STDPConfig()

        # Base STDP
        self.stdp = ClassicSTDP(config)

        # Running rate estimate per neuron
        self.register_buffer(
            "rate_estimate",
            torch.full((num_neurons,), target_rate)
        )

        # Homeostatic scaling factor
        self.register_buffer(
            "scaling",
            torch.ones(num_neurons)
        )

        # Rate decay
        self.rate_decay = 1.0 - 1.0 / homeostatic_tau

    def update_homeostasis(self, spikes: torch.Tensor) -> None:
        """
        Update rate estimates and scaling factors.

        Args:
            spikes: Spike tensor [batch, num_neurons]
        """
        # Update rate estimate (exponential moving average)
        current_rate = spikes.mean(dim=0)  # [num_neurons]
        self.rate_estimate = self.rate_decay * self.rate_estimate + (1 - self.rate_decay) * current_rate

        # Adjust scaling to drive toward target rate
        # If rate too high, reduce scaling; if too low, increase
        rate_error = self.target_rate - self.rate_estimate
        self.scaling = self.scaling * (1 + 0.001 * rate_error)

        # Clamp scaling to reasonable range
        self.scaling = torch.clamp(self.scaling, 0.5, 2.0)

    def scale_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Apply homeostatic scaling to weights."""
        # Scale columns (inputs to each neuron)
        return weights * self.scaling.unsqueeze(1)

    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        pre_trace: torch.Tensor,
        post_trace: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply STDP with homeostatic regulation."""
        # Regular STDP update
        new_weights, dw = self.stdp(
            weights, pre_spikes, post_spikes, pre_trace, post_trace
        )

        # Update homeostasis
        self.update_homeostasis(post_spikes)

        # Apply scaling
        scaled_weights = self.scale_weights(new_weights)

        return scaled_weights, dw

    def get_stats(self) -> Dict[str, float]:
        """Get homeostatic statistics."""
        return {
            "mean_rate": self.rate_estimate.mean().item(),
            "rate_std": self.rate_estimate.std().item(),
            "mean_scaling": self.scaling.mean().item(),
            "scaling_std": self.scaling.std().item(),
        }
