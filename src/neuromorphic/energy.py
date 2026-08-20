"""
Operation-count energy proxies for neuromorphic computing.

Sparse, event-driven execution can reduce operations on compatible hardware,
but this module does not measure wall-plug energy or validate hardware-level
efficiency.

This module provides:
1. Spike counting and rate metrics
2. Synaptic operation counting
3. Energy estimation based on hardware models
4. Efficiency comparison with ANNs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum

import torch
import torch.nn as nn


class HardwareModel(Enum):
    """
    Energy models for different hardware platforms.

    Energy values in picojoules (pJ) per operation.
    """

    # Neuromorphic chips
    LOIHI = "loihi"           # Intel Loihi
    TRUENORTH = "truenorth"   # IBM TrueNorth
    SPINNAKER = "spinnaker"   # SpiNNaker
    DYNAPSE = "dynapse"       # DYNAmic Processor for Spiking nEurons

    # Conventional (for comparison)
    GPU_FP32 = "gpu_fp32"     # GPU floating point
    CPU_FP32 = "cpu_fp32"     # CPU floating point


@dataclass
class EnergyParameters:
    """Energy per operation for a specific hardware."""

    # Core operations (pJ)
    spike_event: float = 1.0          # Energy per spike event
    synaptic_op: float = 10.0         # Energy per synaptic operation (spike × weight)
    membrane_update: float = 1.0      # Energy per neuron update
    memory_access: float = 5.0        # Energy per memory access

    # MAC operations (for comparison with ANNs)
    mac_op: float = 100.0             # Energy per multiply-accumulate

    # Leakage (static power)
    static_power_mw: float = 10.0     # Static power in milliwatts
    timestep_duration_ms: float = 1.0  # Duration of each timestep


# Predefined energy parameters for different hardware
HARDWARE_ENERGY = {
    HardwareModel.LOIHI: EnergyParameters(
        spike_event=0.8,
        synaptic_op=8.0,
        membrane_update=0.5,
        memory_access=4.0,
        mac_op=100.0,
        static_power_mw=15.0,
    ),
    HardwareModel.TRUENORTH: EnergyParameters(
        spike_event=0.3,
        synaptic_op=5.0,
        membrane_update=0.2,
        memory_access=3.0,
        mac_op=100.0,
        static_power_mw=50.0,  # Higher static for larger chip
    ),
    HardwareModel.SPINNAKER: EnergyParameters(
        spike_event=1.0,
        synaptic_op=12.0,
        membrane_update=1.0,
        memory_access=6.0,
        mac_op=100.0,
        static_power_mw=100.0,
    ),
    HardwareModel.DYNAPSE: EnergyParameters(
        spike_event=0.5,
        synaptic_op=6.0,
        membrane_update=0.3,
        memory_access=2.0,
        mac_op=100.0,
        static_power_mw=5.0,
    ),
    HardwareModel.GPU_FP32: EnergyParameters(
        spike_event=50.0,  # Not applicable
        synaptic_op=100.0,
        membrane_update=50.0,
        memory_access=50.0,
        mac_op=100.0,
        static_power_mw=200000.0,  # 200W typical GPU
    ),
    HardwareModel.CPU_FP32: EnergyParameters(
        spike_event=100.0,
        synaptic_op=200.0,
        membrane_update=100.0,
        memory_access=100.0,
        mac_op=200.0,
        static_power_mw=50000.0,  # 50W typical CPU
    ),
}


@dataclass
class SpikeStatistics:
    """Statistics about spiking activity."""

    # Spike counts
    total_spikes: int = 0
    spikes_per_layer: List[int] = field(default_factory=list)
    spikes_per_timestep: List[int] = field(default_factory=list)

    # Rates
    overall_spike_rate: float = 0.0
    layer_spike_rates: List[float] = field(default_factory=list)

    # Sparsity
    active_neuron_ratio: float = 0.0
    temporal_sparsity: float = 0.0

    # Timing
    total_timesteps: int = 0
    total_neurons: int = 0


@dataclass
class EnergyEstimate:
    """Energy consumption estimate."""

    # Dynamic energy (from activity)
    spike_energy_pj: float = 0.0
    synaptic_energy_pj: float = 0.0
    membrane_energy_pj: float = 0.0
    memory_energy_pj: float = 0.0

    # Static energy
    static_energy_pj: float = 0.0

    # Totals
    total_energy_pj: float = 0.0
    energy_per_inference_pj: float = 0.0

    # Comparison with ANN
    equivalent_ann_energy_pj: float = 0.0
    energy_ratio: float = 1.0  # SNN / ANN

    # Efficiency
    spikes_per_nj: float = 0.0  # Spikes per nanojoule
    ops_per_nj: float = 0.0     # Synaptic ops per nanojoule


class SpikeCounter(nn.Module):
    """
    Count spikes in a network.

    Wraps or hooks into layers to track spike activity.
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self) -> None:
        """Reset counters."""
        self.spike_counts: List[int] = []
        self.neuron_counts: List[int] = []
        self.timestep_spikes: List[int] = []
        self.total_spikes = 0
        self.total_timesteps = 0

    def count_layer(
        self,
        spikes: torch.Tensor,
        layer_idx: int = 0,
    ) -> int:
        """
        Count spikes in a layer.

        Args:
            spikes: Spike tensor (any shape)
            layer_idx: Layer index for tracking

        Returns:
            Number of spikes
        """
        count = int(spikes.sum().item())

        # Extend lists if needed
        while len(self.spike_counts) <= layer_idx:
            self.spike_counts.append(0)
            self.neuron_counts.append(0)

        self.spike_counts[layer_idx] += count
        self.neuron_counts[layer_idx] = spikes.numel() // (spikes.shape[0] if spikes.dim() > 1 else 1)

        self.total_spikes += count

        return count

    def count_timestep(self, spikes: torch.Tensor) -> int:
        """Count spikes in a timestep (across all layers)."""
        count = int(spikes.sum().item())
        self.timestep_spikes.append(count)
        self.total_timesteps += 1
        return count

    def get_statistics(self) -> SpikeStatistics:
        """Compute spike statistics."""
        total_neurons = sum(self.neuron_counts) if self.neuron_counts else 0

        # Overall rate
        if total_neurons > 0 and self.total_timesteps > 0:
            overall_rate = self.total_spikes / (total_neurons * self.total_timesteps)
        else:
            overall_rate = 0.0

        # Per-layer rates
        layer_rates = []
        for spikes, neurons in zip(self.spike_counts, self.neuron_counts):
            if neurons > 0 and self.total_timesteps > 0:
                rate = spikes / (neurons * self.total_timesteps)
            else:
                rate = 0.0
            layer_rates.append(rate)

        # Active neuron ratio (simplified)
        active_ratio = overall_rate  # Approximation

        # Temporal sparsity
        if self.timestep_spikes:
            max_possible = total_neurons
            temporal_sparsity = 1.0 - (sum(self.timestep_spikes) / (len(self.timestep_spikes) * max_possible + 1e-8))
        else:
            temporal_sparsity = 1.0

        return SpikeStatistics(
            total_spikes=self.total_spikes,
            spikes_per_layer=self.spike_counts.copy(),
            spikes_per_timestep=self.timestep_spikes.copy(),
            overall_spike_rate=overall_rate,
            layer_spike_rates=layer_rates,
            active_neuron_ratio=active_ratio,
            temporal_sparsity=temporal_sparsity,
            total_timesteps=self.total_timesteps,
            total_neurons=total_neurons,
        )


class EnergyTracker:
    """
    Track energy consumption of spiking networks.

    Estimates energy based on spike counts and network architecture.
    """

    def __init__(
        self,
        hardware: HardwareModel = HardwareModel.LOIHI,
        custom_params: Optional[EnergyParameters] = None,
    ):
        self.hardware = hardware
        self.params = custom_params or HARDWARE_ENERGY[hardware]

        self.spike_counter = SpikeCounter()
        self.synapse_counts: List[int] = []  # Synapses per layer
        self.history: List[EnergyEstimate] = []

    def register_layer(
        self,
        in_features: int,
        out_features: int,
    ) -> int:
        """Register a layer and return its index."""
        layer_idx = len(self.synapse_counts)
        self.synapse_counts.append(in_features * out_features)
        return layer_idx

    def count_spikes(
        self,
        spikes: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Count spikes for energy estimation."""
        self.spike_counter.count_layer(spikes, layer_idx)

    def estimate_energy(
        self,
        num_timesteps: Optional[int] = None,
    ) -> EnergyEstimate:
        """
        Estimate total energy consumption.

        Args:
            num_timesteps: Number of timesteps (uses tracked if None)

        Returns:
            EnergyEstimate with detailed breakdown
        """
        stats = self.spike_counter.get_statistics()
        num_timesteps = num_timesteps or stats.total_timesteps or 1

        # Spike energy (per spike event)
        spike_energy = stats.total_spikes * self.params.spike_event

        # Synaptic energy
        # Each spike causes synaptic ops proportional to fan-out
        synaptic_energy = 0.0
        for layer_idx, layer_spikes in enumerate(stats.spikes_per_layer):
            if layer_idx < len(self.synapse_counts):
                # Average fan-out (synapses / pre-neurons)
                total_synapses = self.synapse_counts[layer_idx]
                synaptic_energy += layer_spikes * self.params.synaptic_op

        # Membrane update energy (all neurons, all timesteps)
        membrane_energy = stats.total_neurons * num_timesteps * self.params.membrane_update

        # Memory access (approximation)
        memory_energy = (stats.total_spikes + stats.total_neurons * num_timesteps) * self.params.memory_access * 0.1

        # Static energy
        static_energy = self.params.static_power_mw * self.params.timestep_duration_ms * num_timesteps * 1e6  # Convert to pJ

        # Total
        total_energy = spike_energy + synaptic_energy + membrane_energy + memory_energy + static_energy

        # Equivalent ANN energy
        # ANN does MAC for every synapse every timestep
        total_macs = sum(self.synapse_counts) * num_timesteps if self.synapse_counts else 0
        ann_energy = total_macs * self.params.mac_op

        # Energy ratio
        energy_ratio = total_energy / ann_energy if ann_energy > 0 else 1.0

        # Efficiency metrics
        spikes_per_nj = stats.total_spikes / (total_energy / 1e3 + 1e-8)
        ops_per_nj = (spike_energy + synaptic_energy) / (total_energy / 1e3 + 1e-8)

        estimate = EnergyEstimate(
            spike_energy_pj=spike_energy,
            synaptic_energy_pj=synaptic_energy,
            membrane_energy_pj=membrane_energy,
            memory_energy_pj=memory_energy,
            static_energy_pj=static_energy,
            total_energy_pj=total_energy,
            energy_per_inference_pj=total_energy,
            equivalent_ann_energy_pj=ann_energy,
            energy_ratio=energy_ratio,
            spikes_per_nj=spikes_per_nj,
            ops_per_nj=ops_per_nj,
        )

        self.history.append(estimate)
        return estimate

    def reset(self) -> None:
        """Reset counters for new inference."""
        self.spike_counter.reset()

    def get_summary(self) -> Dict[str, float]:
        """Get summary of energy tracking."""
        if not self.history:
            return {}

        recent = self.history[-100:]

        return {
            "avg_total_energy_pj": sum(e.total_energy_pj for e in recent) / len(recent),
            "avg_spike_energy_pj": sum(e.spike_energy_pj for e in recent) / len(recent),
            "avg_synaptic_energy_pj": sum(e.synaptic_energy_pj for e in recent) / len(recent),
            "avg_energy_ratio": sum(e.energy_ratio for e in recent) / len(recent),
            "total_inferences": len(self.history),
        }


class EnergyEfficientLoss(nn.Module):
    """
    Loss function that penalizes high spike rates.

    Encourages sparse spike activity; energy impact is hardware-dependent.
    """

    def __init__(
        self,
        target_rate: float = 0.1,
        rate_weight: float = 0.01,
        temporal_weight: float = 0.01,
    ):
        super().__init__()
        self.target_rate = target_rate
        self.rate_weight = rate_weight
        self.temporal_weight = temporal_weight

    def forward(
        self,
        spike_train: torch.Tensor,
        task_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss with energy penalty.

        Args:
            spike_train: Spike tensor [batch, time, neurons]
            task_loss: Primary task loss

        Returns:
            (total_loss, loss_components)
        """
        # Spike rate penalty
        actual_rate = spike_train.mean()
        rate_penalty = (actual_rate - self.target_rate) ** 2

        # Temporal regularization (encourage sparse bursts, not constant firing)
        if spike_train.dim() >= 2:
            temporal_var = spike_train.var(dim=1).mean()
            # Higher variance = more bursty = preferred
            temporal_penalty = -temporal_var
        else:
            temporal_penalty = torch.tensor(0.0)

        # Total loss
        total_loss = task_loss + self.rate_weight * rate_penalty + self.temporal_weight * temporal_penalty

        return total_loss, {
            "task_loss": task_loss.item(),
            "rate_penalty": rate_penalty.item(),
            "temporal_penalty": temporal_penalty.item(),
            "actual_rate": actual_rate.item(),
        }


class EnergyBudget:
    """
    Energy budget manager for constrained inference.

    Limits spike activity to meet energy constraints.
    """

    def __init__(
        self,
        budget_pj_per_inference: float = 1000.0,
        hardware: HardwareModel = HardwareModel.LOIHI,
    ):
        self.budget = budget_pj_per_inference
        self.params = HARDWARE_ENERGY[hardware]

        self.current_energy = 0.0
        self.spike_budget: Optional[int] = None

        self._compute_spike_budget()

    def _compute_spike_budget(self) -> None:
        """Compute maximum allowed spikes."""
        # Assuming most energy goes to spikes and synaptic ops
        avg_cost_per_spike = self.params.spike_event + self.params.synaptic_op
        self.spike_budget = int(self.budget / avg_cost_per_spike)

    def check_budget(self, new_spikes: int) -> bool:
        """Check if adding spikes stays within budget."""
        avg_cost = self.params.spike_event + self.params.synaptic_op
        new_energy = new_spikes * avg_cost

        return (self.current_energy + new_energy) <= self.budget

    def consume(self, spikes: int) -> float:
        """Record energy consumption."""
        avg_cost = self.params.spike_event + self.params.synaptic_op
        energy = spikes * avg_cost
        self.current_energy += energy
        return energy

    def remaining(self) -> float:
        """Get remaining energy budget."""
        return max(0, self.budget - self.current_energy)

    def remaining_spikes(self) -> int:
        """Get approximate remaining spike budget."""
        avg_cost = self.params.spike_event + self.params.synaptic_op
        return int(self.remaining() / avg_cost)

    def reset(self) -> None:
        """Reset for new inference."""
        self.current_energy = 0.0


def compare_energy_efficiency(
    snn_spikes: torch.Tensor,
    ann_activations: torch.Tensor,
    network_params: int,
    num_timesteps: int,
    hardware: HardwareModel = HardwareModel.LOIHI,
) -> Dict[str, float]:
    """
    Compare energy efficiency of SNN vs ANN.

    Args:
        snn_spikes: Spike tensor from SNN
        ann_activations: Activation tensor from equivalent ANN
        network_params: Number of parameters
        num_timesteps: Number of SNN timesteps
        hardware: Target hardware

    Returns:
        Comparison metrics
    """
    params = HARDWARE_ENERGY[hardware]

    # SNN energy
    total_spikes = snn_spikes.sum().item()
    snn_spike_energy = total_spikes * params.spike_event
    snn_synaptic_energy = total_spikes * params.synaptic_op  # Approximation
    snn_total = snn_spike_energy + snn_synaptic_energy

    # ANN energy (every activation does MAC with weights)
    ann_macs = ann_activations.numel() * network_params / ann_activations.shape[-1]  # Approximation
    ann_total = ann_macs * params.mac_op

    # Spike rate
    spike_rate = total_spikes / snn_spikes.numel()

    # Efficiency gain
    efficiency_gain = ann_total / snn_total if snn_total > 0 else float('inf')

    return {
        "snn_energy_pj": snn_total,
        "ann_energy_pj": ann_total,
        "efficiency_gain": efficiency_gain,
        "spike_rate": spike_rate,
        "total_spikes": total_spikes,
        "sparsity": 1.0 - spike_rate,
    }
