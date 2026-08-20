"""
Neuromorphic computing module for SEESWM.

Provides spiking neural network components and operation-count energy proxies.
Actual energy use depends on deployment hardware and must be measured there.

Key Components:
- LIF neurons: Leaky Integrate-and-Fire with temporal dynamics
- STDP: Spike-Timing Dependent Plasticity learning
- SNN layers: Complete spiking network architectures
- Energy tracking: Efficiency metrics for neuromorphic hardware
- Swarm integration: Spiking agents with spike-based communication
"""

# LIF neurons
from .lif import (
    # Configuration
    LIFConfig,
    LIFState,
    ResetMode,
    # Core neurons
    LIFNeuron,
    LIFLayer,
    RecurrentLIFLayer,
    AdaptiveLIFLayer,
    PopulationLIF,
    # Spike function
    spike_fn,
    SurrogateSpike,
)

# STDP learning
from .stdp import (
    # Configuration
    STDPConfig,
    STDPMode,
    # Learning rules
    ClassicSTDP,
    TripletSTDP,
    SymmetricSTDP,
    RewardModulatedSTDP,
    HomeostaticSTDP,
    # Layer with STDP
    STDPLayer,
    STDPTrace,
)

# SNN layers and architectures
from .layers import (
    # Configuration
    SNNConfig,
    # Layers
    SpikingLinear,
    SpikingConv2d,
    SpikingRNN,
    SpikingNetwork,
    LiquidStateMachine,
    # Coding
    TemporalCoding,
    SpikeDecoder,
)

# Energy efficiency
from .energy import (
    # Hardware models
    HardwareModel,
    EnergyParameters,
    HARDWARE_ENERGY,
    # Statistics
    SpikeStatistics,
    EnergyEstimate,
    # Tracking
    SpikeCounter,
    EnergyTracker,
    EnergyBudget,
    # Training
    EnergyEfficientLoss,
    # Comparison
    compare_energy_efficiency,
)

# Swarm integration
from .swarm_integration import (
    # Configuration
    SpikingAgentConfig,
    SpikingSwarmConfig,
    CommunicationMode,
    # Components
    SpikingMicroAgent,
    SpikingSwarmGraph,
    HybridSwarm,
    # Training
    SpikingSwarmTrainer,
)

__all__ = [
    # LIF
    "LIFConfig",
    "LIFState",
    "ResetMode",
    "LIFNeuron",
    "LIFLayer",
    "RecurrentLIFLayer",
    "AdaptiveLIFLayer",
    "PopulationLIF",
    "spike_fn",
    "SurrogateSpike",
    # STDP
    "STDPConfig",
    "STDPMode",
    "ClassicSTDP",
    "TripletSTDP",
    "SymmetricSTDP",
    "RewardModulatedSTDP",
    "HomeostaticSTDP",
    "STDPLayer",
    "STDPTrace",
    # Layers
    "SNNConfig",
    "SpikingLinear",
    "SpikingConv2d",
    "SpikingRNN",
    "SpikingNetwork",
    "LiquidStateMachine",
    "TemporalCoding",
    "SpikeDecoder",
    # Energy
    "HardwareModel",
    "EnergyParameters",
    "HARDWARE_ENERGY",
    "SpikeStatistics",
    "EnergyEstimate",
    "SpikeCounter",
    "EnergyTracker",
    "EnergyBudget",
    "EnergyEfficientLoss",
    "compare_energy_efficiency",
    # Swarm
    "SpikingAgentConfig",
    "SpikingSwarmConfig",
    "CommunicationMode",
    "SpikingMicroAgent",
    "SpikingSwarmGraph",
    "HybridSwarm",
    "SpikingSwarmTrainer",
]
