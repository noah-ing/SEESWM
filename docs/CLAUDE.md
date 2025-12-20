# SEESWM Development Memory

This file tracks progress, decisions, and context for AI-assisted development sessions.

## Project Status

**Current Phase**: Phase 6 - Evolution & Scaling (Complete)
**Last Updated**: Session 7

### Completed

#### Phase 1 - Foundation
- [x] Project structure created following PRD specification
- [x] Core MicroAgent implementation (MLP-based, with plasticity parameters)
- [x] SwarmGraph with multiple topology types (random, small-world, scale-free, modular, hierarchical)
- [x] Message passing infrastructure
- [x] Basic environment (CosmosEnvironment grid world)
- [x] Neuromodulatory system (dopamine, curiosity, fear, uncertainty signals)
- [x] Self-model for capability tracking and "I don't know"
- [x] JEPA-style world model
- [x] Evolutionary optimization framework
- [x] Test suite for agents, swarm, and world model
- [x] Configuration system (YAML configs)

#### Phase 2 - World Modeling
- [x] PPO training loop with GAE (Generalized Advantage Estimation)
- [x] World model integration for curiosity rewards
- [x] Trajectory buffer with proper advantage computation
- [x] Enhanced environment with:
  - Multiple resource types (energy, food, water, material)
  - Resource respawn mechanics
  - Hunger/thirst survival mechanics
  - Internal maze-like walls
  - Comprehensive statistics tracking
- [x] Exploration metrics (coverage, novelty, visitation tracking)
- [x] Alternative curiosity modules (RND, ICM)
- [x] Curiosity-driven training script with comparison experiments

#### Phase 3 - Specialization
- [x] Specialized agent architectures per type:
  - PerceptionNetwork: Attention-based with multi-scale processing, saliency detection
  - ReasoningNetwork: Transformer-style with working memory slots
  - MemoryNetwork: Key-value memory with content-based addressing
  - PlanningNetwork: Goal-conditioned with hierarchical subgoal generation
- [x] SpecializedAgent factory class that builds appropriate networks
- [x] Typed message passing system:
  - MessageType enum (PERCEPT, SALIENCY, INFERENCE, QUERY, MEMORY_RECALL, GOAL, etc.)
  - TypedMessage with semantic routing metadata
  - TypedMessageBus with type-aware filtering
  - MessageEncoder/Decoder for type embedding
  - TypeAwareAggregator for type-specific processing
- [x] Role emergence tracking:
  - Specialization entropy metrics
  - Type concentration per agent
  - Message pattern analysis
- [x] SpecializedSwarmGraph integrating all Phase 3 components
- [x] Specialization training script with comparison experiments
- [x] Comprehensive test suite (32 Phase 3 tests, 72 total)

#### Phase 4 - Self-Model (Meta-Cognition)
- [x] Uncertainty quantification:
  - EnsembleUncertainty: Disagreement between ensemble members
  - MCDropoutUncertainty: Monte Carlo dropout sampling
  - EvidentialNetwork: Single-pass Dirichlet uncertainty
  - UncertaintyAggregator: Combines multiple methods
  - UncertaintyHead: Learnable uncertainty prediction
- [x] Confidence calibration:
  - TemperatureScaling: Post-hoc calibration
  - PlattScaling: Logistic regression on logits
  - FocalLoss: Training-time calibration
  - LabelSmoothing: Implicit calibration
  - CalibrationLoss: Differentiable ECE
  - CalibrationTracker: Monitors calibration over time
- [x] Meta-learning:
  - MAML: Model-Agnostic Meta-Learning for fast adaptation
  - MetaSGD: Learned per-parameter learning rates
  - Reptile: Simplified meta-learning via averaging
  - TaskEmbedding: Task similarity for transfer
  - AdaptiveMetaLearner: Strategy selection
- [x] Theory of Mind:
  - BeliefEncoder: Infer agent beliefs from observations
  - IntentPredictor: Predict agent intentions from behavior
  - TheoryOfMind: Full module with perspective taking
  - CollectiveBeliefAggregator: Aggregate beliefs across agents
- [x] Swarm integration:
  - MetaCognitiveSwarm: Wrapper with all meta-cognitive abilities
  - AgentMetaCognition: Per-agent self-model
  - MetaCognitionTrainer: Training for uncertainty/calibration
- [x] Meta-cognition training script with demos
- [x] Comprehensive test suite (32 Phase 4 tests, 104 total)

#### Phase 5 - Neuromorphic
- [x] LIF (Leaky Integrate-and-Fire) neurons:
  - LIFNeuron: Single neuron with full dynamics
  - LIFLayer: Dense layer with LIF neurons
  - RecurrentLIFLayer: With lateral connections
  - AdaptiveLIFLayer: Spike-frequency adaptation
  - PopulationLIF: Population coding for inputs
  - SurrogateSpike: Differentiable spike function
- [x] STDP learning rules:
  - ClassicSTDP: Pair-based timing-dependent plasticity
  - TripletSTDP: More biologically accurate
  - SymmetricSTDP: Hebbian unsupervised learning
  - RewardModulatedSTDP: For reinforcement learning
  - HomeostaticSTDP: With target firing rate
  - STDPLayer: Layer with online STDP
- [x] SNN architectures:
  - SpikingLinear: Dense spiking layer
  - SpikingConv2d: Convolutional spiking layer
  - SpikingRNN: Recurrent spiking network
  - SpikingNetwork: Multi-layer SNN
  - LiquidStateMachine: Reservoir computing
  - TemporalCoding: Rate/latency/burst encoding
  - SpikeDecoder: Multiple decoding strategies
- [x] Energy efficiency:
  - HardwareModel: Loihi, TrueNorth, SpiNNaker, etc.
  - SpikeCounter: Tracks spike activity
  - EnergyTracker: Estimates energy consumption
  - EnergyBudget: Constrained inference
  - EnergyEfficientLoss: Penalizes high spike rates
  - compare_energy_efficiency: SNN vs ANN comparison
- [x] Spiking swarm integration:
  - SpikingMicroAgent: LIF-based agent with STDP
  - SpikingSwarmGraph: Swarm with spike communication
  - HybridSwarm: Mixed spiking + rate-coded
  - SpikingSwarmTrainer: Training with energy constraints
- [x] Neuromorphic training script with experiments
- [x] Comprehensive test suite (28 Phase 5 tests, 132 total)

#### Phase 6 - Evolution & Scaling
- [x] Distributed training infrastructure:
  - DistributedConfig, ParallelismMode (DATA, MODEL, PIPELINE)
  - DistributedManager: Process group management, device handling
  - DistributedSwarm: DDP wrapper with gradient accumulation
  - ModelParallelSwarm: Shard agents across devices
  - PipelineParallelSwarm: Micro-batch pipeline execution
  - AsyncGradientAggregator: Non-blocking gradient sync
  - DistributedTrainer: Full training loop with checkpointing
- [x] Evolutionary optimization:
  - NEAT-style genome representation (AgentGene, ConnectionGene)
  - InnovationTracker: Track structural innovations
  - SwarmMutator: Add/remove agents, connections, weights
  - SwarmCrossover: Align matching genes, inherit from fitter parent
  - Species: Explicit fitness sharing, representative tracking
  - GeneticOptimizer: Full evolutionary loop with speciation
  - CMAES: Covariance Matrix Adaptation for continuous parameters
- [x] Neural Architecture Search:
  - SearchSpace: Operations (skip, linear, ReLU, GELU, attention, etc.)
  - MixedOperation: Softmax weighted ops for DARTS
  - DARTSCell, DARTSAgent, DARTSSearcher: Differentiable NAS
  - ENASController: RNN policy for architecture sampling
  - ENASSharedNetwork: Weight sharing across architectures
  - ENASSearcher: RL-based architecture search
  - RandomSearchNAS: Baseline comparison
- [x] Scaling to 1000+ agents:
  - AgentPool: Lazy initialization with LRU cache
  - AgentGroup: Local clusters with aggregation
  - HierarchicalSwarm: Multi-level grouping for O(n) messaging
  - SparseMessageGraph: k-nearest neighbor sparse communication
  - CheckpointedSwarm: Gradient checkpointing for memory
  - BatchedAgentProcessor: Efficient batched execution
  - OffloadedSwarm: CPU offloading for large swarms
  - StreamingSwarm: Process agents without loading all
  - estimate_memory_usage: Memory planning utility
- [x] Evolution training script (`experiments/train_evolution.py`):
  - Genetic topology evolution with speciation
  - CMA-ES parameter optimization
  - DARTS and ENAS architecture search
  - Hierarchical swarm demos
  - Sparse messaging demos
  - Scaling benchmarks
- [x] Comprehensive test suite (30 Phase 6 tests, 162 total)

### In Progress
- [ ] Visualization and TensorBoard integration

### Next Steps
1. End-to-end integration testing across all phases
2. Full self-evolving swarm experiments
3. Multi-environment embodiment

## Architecture Decisions

### Agent Design
- Using simple MLP with GRU-style state gating for local memory
- Residual connections for gradient flow
- All agents share same architecture but different specialization types affect I/O routing

### Swarm Topology
- Small-world (Watts-Strogatz) as default - balances clustering and path length
- Hierarchical topology for Phase 3 specialization experiments
- Graph is guaranteed connected (auto-repair if needed)

### Message Passing
- Mean pooling for message aggregation (simple, works well)
- 3 rounds of message passing by default (tunable)
- Messages are tensors, not structured data (keeps it differentiable)

### World Model
- JEPA-inspired (predict in latent space, not observations)
- EMA target encoder to prevent collapse
- Hierarchical prediction at multiple time scales
- Curiosity = prediction error in latent space

### Training (Phase 2)
- PPO with clipped objective for stable updates
- GAE (lambda=0.95) for advantage estimation
- World model trained jointly with policy
- Curiosity coefficient = 0.5 (tunable)

### Environment (Phase 2)
- Multiple resource types with different effects:
  - Energy: +0.2 energy
  - Food: +0.3 energy, reduces hunger
  - Water: reduces thirst
  - Material: for future crafting
- Survival mechanics (hunger/thirst increase over time)
- Resource respawn after configurable delay
- Internal walls for navigation challenge

### Specialized Agents (Phase 3)
- Each agent type has distinct architectural inductive biases
- Perception: Spatial attention + multi-scale feature pyramid
- Reasoning: Transformer blocks + working memory slots
- Memory: Key-value memory bank with content-based addressing
- Planning: Goal-conditioned policy with value estimation

### Typed Messaging (Phase 3)
- Messages carry semantic type (PERCEPT, INFERENCE, GOAL, etc.)
- Agents filter messages based on type relevance
- Control messages (REWARD_SIGNAL, NOVELTY_SIGNAL) modulate processing
- TypeAwareAggregator processes content vs control messages differently

### Neuromorphic Computing (Phase 5)
- LIF neurons with configurable time constants (tau_mem, tau_syn)
- Surrogate gradients (fast sigmoid) for backprop through spikes
- STDP learning: pre-before-post = LTP, post-before-pre = LTD
- Reward-modulated STDP for RL (eligibility traces + dopamine)
- Energy tracking based on neuromorphic hardware models (Loihi, TrueNorth)
- Hybrid swarm: mix of spiking and rate-coded agents
- Target spike rates ~0.1 for energy efficiency

### Evolution & Scaling (Phase 6)
- NEAT-style genome with innovation numbers for structural alignment
- Speciation via genome distance (agents, connections, weights)
- Tournament selection + elitism for evolution
- CMA-ES for continuous parameter tuning (learning rate, sparsity, etc.)
- DARTS: Differentiable NAS with bi-level optimization
- ENAS: Weight sharing with RNN controller (REINFORCE training)
- Hierarchical swarms: 3 levels (agents -> groups -> supergroups)
- Sparse messaging: k-nearest neighbors based on spatial position
- Lazy agent initialization with LRU cache for memory efficiency
- Gradient checkpointing every N agents for large swarms
- CPU offloading for agents not currently processing

## Key Files

| Component | Primary Files |
|-----------|--------------|
| Agents | `src/agents/micro_agent.py`, `src/agents/specializations.py` |
| Swarm | `src/swarm/graph.py`, `src/swarm/specialized_graph.py` |
| Messaging | `src/swarm/messaging.py`, `src/swarm/typed_messaging.py` |
| World Model | `src/world_model/jepa.py` |
| Environment | `src/environment/cosmos.py` |
| Neuromodulation | `src/neuromod/signals.py` |
| Self-Model | `src/self_model/*.py` (capabilities, uncertainty, calibration, meta_learning, theory_of_mind, swarm_integration) |
| Neuromorphic | `src/neuromorphic/lif.py`, `src/neuromorphic/stdp.py`, `src/neuromorphic/layers.py`, `src/neuromorphic/energy.py`, `src/neuromorphic/swarm_integration.py` |
| Evolution | `src/evolution/distributed.py`, `src/evolution/genetic.py`, `src/evolution/nas.py`, `src/evolution/scaling.py` |
| Training | `src/training/__init__.py` (PPO), `src/training/exploration.py` |
| Experiments | `experiments/train_curiosity.py`, `experiments/train_specialization.py`, `experiments/train_metacognition.py`, `experiments/train_neuromorphic.py`, `experiments/train_evolution.py` |
| Tests | `tests/test_*.py` (162 tests total) |

## Performance Notes

- 100 agents with hidden_dim=128: ~2.5M parameters total
- Small swarm (20 agents, dim=64): ~100k parameters, fast iteration
- Large swarm (500 agents, dim=256): ~50M parameters, needs GPU
- Observation dim (32x32 grid, vision_radius=5): ~137 dimensions

## Session Log

### Session 1
- Initialized project from PRD
- Implemented all core components for Phase 1
- Tests written and structured
- Ready for Phase 2 implementation

### Session 2
- Implemented full PPO training with GAE
- Created trajectory buffer and training infrastructure
- Added curiosity integration (world model, RND, ICM options)
- Enhanced environment with:
  - Multiple resource types (energy, food, water, material)
  - Resource respawn mechanics
  - Survival mechanics (hunger, thirst)
  - Internal walls for maze-like navigation
- Created exploration tracking (coverage, novelty metrics)
- Built curiosity-driven training script with comparison mode
- Phase 2 complete, ready for Phase 3

### Session 3
- Migrated to MacBook (CLI teleport)
- Fixed bug in modular topology: nodes not being added explicitly
- Fixed bug in hierarchical topology: same issue
- Fixed test_agent_communication: set networks to eval mode for determinism
- All 40 tests passing
- Verified Phase 2 end-to-end training works:
  - Environment creates correctly (16x16 grid, 137 obs dim)
  - Swarm trains with PPO + curiosity rewards
  - World model loss decreases during training
- Ready for Phase 3 implementation

### Session 4
- Implemented Phase 3: Specialization (full implementation)
- Created specialized agent architectures (`src/agents/specializations.py`):
  - PerceptionNetwork: Multi-head spatial attention, multi-scale processing, saliency detection
  - ReasoningNetwork: Transformer blocks with working memory slots
  - MemoryNetwork: Key-value memory bank with content-based read/write
  - PlanningNetwork: Goal-conditioned policy with hierarchical subgoal generation
  - SpecializedAgent factory class
- Created typed message passing (`src/swarm/typed_messaging.py`):
  - MessageType enum with 12 semantic types
  - TypedMessage with source type, priority, metadata
  - TypedMessageBus with type-aware routing and filtering
  - MessageFilter for flexible message acceptance
  - MessageEncoder/Decoder for type embedding
  - TypeAwareAggregator separating content vs control messages
- Created SpecializedSwarmGraph (`src/swarm/specialized_graph.py`):
  - Integrates specialized agents with typed messaging
  - RoleEmergenceTracker for specialization metrics
  - Supports all topology types
- Created specialization training script (`experiments/train_specialization.py`):
  - SpecializationPPOTrainer adapted for specialized swarms
  - Comparison experiment: specialized vs generic
  - Role emergence analysis tools
- Fixed dimension mismatch bug: aggregators now use output_dim for message content
- All 72 tests passing (32 new Phase 3 tests)
- Phase 3 complete, ready for Phase 4

### Session 5
- Implemented Phase 4: Self-Model (Meta-Cognition)
- Created uncertainty quantification (`src/self_model/uncertainty.py`):
  - EnsembleUncertainty, MCDropoutUncertainty, EvidentialNetwork
  - UncertaintyAggregator for combining methods
  - UncertaintyHead for learned uncertainty prediction
- Created confidence calibration (`src/self_model/calibration.py`):
  - TemperatureScaling, PlattScaling for post-hoc calibration
  - FocalLoss, LabelSmoothing for training-time calibration
  - CalibrationLoss for differentiable ECE optimization
  - CalibrationTracker for monitoring
- Created meta-learning (`src/self_model/meta_learning.py`):
  - MAML for fast adaptation with few examples
  - MetaSGD with learned per-parameter learning rates
  - Reptile as simplified meta-learning alternative
  - TaskEmbedding for task similarity detection
- Created Theory of Mind (`src/self_model/theory_of_mind.py`):
  - BeliefEncoder, IntentPredictor for modeling other agents
  - TheoryOfMind with perspective taking and trust tracking
  - CollectiveBeliefAggregator for swarm-level beliefs
- Created swarm integration (`src/self_model/swarm_integration.py`):
  - MetaCognitiveSwarm wrapping specialized swarm
  - Per-agent uncertainty heads and self-models
  - MetaCognitionTrainer for training meta-cognitive abilities
- Created training script (`experiments/train_metacognition.py`):
  - Uncertainty training, MAML demo, ToM demo
  - Comparison: with vs without meta-cognition
- Fixed bugs: evidential uncertainty clamping, calibration tracker property, trainer gradients
- All 104 tests passing (32 new Phase 4 tests)
- Phase 4 complete, ready for Phase 5

### Session 6
- Implemented Phase 5: Neuromorphic Computing
- Created LIF neurons (`src/neuromorphic/lif.py`):
  - LIFNeuron with full dynamics (membrane, synaptic, refractory)
  - LIFLayer, RecurrentLIFLayer, AdaptiveLIFLayer
  - PopulationLIF for rate/latency coding
  - SurrogateSpike for differentiable training
- Created STDP learning rules (`src/neuromorphic/stdp.py`):
  - ClassicSTDP: Pair-based with exponential windows
  - TripletSTDP: More biologically accurate
  - SymmetricSTDP: Hebbian unsupervised
  - RewardModulatedSTDP: Eligibility traces for RL
  - HomeostaticSTDP: Target firing rate maintenance
  - STDPLayer: Layer with online weight updates
- Created SNN architectures (`src/neuromorphic/layers.py`):
  - SpikingLinear, SpikingConv2d, SpikingRNN
  - SpikingNetwork: Multi-layer SNN
  - LiquidStateMachine: Reservoir computing
  - TemporalCoding, SpikeDecoder for I/O
- Created energy metrics (`src/neuromorphic/energy.py`):
  - Hardware models: Loihi, TrueNorth, SpiNNaker, etc.
  - SpikeCounter, EnergyTracker, EnergyBudget
  - EnergyEfficientLoss for sparse solutions
  - compare_energy_efficiency for SNN vs ANN
- Created spiking swarm (`src/neuromorphic/swarm_integration.py`):
  - SpikingMicroAgent with LIF + STDP
  - SpikingSwarmGraph with spike communication
  - HybridSwarm mixing spiking + rate-coded
  - SpikingSwarmTrainer with energy constraints
- Created training script (`experiments/train_neuromorphic.py`):
  - SNN training with surrogate gradients
  - STDP unsupervised learning
  - Reward-modulated STDP demo
  - Spiking swarm training
  - SNN vs ANN energy comparison
  - Liquid State Machine demo
- Fixed import error: AgentConfig (not MicroAgentConfig)
- All 132 tests passing (28 new Phase 5 tests)
- Phase 5 complete, ready for Phase 6

### Session 7
- Implemented Phase 6: Evolution & Scaling
- Created distributed training infrastructure (`src/evolution/distributed.py`):
  - DistributedConfig, ParallelismMode, DistributedManager
  - DistributedSwarm with DDP wrapper
  - ModelParallelSwarm, PipelineParallelSwarm
  - AsyncGradientAggregator for non-blocking sync
- Created evolutionary optimization (`src/evolution/genetic.py`):
  - NEAT-style genomes (AgentGene, ConnectionGene, SwarmGenome)
  - InnovationTracker for structural alignment
  - SwarmMutator, SwarmCrossover for evolution
  - Species for speciation and fitness sharing
  - GeneticOptimizer with tournament selection
  - CMAES for continuous parameter optimization
- Created Neural Architecture Search (`src/evolution/nas.py`):
  - SearchSpace with configurable operations
  - DARTS: DARTSCell, DARTSAgent, DARTSSearcher
  - ENAS: ENASController, ENASSharedNetwork, ENASSearcher
  - RandomSearchNAS baseline
- Created scaling utilities (`src/evolution/scaling.py`):
  - AgentPool with LRU cache for lazy initialization
  - AgentGroup for local clustering
  - HierarchicalSwarm for multi-level organization
  - SparseMessageGraph for efficient communication
  - CheckpointedSwarm, BatchedAgentProcessor
  - OffloadedSwarm, StreamingSwarm for memory efficiency
- Created training script (`experiments/train_evolution.py`):
  - Topology evolution, CMA-ES optimization
  - DARTS and ENAS architecture search
  - Hierarchical swarm demos, scaling benchmarks
- Fixed bugs:
  - Dataclass inheritance (non-default after default)
  - ENAS log_prob stacking (stack -> cat with view)
  - AgentGroup forward signature (try/except for message arg)
  - BatchedAgentProcessor hidden_dim inference
  - AgentGroup representation shape consistency
- All 162 tests passing (30 new Phase 6 tests)
- Phase 6 complete, all 6 phases implemented

---

*This file is updated by Claude to maintain context across development sessions.*
