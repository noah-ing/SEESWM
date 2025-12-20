# SEESWM Architecture

## Overview

SEESWM (Self-Evolving Embodied Swarm-World-Modeler) is a distributed AGI architecture using swarms of specialized micro-agents with neuromorphic processing, meta-cognition, and evolutionary optimization.

## Core Components

### 1. Micro-Agents (`src/agents/`)

Each agent is a small neural network (10k-1M parameters) with:
- MLP backbone with residual connections
- GRU-style local state gating
- Plasticity parameters modulated by neuromodulatory signals

**Base Agent Types:**
- **Perception**: Receive environment input, extract features
- **Reasoning**: Process integrated information, perform logic
- **Memory**: Store and retrieve episodic/semantic information
- **Planning**: Produce action outputs, goal-directed behavior

**Specialized Architectures (Phase 3):**
- **PerceptionNetwork**: Multi-head spatial attention, multi-scale processing, saliency detection
- **ReasoningNetwork**: Transformer blocks with working memory slots
- **MemoryNetwork**: Key-value memory bank with content-based read/write
- **PlanningNetwork**: Goal-conditioned policy with hierarchical subgoal generation

### 2. Swarm Graph (`src/swarm/`)

Agents are organized in a graph topology:
- **Random** (Erdos-Renyi): Baseline
- **Small-World** (Watts-Strogatz): Efficient + clustered (default)
- **Scale-Free** (Barabasi-Albert): Hub-and-spoke
- **Modular**: Clusters per specialization
- **Hierarchical**: Layer-based (perception → reasoning → planning)

Message passing runs for K rounds, allowing information to propagate through the network.

**Typed Messaging (Phase 3):**
- MessageType enum: PERCEPT, SALIENCY, INFERENCE, QUERY, MEMORY_RECALL, GOAL, etc.
- TypedMessage with source type, priority, metadata
- TypeAwareAggregator separating content vs control messages

### 3. World Model (`src/world_model/`)

JEPA-inspired architecture:
- Encoder: observation → latent state
- Predictor: latent + action → predicted next latent
- Target encoder: EMA-updated for stability

Curiosity signal = prediction error in latent space.

**Alternative Curiosity Modules:**
- RND (Random Network Distillation)
- ICM (Intrinsic Curiosity Module)

### 4. Environment (`src/environment/`)

Grid world simulation:
- Multiple resource types (energy, food, water, material)
- Survival mechanics (hunger, thirst)
- Internal walls for maze-like navigation
- Resource respawn mechanics

Multimodal observation:
- Proprioception (position, energy, hunger, thirst)
- Vision (local grid view)
- Interoception (internal states)

### 5. Neuromodulation (`src/neuromod/`)

Functional "emotions" as learning signals:
- **Dopamine**: Reward/confirmation → boost learning
- **Curiosity**: Novelty/error → drive exploration
- **Fear**: Danger → increase caution
- **Uncertainty**: Disagreement → flag "I don't know"

### 6. Self-Model / Meta-Cognition (`src/self_model/`)

**Uncertainty Quantification:**
- EnsembleUncertainty: Disagreement between ensemble members
- MCDropoutUncertainty: Monte Carlo dropout sampling
- EvidentialNetwork: Single-pass Dirichlet uncertainty
- UncertaintyAggregator: Combines multiple methods

**Confidence Calibration:**
- TemperatureScaling: Post-hoc calibration
- PlattScaling: Logistic regression on logits
- FocalLoss, LabelSmoothing: Training-time calibration
- CalibrationLoss: Differentiable ECE

**Meta-Learning:**
- MAML: Model-Agnostic Meta-Learning for fast adaptation
- MetaSGD: Learned per-parameter learning rates
- Reptile: Simplified meta-learning via averaging
- TaskEmbedding: Task similarity for transfer

**Theory of Mind:**
- BeliefEncoder: Infer agent beliefs from observations
- IntentPredictor: Predict agent intentions from behavior
- TheoryOfMind: Full module with perspective taking
- CollectiveBeliefAggregator: Swarm-level beliefs

### 7. Neuromorphic Computing (`src/neuromorphic/`)

**LIF Neurons:**
- LIFNeuron: Full membrane, synaptic, refractory dynamics
- LIFLayer, RecurrentLIFLayer, AdaptiveLIFLayer
- PopulationLIF: Rate/latency population coding
- SurrogateSpike: Differentiable backprop through spikes

**STDP Learning:**
- ClassicSTDP: Pair-based with exponential windows
- TripletSTDP: More biologically accurate
- RewardModulatedSTDP: Eligibility traces for RL
- HomeostaticSTDP: Target firing rate maintenance

**SNN Architectures:**
- SpikingLinear, SpikingConv2d, SpikingRNN
- SpikingNetwork: Multi-layer SNN
- LiquidStateMachine: Reservoir computing

**Energy Efficiency:**
- Hardware models: Loihi, TrueNorth, SpiNNaker
- EnergyTracker, EnergyBudget
- EnergyEfficientLoss for sparse solutions

**Hybrid Swarms:**
- SpikingMicroAgent: LIF-based with STDP
- SpikingSwarmGraph: Spike communication
- HybridSwarm: Mix spiking + rate-coded agents

### 8. Evolution & Scaling (`src/evolution/`)

**Distributed Training:**
- Data, model, and pipeline parallelism
- DistributedSwarm with DDP wrapper
- AsyncGradientAggregator for non-blocking sync

**Evolutionary Optimization:**
- NEAT-style genomes (AgentGene, ConnectionGene)
- InnovationTracker for structural alignment
- SwarmMutator, SwarmCrossover for evolution
- Species for fitness sharing
- CMAES for continuous parameters

**Neural Architecture Search:**
- DARTS: Differentiable NAS with bi-level optimization
- ENAS: Weight sharing with RNN controller
- RandomSearchNAS: Baseline

**Scaling Utilities:**
- AgentPool: Lazy initialization with LRU cache
- AgentGroup: Local clustering
- HierarchicalSwarm: Multi-level organization (1000+ agents)
- SparseMessageGraph: k-nearest neighbor communication
- CheckpointedSwarm: Gradient checkpointing
- OffloadedSwarm: CPU offloading

## Data Flow

```
Environment Observation
        │
        ▼
┌──────────────────┐
│ Perception Agents│ ← Input agents receive observation
│  (Attention,     │
│   Saliency)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Typed Message    │ ← K rounds of inter-agent communication
│   Passing        │   with semantic routing
│   (K rounds)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Meta-Cognitive   │ ← Uncertainty, calibration, ToM
│   Processing     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Planning Agents  │ → Output agents produce action
│  (Goal-directed, │
│   Hierarchical)  │
└────────┬─────────┘
         │
         ▼
    Environment Action
```

## Synergy Metric

Key hypothesis: collective intelligence > sum of individuals.

```
Synergy = Collective_Performance - Avg_Individual_Performance
```

Positive synergy indicates emergent capabilities.

## Evolution

Genetic algorithm optimizes:
- Number of agents per type
- Hidden dimensions
- Topology type and connectivity
- Learning rates
- Neuromodulation sensitivity
- Agent architectures (via NAS)

Multi-objective fitness: performance + synergy + efficiency + energy

## Scaling Strategies

| Agents | Strategy | Memory |
|--------|----------|--------|
| 10-100 | Direct graph | Full in-memory |
| 100-500 | Sparse messaging | Batched processing |
| 500-1000 | Hierarchical groups | Gradient checkpointing |
| 1000+ | Agent pool + streaming | CPU offloading |
