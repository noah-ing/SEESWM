# SEESWM Development Memory

This file tracks progress, decisions, and context for AI-assisted development sessions.

## Project Status

**Current Phase**: Phase 3 - Specialization (Complete)
**Last Updated**: Session 4

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

### In Progress
- [ ] Visualization and TensorBoard integration

### Next Steps (Phase 4 - Collective Intelligence)
1. Multi-agent coordination tasks
2. Emergent communication protocols
3. Swarm-level goal decomposition
4. Neuromodulation-guided learning

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

## Key Files

| Component | Primary Files |
|-----------|--------------|
| Agents | `src/agents/micro_agent.py`, `src/agents/specializations.py` |
| Swarm | `src/swarm/graph.py`, `src/swarm/specialized_graph.py` |
| Messaging | `src/swarm/messaging.py`, `src/swarm/typed_messaging.py` |
| World Model | `src/world_model/jepa.py` |
| Environment | `src/environment/cosmos.py` |
| Neuromodulation | `src/neuromod/signals.py` |
| Self-Model | `src/self_model/capabilities.py` |
| Training | `src/training/__init__.py` (PPO), `src/training/exploration.py` |
| Experiments | `experiments/train_curiosity.py`, `experiments/train_specialization.py` |
| Tests | `tests/test_*.py` (72 tests total) |

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

---

*This file is updated by Claude to maintain context across development sessions.*
