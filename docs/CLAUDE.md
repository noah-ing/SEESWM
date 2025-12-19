# SEESWM Development Memory

This file tracks progress, decisions, and context for AI-assisted development sessions.

## Project Status

**Current Phase**: Phase 2 - World Modeling (Complete)
**Last Updated**: Session 2

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

### In Progress
- [ ] Visualization and TensorBoard integration

### Next Steps (Phase 3 - Specialization)
1. Agent type specialization with different architectures
2. Hierarchical topology experiments
3. Role emergence metrics
4. Typed message passing

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

## Key Files

| Component | Primary Files |
|-----------|--------------|
| Agents | `src/agents/micro_agent.py` |
| Swarm | `src/swarm/graph.py`, `src/swarm/messaging.py` |
| World Model | `src/world_model/jepa.py` |
| Environment | `src/environment/cosmos.py` |
| Neuromodulation | `src/neuromod/signals.py` |
| Self-Model | `src/self_model/capabilities.py` |
| Training | `src/training/__init__.py` (PPO), `src/training/exploration.py` |
| Experiments | `experiments/train_curiosity.py`, `experiments/train_basic.py` |
| Tests | `tests/test_*.py` |

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

---

*This file is updated by Claude to maintain context across development sessions.*
