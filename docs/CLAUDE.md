# SEESWM Development Memory

This file tracks progress, decisions, and context for AI-assisted development sessions.

## Project Status

**Current Phase**: Phase 1 - Foundation
**Last Updated**: Session 1

### Completed
- [x] Project structure created following PRD specification
- [x] Core MicroAgent implementation (MLP-based, with plasticity parameters)
- [x] SwarmGraph with multiple topology types (random, small-world, scale-free, modular, hierarchical)
- [x] Message passing infrastructure
- [x] Basic environment (CosmosEnvironment grid world)
- [x] Neuromodulatory system (dopamine, curiosity, fear, uncertainty signals)
- [x] Self-model for capability tracking and "I don't know"
- [x] JEPA-style world model (placeholder, functional)
- [x] Evolutionary optimization framework
- [x] Test suite for agents, swarm, and world model
- [x] Configuration system (YAML configs)
- [x] Basic training script

### In Progress
- [ ] Full RL training loop (currently REINFORCE structure only)
- [ ] Integration testing of all components together

### Next Steps (Phase 2)
1. Implement full PPO/A2C training with world model
2. Add curiosity-driven exploration with prediction error
3. Richer environment physics (resource respawn, multi-agent)
4. Visualization and TensorBoard logging

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

## Key Files

| Component | Primary Files |
|-----------|--------------|
| Agents | `src/agents/micro_agent.py` |
| Swarm | `src/swarm/graph.py`, `src/swarm/messaging.py` |
| World Model | `src/world_model/jepa.py` |
| Environment | `src/environment/cosmos.py` |
| Neuromodulation | `src/neuromod/signals.py` |
| Self-Model | `src/self_model/capabilities.py` |
| Training | `experiments/train_basic.py` |
| Tests | `tests/test_*.py` |

## Performance Notes

- 100 agents with hidden_dim=128: ~2.5M parameters total
- Small swarm (20 agents, dim=64): ~100k parameters, fast iteration
- Large swarm (500 agents, dim=256): ~50M parameters, needs GPU

## Session Log

### Session 1
- Initialized project from PRD
- Implemented all core components for Phase 1
- Tests written and structured
- Ready for Phase 2 implementation

---

*This file is updated by Claude to maintain context across development sessions.*
