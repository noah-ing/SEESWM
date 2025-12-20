# Product Requirements Document

## SEESWM: Self-Evolving Embodied Swarm-World-Modeler

A distributed AGI architecture using swarms of specialized micro-agents with neuromorphic processing, meta-cognition, and evolutionary optimization.

## Phase Tracking

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation (agents, swarm, basic env) | Complete |
| 2 | World Modeling (JEPA, curiosity, PPO) | Complete |
| 3 | Specialization (agent types, typed messaging) | Complete |
| 4 | Self-Model (uncertainty, meta-learning, ToM) | Complete |
| 5 | Neuromorphic (SNNs, STDP, energy tracking) | Complete |
| 6 | Evolution & Scaling (distributed, NAS, 1000+ agents) | Complete |

## Phase Details

### Phase 1: Foundation
- MicroAgent with plasticity parameters
- SwarmGraph with 6 topology types
- Message passing infrastructure
- Basic grid environment
- Neuromodulatory signals
- 40 tests

### Phase 2: World Modeling
- JEPA-inspired world model
- PPO with GAE training
- Curiosity modules (RND, ICM)
- Enhanced environment (resources, survival)
- Exploration metrics
- 40 tests

### Phase 3: Specialization
- PerceptionNetwork (attention, saliency)
- ReasoningNetwork (transformer, working memory)
- MemoryNetwork (key-value, content-addressed)
- PlanningNetwork (goal-conditioned, subgoals)
- Typed message passing (12 semantic types)
- Role emergence tracking
- 72 tests

### Phase 4: Self-Model
- Uncertainty quantification (ensemble, MC dropout, evidential)
- Confidence calibration (temperature, Platt, focal loss)
- Meta-learning (MAML, MetaSGD, Reptile)
- Theory of Mind (beliefs, intentions, perspective taking)
- MetaCognitiveSwarm integration
- 104 tests

### Phase 5: Neuromorphic
- LIF neurons with full dynamics
- STDP learning rules (classic, triplet, reward-modulated)
- SNN architectures (linear, conv, RNN, LSM)
- Energy tracking (Loihi, TrueNorth models)
- Hybrid spiking + rate-coded swarms
- 132 tests

### Phase 6: Evolution & Scaling
- Distributed training (data, model, pipeline parallelism)
- NEAT-style evolutionary optimization
- Neural Architecture Search (DARTS, ENAS)
- Hierarchical swarms for 1000+ agents
- Sparse messaging, lazy initialization
- Gradient checkpointing, CPU offloading
- 162 tests

## Success Metrics

- **Synergy**: Collective performance > sum of individual agents
- **Generalization**: Transfer to unseen environments
- **Efficiency**: Energy-efficient neuromorphic inference
- **Scalability**: Linear scaling to 1000+ agents
- **Emergence**: Novel behaviors not explicitly programmed

## Future Directions

1. End-to-end integration testing across all phases
2. Multi-environment embodiment (3D, physics)
3. Language grounding and instruction following
4. Continual learning without catastrophic forgetting
5. Real-world robotics deployment
