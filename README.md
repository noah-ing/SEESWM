# SEESWM: Self-Evolving Embodied Swarm-World-Modeler

A novel AGI architecture that rejects the monolithic LLM paradigm in favor of a **distributed swarm of specialized micro-agents** embedded in simulated worlds, evolving brain-like dynamics through neuromorphic-inspired processing, with functional "emotions" serving as adaptive learning signals.

## Core Hypothesis

**Collective intelligence from many small specialized agents, grounded in world simulation, will exhibit emergent capabilities that equivalent-parameter monolithic models cannot.**

## Key Features

- **Swarm Intelligence**: 10-1000+ micro-agents with specialized roles (perception, reasoning, memory, planning)
- **World Modeling**: JEPA-inspired predictive model with curiosity-driven exploration
- **Meta-Cognition**: Uncertainty quantification, confidence calibration, "I don't know" detection
- **Theory of Mind**: Agents model other agents' beliefs and intentions
- **Neuromorphic Computing**: Spiking neural networks with STDP learning, energy-efficient inference
- **Evolutionary Optimization**: NEAT-style topology evolution, Neural Architecture Search (DARTS/ENAS)
- **Scalability**: Hierarchical swarms, sparse messaging, distributed training

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SWARM ORCHESTRATOR                           │
│         (Hierarchical organization, sparse messaging)               │
└─────────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  PERCEPTION   │    │   REASONING   │    │    MEMORY     │
│    AGENTS     │    │    AGENTS     │    │    AGENTS     │
│  (Attention,  │    │ (Transformer, │    │  (Key-value,  │
│   Saliency)   │    │   Working     │    │   Content-    │
│               │    │   Memory)     │    │   addressed)  │
└───────────────┘    └───────────────┘    └───────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 TYPED MESSAGE PASSING (K rounds)                    │
│    (PERCEPT, INFERENCE, QUERY, GOAL, REWARD_SIGNAL, etc.)           │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   META-COGNITIVE LAYER                              │
│  (Uncertainty, Calibration, MAML, Theory of Mind)                   │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 WORLD MODEL CORE (JEPA-inspired)                    │
│         (Latent prediction, curiosity, RND/ICM options)             │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SIMULATED ENVIRONMENT                            │
│     (Grid world, resources, survival mechanics, obstacles)          │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install torch numpy networkx pyyaml

# Run tests (162 tests)
pytest tests/ -v

# Run experiments
python experiments/train_curiosity.py          # Phase 2: World modeling
python experiments/train_specialization.py    # Phase 3: Agent specialization
python experiments/train_metacognition.py     # Phase 4: Meta-cognition
python experiments/train_neuromorphic.py      # Phase 5: Spiking networks
python experiments/train_evolution.py         # Phase 6: Evolution & scaling
```

## Project Structure

```
seeswm/
├── src/
│   ├── agents/           # MicroAgent, specialized architectures
│   ├── swarm/            # SwarmGraph, typed messaging, topologies
│   ├── world_model/      # JEPA predictor, curiosity modules (RND, ICM)
│   ├── environment/      # Grid world with resources, survival mechanics
│   ├── neuromod/         # Neuromodulatory signals (dopamine, curiosity, fear)
│   ├── self_model/       # Uncertainty, calibration, meta-learning, ToM
│   ├── neuromorphic/     # LIF neurons, STDP, SNNs, energy tracking
│   ├── evolution/        # Distributed training, genetic, NAS, scaling
│   ├── training/         # PPO, exploration metrics
│   └── utils/            # Configuration, logging
├── experiments/          # Training scripts for each phase
├── tests/                # Unit tests (162 total)
├── configs/              # YAML configurations
└── docs/                 # Architecture docs, PRD, development memory
```

## Implementation Phases (All Complete)

| Phase | Description | Key Components |
|-------|-------------|----------------|
| 1 | **Foundation** | MicroAgent, SwarmGraph, message passing, topologies |
| 2 | **World Modeling** | JEPA predictor, PPO+GAE, curiosity (RND/ICM), enhanced environment |
| 3 | **Specialization** | Perception/Reasoning/Memory/Planning networks, typed messaging |
| 4 | **Self-Model** | Uncertainty (ensemble/MC dropout/evidential), calibration, MAML, Theory of Mind |
| 5 | **Neuromorphic** | LIF neurons, STDP learning, SNNs, energy tracking, hybrid swarms |
| 6 | **Evolution & Scaling** | Distributed training, NEAT evolution, DARTS/ENAS, hierarchical swarms |

## Research Foundations

- **ARC Prize 2025**: 7M-parameter TRM achieved 45% on ARC-AGI-1
- **Nature Communications (Nov 2025)**: SNNs achieve 2x adversarial robustness vs ANNs
- **AI Frontiers**: Identifies continual learning and world modeling as critical bottlenecks
- **DeepMind's roadmap**: 50% scaling + 50% innovation

## Performance Notes

- Small swarm (20 agents, dim=64): ~100k parameters, fast iteration
- Medium swarm (100 agents, dim=128): ~2.5M parameters
- Large swarm (500 agents, dim=256): ~50M parameters, benefits from GPU
- Hierarchical scaling: 1000+ agents with O(n) messaging via sparse graphs

## License

MIT
