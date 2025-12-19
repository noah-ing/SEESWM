# AGI Prototype: Self-Evolving Embodied Swarm-World-Modeler (SEESWM)

A novel AGI architecture that rejects the monolithic LLM paradigm in favor of a **distributed swarm of specialized micro-agents** embedded in simulated worlds, evolving brain-like dynamics through neuromorphic-inspired processing, with functional "emotions" serving as adaptive learning signals.

## Core Hypothesis

**Collective intelligence from many small specialized agents, grounded in world simulation, will exhibit emergent capabilities that equivalent-parameter monolithic models cannot.**

## Key Research Foundations

- **ARC Prize 2025**: 7M-parameter TRM achieved 45% on ARC-AGI-1
- **Nature Communications (Nov 2025)**: SNNs achieve 2x adversarial robustness vs ANNs
- **AI Frontiers**: Identifies continual learning and world modeling as critical bottlenecks
- **DeepMind's roadmap**: 50% scaling + 50% innovation

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SWARM ORCHESTRATOR                           │
└─────────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  PERCEPTION   │    │   REASONING   │    │    MEMORY     │
│    AGENTS     │    │    AGENTS     │    │    AGENTS     │
└───────────────┘    └───────────────┘    └───────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      WORLD MODEL CORE (JEPA-inspired)               │
└─────────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SIMULATED ENVIRONMENT                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic training
python experiments/train_basic.py

# Run tests
pytest tests/
```

## Project Structure

```
seeswm/
├── src/
│   ├── agents/         # MicroAgent implementations
│   ├── swarm/          # Swarm topology and messaging
│   ├── world_model/    # JEPA-inspired world model
│   ├── environment/    # Simulated grid world
│   ├── neuromod/       # Neuromodulatory signals
│   ├── self_model/     # Uncertainty and meta-cognition
│   └── utils/          # Configuration and logging
├── experiments/        # Training scripts
├── tests/              # Unit tests
└── configs/            # YAML configurations
```

## Implementation Phases

1. **Foundation** - Basic agents, swarm graph, message passing
2. **World Modeling** - JEPA predictor, curiosity signals
3. **Specialization** - Agent types, hierarchical topology
4. **Self-Model** - Uncertainty quantification, "I don't know"
5. **Neuromorphic** - Spiking neural networks, STDP
6. **Evolution** - Genetic optimization of swarm architecture

## License

MIT
