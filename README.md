# SEESWM: Self-Evolving Embodied Swarm World-Modeler

A distributed architecture for emergent collective intelligence, replacing monolithic neural networks with swarms of specialized micro-agents.

---

## Overview

SEESWM tests the hypothesis that **collective intelligence from many small specialized agents will exhibit emergent capabilities that equivalent-parameter monolithic models cannot achieve**.

After training for 1000 epochs, a 20-agent swarm (2.4M parameters) achieves:
- **10/10 wins** against all baseline architectures (p < 0.001)
- **+1.01 reward delta** vs single-agent, ensemble, and centralized baselines
- **4/7 validation criteria** passed (up from 2/7 untrained)

---

## Quick Start

```bash
# Install dependencies
pip install torch numpy networkx scipy scikit-learn pyyaml tqdm

# Run tests (162 total)
pytest tests/ -v

# Train a swarm
python experiments/train_for_validation.py --epochs 1000 --agents 20

# Validate (untrained)
python experiments/validate_rigorously.py

# Validate (with trained model)
python experiments/validate_rigorously.py --model results/models/swarm_trained_*.pt
```

---

## Architecture

### Micro-Agents

Each agent is a small neural network (10K-100K parameters) with:
- MLP backbone with residual connections
- GRU-style local memory gating
- Plasticity modulated by neuromodulatory signals

**Specializations:**

| Type | Role | Architecture |
|------|------|--------------|
| Perception | Feature extraction | Multi-head spatial attention, saliency detection |
| Reasoning | Inference | Transformer blocks with working memory |
| Memory | Storage/retrieval | Key-value content-addressed memory |
| Planning | Action selection | Goal-conditioned hierarchical policy |

### Swarm Topology

Agents communicate through a configurable graph:

| Topology | Properties | Best For |
|----------|------------|----------|
| Small-World | High clustering, short paths | General purpose (default) |
| Scale-Free | Hub-and-spoke | Information broadcast |
| Modular | Clustered by type | Specialized processing |
| Hierarchical | Layered flow | Sequential tasks |

### Message Passing

Typed messages propagate over K rounds:
- `PERCEPT`: Sensory information
- `INFERENCE`: Logical conclusions
- `QUERY`: Information requests
- `GOAL`: Objectives and subgoals
- `REWARD_SIGNAL`: Learning signals

### World Model

JEPA-inspired predictive architecture:
- Encoder: observation → latent state
- Predictor: latent + action → next latent
- Curiosity = prediction error (drives exploration)

---

## Validation Framework

Rigorous experimental validation with 7 criteria:

| Test | Description | Status |
|------|-------------|--------|
| Ablations | Components matter when removed | Pending |
| Scaling | Performance scales favorably | **Pass** |
| Synergy | Positive collective information gain | Pending |
| Baselines | Beats equivalent architectures | **Pass** |
| Generalization | Transfers to new environments | **Pass** |
| Emergence | Shows emergent behaviors | Pending |
| Interpretability | Provides analysis insights | **Pass** |

### Baseline Comparison (Trained Model)

```
Baseline                  Score      Swarm      Delta       Wins    p-value
----------------------------------------------------------------------
single_agent            -0.0100     1.0000    +1.0100 10/10    0.0000***
ensemble                -0.0100     1.0000    +1.0100 10/10    0.0000***
centralized             -0.0100     1.0000    +1.0100 10/10    0.0000***
independent             -0.1378     1.0000    +1.1378 10/10    0.0000***
random                   0.5774     1.0000    +0.4226 10/10    0.0000***

PASS: Swarm outperforms all required baselines
```

### Training Progress

| Metric | Untrained | Trained (1000 epochs) |
|--------|-----------|----------------------|
| Validation Score | 2/7 | 4/7 |
| Baseline Wins | 6/50 | 50/50 |
| Avg Reward | ~0.5 | ~1.0 |

---

## Project Structure

```
seeswm/
├── src/
│   ├── agents/           # MicroAgent, specializations
│   ├── swarm/            # SwarmGraph, typed messaging
│   ├── world_model/      # JEPA predictor, curiosity
│   ├── environment/      # Grid world simulation
│   ├── self_model/       # Uncertainty, meta-learning, ToM
│   ├── neuromorphic/     # LIF neurons, STDP, SNNs
│   ├── evolution/        # Distributed training, NAS, scaling
│   ├── training/         # PPO, exploration metrics
│   └── validation/       # Ablations, synergy, baselines, stats
├── experiments/
│   ├── train_for_validation.py   # Actor-critic training
│   ├── validate_rigorously.py    # Full validation suite
│   ├── train_curiosity.py        # World model + curiosity
│   ├── train_specialization.py   # Agent role emergence
│   ├── train_metacognition.py    # Self-model training
│   ├── train_neuromorphic.py     # Spiking neural networks
│   └── train_evolution.py        # Genetic + NAS optimization
├── tests/                # 162 unit tests
└── results/              # Saved models and validation outputs
```

---

## Key Components

### Phase 1: Foundation
- SwarmGraph with 5 topology types
- Message passing infrastructure
- Neuromodulatory signals (dopamine, curiosity, fear, uncertainty)

### Phase 2: World Modeling
- JEPA-style latent prediction
- RND/ICM curiosity modules
- PPO training with GAE

### Phase 3: Specialization
- 4 specialized agent architectures
- Typed message passing (12 message types)
- Role emergence tracking

### Phase 4: Meta-Cognition
- Uncertainty quantification (ensemble, MC dropout, evidential)
- Confidence calibration (temperature scaling, focal loss)
- Meta-learning (MAML, MetaSGD, Reptile)
- Theory of Mind (belief encoding, intent prediction)

### Phase 5: Neuromorphic
- LIF neurons with surrogate gradients
- STDP learning (classic, triplet, reward-modulated)
- Energy-efficient spiking networks
- Hybrid spiking/rate-coded swarms

### Phase 6: Evolution & Scaling
- NEAT-style genetic optimization
- CMA-ES for continuous parameters
- DARTS/ENAS neural architecture search
- Hierarchical swarms (1000+ agents)
- Gradient checkpointing and CPU offloading

---

## Validation Modules

### Synergy Measurement
Partial Information Decomposition (PID) quantifies collective intelligence:
```
Synergy = I(X1,...,Xn; Y) - Σ I(Xi; Y)
```
Positive synergy = emergent capability beyond individual contributions.

### Ablation Studies
Systematically removes components to measure importance:
- `no_swarm`: Single large agent with same parameters
- `no_message_passing`: Isolated agents
- `no_memory`: State cleared each step
- `random_topology`: Unstructured graph
- `full_connectivity`: All-to-all messaging

### Statistical Rigor
- Paired t-tests with Bonferroni correction
- Cohen's d effect sizes
- Bootstrap confidence intervals
- Minimum sample size estimation

---

## Environment

Grid world simulation with:
- Multiple resource types (energy, food, water, material)
- Survival mechanics (hunger, thirst)
- Internal walls for navigation
- Multimodal observations (vision, proprioception, interoception)

---

## Performance

| Configuration | Parameters | Training Time | Notes |
|---------------|------------|---------------|-------|
| 20 agents, dim=128 | 2.4M | ~1 hour (CPU) | Default |
| 100 agents, dim=128 | 12M | ~5 hours (CPU) | Large scale |
| 20 agents, dim=64 | 600K | ~20 min (CPU) | Fast iteration |

---

## Citation

```bibtex
@software{seeswm2024,
  title={SEESWM: Self-Evolving Embodied Swarm World-Modeler},
  year={2024},
  url={https://github.com/...}
}
```

---

## License

MIT

---

## References

1. LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. Meta AI.
2. Stanley, K. O., & Miikkulainen, R. (2002). Evolving Neural Networks through Augmenting Topologies. Evolutionary Computation.
3. Maas, W. (1997). Networks of Spiking Neurons: The Third Generation of Neural Network Models. Neural Networks.
4. Williams, P. L., & Beer, R. D. (2010). Nonnegative Decomposition of Multivariate Information. arXiv.
