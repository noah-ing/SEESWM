# SEESWM Architecture

## Overview

SEESWM (Self-Evolving Embodied Swarm-World-Modeler) is a distributed AGI architecture using swarms of specialized micro-agents.

## Core Components

### 1. Micro-Agents (`src/agents/`)

Each agent is a small neural network (10k-1M parameters) with:
- MLP backbone with residual connections
- GRU-style local state gating
- Plasticity parameters modulated by neuromodulatory signals

Agent types:
- **Perception**: Receive environment input, extract features
- **Reasoning**: Process integrated information, perform logic
- **Memory**: Store and retrieve episodic/semantic information
- **Planning**: Produce action outputs, goal-directed behavior

### 2. Swarm Graph (`src/swarm/`)

Agents are organized in a graph topology:
- **Random** (Erdos-Renyi): Baseline
- **Small-World** (Watts-Strogatz): Efficient + clustered
- **Scale-Free** (Barabasi-Albert): Hub-and-spoke
- **Modular**: Clusters per specialization
- **Hierarchical**: Layer-based (perception → reasoning → planning)

Message passing runs for K rounds, allowing information to propagate through the network.

### 3. World Model (`src/world_model/`)

JEPA-inspired architecture:
- Encoder: observation → latent state
- Predictor: latent + action → predicted next latent
- Target encoder: EMA-updated for stability

Curiosity signal = prediction error in latent space.

### 4. Environment (`src/environment/`)

Grid world simulation:
- Resources, hazards, walls, goals
- Multimodal observation (proprioception, vision, interoception)
- Simple physics (movement, collision, energy)

### 5. Neuromodulation (`src/neuromod/`)

Functional "emotions" as learning signals:
- **Dopamine**: Reward/confirmation → boost learning
- **Curiosity**: Novelty/error → drive exploration
- **Fear**: Danger → increase caution
- **Uncertainty**: Disagreement → flag "I don't know"

### 6. Self-Model (`src/self_model/`)

Meta-cognitive capabilities:
- Track capability beliefs per domain
- Bayesian updating after task attempts
- Confidence calibration (ECE)
- "I don't know" detection

## Data Flow

```
Environment Observation
        │
        ▼
┌──────────────────┐
│ Perception Agents│ ← Input agents receive observation
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Message Passing  │ ← K rounds of inter-agent communication
│   (K rounds)     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Planning Agents  │ → Output agents produce action
└────────┬─────────┘
         │
         ▼
    Environment Action
```

## Synergy Metric

Key hypothesis: collective intelligence > sum of individuals.

Synergy = Collective_Performance - Avg_Individual_Performance

Positive synergy indicates emergent capabilities.

## Evolution

Genetic algorithm optimizes:
- Number of agents per type
- Hidden dimensions
- Topology type
- Learning rates
- Neuromodulation sensitivity

Multi-objective fitness: performance + synergy + efficiency
