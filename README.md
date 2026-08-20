# SEESWM

SEESWM is a research prototype for experimenting with small neural agents that
exchange messages in a simulated grid world. It includes multiple agent
specializations, graph topologies, world-model components, training scripts,
and exploratory evaluation utilities.

> **Status:** proof of concept. This repository does not contain a trained model
> checkpoint, peer-reviewed result, safety evaluation, or evidence of a
> production-ready agent system. The checked-in results are small exploratory
> runs and should not be read as validation of emergent intelligence.

## Research question

Can a graph of specialized agents develop useful coordination or division of
labor that a carefully matched monolithic model does not?

The codebase provides infrastructure for investigating that question. The
tracked evidence does not answer it yet: the committed comparison uses
untrained networks and its nominal baseline is not parameter matched.

## What is implemented

- perception, reasoning, memory, and planning agent variants;
- message passing over small-world, modular, hierarchical, and other graphs;
- a configurable resource-and-hazard grid environment;
- experimental world-model, metacognition, neuromorphic, and evolutionary
  modules;
- ablation, scaling, generalization, statistics, and interpretability helpers;
- training and exploratory evaluation scripts under `experiments/`.

These modules are experimental scaffolding. Their presence does not establish
that every proposed mechanism has been trained, validated, or integrated into a
single end-to-end system.

## Tracked evidence

Two JSON artifacts are committed so the current claims can be audited directly.

### Fresh-network comparison

[`results/hypothesis_validation.json`](results/hypothesis_validation.json) was
recorded on 2025-12-19. Its generator constructed new networks and evaluated
them without a training step. The artifact does not record a source revision or
seed. The current
[`validate_hypothesis.py`](experiments/validate_hypothesis.py) retains the
fresh-network scope but corrects the baseline sizing and role-count invariants,
so it is an extension of that experiment rather than a bit-for-bit reproducer.

| Recorded experiment | Scope | Recorded result |
|---|---:|---|
| Random regression comparison | 10 trials | swarm MSE 0.4486 vs baseline MSE 0.4522; swarm lower in 7/10 trials |
| Internal aggregation-score sweep | 5 trials per size | highest mean score at 20 agents (0.0761) |
| Random topology sweep | 10 trials per topology | modular graph had the highest internal score (0.0705) |
| Grid-world rollouts | 10 episodes | reward 0.509 ± 2.235; survival 94 ± 18 steps; exploration 1.56% ± 0.72% |

The comparison is **not equivalent-parameter**: the artifact records 563,520
swarm parameters and 44,752 baseline parameters. The MSE difference is small,
and no confidence interval or significance test is recorded for it.

### Untrained scaling sweep

[`results/emergence/emergence_scaling_results.json`](results/emergence/emergence_scaling_results.json)
was recorded on 2026-01-03 for 4, 10, 20, 50, 100, and 150 agents. The artifact
does not record a checkpoint identifier or invocation arguments, and no model
checkpoint is tracked in this repository. It should therefore be treated as an
untrained/default scaling sweep unless independent provenance is supplied.

The recorded mean reward peaks at 10 agents (1.236) and falls to 0.800 at 100
agents, while both parameter count and message count increase with swarm size.
The historical artifact's “phase transition” labels came from a local
slope-change heuristic; they are descriptive flags, not statistical evidence
of a physical or learned phase transition. The old plot is retained beside the
artifact for provenance, but is not presented here as a research result.

## What these artifacts do not show

- a trained swarm outperforming a trained, compute-matched baseline;
- 100% transfer or generalization to unseen environments;
- statistically validated emergent specialization;
- robustness, graceful degradation, alignment, or safety properties;
- results outside one simulated grid-world family;
- reproducibility across hardware, dependency versions, or independent teams.

## Reproduce and extend

Python 3.10 or 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,analysis,viz]'

# Unit tests
python -m pytest

# Re-run the fresh-network experiments
python experiments/validate_hypothesis.py --device cpu --seed 0 \
  --output results/hypothesis_validation.local.json

# Quick fresh-random behavioral/scaling smoke test
python experiments/emergence_scaling_analysis.py --device cpu --seed 0 --quick \
  --output results/behavioral-local

# Train a schema-v2 candidate checkpoint, then evaluate that exact policy
python experiments/train_for_validation.py --device cpu \
  --save results/models/candidate.pt
python experiments/validate_rigorously.py --device cpu --seeds 10 --episodes 10 \
  --max-steps 100 --model results/models/candidate.pt \
  --output results/validation-local
```

Write new outputs to a separate path so the committed evidence remains intact.
The candidate writers `train_for_validation.py`, `train_specialized.py`, and
`train_specialization.py` write restricted-loader-compatible schema-v2
checkpoints containing the swarm configuration, policy head, environment, and
native-type metrics. Other historical training scripts have not been migrated
and their outputs are not accepted by the schema-v2 evaluator. The evaluator
requires PyTorch 2.10 or newer, uses the restricted loader, bounds checkpoint
size, and rejects older schemas or any configuration, topology, agent, edge, or
policy mismatch instead of falling back to random weights. Its diagnostic
report records the checkpoint SHA-256. These controls do not make arbitrary
third-party `.pt` files safe; use only a checkpoint with trusted provenance and
a verified digest. See [`SECURITY.md`](SECURITY.md). The diagnostic is still not
a substitute for trained controls or independent validation.

## Minimum credible next experiment

1. Train the swarm and a parameter- and compute-matched monolithic baseline.
2. Pre-register primary metrics, seeds, stopping rules, and exclusion criteria.
3. Evaluate on tasks that require information sharing, plus held-out world
   layouts and at least one established multi-agent benchmark.
4. Report per-seed results, confidence intervals, effect sizes, and failures.
5. Publish checkpoints, raw trajectories, environment versions, and an exact
   dependency lock.

Until then, SEESWM is best understood as an experimental framework and source
of testable hypotheses, not evidence that collective intelligence has emerged.

## License

[MIT](LICENSE). This project is not affiliated with or endorsed by any cited
researcher, institution, benchmark, or model provider.
