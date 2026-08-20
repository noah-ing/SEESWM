# SEESWM evidence note

This note describes only the evidence committed to this repository. It replaces
an earlier draft that characterized untracked trained-model results as validated
findings.

## Available artifacts

### `results/hypothesis_validation.json`

- Timestamp: 2025-12-19.
- The generating script creates fresh networks and contains no training step.
- The artifact records neither a source revision nor a seed. The current script
  corrects its baseline sizing and agent-role invariants, so exact reproduction
  of the historical values is not claimed.
- In the ten random-regression trials, swarm MSE was 0.4486 ± 0.0117 and
  baseline MSE was 0.4522 ± 0.0124; the swarm was lower in seven trials.
- The recorded parameter counts are 563,520 for the swarm and 44,752 for the
  baseline, so this is not an equivalent-parameter comparison.
- The remaining experiments are small untrained sweeps over agent count,
  topology, and ten grid-world episodes.

### `results/emergence/emergence_scaling_results.json`

- Timestamp: 2026-01-03.
- Covers 4, 10, 20, 50, 100, and 150 fresh swarms.
- The artifact does not identify a checkpoint or invocation, and no checkpoint
  is tracked in the repository.
- Mean reward peaks at 1.236 for 10 agents in this artifact and is 0.800 for 100
  agents. This is a descriptive untrained scaling observation.
- Historical “phase transition” entries are outputs of a slope-change
  heuristic, not hypothesis-test results. The current script calls them
  `slope_change_flags`.

## Checkpoint and evaluator status

The candidate writers `train_for_validation.py`, `train_specialized.py`, and
`train_specialization.py` use a versioned primitive/tensor-only checkpoint
schema; older training scripts remain legacy-only. The evaluators require
PyTorch 2.10 or newer, use its restricted loader, cap accepted file size,
reconstruct the saved topology and policy head, strict-load every component
used by the evaluated policy, and reject mismatches. Value-head and world-model
state may be retained for provenance but are not evaluated by the policy
diagnostic. Legacy schemas are rejected. A `.pt` file still requires trusted
provenance and a verified digest; these controls are not a sandbox. The report
records the checkpoint SHA-256 and does not certify emergence, generalization,
or baseline superiority.

## Unsupported conclusions

The committed artifacts do not support claims of trained emergent
specialization, a 100-fold performance advantage, 100% transfer efficiency,
robustness, scalable oversight, or superiority to trained standard baselines.
They also do not establish that agent messages are causally useful.

## Required evidence for stronger claims

- versioned trained checkpoints and full training configurations;
- matched parameter, compute, data, and optimization budgets;
- established MARL baselines and tasks requiring coordination;
- multiple seeds with uncertainty estimates and predefined tests;
- raw trajectories, evaluation logs, environment versions, and dependency lock;
- explicit message-passing and specialization ablations.

See the repository README for reproduction commands and the current project
scope.
