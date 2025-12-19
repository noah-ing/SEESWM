"""
Metrics for measuring swarm collective intelligence.

Key metrics:
- Synergy: collective > sum of individuals
- Redundancy: overlap in agent representations
- Specialization: how distinct are agent roles
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SynergyMetrics:
    """Metrics quantifying emergent collective intelligence."""

    synergy: float  # Collective - sum of individuals (positive = emergent)
    redundancy: float  # How much agents overlap
    specialization: float  # How distinct are agent outputs
    collective_accuracy: float
    individual_accuracy: float


def compute_mutual_information(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 20,
) -> float:
    """
    Estimate mutual information between predictions and targets.

    Uses histogram-based estimation for continuous variables.
    """
    # Discretize into bins
    pred_bins = torch.bucketize(
        predictions.flatten(),
        torch.linspace(predictions.min(), predictions.max(), num_bins),
    )
    target_bins = torch.bucketize(
        targets.flatten(),
        torch.linspace(targets.min(), targets.max(), num_bins),
    )

    # Joint histogram
    joint = torch.zeros(num_bins, num_bins)
    for p, t in zip(pred_bins, target_bins):
        if p < num_bins and t < num_bins:
            joint[p, t] += 1

    # Normalize to get probabilities
    joint = joint / joint.sum()
    p_pred = joint.sum(dim=1)
    p_target = joint.sum(dim=0)

    # MI = sum p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = 0.0
    for i in range(num_bins):
        for j in range(num_bins):
            if joint[i, j] > 0 and p_pred[i] > 0 and p_target[j] > 0:
                mi += joint[i, j] * torch.log(
                    joint[i, j] / (p_pred[i] * p_target[j])
                )

    return mi.item()


def compute_representation_similarity(
    outputs: list[torch.Tensor],
) -> float:
    """
    Compute average cosine similarity between agent outputs.

    High similarity = high redundancy = agents doing similar things.
    """
    if len(outputs) < 2:
        return 0.0

    similarities = []
    for i, out1 in enumerate(outputs):
        for out2 in outputs[i + 1 :]:
            # Flatten and compute cosine similarity
            v1 = out1.flatten()
            v2 = out2.flatten()
            sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0))
            similarities.append(sim.item())

    return sum(similarities) / len(similarities)


def compute_specialization_score(
    outputs: list[torch.Tensor],
    agent_types: list[str],
) -> float:
    """
    Measure how well agent outputs cluster by type.

    High score = agents of same type produce similar outputs,
    different types produce different outputs.
    """
    if len(outputs) < 2:
        return 0.0

    within_type_sim = []
    between_type_sim = []

    for i, (out1, type1) in enumerate(zip(outputs, agent_types)):
        for j, (out2, type2) in enumerate(zip(outputs[i + 1 :], agent_types[i + 1 :])):
            v1 = out1.flatten()
            v2 = out2.flatten()
            sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

            if type1 == type2:
                within_type_sim.append(sim)
            else:
                between_type_sim.append(sim)

    avg_within = sum(within_type_sim) / max(1, len(within_type_sim))
    avg_between = sum(between_type_sim) / max(1, len(between_type_sim))

    # Specialization = within-type similarity - between-type similarity
    return avg_within - avg_between
