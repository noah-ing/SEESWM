"""
Self-model for tracking agent capabilities and uncertainty.

Enables:
- Appropriate uncertainty ("I don't know")
- Capability awareness (what tasks I'm good/bad at)
- Confidence calibration
- Meta-learning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class CapabilityBelief:
    """Bayesian belief about capability in a domain."""

    mean: float = 0.5
    variance: float = 0.25
    sample_count: int = 0

    def update(self, outcome: float, learning_rate: float = 0.1) -> None:
        """Bayesian update after observing outcome."""
        self.sample_count += 1

        # Simple online update
        error = outcome - self.mean
        self.mean += learning_rate * error
        self.variance = (1 - learning_rate) * self.variance + \
                       learning_rate * error * error

        # Clamp
        self.mean = max(0.0, min(1.0, self.mean))
        self.variance = max(0.01, min(0.25, self.variance))


@dataclass
class TaskOutcome:
    """Record of a task attempt."""

    domain: int
    outcome: float  # 0-1
    confidence: float  # predicted confidence
    timestamp: int = 0


class SelfModel:
    """
    Model of the agent's own capabilities and limitations.

    Tracks:
    - Capability beliefs per domain
    - History of outcomes
    - Confidence calibration

    Enables:
    - "I don't know" responses
    - Transfer learning decisions
    - Meta-learning improvements
    """

    def __init__(
        self,
        num_domains: int = 10,
        idk_capability_threshold: float = 0.4,
        idk_variance_threshold: float = 0.2,
        ensemble_disagreement_threshold: float = 0.3,
    ):
        self.num_domains = num_domains
        self.idk_capability_threshold = idk_capability_threshold
        self.idk_variance_threshold = idk_variance_threshold
        self.ensemble_disagreement_threshold = ensemble_disagreement_threshold

        # Capability beliefs per domain
        self.capabilities: dict[int, CapabilityBelief] = {
            d: CapabilityBelief() for d in range(num_domains)
        }

        # History
        self.history: list[TaskOutcome] = []
        self.timestamp = 0

        # Calibration tracking
        self.calibration_bins: dict[int, list[tuple[float, float]]] = {
            i: [] for i in range(10)  # 10 confidence bins
        }

    def update_beliefs(self, domain: int, outcome: float) -> None:
        """Update capability belief after task attempt."""
        if domain not in self.capabilities:
            self.capabilities[domain] = CapabilityBelief()

        self.capabilities[domain].update(outcome)
        self.timestamp += 1

    def should_say_idk(
        self,
        domain: int,
        ensemble_outputs: Optional[list[torch.Tensor]] = None,
    ) -> tuple[bool, dict]:
        """
        Determine if uncertainty is high enough to flag "I don't know".

        Triggers when:
        1. Low capability belief in domain
        2. High variance in capability belief
        3. High ensemble disagreement
        """
        belief = self.capabilities.get(domain, CapabilityBelief())

        reasons = {}

        # Check capability threshold
        if belief.mean < self.idk_capability_threshold:
            reasons["low_capability"] = belief.mean

        # Check variance threshold
        if belief.variance > self.idk_variance_threshold:
            reasons["high_variance"] = belief.variance

        # Check ensemble disagreement
        if ensemble_outputs and len(ensemble_outputs) >= 2:
            stacked = torch.stack(ensemble_outputs)
            disagreement = stacked.var(dim=0).mean().item()
            if disagreement > self.ensemble_disagreement_threshold:
                reasons["ensemble_disagreement"] = disagreement

        should_idk = len(reasons) > 0

        info = {
            "should_idk": should_idk,
            "capability_mean": belief.mean,
            "capability_variance": belief.variance,
            "sample_count": belief.sample_count,
            "reasons": reasons,
        }

        return should_idk, info

    def record_outcome(
        self,
        domain: int,
        outcome: float,
        confidence: float,
    ) -> None:
        """Record task outcome for calibration tracking."""
        self.update_beliefs(domain, outcome)

        self.history.append(TaskOutcome(
            domain=domain,
            outcome=outcome,
            confidence=confidence,
            timestamp=self.timestamp,
        ))

        # Update calibration bins
        bin_idx = min(9, int(confidence * 10))
        self.calibration_bins[bin_idx].append((confidence, outcome))

    def get_calibration_error(self) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Good calibration: predicted confidence matches actual accuracy.
        """
        total_error = 0.0
        total_samples = 0

        for bin_idx, samples in self.calibration_bins.items():
            if not samples:
                continue

            confidences, outcomes = zip(*samples)
            avg_confidence = sum(confidences) / len(confidences)
            avg_accuracy = sum(outcomes) / len(outcomes)

            bin_error = abs(avg_confidence - avg_accuracy)
            total_error += bin_error * len(samples)
            total_samples += len(samples)

        if total_samples == 0:
            return 0.0

        return total_error / total_samples

    def get_domain_summary(self, domain: int) -> dict:
        """Get summary of capability belief for a domain."""
        belief = self.capabilities.get(domain, CapabilityBelief())
        return {
            "domain": domain,
            "capability_mean": belief.mean,
            "capability_variance": belief.variance,
            "sample_count": belief.sample_count,
            "would_say_idk": belief.mean < self.idk_capability_threshold,
        }

    def get_stats(self) -> dict:
        """Get overall self-model statistics."""
        return {
            "num_domains": self.num_domains,
            "total_outcomes": len(self.history),
            "calibration_error": self.get_calibration_error(),
            "avg_capability": sum(c.mean for c in self.capabilities.values()) / max(1, len(self.capabilities)),
        }
