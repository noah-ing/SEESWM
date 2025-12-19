"""
Neuromodulatory signals for adaptive learning.

Functional "emotions" that modulate agent learning and behavior:
- Dopamine: reward/prediction confirmation -> boost learning
- Curiosity: novelty/prediction error -> drive exploration
- Fear: danger signals -> increase caution
- Uncertainty: ensemble disagreement -> flag "I don't know"
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch


class SignalType(Enum):
    """Types of neuromodulatory signals."""

    DOPAMINE = auto()  # Reward, confirmation
    CURIOSITY = auto()  # Novelty, prediction error
    FEAR = auto()  # Danger, rapid loss
    UNCERTAINTY = auto()  # Ensemble disagreement


@dataclass
class NeuromodulatorState:
    """Current state of all neuromodulatory signals."""

    dopamine: float = 0.0
    curiosity: float = 0.0
    fear: float = 0.0
    uncertainty: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "dopamine": self.dopamine,
            "curiosity": self.curiosity,
            "fear": self.fear,
            "uncertainty": self.uncertainty,
        }


class NeuromodulatorSystem:
    """
    Central neuromodulatory system for the swarm.

    Computes signals based on:
    - External rewards (dopamine)
    - Prediction errors (curiosity)
    - Energy/damage changes (fear)
    - Ensemble disagreement (uncertainty)

    Then broadcasts these to modulate agent plasticity.
    """

    def __init__(
        self,
        dopamine_scale: float = 1.0,
        curiosity_scale: float = 1.0,
        fear_scale: float = 1.0,
        uncertainty_threshold: float = 0.3,
        decay_rate: float = 0.9,
    ):
        self.dopamine_scale = dopamine_scale
        self.curiosity_scale = curiosity_scale
        self.fear_scale = fear_scale
        self.uncertainty_threshold = uncertainty_threshold
        self.decay_rate = decay_rate

        self.state = NeuromodulatorState()
        self.history: list[NeuromodulatorState] = []

    def compute_dopamine(
        self,
        reward: float,
        predicted_reward: Optional[float] = None,
    ) -> float:
        """
        Compute dopamine signal.

        Reward prediction error model:
        - Positive surprise (reward > predicted) -> high dopamine
        - Expected reward -> baseline dopamine
        - Negative surprise (reward < predicted) -> low dopamine
        """
        if predicted_reward is None:
            # Simple reward magnitude
            signal = max(0, reward) * self.dopamine_scale
        else:
            # Reward prediction error
            rpe = reward - predicted_reward
            signal = (1.0 + rpe) * self.dopamine_scale

        return max(0, min(2.0, signal))  # Clamp to [0, 2]

    def compute_curiosity(
        self,
        prediction_error: float,
        novelty_score: Optional[float] = None,
    ) -> float:
        """
        Compute curiosity signal.

        High prediction error -> high curiosity -> explore more.
        """
        signal = prediction_error * self.curiosity_scale
        if novelty_score is not None:
            signal = 0.7 * signal + 0.3 * novelty_score

        return max(0, min(1.0, signal))

    def compute_fear(
        self,
        energy_delta: float,
        damage_delta: float,
        near_hazard: bool = False,
    ) -> float:
        """
        Compute fear signal.

        Triggers on:
        - Rapid energy loss
        - Damage received
        - Proximity to known hazards
        """
        signal = 0.0

        # Energy loss
        if energy_delta < 0:
            signal += abs(energy_delta) * 2.0

        # Damage
        signal += damage_delta * 3.0

        # Hazard proximity
        if near_hazard:
            signal += 0.3

        return max(0, min(1.0, signal * self.fear_scale))

    def compute_uncertainty(
        self,
        ensemble_outputs: list[torch.Tensor],
    ) -> float:
        """
        Compute uncertainty from ensemble disagreement.

        High variance across ensemble -> high uncertainty.
        """
        if len(ensemble_outputs) < 2:
            return 0.0

        stacked = torch.stack(ensemble_outputs)
        variance = stacked.var(dim=0).mean().item()

        return min(1.0, variance / self.uncertainty_threshold)

    def update(
        self,
        reward: float = 0.0,
        prediction_error: float = 0.0,
        energy_delta: float = 0.0,
        damage_delta: float = 0.0,
        near_hazard: bool = False,
        ensemble_outputs: Optional[list[torch.Tensor]] = None,
    ) -> NeuromodulatorState:
        """
        Update all neuromodulatory signals.

        Returns the new state.
        """
        # Decay previous state
        self.state.dopamine *= self.decay_rate
        self.state.curiosity *= self.decay_rate
        self.state.fear *= self.decay_rate
        self.state.uncertainty *= self.decay_rate

        # Compute new signals
        dopamine = self.compute_dopamine(reward)
        curiosity = self.compute_curiosity(prediction_error)
        fear = self.compute_fear(energy_delta, damage_delta, near_hazard)

        uncertainty = 0.0
        if ensemble_outputs:
            uncertainty = self.compute_uncertainty(ensemble_outputs)

        # Update with momentum
        alpha = 0.3
        self.state.dopamine = alpha * dopamine + (1 - alpha) * self.state.dopamine
        self.state.curiosity = alpha * curiosity + (1 - alpha) * self.state.curiosity
        self.state.fear = alpha * fear + (1 - alpha) * self.state.fear
        self.state.uncertainty = alpha * uncertainty + (1 - alpha) * self.state.uncertainty

        # Record history
        self.history.append(NeuromodulatorState(
            dopamine=self.state.dopamine,
            curiosity=self.state.curiosity,
            fear=self.state.fear,
            uncertainty=self.state.uncertainty,
        ))

        return self.state

    def get_modulation_factors(self) -> dict[str, float]:
        """Get factors for modulating agent plasticity."""
        return {
            "learning_rate_multiplier": 1.0 + self.state.dopamine - 0.5 * self.state.fear,
            "exploration_bonus": self.state.curiosity,
            "caution_factor": self.state.fear,
            "should_flag_idk": self.state.uncertainty > 0.5,
        }

    def reset(self) -> None:
        """Reset to initial state."""
        self.state = NeuromodulatorState()
        self.history = []
