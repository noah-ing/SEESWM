"""
Uncertainty quantification methods for meta-cognition.

Three complementary approaches:
1. Ensemble uncertainty: Disagreement between multiple models
2. Monte Carlo dropout: Approximate Bayesian inference via dropout sampling
3. Evidential uncertainty: Single-pass uncertainty via Dirichlet prior

These enable agents to know what they don't know.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UncertaintyEstimate:
    """Structured uncertainty estimate."""

    # Point estimate
    prediction: torch.Tensor

    # Uncertainty components
    aleatoric: float = 0.0  # Data uncertainty (irreducible)
    epistemic: float = 0.0  # Model uncertainty (reducible with more data)
    total: float = 0.0

    # Additional info
    confidence: float = 1.0  # 1 - total_uncertainty (clamped)
    entropy: float = 0.0

    def should_abstain(self, threshold: float = 0.5) -> bool:
        """Should the agent abstain from this prediction?"""
        return self.confidence < threshold


class EnsembleUncertainty(nn.Module):
    """
    Ensemble-based uncertainty estimation.

    Uses disagreement between ensemble members to estimate epistemic uncertainty.
    Each member sees same input but has different initialization/training.
    """

    def __init__(
        self,
        base_network_fn,  # Callable that returns a network
        num_members: int = 5,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_members = num_members
        self.device = device

        # Create ensemble members
        self.members = nn.ModuleList([
            base_network_fn() for _ in range(num_members)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, UncertaintyEstimate]:
        """
        Forward pass through all ensemble members.

        Returns mean prediction and uncertainty estimate.
        """
        # Collect predictions from all members
        predictions = []
        for member in self.members:
            pred = member(x)
            predictions.append(pred)

        stacked = torch.stack(predictions, dim=0)  # [num_members, batch, ...]

        # Mean prediction
        mean_pred = stacked.mean(dim=0)

        # Epistemic uncertainty = variance across ensemble
        variance = stacked.var(dim=0)
        epistemic = variance.mean().item()

        # For classification: also compute predictive entropy
        if mean_pred.dim() >= 2:  # Logits
            probs = F.softmax(mean_pred, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
        else:
            entropy = 0.0

        estimate = UncertaintyEstimate(
            prediction=mean_pred,
            epistemic=epistemic,
            aleatoric=0.0,  # Would need separate estimation
            total=epistemic,
            confidence=max(0.0, 1.0 - min(1.0, epistemic)),
            entropy=entropy,
        )

        return mean_pred, estimate

    def get_member_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get raw predictions from each member."""
        return [member(x) for member in self.members]

    def parameters_per_member(self) -> List:
        """Get parameters grouped by member (for diverse training)."""
        return [list(member.parameters()) for member in self.members]


class MCDropoutUncertainty(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.

    Keeps dropout enabled at inference time and samples multiple times
    to approximate Bayesian posterior.
    """

    def __init__(
        self,
        network: nn.Module,
        num_samples: int = 10,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.network = network
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate

        # Add dropout layers if not present
        self._add_dropout_layers()

    def _add_dropout_layers(self) -> None:
        """Ensure network has dropout layers."""
        # This is a simple approach - wraps the network
        # A more sophisticated approach would modify internal layers
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def forward(
        self,
        x: torch.Tensor,
        return_samples: bool = False,
    ) -> Tuple[torch.Tensor, UncertaintyEstimate]:
        """
        MC Dropout forward pass.

        Samples multiple times with dropout enabled.
        """
        # Ensure dropout is active
        self.train()  # Keep dropout active

        samples = []
        for _ in range(self.num_samples):
            # Apply dropout to input
            x_dropped = self.dropout(x)
            pred = self.network(x_dropped)
            samples.append(pred)

        stacked = torch.stack(samples, dim=0)

        # Statistics
        mean_pred = stacked.mean(dim=0)
        variance = stacked.var(dim=0)

        epistemic = variance.mean().item()

        # Entropy for classification
        if mean_pred.dim() >= 2:
            probs = F.softmax(mean_pred, dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
        else:
            entropy = 0.0

        estimate = UncertaintyEstimate(
            prediction=mean_pred,
            epistemic=epistemic,
            total=epistemic,
            confidence=max(0.0, 1.0 - min(1.0, epistemic)),
            entropy=entropy,
        )

        if return_samples:
            return mean_pred, estimate, samples
        return mean_pred, estimate


class EvidentialNetwork(nn.Module):
    """
    Evidential Deep Learning for single-pass uncertainty.

    Instead of predicting class probabilities directly, predicts parameters
    of a Dirichlet distribution over class probabilities. This allows
    distinguishing aleatoric (data) from epistemic (model) uncertainty.

    Reference: "Evidential Deep Learning to Quantify Classification Uncertainty"
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Network outputs evidence (positive values) for each class
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softplus(),  # Ensure positive evidence
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, UncertaintyEstimate]:
        """
        Compute Dirichlet parameters and uncertainty.

        Returns class probabilities and uncertainty estimate.
        """
        # Get evidence (positive values)
        evidence = self.network(x)

        # Dirichlet parameters: alpha = evidence + 1
        alpha = evidence + 1.0

        # Dirichlet strength
        S = alpha.sum(dim=-1, keepdim=True)

        # Expected probabilities (mean of Dirichlet)
        probs = alpha / S

        # Uncertainty decomposition
        # Aleatoric: expected entropy of categorical under Dirichlet
        # Epistemic: mutual information (how much we'd learn from seeing the label)

        # Total uncertainty (entropy of expected)
        total_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)

        # Aleatoric (expected entropy under Dirichlet)
        digamma_S = torch.digamma(S)
        digamma_alpha = torch.digamma(alpha)
        aleatoric = -(alpha / S * (digamma_alpha - digamma_S)).sum(dim=-1)

        # Epistemic = Total - Aleatoric (clamped to be non-negative)
        epistemic = torch.clamp(total_entropy - aleatoric, min=0.0)

        estimate = UncertaintyEstimate(
            prediction=probs,
            aleatoric=max(0.0, aleatoric.mean().item()),
            epistemic=max(0.0, epistemic.mean().item()),
            total=max(0.0, total_entropy.mean().item()),
            confidence=max(0.0, min(1.0, 1.0 - (epistemic.mean().item() / max(0.01, math.log(self.num_classes))))),
            entropy=total_entropy.mean().item(),
        )

        return probs, estimate

    def compute_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        kl_weight: float = 0.1,
    ) -> torch.Tensor:
        """
        Evidential loss: NLL + KL regularization.

        KL term encourages high uncertainty on wrong predictions.
        """
        evidence = self.network(x)
        alpha = evidence + 1.0
        S = alpha.sum(dim=-1, keepdim=True)

        # One-hot targets
        if targets.dim() == 1:
            targets = F.one_hot(targets, self.num_classes).float()

        # Type II Maximum Likelihood loss
        loss_nll = (
            targets * (torch.digamma(S) - torch.digamma(alpha))
        ).sum(dim=-1).mean()

        # KL divergence from uniform Dirichlet
        # Encourages evidence to be low for incorrect classes
        alpha_tilde = targets + (1 - targets) * alpha
        S_tilde = alpha_tilde.sum(dim=-1, keepdim=True)

        kl = (
            torch.lgamma(S_tilde).squeeze() - torch.lgamma(alpha_tilde).sum(dim=-1)
            - torch.lgamma(torch.tensor(self.num_classes, dtype=torch.float32))
            + ((alpha_tilde - 1) * (
                torch.digamma(alpha_tilde) - torch.digamma(S_tilde)
            )).sum(dim=-1)
        ).mean()

        return loss_nll + kl_weight * kl


class UncertaintyAggregator:
    """
    Aggregate uncertainty estimates from multiple sources.

    Combines ensemble, MC dropout, and evidential uncertainty
    for robust meta-cognitive assessment.
    """

    def __init__(
        self,
        ensemble_weight: float = 0.4,
        mcdropout_weight: float = 0.3,
        evidential_weight: float = 0.3,
    ):
        self.weights = {
            "ensemble": ensemble_weight,
            "mcdropout": mcdropout_weight,
            "evidential": evidential_weight,
        }

    def aggregate(
        self,
        estimates: Dict[str, UncertaintyEstimate],
    ) -> UncertaintyEstimate:
        """Weighted aggregation of uncertainty estimates."""
        total_weight = 0.0
        weighted_epistemic = 0.0
        weighted_aleatoric = 0.0
        weighted_entropy = 0.0

        # Use first available prediction
        prediction = None

        for source, estimate in estimates.items():
            weight = self.weights.get(source, 0.0)
            if weight > 0:
                weighted_epistemic += weight * estimate.epistemic
                weighted_aleatoric += weight * estimate.aleatoric
                weighted_entropy += weight * estimate.entropy
                total_weight += weight

                if prediction is None:
                    prediction = estimate.prediction

        if total_weight > 0:
            weighted_epistemic /= total_weight
            weighted_aleatoric /= total_weight
            weighted_entropy /= total_weight

        total = weighted_epistemic + weighted_aleatoric

        return UncertaintyEstimate(
            prediction=prediction if prediction is not None else torch.tensor([]),
            epistemic=weighted_epistemic,
            aleatoric=weighted_aleatoric,
            total=total,
            confidence=max(0.0, 1.0 - min(1.0, total)),
            entropy=weighted_entropy,
        )


class UncertaintyHead(nn.Module):
    """
    Lightweight uncertainty head that can be added to any network.

    Learns to predict its own uncertainty from intermediate features.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        super().__init__()

        # Predict uncertainty from features
        self.uncertainty_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # [aleatoric, epistemic]
            nn.Softplus(),  # Ensure positive
        )

        # Track calibration
        self.prediction_history: List[Tuple[float, float]] = []

    def forward(self, features: torch.Tensor) -> Tuple[float, float]:
        """Predict aleatoric and epistemic uncertainty."""
        uncertainties = self.uncertainty_net(features)
        aleatoric = uncertainties[..., 0].mean().item()
        epistemic = uncertainties[..., 1].mean().item()
        return aleatoric, epistemic

    def update_calibration(
        self,
        predicted_uncertainty: float,
        actual_error: float,
    ) -> None:
        """Track predicted vs actual for calibration."""
        self.prediction_history.append((predicted_uncertainty, actual_error))
        # Keep last 1000
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]

    def get_calibration_error(self) -> float:
        """Compute calibration error between predicted uncertainty and actual error."""
        if len(self.prediction_history) < 10:
            return 0.0

        predictions, actuals = zip(*self.prediction_history)
        pred_tensor = torch.tensor(predictions)
        actual_tensor = torch.tensor(actuals)

        # Correlation would be ideal = 1.0
        # We compute mean squared difference as error
        return F.mse_loss(pred_tensor, actual_tensor).item()
