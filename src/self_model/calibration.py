"""
Confidence calibration methods.

Ensures predicted confidence matches actual accuracy.
A well-calibrated model saying "80% confident" should be correct 80% of the time.

Methods:
1. Temperature Scaling: Simple post-hoc calibration
2. Platt Scaling: Logistic regression on logits
3. Focal Loss: Training-time calibration
4. Label Smoothing: Implicit calibration via soft targets
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration quality."""

    ece: float = 0.0  # Expected Calibration Error
    mce: float = 0.0  # Maximum Calibration Error
    brier: float = 0.0  # Brier score
    nll: float = 0.0  # Negative log-likelihood

    # Per-bin details
    bin_accuracies: List[float] = field(default_factory=list)
    bin_confidences: List[float] = field(default_factory=list)
    bin_counts: List[int] = field(default_factory=list)


def compute_calibration_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_bins: int = 15,
) -> CalibrationMetrics:
    """
    Compute comprehensive calibration metrics.

    Args:
        logits: Model output logits [batch, num_classes]
        targets: Ground truth labels [batch]
        num_bins: Number of bins for ECE computation

    Returns:
        CalibrationMetrics with ECE, MCE, Brier, NLL
    """
    probs = F.softmax(logits, dim=-1)
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(targets).float()

    # Binning
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    ece = 0.0
    mce = 0.0
    total_samples = len(targets)

    for i in range(num_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        bin_size = in_bin.sum().item()

        if bin_size > 0:
            bin_acc = accuracies[in_bin].mean().item()
            bin_conf = confidences[in_bin].mean().item()

            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            bin_counts.append(bin_size)

            # ECE contribution
            gap = abs(bin_acc - bin_conf)
            ece += (bin_size / total_samples) * gap
            mce = max(mce, gap)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)

    # Brier score
    one_hot = F.one_hot(targets, probs.shape[-1]).float()
    brier = ((probs - one_hot) ** 2).sum(dim=-1).mean().item()

    # NLL
    nll = F.cross_entropy(logits, targets).item()

    return CalibrationMetrics(
        ece=ece,
        mce=mce,
        brier=brier,
        nll=nll,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for post-hoc calibration.

    Divides logits by a learned temperature parameter.
    Simple but very effective for neural networks.

    Reference: "On Calibration of Modern Neural Networks"
    """

    def __init__(self, initial_temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([initial_temperature]))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        max_iter: int = 50,
    ) -> float:
        """
        Find optimal temperature on validation set.

        Uses LBFGS optimization to minimize NLL.
        """
        # Use LBFGS for optimization
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = F.cross_entropy(scaled, targets)
            loss.backward()
            return loss

        optimizer.step(closure)

        return self.temperature.item()

    def get_calibrated_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Get calibrated probabilities."""
        return F.softmax(self.forward(logits), dim=-1)


class PlattScaling(nn.Module):
    """
    Platt scaling for calibration.

    Fits a logistic regression on the logits (or max logit for binary).
    More flexible than temperature scaling but may overfit.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        # Affine transformation per class
        self.weights = nn.Parameter(torch.ones(num_classes))
        self.biases = nn.Parameter(torch.zeros(num_classes))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Platt scaling."""
        return logits * self.weights + self.biases

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> Dict[str, float]:
        """Fit Platt scaling parameters."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        best_loss = float('inf')
        for _ in range(max_iter):
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = F.cross_entropy(scaled, targets)

            if loss.item() < best_loss:
                best_loss = loss.item()

            loss.backward()
            optimizer.step()

        return {"final_nll": best_loss}


class FocalLoss(nn.Module):
    """
    Focal Loss for training-time calibration.

    Down-weights easy examples, focusing on hard ones.
    This implicitly improves calibration by preventing overconfidence.

    Reference: "Focal Loss for Dense Object Detection"
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss."""
        probs = F.softmax(logits, dim=-1)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Get probability of true class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Apply class weights if provided
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothing(nn.Module):
    """
    Label smoothing for implicit calibration.

    Softens one-hot targets to prevent overconfidence.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        num_classes: int = 10,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-entropy with label smoothing."""
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (self.num_classes - 1)

        # Create smoothed targets
        one_hot = F.one_hot(targets, self.num_classes).float()
        smooth_targets = one_hot * confidence + (1 - one_hot) * smooth_value

        # Cross entropy with soft targets
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)

        return loss.mean()


class CalibrationLoss(nn.Module):
    """
    Explicit calibration loss for end-to-end training.

    Combines task loss with calibration penalty.
    """

    def __init__(
        self,
        num_bins: int = 15,
        calibration_weight: float = 0.1,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.calibration_weight = calibration_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Returns (total_loss, {task_loss, calibration_loss, ece})
        """
        # Task loss
        task_loss = F.cross_entropy(logits, targets)

        # Calibration penalty (differentiable approximation of ECE)
        probs = F.softmax(logits, dim=-1)
        confidences, predictions = probs.max(dim=-1)
        accuracies = predictions.eq(targets).float()

        # Soft binning for differentiability
        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1, device=logits.device)
        calibration_loss = torch.tensor(0.0, device=logits.device)

        for i in range(self.num_bins):
            # Soft membership (using sigmoid instead of hard threshold)
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]

            # Approximate bin membership
            in_bin_soft = torch.sigmoid(10 * (confidences - lower)) * \
                         torch.sigmoid(10 * (upper - confidences))

            bin_weight = in_bin_soft.sum() + 1e-8

            bin_acc = (in_bin_soft * accuracies).sum() / bin_weight
            bin_conf = (in_bin_soft * confidences).sum() / bin_weight

            calibration_loss += in_bin_soft.sum() * (bin_acc - bin_conf).abs()

        calibration_loss = calibration_loss / len(targets)

        total_loss = task_loss + self.calibration_weight * calibration_loss

        return total_loss, {
            "task_loss": task_loss.item(),
            "calibration_loss": calibration_loss.item(),
        }


class AdaptiveCalibrator(nn.Module):
    """
    Adaptive calibration that learns when to trust predictions.

    Learns a separate network to predict calibration adjustments
    based on input features.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 32,
    ):
        super().__init__()

        # Predict temperature adjustment from features
        self.adjustment_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensure positive temperature
        )

        # Base temperature
        self.base_temperature = nn.Parameter(torch.tensor([1.0]))

    def forward(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive temperature scaling."""
        # Predict per-sample temperature adjustment
        adjustment = self.adjustment_net(features)

        # Effective temperature
        temperature = self.base_temperature * (1 + adjustment)

        # Apply to logits
        return logits / temperature

    def get_effective_temperature(self, features: torch.Tensor) -> torch.Tensor:
        """Get the effective temperature for given features."""
        adjustment = self.adjustment_net(features)
        return self.base_temperature * (1 + adjustment)


class CalibrationTracker:
    """
    Track calibration metrics over training.

    Maintains running statistics for monitoring calibration.
    """

    def __init__(self, num_bins: int = 15):
        self.num_bins = num_bins
        self.history: List[CalibrationMetrics] = []

    @property
    def total_updates(self) -> int:
        """Total number of updates."""
        return len(self.history)

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> CalibrationMetrics:
        """Compute and store calibration metrics."""
        metrics = compute_calibration_metrics(logits, targets, self.num_bins)
        self.history.append(metrics)
        return metrics

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        if not self.history:
            return {}

        recent = self.history[-100:]  # Last 100 updates

        return {
            "avg_ece": sum(m.ece for m in recent) / len(recent),
            "avg_mce": sum(m.mce for m in recent) / len(recent),
            "avg_brier": sum(m.brier for m in recent) / len(recent),
            "avg_nll": sum(m.nll for m in recent) / len(recent),
            "total_updates": len(self.history),
        }

    def is_well_calibrated(self, ece_threshold: float = 0.05) -> bool:
        """Check if model is well-calibrated."""
        if not self.history:
            return False
        return self.history[-1].ece < ece_threshold
