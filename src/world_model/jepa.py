"""
JEPA-inspired World Model.

Joint Embedding Predictive Architecture for learning world dynamics
by predicting in latent space rather than pixel/observation space.

To be implemented in Phase 2.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """Encode observations to latent space."""

    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class Predictor(nn.Module):
    """Predict next latent state given current latent and action."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([latent, action], dim=-1)
        return self.network(combined)


class WorldModel(nn.Module):
    """
    JEPA-style world model.

    Learns to predict future states in latent space.
    Uses EMA target encoder to prevent representation collapse.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 256,
        num_hierarchy_levels: int = 3,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay

        # Encoder
        self.encoder = Encoder(obs_dim, latent_dim)

        # Target encoder (EMA updated)
        self.target_encoder = copy.deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Hierarchical predictors (different time horizons)
        self.predictors = nn.ModuleList([
            Predictor(latent_dim, action_dim)
            for _ in range(num_hierarchy_levels)
        ])

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Encode observation and predict future latents.

        Returns:
            current_latent: Encoded current state
            predictions: List of predicted future latents at different horizons
        """
        current_latent = self.encoder(observation)
        predictions = [pred(current_latent, action) for pred in self.predictors]
        return current_latent, predictions

    def compute_curiosity(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
    ) -> float:
        """
        Compute curiosity signal as prediction error.

        High error = novel/surprising state = high curiosity.
        """
        _, predictions = self.forward(observation, action)

        with torch.no_grad():
            target_latent = self.target_encoder(next_observation)

        # Prediction error at finest granularity
        error = F.mse_loss(predictions[0], target_latent)
        return error.item()

    def compute_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        next_observation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training loss."""
        _, predictions = self.forward(observation, action)

        with torch.no_grad():
            target_latent = self.target_encoder(next_observation)

        # Sum losses across hierarchy
        loss = sum(F.mse_loss(pred, target_latent) for pred in predictions)
        return loss

    @torch.no_grad()
    def update_target(self) -> None:
        """EMA update of target encoder."""
        for param, target_param in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            target_param.data = (
                self.ema_decay * target_param.data + (1 - self.ema_decay) * param.data
            )
