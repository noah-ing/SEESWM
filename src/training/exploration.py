"""
Exploration metrics and novelty detection for curiosity-driven learning.

Tracks:
- State visitation counts
- Coverage of the environment
- Novelty scores
- Exploration efficiency
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
import numpy as np
import torch


@dataclass
class StateVisitation:
    """Track state visitation for exploration analysis."""

    # Grid-based visitation (for spatial exploration)
    grid_visits: Dict[Tuple[int, int], int] = field(default_factory=dict)

    # Hash-based visitation (for general states)
    state_visits: Dict[int, int] = field(default_factory=dict)

    # Trajectory of visited positions
    trajectory: List[Tuple[int, int]] = field(default_factory=list)

    # Unique states encountered
    unique_states: Set[int] = field(default_factory=set)

    def record_visit(self, position: Tuple[int, int], state_hash: int) -> None:
        """Record a state visit."""
        # Grid visitation
        self.grid_visits[position] = self.grid_visits.get(position, 0) + 1
        self.trajectory.append(position)

        # State visitation
        self.state_visits[state_hash] = self.state_visits.get(state_hash, 0) + 1
        self.unique_states.add(state_hash)

    def get_novelty_bonus(self, position: Tuple[int, int], state_hash: int) -> float:
        """
        Compute novelty bonus based on visitation count.

        Uses count-based exploration bonus: 1 / sqrt(N(s))
        """
        visit_count = self.state_visits.get(state_hash, 0)
        return 1.0 / np.sqrt(visit_count + 1)

    def get_coverage(self, grid_size: int) -> float:
        """Compute fraction of grid cells visited."""
        # Exclude border cells (walls)
        explorable_cells = (grid_size - 2) ** 2
        visited_cells = len([
            pos for pos in self.grid_visits
            if 0 < pos[0] < grid_size - 1 and 0 < pos[1] < grid_size - 1
        ])
        return visited_cells / explorable_cells

    def get_heatmap(self, grid_size: int) -> np.ndarray:
        """Generate visitation heatmap."""
        heatmap = np.zeros((grid_size, grid_size))
        for (x, y), count in self.grid_visits.items():
            heatmap[x, y] = count
        return heatmap

    def reset(self) -> None:
        """Reset visitation tracking."""
        self.grid_visits.clear()
        self.state_visits.clear()
        self.trajectory.clear()
        self.unique_states.clear()


class ExplorationTracker:
    """
    Comprehensive exploration tracking for curiosity-driven learning.

    Measures:
    - Coverage: fraction of environment explored
    - Novelty: how often agent visits new states
    - Efficiency: resources collected per step
    - Dispersion: how spread out the exploration is
    """

    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        self.visitation = StateVisitation()

        # Episode statistics
        self.episode_coverages: List[float] = []
        self.episode_unique_states: List[int] = []
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

        # Running statistics
        self.total_steps = 0
        self.total_resources = 0
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

    def record_step(
        self,
        position: Tuple[int, int],
        observation: torch.Tensor,
        reward: float,
    ) -> Dict[str, float]:
        """
        Record a step and compute exploration metrics.

        Returns novelty bonus and current metrics.
        """
        # Hash the observation for state tracking
        state_hash = hash(tuple(observation.flatten().tolist()[:20]))  # Use first 20 elements

        # Get novelty before recording
        novelty = self.visitation.get_novelty_bonus(position, state_hash)

        # Record visit
        self.visitation.record_visit(position, state_hash)

        # Update statistics
        self.total_steps += 1
        self.current_episode_reward += reward
        self.current_episode_length += 1

        if reward > 0:  # Resource collected
            self.total_resources += 1

        return {
            "novelty_bonus": novelty,
            "coverage": self.visitation.get_coverage(self.grid_size),
            "unique_states": len(self.visitation.unique_states),
        }

    def end_episode(self) -> Dict[str, float]:
        """End current episode and record statistics."""
        coverage = self.visitation.get_coverage(self.grid_size)
        unique = len(self.visitation.unique_states)

        self.episode_coverages.append(coverage)
        self.episode_unique_states.append(unique)
        self.episode_rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)

        stats = {
            "episode_coverage": coverage,
            "episode_unique_states": unique,
            "episode_reward": self.current_episode_reward,
            "episode_length": self.current_episode_length,
        }

        # Reset episode stats
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

        return stats

    def reset_episode(self) -> None:
        """Reset for new episode (keeps history)."""
        self.visitation.reset()
        self.current_episode_reward = 0.0
        self.current_episode_length = 0

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics."""
        return {
            "total_steps": self.total_steps,
            "total_episodes": len(self.episode_coverages),
            "avg_coverage": np.mean(self.episode_coverages) if self.episode_coverages else 0.0,
            "max_coverage": max(self.episode_coverages) if self.episode_coverages else 0.0,
            "avg_unique_states": np.mean(self.episode_unique_states) if self.episode_unique_states else 0.0,
            "avg_reward": np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            "avg_length": np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            "resources_per_step": self.total_resources / max(1, self.total_steps),
        }


class RNDNetwork(torch.nn.Module):
    """
    Random Network Distillation for novelty detection.

    Uses prediction error between a fixed random target network
    and a trained predictor network as novelty signal.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64):
        super().__init__()

        # Target network (fixed, random)
        self.target = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

        # Predictor network (trained)
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

        # Freeze target network
        for param in self.target.parameters():
            param.requires_grad = False

        # Initialize target with random weights
        self._init_target()

    def _init_target(self) -> None:
        """Initialize target network with random weights."""
        for layer in self.target:
            if hasattr(layer, 'weight'):
                torch.nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning target and predictor outputs."""
        with torch.no_grad():
            target_out = self.target(x)
        predictor_out = self.predictor(x)
        return target_out, predictor_out

    def compute_novelty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute novelty score as prediction error."""
        target_out, predictor_out = self.forward(x)
        error = (target_out - predictor_out).pow(2).mean(dim=-1)
        return error

    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute loss for training predictor."""
        target_out, predictor_out = self.forward(x)
        loss = (target_out - predictor_out).pow(2).mean()
        return loss


class ICMModule(torch.nn.Module):
    """
    Intrinsic Curiosity Module (ICM).

    Combines:
    - Forward model: predict next state from current state + action
    - Inverse model: predict action from current + next state

    Curiosity = forward model prediction error
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        feature_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Feature encoder (shared)
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, feature_dim),
        )

        # Forward model: predict next feature from current feature + action
        self.forward_model = torch.nn.Sequential(
            torch.nn.Linear(feature_dim + action_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, feature_dim),
        )

        # Inverse model: predict action from current + next feature
        self.inverse_model = torch.nn.Sequential(
            torch.nn.Linear(feature_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns: (predicted_next_features, predicted_action_logits, intrinsic_reward)
        """
        # Encode observations
        features = self.encoder(obs)
        next_features = self.encoder(next_obs)

        # Forward model
        forward_input = torch.cat([features, action], dim=-1)
        predicted_next = self.forward_model(forward_input)

        # Inverse model
        inverse_input = torch.cat([features, next_features], dim=-1)
        predicted_action = self.inverse_model(inverse_input)

        # Intrinsic reward = forward prediction error
        intrinsic_reward = (predicted_next - next_features.detach()).pow(2).mean(dim=-1)

        return predicted_next, predicted_action, intrinsic_reward

    def compute_loss(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        action_labels: torch.Tensor,
        forward_weight: float = 0.8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute ICM loss.

        Returns: (forward_loss, inverse_loss)
        """
        predicted_next, predicted_action, _ = self.forward(obs, next_obs, action)

        # Forward loss
        next_features = self.encoder(next_obs).detach()
        forward_loss = (predicted_next - next_features).pow(2).mean()

        # Inverse loss
        inverse_loss = torch.nn.functional.cross_entropy(predicted_action, action_labels)

        return forward_loss, inverse_loss

    def get_intrinsic_reward(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Get intrinsic curiosity reward."""
        _, _, intrinsic_reward = self.forward(obs, next_obs, action)
        return intrinsic_reward
