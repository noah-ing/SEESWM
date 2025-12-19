"""
Reinforcement Learning infrastructure for SEESWM.

Implements PPO (Proximal Policy Optimization) adapted for swarm-based agents
with integrated curiosity rewards from the world model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class Transition:
    """A single environment transition."""
    observation: torch.Tensor
    action: int
    reward: float
    next_observation: torch.Tensor
    done: bool
    log_prob: float
    value: float

    # Computed during training
    advantage: float = 0.0
    returns: float = 0.0


@dataclass
class TrajectoryBuffer:
    """Buffer for storing and processing trajectories."""

    transitions: List[Transition] = field(default_factory=list)
    gamma: float = 0.99
    gae_lambda: float = 0.95

    def add(self, transition: Transition) -> None:
        """Add a transition to the buffer."""
        self.transitions.append(transition)

    def compute_advantages(self, final_value: float = 0.0) -> None:
        """
        Compute GAE (Generalized Advantage Estimation) for all transitions.

        GAE provides a good bias-variance tradeoff for advantage estimation.
        """
        last_gae = 0.0

        # Process in reverse order
        for i in reversed(range(len(self.transitions))):
            t = self.transitions[i]

            if t.done:
                next_value = 0.0
                last_gae = 0.0
            elif i == len(self.transitions) - 1:
                next_value = final_value
            else:
                next_value = self.transitions[i + 1].value

            # TD error
            delta = t.reward + self.gamma * next_value - t.value

            # GAE
            last_gae = delta + self.gamma * self.gae_lambda * last_gae
            t.advantage = last_gae
            t.returns = last_gae + t.value

    def get_batches(self, batch_size: int) -> List[List[Transition]]:
        """Split transitions into mini-batches for training."""
        indices = np.random.permutation(len(self.transitions))
        batches = []

        for start in range(0, len(self.transitions), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch = [self.transitions[i] for i in batch_indices]
            batches.append(batch)

        return batches

    def clear(self) -> None:
        """Clear the buffer."""
        self.transitions = []

    def __len__(self) -> int:
        return len(self.transitions)


class PolicyHead(nn.Module):
    """Policy head for action selection."""

    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return action logits."""
        return self.network(x)

    def get_action(self, x: torch.Tensor) -> Tuple[int, float]:
        """Sample action and return log probability."""
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item()

    def evaluate(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update."""
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy


class ValueHead(nn.Module):
    """Value head for state value estimation."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return state value."""
        return self.network(x).squeeze(-1)


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    # Core hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2

    # Training
    num_epochs: int = 4
    batch_size: int = 64
    max_grad_norm: float = 0.5

    # Loss coefficients
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # Curiosity
    curiosity_coef: float = 0.1  # Weight for intrinsic curiosity reward

    # Device
    device: str = "cpu"


class PPOTrainer:
    """
    PPO trainer for swarm-based RL.

    Combines extrinsic rewards from environment with intrinsic
    curiosity rewards from world model prediction error.
    """

    def __init__(
        self,
        swarm,  # SwarmGraph
        world_model,  # WorldModel
        config: Optional[PPOConfig] = None,
    ):
        self.swarm = swarm
        self.world_model = world_model
        self.config = config or PPOConfig()
        self.device = self.config.device

        # Get dimensions
        self.obs_dim = swarm.config.input_dim
        self.output_dim = swarm.config.output_dim
        self.num_actions = 5  # stay, up, down, left, right

        # Create policy and value heads
        self.policy_head = PolicyHead(
            self.output_dim, self.num_actions
        ).to(self.device)

        self.value_head = ValueHead(self.output_dim).to(self.device)

        # Collect all parameters
        self.all_params = list(self.policy_head.parameters())
        self.all_params += list(self.value_head.parameters())

        # Add swarm agent parameters
        for agent in self.swarm.agents.values():
            self.all_params += list(agent.network.parameters())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.all_params, lr=self.config.learning_rate
        )

        # World model optimizer (separate)
        if self.world_model is not None:
            self.world_model_optimizer = torch.optim.Adam(
                self.world_model.parameters(), lr=self.config.learning_rate
            )

        # Buffer
        self.buffer = TrajectoryBuffer(
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        # Statistics
        self.total_steps = 0
        self.episode_rewards: List[float] = []
        self.curiosity_rewards: List[float] = []

    def get_action(self, observation: torch.Tensor) -> Tuple[int, float, float]:
        """
        Get action from swarm output.

        Returns: (action, log_prob, value)
        """
        # Run through swarm
        swarm_output = self.swarm.step(observation)

        # Get action from policy head
        action, log_prob = self.policy_head.get_action(swarm_output)

        # Get value
        value = self.value_head(swarm_output).item()

        return action, log_prob, value

    def compute_curiosity_reward(
        self,
        obs: torch.Tensor,
        action: int,
        next_obs: torch.Tensor,
    ) -> float:
        """
        Compute intrinsic curiosity reward from world model.

        High prediction error = novel state = high curiosity reward.
        """
        if self.world_model is None:
            return 0.0

        # Convert action to tensor
        action_onehot = torch.zeros(1, self.num_actions, device=self.device)
        action_onehot[0, action] = 1.0

        # Compute prediction error
        curiosity = self.world_model.compute_curiosity(obs, action_onehot, next_obs)

        return curiosity * self.config.curiosity_coef

    def collect_rollout(
        self,
        env,  # CosmosEnvironment
        num_steps: int,
    ) -> dict:
        """
        Collect experience by running the agent in the environment.

        Returns rollout statistics.
        """
        observations = env.reset()
        obs_tensor = observations[0].to_tensor(self.device).unsqueeze(0)
        self.swarm.reset(batch_size=1)

        episode_reward = 0.0
        episode_curiosity = 0.0
        episode_length = 0

        for step in range(num_steps):
            # Get action
            action, log_prob, value = self.get_action(obs_tensor)

            # Environment step
            next_observations, rewards, dones = env.step([action])
            next_obs_tensor = next_observations[0].to_tensor(self.device).unsqueeze(0)

            # Compute curiosity reward
            curiosity_reward = self.compute_curiosity_reward(
                obs_tensor, action, next_obs_tensor
            )

            # Combined reward
            total_reward = rewards[0] + curiosity_reward

            # Store transition
            transition = Transition(
                observation=obs_tensor.squeeze(0).clone(),
                action=action,
                reward=total_reward,
                next_observation=next_obs_tensor.squeeze(0).clone(),
                done=dones[0],
                log_prob=log_prob,
                value=value,
            )
            self.buffer.add(transition)

            # Track statistics
            episode_reward += rewards[0]
            episode_curiosity += curiosity_reward
            episode_length += 1
            self.total_steps += 1

            if dones[0]:
                # Episode ended
                self.episode_rewards.append(episode_reward)
                self.curiosity_rewards.append(episode_curiosity)

                # Reset
                observations = env.reset()
                obs_tensor = observations[0].to_tensor(self.device).unsqueeze(0)
                self.swarm.reset(batch_size=1)
                episode_reward = 0.0
                episode_curiosity = 0.0
                episode_length = 0
            else:
                obs_tensor = next_obs_tensor

        # Compute advantages
        with torch.no_grad():
            final_output = self.swarm.step(obs_tensor)
            final_value = self.value_head(final_output).item()

        self.buffer.compute_advantages(final_value)

        return {
            "steps": num_steps,
            "episodes_completed": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0,
            "avg_curiosity": np.mean(self.curiosity_rewards[-10:]) if self.curiosity_rewards else 0.0,
        }

    def update(self) -> dict:
        """
        Perform PPO update on collected experience.

        Returns training statistics.
        """
        if len(self.buffer) == 0:
            return {}

        # Training statistics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        world_model_losses = []

        for epoch in range(self.config.num_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                # Prepare batch tensors
                obs = torch.stack([t.observation for t in batch]).to(self.device)
                actions = torch.tensor([t.action for t in batch], device=self.device)
                old_log_probs = torch.tensor([t.log_prob for t in batch], device=self.device)
                advantages = torch.tensor([t.advantage for t in batch], dtype=torch.float32, device=self.device)
                returns = torch.tensor([t.returns for t in batch], dtype=torch.float32, device=self.device)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Forward pass through swarm (batch)
                self.swarm.reset(batch_size=len(batch))
                swarm_output = self.swarm.step(obs)

                # Policy loss (PPO clipped objective)
                log_probs, entropy = self.policy_head.evaluate(swarm_output, actions)
                ratio = torch.exp(log_probs - old_log_probs)

                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values = self.value_head(swarm_output)
                value_loss = F.mse_loss(values, returns)

                # Entropy bonus (for exploration)
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.all_params, self.config.max_grad_norm)
                self.optimizer.step()

                # Track losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())

                # Update world model
                if self.world_model is not None:
                    next_obs = torch.stack([t.next_observation for t in batch]).to(self.device)
                    action_onehot = F.one_hot(actions, self.num_actions).float()

                    wm_loss = self.world_model.compute_loss(obs, action_onehot, next_obs)

                    self.world_model_optimizer.zero_grad()
                    wm_loss.backward()
                    self.world_model_optimizer.step()
                    self.world_model.update_target()

                    world_model_losses.append(wm_loss.item())

        # Clear buffer
        self.buffer.clear()

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
            "world_model_loss": np.mean(world_model_losses) if world_model_losses else 0.0,
        }

    def train_step(self, env, rollout_steps: int = 256) -> dict:
        """
        Perform one training step: collect rollout + update.

        Returns combined statistics.
        """
        rollout_stats = self.collect_rollout(env, rollout_steps)
        update_stats = self.update()

        return {**rollout_stats, **update_stats}

    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            "total_steps": self.total_steps,
            "episodes": len(self.episode_rewards),
            "avg_reward_10": np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0,
            "avg_reward_100": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            "avg_curiosity_10": np.mean(self.curiosity_rewards[-10:]) if self.curiosity_rewards else 0.0,
        }
