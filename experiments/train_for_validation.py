"""
Training script for SEESWM validation.

Trains the swarm using REINFORCE policy gradient and saves for validation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import argparse
import json
from datetime import datetime

from src.swarm.graph import SwarmGraph, SwarmConfig, TopologyType
from src.environment.cosmos import CosmosEnvironment


class PolicyGradientTrainer:
    """REINFORCE trainer with baseline for SwarmGraph."""

    def __init__(
        self,
        swarm: SwarmGraph,
        env: CosmosEnvironment,
        lr: float = 1e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        device: str = "cpu",
    ):
        self.swarm = swarm
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = device

        # Collect all trainable parameters
        all_params = []
        for agent in swarm.agents.values():
            all_params.extend(agent.network.parameters())

        # Action head to convert swarm output to action logits
        output_dim = swarm.config.output_dim
        self.action_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        ).to(device)

        # Value head for baseline (reduces variance)
        self.value_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)

        all_params.extend(self.action_head.parameters())
        all_params.extend(self.value_head.parameters())

        self.optimizer = optim.Adam(all_params, lr=lr)

    def get_action(self, obs_tensor: torch.Tensor) -> tuple:
        """Get action from swarm output."""
        output = self.swarm.step(obs_tensor)
        action_logits = self.action_head(output)
        value = self.value_head(output)

        # Sample action from policy
        probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action.item(), log_prob, entropy, value.squeeze()

    def compute_returns(self, rewards: list) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def train_episode(self) -> dict:
        """Train on one episode using actor-critic."""
        observations = self.env.reset()
        obs_tensor = observations[0].to_tensor(self.device).unsqueeze(0)

        self.swarm.reset(batch_size=1)

        log_probs = []
        entropies = []
        values = []
        rewards = []

        for step in range(200):  # Max steps per episode
            action, log_prob, entropy, value = self.get_action(obs_tensor)

            observations, reward_list, dones = self.env.step([action])
            reward = reward_list[0]
            done = dones[0]

            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            rewards.append(reward)

            if done:
                break

            obs_tensor = observations[0].to_tensor(self.device).unsqueeze(0)

        # Compute returns and advantages
        returns = self.compute_returns(rewards)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        values = torch.stack(values)

        # Advantage = return - baseline value
        advantages = returns - values.detach()

        # Policy gradient loss with baseline
        policy_loss = -(log_probs * advantages).mean()

        # Value loss
        value_loss = 0.5 * ((values - returns) ** 2).mean()

        # Entropy bonus (encourages exploration)
        entropy_loss = -self.entropy_coef * entropies.mean()

        total_loss = policy_loss + value_loss + entropy_loss

        # Update
        self.optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients for all parameters
        all_params = []
        for agent in self.swarm.agents.values():
            all_params.extend(agent.network.parameters())
        all_params.extend(self.action_head.parameters())
        all_params.extend(self.value_head.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.5)

        self.optimizer.step()

        return {
            'reward': sum(rewards),
            'length': len(rewards),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropies.mean().item(),
        }


def train_swarm(
    num_epochs: int = 500,
    num_agents: int = 20,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    device: str = "cpu",
    save_path: str = None,
    verbose: bool = True,
):
    """Train swarm and return trained model."""

    # Create swarm
    swarm_config = SwarmConfig(
        num_agents=num_agents,
        input_dim=137,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,  # Will project to actions
        message_dim=hidden_dim,
        topology=TopologyType.SMALL_WORLD,
        num_perception=num_agents // 4,
        num_reasoning=num_agents // 4,
        num_memory=num_agents // 4,
        num_planning=num_agents - 3 * (num_agents // 4),
    )
    swarm = SwarmGraph(swarm_config, device=device)

    if verbose:
        print(f"Created swarm with {swarm.total_parameters:,} parameters")

    # Create environment
    env = CosmosEnvironment(
        grid_size=32,
        num_resources=20,
        num_hazards=10,
    )

    # Create trainer
    trainer = PolicyGradientTrainer(
        swarm=swarm,
        env=env,
        lr=lr,
        gamma=0.99,
        entropy_coef=0.01,
        device=device,
    )

    # Training loop
    reward_history = deque(maxlen=100)
    best_avg_reward = float('-inf')

    if verbose:
        print(f"\nTraining for {num_epochs} epochs...")
        print("-" * 60)

    for epoch in range(num_epochs):
        metrics = trainer.train_episode()
        reward_history.append(metrics['reward'])
        avg_reward = np.mean(reward_history)

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward

        if verbose and (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1:4d} | "
                  f"Reward: {metrics['reward']:7.2f} | "
                  f"Avg(100): {avg_reward:7.2f} | "
                  f"Best: {best_avg_reward:7.2f} | "
                  f"Entropy: {metrics['entropy']:.3f}")

    if verbose:
        print("-" * 60)
        print(f"Training complete! Final avg reward: {avg_reward:.2f}")

    # Save model
    if save_path:
        save_data = {
            'swarm_config': swarm_config.__dict__,
            'swarm_state': swarm.state_dict(),
            'action_head_state': trainer.action_head.state_dict(),
            'training_metrics': {
                'final_avg_reward': avg_reward,
                'best_avg_reward': best_avg_reward,
                'num_epochs': num_epochs,
            }
        }
        torch.save(save_data, save_path)
        if verbose:
            print(f"Saved trained model to {save_path}")

    return swarm, trainer.action_head, avg_reward


def main():
    parser = argparse.ArgumentParser(description="Train SEESWM for validation")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--agents", type=int, default=20, help="Number of agents")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--save", type=str, default=None, help="Save path")
    parser.add_argument("--validate", action="store_true", help="Run validation after training")
    args = parser.parse_args()

    # Default save path
    if args.save is None:
        Path("results/models").mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save = f"results/models/swarm_trained_{timestamp}.pt"

    # Train
    swarm, action_head, final_reward = train_swarm(
        num_epochs=args.epochs,
        num_agents=args.agents,
        hidden_dim=args.hidden,
        lr=args.lr,
        device=args.device,
        save_path=args.save,
        verbose=True,
    )

    # Optionally run validation
    if args.validate:
        print("\n" + "=" * 60)
        print("RUNNING VALIDATION")
        print("=" * 60)

        import subprocess
        result = subprocess.run(
            ["python", "experiments/validate_rigorously.py", "--mode", "quick"],
            capture_output=False,
        )


if __name__ == "__main__":
    main()
