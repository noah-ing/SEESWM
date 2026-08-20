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
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as package_version
import platform
import random
import subprocess

from src.swarm.graph import (
    SWARM_STATE_SCHEMA_VERSION,
    SwarmGraph,
    SwarmConfig,
    TopologyType,
    swarm_config_to_dict,
)
from src.environment.cosmos import CosmosEnvironment


def _source_revision() -> str | None:
    """Read the local Git revision without making checkpointing depend on Git."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def _source_dirty() -> bool | None:
    """Report tracked or untracked worktree changes, or None without Git."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=Path(__file__).resolve().parent.parent,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for distribution in ("torch", "numpy", "networkx"):
        try:
            versions[distribution] = str(package_version(distribution))
        except PackageNotFoundError:
            versions[distribution] = None
    return versions


def _environment_config_to_dict(env: CosmosEnvironment) -> dict:
    config = env.config
    return {
        "grid_size": int(config.grid_size),
        "num_resources": int(config.num_resources),
        "num_hazards": int(config.num_hazards),
        "num_agents": int(config.num_agents),
        "vision_radius": int(config.vision_radius),
        "enable_respawn": bool(config.enable_respawn),
        "respawn_delay": int(config.respawn_delay),
        "num_food": int(config.num_food),
        "num_water": int(config.num_water),
        "num_material": int(config.num_material),
        "hunger_rate": float(config.hunger_rate),
        "thirst_rate": float(config.thirst_rate),
        "starvation_threshold": float(config.starvation_threshold),
        "movement_cost": float(config.movement_cost),
        "stay_cost": float(config.stay_cost),
        "max_steps": int(config.max_steps),
    }


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
    seed: int = 42,
):
    """Train swarm and return trained model."""

    if num_epochs < 1 or num_agents < 1 or hidden_dim < 1:
        raise ValueError("epochs, agents, and hidden dimension must be positive")
    if lr <= 0 or not np.isfinite(lr):
        raise ValueError("learning rate must be finite and positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create swarm
    swarm_config = SwarmConfig(
        num_agents=num_agents,
        input_dim=137,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,  # Will project to actions
        message_dim=hidden_dim,
        topology=TopologyType.SMALL_WORLD,
        small_world_k=min(4, num_agents),
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
    avg_reward = 0.0

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
        save_file = Path(save_path).expanduser()
        save_file.parent.mkdir(parents=True, exist_ok=True)
        save_data = {
            'schema_version': SWARM_STATE_SCHEMA_VERSION,
            'checkpoint_type': 'training',
            'swarm_config': swarm_config_to_dict(swarm_config),
            'swarm_state': swarm.state_dict(),
            'policy_head': {
                'type': 'mlp_relu',
                'input_dim': int(swarm_config.output_dim),
                'hidden_dim': 64,
                'num_actions': 5,
                'state_dict': dict(trainer.action_head.state_dict()),
            },
            'value_head': {
                'type': 'mlp_relu',
                'input_dim': int(swarm_config.output_dim),
                'hidden_dim': 64,
                'state_dict': dict(trainer.value_head.state_dict()),
            },
            'environment_config': _environment_config_to_dict(env),
            'training_config': {
                'algorithm': 'reinforce_with_baseline',
                'num_epochs': int(num_epochs),
                'learning_rate': float(lr),
                'gamma': float(trainer.gamma),
                'entropy_coef': float(trainer.entropy_coef),
                'max_episode_steps': 200,
                'device': str(device),
            },
            'training_metrics': {
                'final_avg_reward': float(avg_reward),
                'best_avg_reward': float(best_avg_reward),
                'num_epochs': int(num_epochs),
            },
            'seed': int(seed),
            'source_revision': _source_revision(),
            'source_dirty': _source_dirty(),
            'dependency_versions': _dependency_versions(),
        }
        torch.save(save_data, save_file)
        if verbose:
            print(f"Saved trained model to {save_file}")

    return swarm, trainer.action_head, avg_reward


def main():
    parser = argparse.ArgumentParser(description="Train SEESWM for validation")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument("--agents", type=int, default=20, help="Number of agents")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--save", type=str, default=None, help="Save path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
        seed=args.seed,
    )

    # Optionally run validation
    if args.validate:
        print("\n" + "=" * 60)
        print("RUNNING VALIDATION")
        print("=" * 60)

        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("validate_rigorously.py")),
                "--model",
                str(args.save),
                "--device",
                args.device,
                "--seeds",
                "3",
                "--episodes",
                "3",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
