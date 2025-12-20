"""
Advanced training with specialization pressure and diversity rewards.

This script aims to produce swarms that:
1. Show measurable synergy (agents contribute unique information)
2. Develop emergent specialization (different agents do different things)
3. Are sensitive to ablation (removing components hurts performance)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional
import argparse
from datetime import datetime

from src.swarm.graph import SwarmGraph, SwarmConfig, TopologyType
from src.environment.cosmos import CosmosEnvironment


@dataclass
class SpecializationMetrics:
    """Track how specialized agents become."""
    action_entropy_per_agent: List[float]  # Low = specialized
    message_variance_per_agent: List[float]  # High = unique outputs
    pairwise_correlation: float  # Low = agents are different
    specialization_score: float  # Combined metric


class DiversityReward:
    """Compute intrinsic rewards for agent diversity."""

    def __init__(self, num_agents: int, hidden_dim: int, device: str = "cpu"):
        self.num_agents = num_agents
        self.device = device

        # Track running statistics of agent outputs
        self.output_mean = torch.zeros(num_agents, hidden_dim, device=device)
        self.output_var = torch.ones(num_agents, hidden_dim, device=device)
        self.update_count = 0

    def update_and_reward(self, agent_outputs: Dict[int, torch.Tensor]) -> float:
        """
        Update statistics and return diversity reward.

        Diversity = how different are agents from each other?
        """
        if len(agent_outputs) < 2:
            return 0.0

        # Stack outputs: (num_agents, hidden_dim)
        outputs = torch.stack([agent_outputs[i].mean(dim=0) for i in sorted(agent_outputs.keys())])

        # Update running mean/var
        self.update_count += 1
        alpha = 1.0 / self.update_count
        self.output_mean = (1 - alpha) * self.output_mean + alpha * outputs.detach()
        self.output_var = (1 - alpha) * self.output_var + alpha * (outputs.detach() - self.output_mean) ** 2

        # Diversity reward: high variance across agents = good
        # Measure pairwise distances between agent outputs
        pairwise_dist = torch.cdist(outputs.unsqueeze(0), outputs.unsqueeze(0)).squeeze()
        mean_dist = pairwise_dist.sum() / (self.num_agents * (self.num_agents - 1) + 1e-8)

        # Also reward individual agent consistency (low variance over time = specialized)
        consistency_bonus = -self.output_var.mean().item() * 0.1

        return mean_dist.item() * 0.1 + consistency_bonus


class SpecializationTrainer:
    """Actor-critic trainer with specialization pressure."""

    def __init__(
        self,
        swarm: SwarmGraph,
        env: CosmosEnvironment,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        diversity_coef: float = 0.1,
        device: str = "cpu",
    ):
        self.swarm = swarm
        self.env = env
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.diversity_coef = diversity_coef
        self.device = device

        # Collect parameters
        all_params = []
        for agent in swarm.agents.values():
            all_params.extend(agent.network.parameters())

        # Policy and value heads
        output_dim = swarm.config.output_dim
        self.action_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5),
        ).to(device)

        self.value_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        ).to(device)

        all_params.extend(self.action_head.parameters())
        all_params.extend(self.value_head.parameters())

        self.optimizer = optim.Adam(all_params, lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=5000, eta_min=1e-5
        )

        # Diversity reward tracker
        self.diversity_reward = DiversityReward(
            len(swarm.agents), output_dim, device
        )

        # Track per-agent action distributions for specialization
        self.agent_action_counts = {i: torch.zeros(5) for i in swarm.agents.keys()}

    def get_action(self, obs_tensor: torch.Tensor) -> tuple:
        """Get action and track per-agent contributions."""
        output = self.swarm.step(obs_tensor)

        # Get individual agent outputs for diversity reward
        agent_outputs = {}
        for i, agent in self.swarm.agents.items():
            agent_outputs[i] = agent.last_output if hasattr(agent, 'last_output') else output

        action_logits = self.action_head(output)
        value = self.value_head(output)

        probs = torch.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        # Compute diversity reward
        div_reward = self.diversity_reward.update_and_reward(agent_outputs)

        return action.item(), log_prob, entropy, value.squeeze(), div_reward

    def compute_returns(self, rewards: list) -> torch.Tensor:
        """Compute discounted returns."""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def train_episode(self) -> Dict:
        """Train one episode with specialization rewards."""
        observations = self.env.reset()
        obs_tensor = observations[0].to_tensor(self.device).unsqueeze(0)

        self.swarm.reset(batch_size=1)

        log_probs = []
        entropies = []
        values = []
        rewards = []
        diversity_rewards = []

        for step in range(200):
            action, log_prob, entropy, value, div_reward = self.get_action(obs_tensor)

            observations, reward_list, dones = self.env.step([action])
            reward = reward_list[0]
            done = dones[0]

            # Combine extrinsic and diversity rewards
            total_reward = reward + self.diversity_coef * div_reward

            log_probs.append(log_prob)
            entropies.append(entropy)
            values.append(value)
            rewards.append(total_reward)
            diversity_rewards.append(div_reward)

            if done:
                break

            obs_tensor = observations[0].to_tensor(self.device).unsqueeze(0)

        # Compute loss
        returns = self.compute_returns(rewards)
        log_probs = torch.stack(log_probs)
        entropies = torch.stack(entropies)
        values = torch.stack(values)

        advantages = returns - values.detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        entropy_loss = -self.entropy_coef * entropies.mean()

        total_loss = policy_loss + value_loss + entropy_loss

        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        all_params = []
        for agent in self.swarm.agents.values():
            all_params.extend(agent.network.parameters())
        all_params.extend(self.action_head.parameters())
        all_params.extend(self.value_head.parameters())
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=0.5)

        self.optimizer.step()
        self.scheduler.step()

        return {
            'reward': sum(rewards),
            'extrinsic_reward': sum(rewards) - self.diversity_coef * sum(diversity_rewards),
            'diversity_reward': sum(diversity_rewards),
            'length': len(rewards),
            'entropy': entropies.mean().item(),
            'lr': self.scheduler.get_last_lr()[0],
        }

    def compute_specialization_metrics(self) -> SpecializationMetrics:
        """Compute how specialized the agents have become."""
        # Run several episodes and track agent behaviors
        agent_outputs_all = {i: [] for i in self.swarm.agents.keys()}

        for _ in range(10):
            obs = self.env.reset()
            obs_tensor = obs[0].to_tensor(self.device).unsqueeze(0)
            self.swarm.reset(batch_size=1)

            for step in range(50):
                with torch.no_grad():
                    output = self.swarm.step(obs_tensor)

                    # Collect per-agent outputs
                    for i, agent in self.swarm.agents.items():
                        if hasattr(agent, 'last_output'):
                            agent_outputs_all[i].append(agent.last_output.cpu())
                        else:
                            agent_outputs_all[i].append(output.cpu())

                action = self.action_head(output).argmax(dim=-1).item()
                obs, _, dones = self.env.step([action])
                if dones[0]:
                    break
                obs_tensor = obs[0].to_tensor(self.device).unsqueeze(0)

        # Compute metrics
        action_entropies = []
        message_variances = []

        for i, outputs in agent_outputs_all.items():
            if len(outputs) > 1:
                stacked = torch.stack(outputs)
                # Variance of this agent's outputs over time
                var = stacked.var(dim=0).mean().item()
                message_variances.append(var)
                action_entropies.append(var)  # Simplified

        # Pairwise correlation between agents
        if len(agent_outputs_all) >= 2:
            agent_means = []
            for outputs in agent_outputs_all.values():
                if outputs:
                    agent_means.append(torch.stack(outputs).mean(dim=0))

            if len(agent_means) >= 2:
                stacked_means = torch.stack(agent_means)
                # Flatten to (num_agents, features) if needed
                if stacked_means.dim() > 2:
                    stacked_means = stacked_means.view(stacked_means.size(0), -1)
                # Cosine similarity
                norms = stacked_means.norm(dim=-1, keepdim=True)
                normalized = stacked_means / (norms + 1e-8)
                # Use transpose instead of .T for clarity
                similarity = (normalized @ normalized.transpose(-2, -1)).mean().item()
                pairwise_corr = similarity
            else:
                pairwise_corr = 1.0
        else:
            pairwise_corr = 1.0

        # Specialization score: low correlation + high variance = good
        spec_score = (1 - pairwise_corr) * np.mean(message_variances) if message_variances else 0.0

        return SpecializationMetrics(
            action_entropy_per_agent=action_entropies,
            message_variance_per_agent=message_variances,
            pairwise_correlation=pairwise_corr,
            specialization_score=spec_score,
        )


def train_with_specialization(
    num_epochs: int = 5000,
    num_agents: int = 20,
    hidden_dim: int = 128,
    lr: float = 3e-4,
    diversity_coef: float = 0.1,
    device: str = "cpu",
    save_path: str = None,
    verbose: bool = True,
):
    """Train swarm with specialization pressure."""

    # Create swarm
    swarm_config = SwarmConfig(
        num_agents=num_agents,
        input_dim=137,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        message_dim=hidden_dim,
        topology=TopologyType.MODULAR,  # Modular encourages specialization
        num_perception=num_agents // 4,
        num_reasoning=num_agents // 4,
        num_memory=num_agents // 4,
        num_planning=num_agents - 3 * (num_agents // 4),
    )
    swarm = SwarmGraph(swarm_config, device=device)

    if verbose:
        print(f"Created swarm: {swarm.total_parameters:,} parameters")
        print(f"Topology: MODULAR (encourages specialization)")
        print(f"Diversity coefficient: {diversity_coef}")

    # Create environment
    env = CosmosEnvironment(grid_size=32, num_resources=20, num_hazards=10)

    # Create trainer
    trainer = SpecializationTrainer(
        swarm=swarm,
        env=env,
        lr=lr,
        diversity_coef=diversity_coef,
        device=device,
    )

    # Training loop
    reward_history = deque(maxlen=100)
    best_avg_reward = float('-inf')

    if verbose:
        print(f"\nTraining for {num_epochs} epochs...")
        print("-" * 70)

    for epoch in range(num_epochs):
        metrics = trainer.train_episode()
        reward_history.append(metrics['extrinsic_reward'])
        avg_reward = np.mean(reward_history)

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward

        if verbose and (epoch + 1) % 100 == 0:
            spec_metrics = trainer.compute_specialization_metrics()
            print(f"Epoch {epoch+1:5d} | "
                  f"Reward: {metrics['extrinsic_reward']:7.2f} | "
                  f"Avg: {avg_reward:7.2f} | "
                  f"Div: {metrics['diversity_reward']:6.3f} | "
                  f"Spec: {spec_metrics.specialization_score:.4f} | "
                  f"Corr: {spec_metrics.pairwise_correlation:.3f}")

    if verbose:
        print("-" * 70)
        print(f"Training complete! Final avg reward: {avg_reward:.2f}")

        # Final specialization analysis
        final_metrics = trainer.compute_specialization_metrics()
        print(f"\nSpecialization Analysis:")
        print(f"  Pairwise correlation: {final_metrics.pairwise_correlation:.4f} (lower = more diverse)")
        print(f"  Specialization score: {final_metrics.specialization_score:.4f} (higher = more specialized)")

    # Save
    if save_path:
        save_data = {
            'swarm_config': swarm_config.__dict__,
            'swarm_state': swarm.state_dict(),
            'action_head_state': trainer.action_head.state_dict(),
            'value_head_state': trainer.value_head.state_dict(),
            'training_metrics': {
                'final_avg_reward': avg_reward,
                'best_avg_reward': best_avg_reward,
                'num_epochs': num_epochs,
                'specialization_score': final_metrics.specialization_score,
                'pairwise_correlation': final_metrics.pairwise_correlation,
            }
        }
        torch.save(save_data, save_path)
        if verbose:
            print(f"\nSaved to {save_path}")

    return swarm, trainer, avg_reward


def main():
    parser = argparse.ArgumentParser(description="Train SEESWM with specialization")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs")
    parser.add_argument("--agents", type=int, default=20, help="Number of agents")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--diversity", type=float, default=0.1, help="Diversity reward coefficient")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--save", type=str, default=None, help="Save path")
    args = parser.parse_args()

    if args.save is None:
        Path("results/models").mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save = f"results/models/swarm_specialized_{timestamp}.pt"

    train_with_specialization(
        num_epochs=args.epochs,
        num_agents=args.agents,
        hidden_dim=args.hidden,
        lr=args.lr,
        diversity_coef=args.diversity,
        device=args.device,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
