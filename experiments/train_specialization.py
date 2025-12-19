"""
Specialization training for SEESWM.

Phase 3: Agent specialization with typed messaging and role emergence tracking.
Trains specialized agent architectures and measures emergent division of labor.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.swarm.graph import SwarmGraph, SwarmConfig, TopologyType
from src.swarm.specialized_graph import (
    SpecializedSwarmGraph,
    SpecializedSwarmConfig,
)
from src.swarm.typed_messaging import MessageType, AGENT_MESSAGE_TYPES
from src.world_model.jepa import WorldModel
from src.environment.cosmos import CosmosEnvironment, EnvironmentConfig
from src.training import PPOConfig, TrajectoryBuffer, PolicyHead, ValueHead, Transition
from src.agents.micro_agent import AgentType
from src.utils.config import Config
from src.utils.logging import setup_logger, MetricsLogger


class SpecializationPPOTrainer:
    """
    PPO trainer adapted for specialized swarms.

    Key differences from base PPOTrainer:
    - Uses SpecializedSwarmGraph with typed messaging
    - Tracks role emergence metrics
    - Reports specialization-specific statistics
    """

    def __init__(
        self,
        swarm: SpecializedSwarmGraph,
        world_model: Optional[WorldModel],
        config: Optional[PPOConfig] = None,
    ):
        self.swarm = swarm
        self.world_model = world_model
        self.config = config or PPOConfig()
        self.device = self.config.device

        # Dimensions
        self.obs_dim = swarm.config.input_dim
        self.output_dim = swarm.config.output_dim
        self.num_actions = 5

        # Policy and value heads
        self.policy_head = PolicyHead(self.output_dim, self.num_actions).to(self.device)
        self.value_head = ValueHead(self.output_dim).to(self.device)

        # Collect parameters
        self.all_params = list(self.policy_head.parameters())
        self.all_params += list(self.value_head.parameters())
        self.all_params += list(self.swarm.parameters())

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.all_params, lr=self.config.learning_rate
        )

        # World model optimizer
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
        self.specialization_stats: List[Dict] = []

    def get_action(self, observation: torch.Tensor) -> tuple:
        """Get action from specialized swarm."""
        swarm_output = self.swarm.step(observation)
        action, log_prob = self.policy_head.get_action(swarm_output)
        value = self.value_head(swarm_output).item()
        return action, log_prob, value

    def compute_curiosity_reward(
        self,
        obs: torch.Tensor,
        action: int,
        next_obs: torch.Tensor,
    ) -> float:
        """Compute curiosity from world model."""
        if self.world_model is None:
            return 0.0

        action_onehot = torch.zeros(1, self.num_actions, device=self.device)
        action_onehot[0, action] = 1.0

        curiosity = self.world_model.compute_curiosity(obs, action_onehot, next_obs)
        return curiosity * self.config.curiosity_coef

    def collect_rollout(self, env, num_steps: int) -> Dict:
        """Collect experience from specialized swarm."""
        observations = env.reset()
        obs_tensor = observations[0].to_tensor(self.device).unsqueeze(0)
        self.swarm.reset(batch_size=1)

        episode_reward = 0.0
        episode_curiosity = 0.0

        for step in range(num_steps):
            action, log_prob, value = self.get_action(obs_tensor)

            next_observations, rewards, dones = env.step([action])
            next_obs_tensor = next_observations[0].to_tensor(self.device).unsqueeze(0)

            curiosity_reward = self.compute_curiosity_reward(
                obs_tensor, action, next_obs_tensor
            )

            total_reward = rewards[0] + curiosity_reward

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

            episode_reward += rewards[0]
            episode_curiosity += curiosity_reward
            self.total_steps += 1

            if dones[0]:
                self.episode_rewards.append(episode_reward)
                self.curiosity_rewards.append(episode_curiosity)

                # Track role metrics
                if self.swarm.config.track_role_emergence:
                    self.specialization_stats.append(
                        self.swarm.get_role_emergence_metrics()
                    )

                observations = env.reset()
                obs_tensor = observations[0].to_tensor(self.device).unsqueeze(0)
                self.swarm.reset(batch_size=1)
                episode_reward = 0.0
                episode_curiosity = 0.0
            else:
                obs_tensor = next_obs_tensor

        # Compute advantages
        with torch.no_grad():
            final_output = self.swarm.step(obs_tensor)
            final_value = self.value_head(final_output).item()

        self.buffer.compute_advantages(final_value)

        return {
            "steps": num_steps,
            "avg_reward": np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0,
            "avg_curiosity": np.mean(self.curiosity_rewards[-10:]) if self.curiosity_rewards else 0.0,
        }

    def update(self) -> Dict:
        """PPO update with specialized swarm."""
        if len(self.buffer) == 0:
            return {}

        policy_losses = []
        value_losses = []
        entropy_losses = []
        world_model_losses = []

        for epoch in range(self.config.num_epochs):
            for batch in self.buffer.get_batches(self.config.batch_size):
                obs = torch.stack([t.observation for t in batch]).to(self.device)
                actions = torch.tensor([t.action for t in batch], device=self.device)
                old_log_probs = torch.tensor([t.log_prob for t in batch], device=self.device)
                advantages = torch.tensor(
                    [t.advantage for t in batch], dtype=torch.float32, device=self.device
                )
                returns = torch.tensor(
                    [t.returns for t in batch], dtype=torch.float32, device=self.device
                )

                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                self.swarm.reset(batch_size=len(batch))
                swarm_output = self.swarm.step(obs)

                log_probs, entropy = self.policy_head.evaluate(swarm_output, actions)
                ratio = torch.exp(log_probs - old_log_probs)

                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                values = self.value_head(swarm_output)
                value_loss = F.mse_loss(values, returns)

                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.all_params, self.config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(-entropy_loss.item())

                if self.world_model is not None:
                    next_obs = torch.stack([t.next_observation for t in batch]).to(self.device)
                    action_onehot = F.one_hot(actions, self.num_actions).float()

                    wm_loss = self.world_model.compute_loss(obs, action_onehot, next_obs)

                    self.world_model_optimizer.zero_grad()
                    wm_loss.backward()
                    self.world_model_optimizer.step()
                    self.world_model.update_target()

                    world_model_losses.append(wm_loss.item())

        self.buffer.clear()

        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropy_losses),
            "world_model_loss": np.mean(world_model_losses) if world_model_losses else 0.0,
        }

    def train_step(self, env, rollout_steps: int = 256) -> Dict:
        """One training step."""
        rollout_stats = self.collect_rollout(env, rollout_steps)
        update_stats = self.update()
        return {**rollout_stats, **update_stats}

    def get_stats(self) -> Dict:
        """Get training statistics including specialization metrics."""
        base_stats = {
            "total_steps": self.total_steps,
            "episodes": len(self.episode_rewards),
            "avg_reward_10": np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0,
            "avg_reward_100": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            "avg_curiosity_10": np.mean(self.curiosity_rewards[-10:]) if self.curiosity_rewards else 0.0,
        }

        # Add role emergence metrics
        if self.specialization_stats:
            latest = self.specialization_stats[-1]
            base_stats["specialization_entropy"] = latest.get("avg_specialization_entropy", 0.0)

        return base_stats


def train_specialized(
    num_iterations: int = 1000,
    rollout_steps: int = 256,
    topology: str = "hierarchical",
    device: str = "cpu",
    log_dir: str = "logs/specialization",
    seed: int = 42,
):
    """
    Train specialized swarm with role emergence tracking.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_dir) / f"specialized_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("seeswm_specialization", log_dir=str(run_dir))
    metrics = MetricsLogger(str(run_dir))

    logger.info("=" * 60)
    logger.info("SEESWM Phase 3: Specialized Agent Training")
    logger.info("=" * 60)

    # Topology mapping
    topology_map = {
        "hierarchical": TopologyType.HIERARCHICAL,
        "modular": TopologyType.MODULAR,
        "small_world": TopologyType.SMALL_WORLD,
        "scale_free": TopologyType.SCALE_FREE,
    }

    # Environment
    env_config = EnvironmentConfig(
        grid_size=32,
        num_resources=15,
        num_hazards=5,
        num_food=10,
        num_water=10,
        num_material=5,
        enable_respawn=True,
        respawn_delay=30,
        max_steps=300,
    )
    env = CosmosEnvironment(config=env_config)
    logger.info(f"Environment: {env_config.grid_size}x{env_config.grid_size}")
    logger.info(f"Observation dim: {env.observation_dim}")

    # Specialized swarm
    swarm_config = SpecializedSwarmConfig(
        num_agents=20,
        num_perception=5,
        num_reasoning=5,
        num_memory=5,
        num_planning=5,
        topology=topology_map.get(topology, TopologyType.HIERARCHICAL),
        hidden_dim=64,
        input_dim=env.observation_dim,
        output_dim=32,
        message_passing_rounds=3,
        use_typed_messaging=True,
        encode_message_types=True,
        track_role_emergence=True,
    )
    swarm = SpecializedSwarmGraph(config=swarm_config, device=device)

    logger.info(f"Specialized Swarm: {swarm_config.num_agents} agents")
    logger.info(f"  - Perception: {swarm_config.num_perception}")
    logger.info(f"  - Reasoning: {swarm_config.num_reasoning}")
    logger.info(f"  - Memory: {swarm_config.num_memory}")
    logger.info(f"  - Planning: {swarm_config.num_planning}")
    logger.info(f"Topology: {topology}")
    logger.info(f"Total parameters: {swarm.total_parameters:,}")
    logger.info(f"Typed messaging: {swarm_config.use_typed_messaging}")

    # World model
    world_model = WorldModel(
        obs_dim=env.observation_dim,
        action_dim=5,
        latent_dim=64,
        num_hierarchy_levels=2,
    ).to(device)
    logger.info(f"World model: {sum(p.numel() for p in world_model.parameters()):,} parameters")

    # Trainer
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        num_epochs=4,
        batch_size=64,
        curiosity_coef=0.5,
        device=device,
    )
    trainer = SpecializationPPOTrainer(swarm, world_model, ppo_config)

    # Training
    logger.info("\nStarting training...")
    best_avg_reward = float("-inf")

    for iteration in tqdm(range(num_iterations), desc="Training"):
        stats = trainer.train_step(env, rollout_steps)

        metrics.log(
            iteration,
            reward=stats.get("avg_reward", 0),
            curiosity=stats.get("avg_curiosity", 0),
            policy_loss=stats.get("policy_loss", 0),
            value_loss=stats.get("value_loss", 0),
            entropy=stats.get("entropy", 0),
            world_model_loss=stats.get("world_model_loss", 0),
        )

        if iteration % 10 == 0:
            trainer_stats = trainer.get_stats()
            env_stats = env.get_stats()
            msg_stats = swarm.get_message_stats()
            role_metrics = swarm.get_role_emergence_metrics()

            logger.info(
                f"Iter {iteration}: "
                f"reward={trainer_stats['avg_reward_10']:.3f}, "
                f"coverage={env_stats['coverage']:.1%}, "
                f"msgs={msg_stats['total_messages']}, "
                f"spec_entropy={role_metrics.get('avg_specialization_entropy', 0):.3f}"
            )

            if trainer_stats["avg_reward_10"] > best_avg_reward:
                best_avg_reward = trainer_stats["avg_reward_10"]
                torch.save({
                    "swarm": swarm.state_dict(),
                    "world_model": world_model.state_dict(),
                    "iteration": iteration,
                    "avg_reward": best_avg_reward,
                    "role_metrics": role_metrics,
                }, run_dir / "best_model.pt")

        if iteration % 100 == 0 and iteration > 0:
            torch.save({
                "swarm": swarm.state_dict(),
                "world_model": world_model.state_dict(),
                "iteration": iteration,
            }, run_dir / f"checkpoint_{iteration}.pt")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    final_stats = trainer.get_stats()
    final_role_metrics = swarm.get_role_emergence_metrics()

    logger.info(f"Total steps: {final_stats['total_steps']:,}")
    logger.info(f"Total episodes: {final_stats['episodes']}")
    logger.info(f"Final avg reward: {final_stats['avg_reward_100']:.3f}")
    logger.info(f"Best avg reward: {best_avg_reward:.3f}")

    logger.info("\nRole Emergence Metrics:")
    logger.info(f"  Specialization entropy: {final_role_metrics.get('avg_specialization_entropy', 0):.3f}")

    type_concentrations = final_role_metrics.get("type_concentrations", {})
    for agent_type, concentration in type_concentrations.items():
        logger.info(f"  {agent_type} concentration: {concentration:.3f}")

    torch.save({
        "swarm": swarm.state_dict(),
        "world_model": world_model.state_dict(),
        "final_stats": final_stats,
        "role_metrics": final_role_metrics,
    }, run_dir / "final_model.pt")

    metrics.save()
    logger.info(f"\nResults saved to: {run_dir}")

    return trainer, swarm, world_model


def compare_specialized_vs_generic(
    num_iterations: int = 500,
    device: str = "cpu",
):
    """
    Compare specialized swarm vs generic swarm performance.
    """
    print("=" * 60)
    print("Comparison: Specialized vs Generic Swarm")
    print("=" * 60)

    results = {}

    # Environment config
    env_config = EnvironmentConfig(
        grid_size=24,
        num_resources=10,
        num_hazards=3,
        max_steps=200,
    )

    for swarm_type in ["specialized", "generic"]:
        print(f"\nTraining {swarm_type} swarm...")

        env = CosmosEnvironment(config=env_config)

        if swarm_type == "specialized":
            swarm_config = SpecializedSwarmConfig(
                num_agents=16,
                num_perception=4,
                num_reasoning=4,
                num_memory=4,
                num_planning=4,
                topology=TopologyType.HIERARCHICAL,
                hidden_dim=32,
                input_dim=env.observation_dim,
                output_dim=16,
                use_typed_messaging=True,
                track_role_emergence=True,
            )
            swarm = SpecializedSwarmGraph(config=swarm_config, device=device)
        else:
            swarm_config = SwarmConfig(
                num_agents=16,
                num_perception=4,
                num_reasoning=4,
                num_memory=4,
                num_planning=4,
                topology=TopologyType.SMALL_WORLD,
                hidden_dim=32,
                input_dim=env.observation_dim,
                output_dim=16,
            )
            swarm = SwarmGraph(config=swarm_config, device=device)

        world_model = WorldModel(
            obs_dim=env.observation_dim,
            action_dim=5,
            latent_dim=32,
        ).to(device)

        ppo_config = PPOConfig(
            learning_rate=3e-4,
            curiosity_coef=0.5,
            device=device,
        )

        if swarm_type == "specialized":
            trainer = SpecializationPPOTrainer(swarm, world_model, ppo_config)
        else:
            from src.training import PPOTrainer
            trainer = PPOTrainer(swarm, world_model, ppo_config)

        rewards = []
        coverages = []

        for i in tqdm(range(num_iterations), desc=swarm_type):
            stats = trainer.train_step(env, rollout_steps=128)
            trainer_stats = trainer.get_stats()
            env_stats = env.get_stats()

            if i % 50 == 0:
                rewards.append(trainer_stats["avg_reward_10"])
                coverages.append(env_stats["coverage"])

        results[swarm_type] = {
            "final_reward": trainer.get_stats()["avg_reward_100"],
            "final_coverage": env.get_stats()["coverage"],
            "rewards": rewards,
            "coverages": coverages,
            "total_params": (
                swarm.total_parameters if hasattr(swarm, "total_parameters")
                else sum(
                    sum(p.numel() for p in a.network.parameters())
                    for a in swarm.agents.values()
                )
            ),
        }

        if swarm_type == "specialized":
            results[swarm_type]["role_metrics"] = swarm.get_role_emergence_metrics()

    # Print comparison
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    for label, data in results.items():
        print(f"\n{label}:")
        print(f"  Final avg reward: {data['final_reward']:.3f}")
        print(f"  Final coverage: {data['final_coverage']:.1%}")
        print(f"  Total parameters: {data['total_params']:,}")

        if "role_metrics" in data:
            rm = data["role_metrics"]
            print(f"  Specialization entropy: {rm.get('avg_specialization_entropy', 0):.3f}")

    # Compute improvement
    if "specialized" in results and "generic" in results:
        reward_improvement = (
            results["specialized"]["final_reward"] - results["generic"]["final_reward"]
        ) / max(abs(results["generic"]["final_reward"]), 0.01) * 100

        print(f"\nSpecialized vs Generic improvement: {reward_improvement:+.1f}%")

    return results


def analyze_role_emergence(
    checkpoint_path: str,
    device: str = "cpu",
):
    """
    Analyze role emergence from a trained checkpoint.
    """
    print("=" * 60)
    print("Role Emergence Analysis")
    print("=" * 60)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    role_metrics = checkpoint.get("role_metrics", {})

    print("\nRole Emergence Metrics:")
    print(f"  Avg specialization entropy: {role_metrics.get('avg_specialization_entropy', 0):.4f}")

    type_concentrations = role_metrics.get("type_concentrations", {})
    print("\nType Concentrations (higher = more specialized):")
    for agent_type, concentration in type_concentrations.items():
        bar = "=" * int(concentration * 50)
        print(f"  {agent_type:12s}: {concentration:.3f} |{bar}|")

    print("\nInterpretation:")
    print("  - Low entropy = agents are highly specialized")
    print("  - High concentration = agents send their expected message types")

    return role_metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Specialized Agent Training")
    parser.add_argument("--iterations", type=int, default=500, help="Training iterations")
    parser.add_argument("--rollout-steps", type=int, default=256, help="Steps per rollout")
    parser.add_argument("--topology", type=str, default="hierarchical",
                       choices=["hierarchical", "modular", "small_world", "scale_free"])
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--log-dir", type=str, default="logs/specialization", help="Log dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compare", action="store_true", help="Compare specialized vs generic")
    parser.add_argument("--analyze", type=str, default=None, help="Analyze checkpoint")
    args = parser.parse_args()

    if args.analyze:
        analyze_role_emergence(args.analyze, device=args.device)
    elif args.compare:
        compare_specialized_vs_generic(
            num_iterations=args.iterations,
            device=args.device,
        )
    else:
        train_specialized(
            num_iterations=args.iterations,
            rollout_steps=args.rollout_steps,
            topology=args.topology,
            device=args.device,
            log_dir=args.log_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
