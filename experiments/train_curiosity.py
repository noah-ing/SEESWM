"""
Curiosity-driven training for SEESWM.

Phase 2: World model integrated with intrinsic curiosity motivation.
Agents learn to explore efficiently using prediction error as reward.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm

from src.swarm.graph import SwarmGraph, SwarmConfig, TopologyType
from src.world_model.jepa import WorldModel
from src.environment.cosmos import CosmosEnvironment, EnvironmentConfig
from src.training import PPOTrainer, PPOConfig
from src.training.exploration import ExplorationTracker, RNDNetwork, ICMModule
from src.utils.config import Config
from src.utils.logging import setup_logger, MetricsLogger


def train_curiosity(
    num_iterations: int = 1000,
    rollout_steps: int = 256,
    curiosity_type: str = "world_model",  # "world_model", "rnd", "icm"
    device: str = "cpu",
    log_dir: str = "logs/curiosity",
    seed: int = 42,
):
    """
    Run curiosity-driven training.

    Args:
        num_iterations: Number of training iterations
        rollout_steps: Steps per rollout
        curiosity_type: Type of curiosity module ("world_model", "rnd", "icm")
        device: Device to run on
        log_dir: Directory for logs
        seed: Random seed
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("seeswm_curiosity", log_dir=str(run_dir))
    metrics = MetricsLogger(str(run_dir))

    logger.info("=" * 60)
    logger.info("SEESWM Curiosity-Driven Training")
    logger.info("=" * 60)
    logger.info(f"Curiosity type: {curiosity_type}")
    logger.info(f"Device: {device}")
    logger.info(f"Rollout steps: {rollout_steps}")
    logger.info(f"Iterations: {num_iterations}")

    # Create environment
    env_config = EnvironmentConfig(
        grid_size=32,  # Smaller grid for faster iteration
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
    logger.info(f"Environment: {env_config.grid_size}x{env_config.grid_size} grid")
    logger.info(f"Observation dim: {env.observation_dim}")

    # Create swarm
    swarm_config = SwarmConfig(
        num_agents=20,
        num_perception=5,
        num_reasoning=5,
        num_memory=5,
        num_planning=5,
        topology=TopologyType.SMALL_WORLD,
        hidden_dim=64,
        input_dim=env.observation_dim,
        output_dim=32,
        message_passing_rounds=2,
    )
    swarm = SwarmGraph(config=swarm_config, device=device)
    logger.info(f"Swarm: {swarm_config.num_agents} agents, {swarm.total_parameters:,} parameters")
    logger.info(f"Topology: {swarm.get_topology_stats()}")

    # Create world model
    world_model = WorldModel(
        obs_dim=env.observation_dim,
        action_dim=5,  # 5 actions
        latent_dim=64,
        num_hierarchy_levels=2,
    ).to(device)
    logger.info(f"World model: {sum(p.numel() for p in world_model.parameters()):,} parameters")

    # Create PPO trainer
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        num_epochs=4,
        batch_size=64,
        curiosity_coef=0.5,  # Weight for curiosity reward
        device=device,
    )
    trainer = PPOTrainer(swarm, world_model, ppo_config)
    logger.info(f"PPO config: lr={ppo_config.learning_rate}, gamma={ppo_config.gamma}")

    # Create exploration tracker
    exploration_tracker = ExplorationTracker(grid_size=env_config.grid_size)

    # Optional: Additional curiosity modules
    rnd_network = None
    icm_module = None

    if curiosity_type == "rnd":
        rnd_network = RNDNetwork(env.observation_dim, hidden_dim=64).to(device)
        rnd_optimizer = torch.optim.Adam(rnd_network.predictor.parameters(), lr=1e-4)
        logger.info("Using RND for additional novelty detection")

    elif curiosity_type == "icm":
        icm_module = ICMModule(env.observation_dim, action_dim=5).to(device)
        icm_optimizer = torch.optim.Adam(icm_module.parameters(), lr=1e-4)
        logger.info("Using ICM for curiosity")

    # Training loop
    logger.info("\nStarting training...")
    best_avg_reward = float("-inf")

    for iteration in tqdm(range(num_iterations), desc="Training"):
        # Collect rollout and update
        stats = trainer.train_step(env, rollout_steps)

        # Log metrics
        metrics.log(
            iteration,
            reward=stats.get("avg_reward", 0),
            curiosity=stats.get("avg_curiosity", 0),
            policy_loss=stats.get("policy_loss", 0),
            value_loss=stats.get("value_loss", 0),
            entropy=stats.get("entropy", 0),
            world_model_loss=stats.get("world_model_loss", 0),
        )

        # Periodic logging
        if iteration % 10 == 0:
            trainer_stats = trainer.get_stats()
            env_stats = env.get_stats()

            logger.info(
                f"Iter {iteration}: "
                f"reward={trainer_stats['avg_reward_10']:.3f}, "
                f"curiosity={trainer_stats['avg_curiosity_10']:.3f}, "
                f"coverage={env_stats['coverage']:.1%}, "
                f"wm_loss={stats.get('world_model_loss', 0):.4f}"
            )

            # Track best model
            if trainer_stats["avg_reward_10"] > best_avg_reward:
                best_avg_reward = trainer_stats["avg_reward_10"]
                # Save best model
                torch.save({
                    "swarm": swarm.state_dict(),
                    "world_model": world_model.state_dict(),
                    "iteration": iteration,
                    "avg_reward": best_avg_reward,
                }, run_dir / "best_model.pt")

        # Save checkpoint periodically
        if iteration % 100 == 0 and iteration > 0:
            torch.save({
                "swarm": swarm.state_dict(),
                "world_model": world_model.state_dict(),
                "iteration": iteration,
                "trainer_stats": trainer.get_stats(),
            }, run_dir / f"checkpoint_{iteration}.pt")

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    final_stats = trainer.get_stats()
    logger.info(f"Total steps: {final_stats['total_steps']:,}")
    logger.info(f"Total episodes: {final_stats['episodes']}")
    logger.info(f"Final avg reward (100 ep): {final_stats['avg_reward_100']:.3f}")
    logger.info(f"Best avg reward: {best_avg_reward:.3f}")

    # Save final model
    torch.save({
        "swarm": swarm.state_dict(),
        "world_model": world_model.state_dict(),
        "final_stats": final_stats,
    }, run_dir / "final_model.pt")

    metrics.save()
    logger.info(f"\nResults saved to: {run_dir}")

    return trainer, swarm, world_model


def compare_curiosity_vs_no_curiosity(
    num_iterations: int = 500,
    device: str = "cpu",
):
    """
    Run an exploratory curiosity-versus-standard-RL comparison.

    A single run does not validate a general improvement claim.
    """
    print("=" * 60)
    print("Comparison: Curiosity vs No Curiosity")
    print("=" * 60)

    results = {}

    for curiosity_coef in [0.0, 0.5]:
        label = "with_curiosity" if curiosity_coef > 0 else "no_curiosity"
        print(f"\nTraining {label}...")

        # Create environment
        env_config = EnvironmentConfig(
            grid_size=24,
            num_resources=10,
            num_hazards=3,
            max_steps=200,
        )
        env = CosmosEnvironment(config=env_config)

        # Create swarm
        swarm_config = SwarmConfig(
            num_agents=16,
            topology=TopologyType.SMALL_WORLD,
            hidden_dim=32,
            input_dim=env.observation_dim,
            output_dim=16,
        )
        swarm = SwarmGraph(config=swarm_config, device=device)

        # Create world model (only used if curiosity > 0)
        world_model = WorldModel(
            obs_dim=env.observation_dim,
            action_dim=5,
            latent_dim=32,
        ).to(device) if curiosity_coef > 0 else None

        # Create trainer
        ppo_config = PPOConfig(
            learning_rate=3e-4,
            curiosity_coef=curiosity_coef,
            device=device,
        )
        trainer = PPOTrainer(swarm, world_model, ppo_config)

        # Train
        rewards = []
        coverages = []

        for i in tqdm(range(num_iterations), desc=label):
            stats = trainer.train_step(env, rollout_steps=128)
            trainer_stats = trainer.get_stats()
            env_stats = env.get_stats()

            if i % 50 == 0:
                rewards.append(trainer_stats["avg_reward_10"])
                coverages.append(env_stats["coverage"])

        results[label] = {
            "final_reward": trainer.get_stats()["avg_reward_100"],
            "final_coverage": env.get_stats()["coverage"],
            "rewards": rewards,
            "coverages": coverages,
        }

    # Print comparison
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for label, data in results.items():
        print(f"\n{label}:")
        print(f"  Final avg reward: {data['final_reward']:.3f}")
        print(f"  Final coverage: {data['final_coverage']:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Curiosity-Driven SEESWM Training")
    parser.add_argument("--iterations", type=int, default=500, help="Training iterations")
    parser.add_argument("--rollout-steps", type=int, default=256, help="Steps per rollout")
    parser.add_argument("--curiosity", type=str, default="world_model",
                       choices=["world_model", "rnd", "icm"], help="Curiosity type")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--log-dir", type=str, default="logs/curiosity", help="Log directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--compare", action="store_true", help="Run comparison experiment")
    args = parser.parse_args()

    if args.compare:
        compare_curiosity_vs_no_curiosity(
            num_iterations=args.iterations,
            device=args.device,
        )
    else:
        train_curiosity(
            num_iterations=args.iterations,
            rollout_steps=args.rollout_steps,
            curiosity_type=args.curiosity,
            device=args.device,
            log_dir=args.log_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
