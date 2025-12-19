"""
Basic training script for SEESWM.

Phase 1: Get the swarm learning to collect resources in the grid world.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from tqdm import tqdm

from src.swarm.graph import SwarmGraph, SwarmConfig, TopologyType
from src.environment.cosmos import CosmosEnvironment
from src.utils.config import load_config, Config
from src.utils.logging import setup_logger, MetricsLogger


def train(config: Config):
    """Run basic training loop."""
    logger = setup_logger("seeswm", log_dir=config.log_dir)
    metrics = MetricsLogger(config.log_dir)

    logger.info("Initializing SEESWM training...")
    logger.info(f"Config: {config.to_dict()}")

    # Create swarm
    topology_map = {
        "random": TopologyType.RANDOM,
        "small_world": TopologyType.SMALL_WORLD,
        "scale_free": TopologyType.SCALE_FREE,
        "modular": TopologyType.MODULAR,
        "hierarchical": TopologyType.HIERARCHICAL,
    }

    swarm_config = SwarmConfig(
        num_agents=config.num_agents,
        topology=topology_map.get(config.topology, TopologyType.SMALL_WORLD),
        hidden_dim=config.hidden_dim,
        message_passing_rounds=config.message_passing_rounds,
    )

    swarm = SwarmGraph(config=swarm_config, device=config.device)
    logger.info(f"Created swarm with {swarm.total_parameters:,} parameters")
    logger.info(f"Topology: {swarm.get_topology_stats()}")

    # Create environment
    env = CosmosEnvironment(
        grid_size=config.grid_size,
        num_resources=config.num_resources,
        num_hazards=config.num_hazards,
    )
    logger.info(f"Created environment: {config.grid_size}x{config.grid_size} grid")

    # Collect all parameters for optimizer
    all_params = []
    for agent in swarm.agents.values():
        all_params.extend(agent.network.parameters())

    optimizer = optim.Adam(all_params, lr=config.learning_rate)

    # Training loop
    episode_rewards = []

    for epoch in range(config.num_epochs):
        observations = env.reset()
        obs_tensor = observations[0].to_tensor(config.device).unsqueeze(0)

        swarm.reset(batch_size=1)
        episode_reward = 0.0
        episode_length = 0

        for step in range(200):  # Max 200 steps per episode
            # Get action from swarm output
            with torch.no_grad():
                output = swarm.step(obs_tensor)

            # Convert output to action (simple argmax over 5 actions)
            # Project output to 5-dim for action selection
            action_logits = output[:, :5] if output.shape[1] >= 5 else output
            action = action_logits.argmax(dim=-1).item()

            # Environment step
            observations, rewards, dones = env.step([action])

            episode_reward += rewards[0]
            episode_length += 1

            if dones[0]:
                break

            obs_tensor = observations[0].to_tensor(config.device).unsqueeze(0)

        episode_rewards.append(episode_reward)

        # Simple policy gradient update (REINFORCE-style)
        # For now, just demonstrate the training loop structure
        # Full RL will be implemented in Phase 2

        if epoch % config.log_every == 0:
            avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            logger.info(
                f"Epoch {epoch}: reward={episode_reward:.2f}, "
                f"avg_reward={avg_reward:.2f}, length={episode_length}"
            )
            metrics.log(epoch, reward=episode_reward, avg_reward=avg_reward, length=episode_length)

        if epoch % config.save_every == 0 and epoch > 0:
            save_path = Path(config.log_dir) / f"swarm_epoch_{epoch}.pt"
            torch.save(swarm.state_dict(), save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    # Final stats
    logger.info("Training complete!")
    logger.info(f"Final avg reward (last 10): {sum(episode_rewards[-10:])/10:.2f}")

    metrics.save()

    return swarm, episode_rewards


def main():
    parser = argparse.ArgumentParser(description="Train SEESWM")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs")
    parser.add_argument("--device", type=str, default=None, help="Override device")
    args = parser.parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = Config()

    if args.epochs:
        config.num_epochs = args.epochs
    if args.device:
        config.device = args.device

    train(config)


if __name__ == "__main__":
    main()
