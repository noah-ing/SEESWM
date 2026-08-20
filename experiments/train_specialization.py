"""
Specialization training for SEESWM.

Phase 3: Agent specialization with typed messaging and role-pattern tracking.
Trains specialized architectures and records descriptive communication metrics.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from typing import Optional, Dict, List
import math
import platform
import random
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.swarm.graph import (
    SWARM_STATE_SCHEMA_VERSION,
    SwarmGraph,
    SwarmConfig,
    TopologyType,
)
from src.swarm.specialized_graph import (
    SpecializedSwarmGraph,
    SpecializedSwarmConfig,
    specialized_swarm_config_to_dict,
)
from src.world_model.jepa import WorldModel
from src.environment.cosmos import CosmosEnvironment, EnvironmentConfig
from src.training import PPOConfig, TrajectoryBuffer, PolicyHead, ValueHead, Transition
from src.utils.logging import setup_logger, MetricsLogger
from experiments.validate_rigorously import load_checkpoint_with_digest


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


def _environment_config_to_dict(config: EnvironmentConfig) -> dict:
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


def _ppo_config_to_dict(config: PPOConfig) -> dict:
    return {
        "learning_rate": float(config.learning_rate),
        "gamma": float(config.gamma),
        "gae_lambda": float(config.gae_lambda),
        "clip_epsilon": float(config.clip_epsilon),
        "num_epochs": int(config.num_epochs),
        "batch_size": int(config.batch_size),
        "max_grad_norm": float(config.max_grad_norm),
        "value_coef": float(config.value_coef),
        "entropy_coef": float(config.entropy_coef),
        "curiosity_coef": float(config.curiosity_coef),
        "device": str(config.device),
    }


def _to_native(value: object) -> object:
    """Recursively reject custom objects and normalize NumPy scalar metrics."""
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Checkpoint metrics must be finite")
        return value
    if isinstance(value, np.generic):
        return _to_native(value.item())
    if type(value) is dict:
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(
                    "Checkpoint metric keys must be strings, "
                    f"got {type(key)}"
                )
            result[key] = _to_native(item)
        return result
    if type(value) in (list, tuple):
        return [_to_native(item) for item in value]
    raise TypeError(f"Unsupported checkpoint metric value: {type(value)}")


def _build_checkpoint(
    *,
    trainer: "SpecializationPPOTrainer",
    swarm: SpecializedSwarmGraph,
    world_model: WorldModel,
    env_config: EnvironmentConfig,
    num_iterations: int,
    rollout_steps: int,
    topology: str,
    iteration: int,
    metrics: dict,
    role_metrics: dict,
    seed: int,
    source_revision: str | None,
    source_dirty: bool | None,
    dependency_versions: dict[str, str | None],
) -> dict:
    return {
        "schema_version": SWARM_STATE_SCHEMA_VERSION,
        "checkpoint_type": "specialized_ppo_training",
        "swarm_config": specialized_swarm_config_to_dict(swarm.config),
        "swarm_state": swarm.state_dict(),
        "policy_head": {
            "type": "policy_head_tanh",
            "input_dim": int(trainer.output_dim),
            "hidden_dim": 64,
            "num_actions": int(trainer.num_actions),
            "state_dict": dict(trainer.policy_head.state_dict()),
        },
        "value_head": {
            "type": "value_head_tanh",
            "input_dim": int(trainer.output_dim),
            "hidden_dim": 64,
            "state_dict": dict(trainer.value_head.state_dict()),
        },
        "world_model": {
            "type": "jepa",
            "obs_dim": int(world_model.obs_dim),
            "action_dim": int(world_model.action_dim),
            "latent_dim": int(world_model.latent_dim),
            "num_hierarchy_levels": int(len(world_model.predictors)),
            "ema_decay": float(world_model.ema_decay),
            "state_dict": dict(world_model.state_dict()),
        },
        "environment_config": _environment_config_to_dict(env_config),
        "training_config": {
            "algorithm": "ppo_with_curiosity",
            "num_iterations": int(num_iterations),
            "rollout_steps": int(rollout_steps),
            "topology": str(topology),
            "ppo": _ppo_config_to_dict(trainer.config),
        },
        "iteration": int(iteration),
        "training_metrics": _to_native(metrics),
        "role_metrics": _to_native(role_metrics),
        "seed": int(seed),
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "dependency_versions": dict(dependency_versions),
    }


class SpecializationPPOTrainer:
    """
    PPO trainer adapted for specialized swarms.

    Key differences from base PPOTrainer:
    - Uses SpecializedSwarmGraph with typed messaging
    - Tracks descriptive role-pattern metrics
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
    Train a specialized swarm with role-pattern tracking.
    """
    if num_iterations < 1 or rollout_steps < 1:
        raise ValueError("iterations and rollout steps must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    supported_topologies = {
        "hierarchical",
        "modular",
        "small_world",
        "scale_free",
    }
    if topology not in supported_topologies:
        raise ValueError(f"unsupported topology: {topology!r}")

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    source_revision = _source_revision()
    source_dirty = _source_dirty()
    dependency_versions = _dependency_versions()

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
                torch.save(
                    _build_checkpoint(
                        trainer=trainer,
                        swarm=swarm,
                        world_model=world_model,
                        env_config=env_config,
                        num_iterations=num_iterations,
                        rollout_steps=rollout_steps,
                        topology=topology,
                        iteration=iteration,
                        metrics={
                            **trainer_stats,
                            "best_avg_reward": best_avg_reward,
                        },
                        role_metrics=role_metrics,
                        seed=seed,
                        source_revision=source_revision,
                        source_dirty=source_dirty,
                        dependency_versions=dependency_versions,
                    ),
                    run_dir / "best_model.pt",
                )

        if iteration % 100 == 0 and iteration > 0:
            torch.save(
                _build_checkpoint(
                    trainer=trainer,
                    swarm=swarm,
                    world_model=world_model,
                    env_config=env_config,
                    num_iterations=num_iterations,
                    rollout_steps=rollout_steps,
                    topology=topology,
                    iteration=iteration,
                    metrics=trainer.get_stats(),
                    role_metrics=swarm.get_role_emergence_metrics(),
                    seed=seed,
                    source_revision=source_revision,
                    source_dirty=source_dirty,
                    dependency_versions=dependency_versions,
                ),
                run_dir / f"checkpoint_{iteration}.pt",
            )

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

    logger.info("\nRole-pattern metrics:")
    logger.info(f"  Specialization entropy: {final_role_metrics.get('avg_specialization_entropy', 0):.3f}")

    type_concentrations = final_role_metrics.get("type_concentrations", {})
    for agent_type, concentration in type_concentrations.items():
        logger.info(f"  {agent_type} concentration: {concentration:.3f}")

    torch.save(
        _build_checkpoint(
            trainer=trainer,
            swarm=swarm,
            world_model=world_model,
            env_config=env_config,
            num_iterations=num_iterations,
            rollout_steps=rollout_steps,
            topology=topology,
            iteration=max(num_iterations - 1, 0),
            metrics={
                **final_stats,
                "best_avg_reward": best_avg_reward,
            },
            role_metrics=final_role_metrics,
            seed=seed,
            source_revision=source_revision,
            source_dirty=source_dirty,
            dependency_versions=dependency_versions,
        ),
        run_dir / "final_model.pt",
    )

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
    Analyze descriptive role-pattern metrics from a trained checkpoint.
    """
    print("=" * 60)
    print("Role-pattern analysis")
    print("=" * 60)

    checkpoint, digest = load_checkpoint_with_digest(Path(checkpoint_path))
    print(f"Checkpoint SHA-256: {digest}")
    role_metrics = checkpoint.get("role_metrics")
    if type(role_metrics) is not dict:
        raise TypeError("checkpoint role_metrics must be a native dict")

    print("\nRole-pattern metrics:")
    print(f"  Avg specialization entropy: {role_metrics.get('avg_specialization_entropy', 0):.4f}")

    type_concentrations = role_metrics.get("type_concentrations", {})
    print("\nType concentrations (higher = more concentrated):")
    for agent_type, concentration in type_concentrations.items():
        bar = "=" * int(concentration * 50)
        print(f"  {agent_type:12s}: {concentration:.3f} |{bar}|")

    print("\nInterpretation:")
    print("  - Lower entropy means message types were more concentrated")
    print("  - Concentration reports use of architecture-associated message types")

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
