#!/usr/bin/env python3
"""Fail-closed evaluation of a trained SEESWM policy checkpoint.

This command deliberately evaluates only the policy contained in a schema-v2
checkpoint. It does not claim to establish emergence, generalization,
publication readiness, or superiority to baselines. Those questions require
separately trained, capacity-matched controls and a preregistered protocol.

Legacy pickle checkpoints are not loaded. Recreate them with a current
training script in a trusted environment before evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

# Add project root to path when the script is run directly.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.cosmos import (
    CosmosEnvironment,
    EnvironmentConfig,
    validate_environment_capacity,
)
from src.swarm.graph import (
    MAX_PERSISTED_DIMENSION,
    SWARM_STATE_SCHEMA_VERSION,
    SwarmGraph,
    _validate_tensor_state_dict,
    swarm_config_from_dict,
)
from src.utils.checkpoint import load_bounded_weights_only


CHECKPOINT_SCHEMA_VERSION = 2
NUM_ACTIONS = 5
MAX_METADATA_NODES = 100_000
MAX_METADATA_STRING_LENGTH = 65_536
MAX_EVALUATION_GRID_SIZE = 128
MAX_EVALUATION_VISION_RADIUS = 31
MAX_EVALUATION_OBJECTS = 1_024

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("seeswm_checkpoint_evaluation")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_native_int(
    value: Any,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a native int")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _validate_json_metadata(
    value: Any,
    name: str,
    depth: int = 0,
    budget: list[int] | None = None,
) -> Any:
    """Return bounded JSON-native metadata or reject it."""
    if budget is None:
        budget = [MAX_METADATA_NODES]
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("checkpoint metadata exceeds the total node limit")
    if depth > 8:
        raise ValueError(f"{name} exceeds the metadata nesting limit")
    if value is None or type(value) in (bool, int):
        return value
    if type(value) is str:
        if len(value) > MAX_METADATA_STRING_LENGTH:
            raise ValueError(f"{name} exceeds the metadata string-length limit")
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must not contain NaN or infinity")
        return value
    if type(value) is list:
        if len(value) > 10_000:
            raise ValueError(f"{name} exceeds the metadata item limit")
        return [
            _validate_json_metadata(
                item,
                f"{name}[{index}]",
                depth + 1,
                budget,
            )
            for index, item in enumerate(value)
        ]
    if type(value) is dict:
        if len(value) > 10_000:
            raise ValueError(f"{name} exceeds the metadata item limit")
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{name} keys must be native strings")
            if len(key) > 256:
                raise ValueError(f"{name} contains an overlong metadata key")
            result[key] = _validate_json_metadata(
                item,
                f"{name}[{key!r}]",
                depth + 1,
                budget,
            )
        return result
    raise TypeError(f"{name} contains unsupported value {type(value).__name__}")


def load_checkpoint_with_digest(
    path: Path,
) -> tuple[Mapping[str, Any], str]:
    """Load a schema-v2 envelope and return the digest of the loaded bytes."""
    checkpoint, _, digest = load_bounded_weights_only(path, map_location="cpu")
    if type(checkpoint) is not dict:
        raise TypeError("checkpoint must be a native dict")

    schema_version = checkpoint.get("schema_version")
    if type(schema_version) is not int or schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported checkpoint schema; expected schema_version=2. "
            "Legacy checkpoints are intentionally rejected"
        )

    required = {"swarm_config", "swarm_state", "policy_head", "environment_config"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing required fields: {', '.join(missing)}")

    allowed = required | {
        "schema_version",
        "checkpoint_type",
        "value_head",
        "world_model",
        "training_config",
        "training_metrics",
        "role_metrics",
        "iteration",
        "seed",
        "source_revision",
        "source_dirty",
        "dependency_versions",
    }
    extra = sorted(set(checkpoint) - allowed)
    if extra:
        raise ValueError(f"checkpoint contains unexpected fields: {', '.join(extra)}")

    swarm_config = checkpoint["swarm_config"]
    if type(swarm_config) is not dict:
        raise TypeError("swarm_config must be a native dict")

    swarm_state = checkpoint["swarm_state"]
    if type(swarm_state) is not dict:
        raise TypeError("swarm_state must be a native dict")
    if swarm_state.get("schema_version") != SWARM_STATE_SCHEMA_VERSION:
        raise ValueError(
            "checkpoint contains an unsupported swarm state schema: "
            f"{swarm_state.get('schema_version')!r}"
        )
    if swarm_state.get("config") != swarm_config:
        raise ValueError("swarm_config does not match swarm_state config")

    if type(checkpoint["policy_head"]) is not dict:
        raise TypeError("policy_head must be a native dict")
    if type(checkpoint["environment_config"]) is not dict:
        raise TypeError("environment_config must be a native dict")

    metadata_budget = [MAX_METADATA_NODES]
    for field_name in (
        "checkpoint_type",
        "training_config",
        "training_metrics",
        "role_metrics",
        "iteration",
        "seed",
        "source_revision",
        "source_dirty",
        "dependency_versions",
    ):
        if field_name in checkpoint:
            _validate_json_metadata(
                checkpoint[field_name],
                field_name,
                budget=metadata_budget,
            )

    return checkpoint, digest


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    """Load and validate a schema-v2 checkpoint envelope."""
    checkpoint, _ = load_checkpoint_with_digest(path)
    return checkpoint


def build_policy_head(specification: Mapping[str, Any], device: str) -> nn.Module:
    """Reconstruct a known policy-head architecture and strict-load its weights."""
    if type(specification) is not dict:
        raise TypeError("policy_head must be a native dict")
    required = {"type", "input_dim", "hidden_dim", "num_actions", "state_dict"}
    actual = set(specification)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise ValueError(
            f"policy_head fields do not match; missing={missing}, extra={extra}"
        )

    policy_type = specification["type"]
    if type(policy_type) is not str:
        raise TypeError("policy head type must be a native string")
    input_dim = _require_native_int(
        specification["input_dim"],
        "policy head input_dim",
        minimum=1,
        maximum=MAX_PERSISTED_DIMENSION,
    )
    hidden_dim = _require_native_int(
        specification["hidden_dim"],
        "policy head hidden_dim",
        minimum=1,
        maximum=MAX_PERSISTED_DIMENSION,
    )
    num_actions = _require_native_int(
        specification["num_actions"],
        "policy head num_actions",
        minimum=NUM_ACTIONS,
        maximum=NUM_ACTIONS,
    )
    if num_actions != NUM_ACTIONS:
        raise ValueError(
            f"Cosmos evaluation requires exactly {NUM_ACTIONS} actions, got {num_actions}"
        )

    if policy_type == "mlp_relu":
        policy = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        ).to(device)
    elif policy_type == "policy_head_tanh":
        from src.training import PolicyHead

        policy = PolicyHead(input_dim, num_actions, hidden_dim=hidden_dim).to(device)
    else:
        raise ValueError(f"unsupported policy head type: {policy_type!r}")
    state = _validate_tensor_state_dict(
        specification["state_dict"], "policy head state"
    )
    policy.load_state_dict(state, strict=True)
    policy.eval()
    return policy


def environment_config_to_dict(configuration: EnvironmentConfig) -> dict[str, Any]:
    """Convert every environment field to a native checkpoint primitive."""
    return {
        item.name: getattr(configuration, item.name)
        for item in fields(EnvironmentConfig)
    }


def build_environment(configuration: Mapping[str, Any]) -> CosmosEnvironment:
    """Reconstruct an environment from an exact, native-type config mapping."""
    configuration = _require_mapping(configuration, "environment_config")
    defaults = EnvironmentConfig()
    expected = {item.name for item in fields(EnvironmentConfig)}
    actual = set(configuration)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"environment_config keys do not match; missing={missing}, extra={extra}"
        )

    for name in sorted(expected):
        expected_type = type(getattr(defaults, name))
        if type(configuration[name]) is not expected_type:
            raise TypeError(
                f"environment_config {name} must be a native "
                f"{expected_type.__name__}"
            )

    config = EnvironmentConfig(**dict(configuration))
    if config.num_agents != 1:
        raise ValueError("this evaluator currently supports exactly one environment agent")
    if not 4 <= config.grid_size <= MAX_EVALUATION_GRID_SIZE:
        raise ValueError(
            "evaluation grid_size must be in "
            f"[4, {MAX_EVALUATION_GRID_SIZE}]"
        )
    if not 1 <= config.vision_radius <= MAX_EVALUATION_VISION_RADIUS:
        raise ValueError(
            "evaluation vision_radius must be in "
            f"[1, {MAX_EVALUATION_VISION_RADIUS}]"
        )
    if config.max_steps < 1:
        raise ValueError("environment dimensions, vision radius, and max_steps are invalid")
    nonnegative = (
        config.num_resources,
        config.num_hazards,
        config.respawn_delay,
        config.num_food,
        config.num_water,
        config.num_material,
        config.hunger_rate,
        config.thirst_rate,
        config.movement_cost,
        config.stay_cost,
    )
    float_values = (
        config.hunger_rate,
        config.thirst_rate,
        config.starvation_threshold,
        config.movement_cost,
        config.stay_cost,
    )
    if not all(math.isfinite(value) for value in float_values):
        raise ValueError("environment rates, costs, and thresholds must be finite")
    if any(value < 0 for value in nonnegative) or config.starvation_threshold <= 0:
        raise ValueError("environment counts, rates, costs, and thresholds are invalid")
    object_count = sum(
        (
            config.num_resources,
            config.num_hazards,
            config.num_food,
            config.num_water,
            config.num_material,
        )
    )
    if object_count > MAX_EVALUATION_OBJECTS:
        raise ValueError(
            "evaluation environment exceeds the object-count limit: "
            f"{object_count} > {MAX_EVALUATION_OBJECTS}"
        )
    validate_environment_capacity(config)
    return CosmosEnvironment(config=config)


def set_swarm_eval(swarm: Any) -> None:
    """Disable dropout in every trainable swarm component used for evaluation."""
    for agent in swarm.agents.values():
        agent.network.eval()
    for aggregator in getattr(swarm, "aggregators", {}).values():
        aggregator.eval()
    message_encoder = getattr(swarm, "message_encoder", None)
    if message_encoder is not None:
        message_encoder.eval()


def build_candidate(
    checkpoint: Mapping[str, Any], device: str
) -> tuple[Any, nn.Module, dict[str, Any]]:
    """Reconstruct the exact saved swarm configuration and trained policy head."""
    saved_config = _require_mapping(checkpoint["swarm_config"], "swarm_config")
    policy_spec = _require_mapping(checkpoint["policy_head"], "policy_head")
    if policy_spec.get("type") == "policy_head_tanh":
        from src.swarm.specialized_graph import (
            SpecializedSwarmGraph,
            specialized_swarm_config_from_dict,
        )

        config = specialized_swarm_config_from_dict(saved_config)
        swarm = SpecializedSwarmGraph(config=config, device=device)
    else:
        config = swarm_config_from_dict(saved_config)
        swarm = SwarmGraph(config=config, device=device)
    swarm.load_state_dict(checkpoint["swarm_state"])
    set_swarm_eval(swarm)

    if int(policy_spec.get("input_dim", -1)) != config.output_dim:
        raise ValueError(
            "policy-head input dimension does not match the swarm output dimension"
        )
    policy = build_policy_head(policy_spec, device)

    environment_config = dict(
        _require_mapping(checkpoint["environment_config"], "environment_config")
    )
    environment = build_environment(environment_config)
    if environment.observation_dim != config.input_dim:
        raise ValueError(
            "environment observation dimension does not match the saved swarm input dimension"
        )

    return swarm, policy, environment_config


def evaluate_seed(
    swarm: SwarmGraph,
    policy: nn.Module,
    environment_config: Mapping[str, Any],
    *,
    device: str,
    seed: int,
    episodes: int,
    max_steps: int,
) -> dict[str, Any]:
    """Run deterministic greedy-policy episodes for one declared seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    environment = build_environment(environment_config)
    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    episode_coverage: list[float] = []

    for _ in range(episodes):
        observations = environment.reset()
        swarm.reset(batch_size=1)
        reward_total = 0.0

        for step_index in range(max_steps):
            observation = observations[0].to_tensor(device).unsqueeze(0)
            with torch.no_grad():
                representation = swarm.step(observation)
                logits = policy(representation)
                if logits.ndim != 2 or logits.shape != (1, NUM_ACTIONS):
                    raise ValueError(
                        "policy head returned an invalid action-logit shape: "
                        f"{tuple(logits.shape)}"
                    )
                action = int(logits.argmax(dim=-1).item())

            if not 0 <= action < NUM_ACTIONS:
                raise ValueError(f"policy produced an out-of-range action: {action}")

            observations, rewards, dones = environment.step([action])
            reward_total += float(rewards[0]) if rewards else 0.0
            if dones and dones[0]:
                break

        steps = step_index + 1
        environment_stats = environment.get_stats()
        episode_rewards.append(reward_total)
        episode_steps.append(steps)
        episode_coverage.append(float(environment_stats.get("coverage", 0.0)))

    return {
        "seed": seed,
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "episode_coverage": episode_coverage,
        "mean_reward": float(np.mean(episode_rewards)),
        "mean_steps": float(np.mean(episode_steps)),
        "mean_coverage": float(np.mean(episode_coverage)),
    }


def main() -> dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Evaluate one schema-v2 SEESWM policy checkpoint (diagnostic only)"
    )
    parser.add_argument("--model", required=True, help="Path to a schema-v2 checkpoint")
    parser.add_argument("--device", default="cpu", help="PyTorch device")
    parser.add_argument("--seeds", type=int, default=10, help="Number of declared seeds")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per seed")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument(
        "--output",
        default="results/validation-local",
        help="Output directory",
    )
    args = parser.parse_args()

    if args.seeds <= 0 or args.episodes <= 0 or args.max_steps <= 0:
        parser.error("--seeds, --episodes, and --max-steps must all be positive")

    checkpoint_path = Path(args.model).expanduser().resolve(strict=True)
    checkpoint, digest = load_checkpoint_with_digest(checkpoint_path)
    swarm, policy, environment_config = build_candidate(checkpoint, args.device)

    logger.info(
        "Loaded schema-v2 checkpoint %s (sha256=%s...)",
        checkpoint_path.name,
        digest[:12],
    )
    seed_results = [
        evaluate_seed(
            swarm,
            policy,
            environment_config,
            device=args.device,
            seed=seed,
            episodes=args.episodes,
            max_steps=args.max_steps,
        )
        for seed in range(args.seeds)
    ]

    mean_rewards = [result["mean_reward"] for result in seed_results]
    report = {
        "report_schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "in-distribution deterministic policy diagnostic",
        "limitations": [
            "No trained capacity-matched baselines are evaluated.",
            "This result does not establish emergence, generalization, or publication readiness.",
            "The environment and training implementation have not been independently validated.",
        ],
        "checkpoint": {
            "file": checkpoint_path.name,
            "sha256": digest,
            "schema_version": int(checkpoint["schema_version"]),
            "source_revision": checkpoint.get("source_revision"),
            "source_dirty": checkpoint.get("source_dirty"),
            "training_seed": checkpoint.get("seed"),
            "dependency_versions": checkpoint.get("dependency_versions"),
            "loaded_successfully": True,
        },
        "configuration": {
            "swarm": dict(checkpoint["swarm_config"]),
            "environment": environment_config,
            "device": args.device,
            "seeds": list(range(args.seeds)),
            "episodes_per_seed": args.episodes,
            "max_steps": args.max_steps,
            "action_selection": "greedy_argmax",
            "recorded_training": checkpoint.get("training_config"),
        },
        "results": {
            "mean_reward_across_seeds": float(np.mean(mean_rewards)),
            "std_reward_across_seeds": float(np.std(mean_rewards)),
            "per_seed": seed_results,
        },
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"checkpoint_evaluation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Diagnostic report saved to %s", output_path)
    return report


if __name__ == "__main__":
    main()
