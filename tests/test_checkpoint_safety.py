"""Focused safety tests for schema-v2 checkpoint persistence."""

import hashlib
from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from experiments.validate_rigorously import (
    CHECKPOINT_SCHEMA_VERSION,
    NUM_ACTIONS,
    build_environment,
    build_policy_head,
    environment_config_to_dict,
    load_checkpoint,
)
from src.agents.micro_agent import (
    AGENT_STATE_SCHEMA_VERSION,
    AgentConfig,
    MicroAgent,
)
from src.environment.cosmos import EnvironmentConfig
from src.swarm.graph import (
    SWARM_STATE_SCHEMA_VERSION,
    SwarmConfig,
    SwarmGraph,
    TopologyType,
    swarm_config_from_dict,
    swarm_config_to_dict,
)
from src.utils import checkpoint as checkpoint_utils


def _tiny_agent_config() -> AgentConfig:
    return AgentConfig(
        input_dim=4,
        hidden_dim=8,
        output_dim=3,
        message_dim=3,
        num_layers=1,
        dropout=0.0,
        state_dim=2,
    )


def _tiny_swarm_config() -> SwarmConfig:
    return SwarmConfig(
        num_agents=4,
        topology=TopologyType.FULLY_CONNECTED,
        message_passing_rounds=1,
        input_dim=4,
        hidden_dim=8,
        output_dim=3,
        message_dim=3,
        num_perception=1,
        num_reasoning=1,
        num_memory=1,
        num_planning=1,
    )


def _assert_agent_networks_equal(expected: MicroAgent, actual: MicroAgent) -> None:
    expected_state = expected.network.state_dict()
    actual_state = actual.network.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name, expected_tensor in expected_state.items():
        torch.testing.assert_close(actual_state[name], expected_tensor)


def _checkpoint_payload() -> dict:
    swarm = SwarmGraph(_tiny_swarm_config())
    policy = nn.Sequential(
        nn.Linear(swarm.config.output_dim, 7),
        nn.ReLU(),
        nn.Linear(7, NUM_ACTIONS),
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "swarm_config": swarm_config_to_dict(swarm.config),
        "swarm_state": swarm.state_dict(),
        "policy_head": {
            "type": "mlp_relu",
            "input_dim": swarm.config.output_dim,
            "hidden_dim": 7,
            "num_actions": NUM_ACTIONS,
            "state_dict": dict(policy.state_dict()),
        },
        "environment_config": {},
    }


def test_micro_agent_weights_only_roundtrip_resets_ephemeral_state(tmp_path):
    config = _tiny_agent_config()
    source = MicroAgent(agent_id=7, config=config)
    source.forward(torch.randn(2, config.input_dim), [])
    assert source.local_state is not None

    state = source.state_dict()
    assert state["schema_version"] == AGENT_STATE_SCHEMA_VERSION
    assert "local_state" not in state
    assert "last_output" not in state

    checkpoint_path = tmp_path / "agent.pt"
    torch.save(state, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    restored = MicroAgent(agent_id=7, config=config)
    restored.forward(torch.randn(1, config.input_dim), [])
    assert restored.local_state is not None
    restored.load_state_dict(loaded)

    _assert_agent_networks_equal(source, restored)
    assert restored.local_state is None
    assert restored.last_output is None
    assert restored._batch_size is None


def test_swarm_weights_only_roundtrip_resets_all_agent_local_state(tmp_path):
    config = _tiny_swarm_config()
    source = SwarmGraph(config)
    source.step(torch.randn(2, config.input_dim))
    assert all(agent.local_state is not None for agent in source.agents.values())

    state = source.state_dict()
    assert state["schema_version"] == SWARM_STATE_SCHEMA_VERSION
    assert all("local_state" not in agent for agent in state["agents"].values())

    checkpoint_path = tmp_path / "swarm.pt"
    torch.save(state, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    restored = SwarmGraph(config)
    restored.step(torch.randn(1, config.input_dim))
    restored.load_state_dict(loaded)

    assert restored.state_dict()["graph_edges"] == loaded["graph_edges"]
    for agent_id, source_agent in source.agents.items():
        _assert_agent_networks_equal(source_agent, restored.agents[agent_id])
    assert all(agent.local_state is None for agent in restored.agents.values())


def test_micro_agent_rejects_config_mismatch():
    config = _tiny_agent_config()
    state = MicroAgent(agent_id=0, config=config).state_dict()
    mismatched = MicroAgent(agent_id=0, config=replace(config, hidden_dim=9))

    with pytest.raises(ValueError, match="Agent config mismatch"):
        mismatched.load_state_dict(state)


def test_swarm_rejects_config_mismatch():
    config = _tiny_swarm_config()
    state = SwarmGraph(config).state_dict()
    mismatched = SwarmGraph(
        replace(config, message_passing_rounds=config.message_passing_rounds + 1)
    )

    with pytest.raises(ValueError, match="Swarm config mismatch"):
        mismatched.load_state_dict(state)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_swarm_rejects_missing_or_extra_agent_ids(mutation):
    config = _tiny_swarm_config()
    state = SwarmGraph(config).state_dict()
    state["agents"] = dict(state["agents"])

    if mutation == "missing":
        state["agents"].pop(config.num_agents - 1)
    else:
        state["agents"][config.num_agents] = state["agents"][0]

    with pytest.raises(ValueError, match="swarm agent states IDs do not match"):
        SwarmGraph(config).load_state_dict(state)


@pytest.mark.parametrize(
    ("graph_edges", "message"),
    [
        ([[0, 1], [1, 2], [2, 4]], "endpoint outside"),
        ([[0, 1], [1, 0], [1, 2], [2, 3]], "duplicates edge"),
        ([[0, 1], [2, 3]], "connected graph"),
    ],
    ids=["invalid-endpoint", "duplicate", "disconnected"],
)
def test_swarm_rejects_invalid_graph_edges(graph_edges, message):
    config = _tiny_swarm_config()
    state = SwarmGraph(config).state_dict()
    state["graph_edges"] = graph_edges

    with pytest.raises(ValueError, match=message):
        SwarmGraph(config).load_state_dict(state)


def test_checkpoint_reconstructs_policy_with_exactly_five_actions(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(_checkpoint_payload(), checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path)
    policy = build_policy_head(checkpoint["policy_head"], device="cpu")
    logits = policy(torch.randn(3, checkpoint["policy_head"]["input_dim"]))

    assert logits.shape == (3, NUM_ACTIONS)
    assert NUM_ACTIONS == 5


def test_persisted_swarm_config_rejects_role_count_mismatch():
    serialized = swarm_config_to_dict(_tiny_swarm_config())
    serialized["num_planning"] = 0

    with pytest.raises(ValueError, match="role counts must sum exactly to num_agents"):
        swarm_config_from_dict(serialized)


def test_persisted_swarm_config_rejects_combined_allocation_bomb():
    serialized = swarm_config_to_dict(_tiny_swarm_config())
    serialized["hidden_dim"] = 4096

    with pytest.raises(ValueError, match="allocation budget"):
        swarm_config_from_dict(serialized)


def test_policy_spec_rejects_bool_dimensions():
    specification = _checkpoint_payload()["policy_head"]
    specification["input_dim"] = True

    with pytest.raises(TypeError, match="input_dim must be a native int"):
        build_policy_head(specification, device="cpu")


def test_checkpoint_rejects_unexpected_envelope_field(tmp_path):
    checkpoint = _checkpoint_payload()
    checkpoint["unreviewed_extension"] = {"enabled": True}
    checkpoint_path = tmp_path / "unexpected-field.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="unexpected fields"):
        load_checkpoint(checkpoint_path)


def test_checkpoint_loader_rejects_vulnerable_torch_runtime(monkeypatch):
    monkeypatch.setattr(checkpoint_utils.torch, "__version__", "2.9.1")

    with pytest.raises(RuntimeError, match="PyTorch 2.10.0 or newer"):
        checkpoint_utils.require_restricted_loader_runtime()


def test_checkpoint_loader_hashes_the_bytes_it_loads(tmp_path):
    checkpoint_path = tmp_path / "digest.pt"
    torch.save(_checkpoint_payload(), checkpoint_path)

    _, loaded_path, digest = checkpoint_utils.load_bounded_weights_only(
        checkpoint_path
    )

    assert loaded_path == checkpoint_path.resolve()
    assert digest == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()


def test_checkpoint_loader_enforces_file_size_limit(tmp_path):
    checkpoint_path = tmp_path / "size-limit.pt"
    torch.save(_checkpoint_payload(), checkpoint_path)

    with pytest.raises(ValueError, match="evaluator limit"):
        checkpoint_utils.load_bounded_weights_only(
            checkpoint_path,
            max_bytes=checkpoint_path.stat().st_size - 1,
        )


def test_evaluator_rejects_environment_without_worst_case_wall_capacity():
    config = EnvironmentConfig(
        grid_size=12,
        num_resources=96,
        num_hazards=0,
        num_agents=1,
        vision_radius=1,
        num_food=0,
        num_water=0,
        num_material=0,
    )

    with pytest.raises(ValueError, match="worst-case internal walls"):
        build_environment(environment_config_to_dict(config))


def test_evaluator_rejects_oversized_grid_before_allocation():
    config = environment_config_to_dict(EnvironmentConfig())
    config["grid_size"] = 129

    with pytest.raises(ValueError, match="evaluation grid_size"):
        build_environment(config)


def test_checkpoint_rejects_overlong_metadata(tmp_path):
    checkpoint = _checkpoint_payload()
    checkpoint["source_revision"] = "a" * 65_537
    checkpoint_path = tmp_path / "overlong-metadata.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="string-length limit"):
        load_checkpoint(checkpoint_path)


@pytest.mark.parametrize("invalidity", ["old-schema", "missing-field"])
def test_load_checkpoint_rejects_old_schema_or_missing_required_field(
    tmp_path, invalidity
):
    checkpoint = _checkpoint_payload()
    if invalidity == "old-schema":
        checkpoint["schema_version"] = CHECKPOINT_SCHEMA_VERSION - 1
        message = "unsupported checkpoint schema"
    else:
        checkpoint.pop("policy_head")
        message = "missing required fields: policy_head"

    checkpoint_path = tmp_path / f"{invalidity}.pt"
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match=message):
        load_checkpoint(checkpoint_path)
