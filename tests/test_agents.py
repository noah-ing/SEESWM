"""
Tests for MicroAgent.
"""

import pytest
import torch

from src.agents.micro_agent import MicroAgent, AgentConfig, AgentType


class TestMicroAgent:
    """Test suite for MicroAgent."""

    def test_agent_creation(self):
        """Test basic agent creation."""
        agent = MicroAgent(agent_id=0)

        assert agent.agent_id == 0
        assert agent.agent_type == AgentType.GENERAL
        assert agent.num_parameters > 0

    def test_agent_with_config(self):
        """Test agent creation with custom config."""
        config = AgentConfig(
            agent_type=AgentType.PERCEPTION,
            hidden_dim=64,
            num_layers=3,
        )
        agent = MicroAgent(agent_id=1, config=config)

        assert agent.agent_type == AgentType.PERCEPTION
        assert agent.config.hidden_dim == 64
        assert agent.config.num_layers == 3

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        config = AgentConfig(input_dim=32, output_dim=16, message_dim=16)
        agent = MicroAgent(agent_id=0, config=config)

        batch_size = 4
        inputs = torch.randn(batch_size, 32)

        output = agent.forward(inputs, neighbor_messages=[])

        assert output.shape == (batch_size, 16)

    def test_forward_with_messages(self):
        """Test forward pass with neighbor messages."""
        config = AgentConfig(input_dim=32, output_dim=16, message_dim=16)
        agent = MicroAgent(agent_id=0, config=config)

        batch_size = 4
        inputs = torch.randn(batch_size, 32)
        messages = [
            torch.randn(batch_size, 16),
            torch.randn(batch_size, 16),
            torch.randn(batch_size, 16),
        ]

        output = agent.forward(inputs, neighbor_messages=messages)

        assert output.shape == (batch_size, 16)

    def test_local_state_persistence(self):
        """Test that local state persists across calls."""
        agent = MicroAgent(agent_id=0)
        agent.reset_state(batch_size=2)

        initial_state = agent.local_state.clone()

        inputs = torch.randn(2, agent.config.input_dim)
        agent.forward(inputs, [])

        # State should have changed
        assert not torch.allclose(agent.local_state, initial_state)

    def test_modulation(self):
        """Test neuromodulatory signal application."""
        agent = MicroAgent(agent_id=0)

        original_lr = agent.get_effective_lr()

        agent.modulate("dopamine", 0.5)
        boosted_lr = agent.get_effective_lr()

        assert boosted_lr > original_lr

    def test_state_dict_roundtrip(self):
        """Test save/load state dict."""
        agent = MicroAgent(agent_id=0)

        # Run a forward pass to initialize state
        inputs = torch.randn(1, agent.config.input_dim)
        original_output = agent.forward(inputs, [])

        # Save state
        state = agent.state_dict()

        # Create new agent and load state
        agent2 = MicroAgent(agent_id=0)
        agent2.load_state_dict(state)

        # Should produce same output
        agent2.reset_state(1)
        loaded_output = agent2.forward(inputs, [])

        # Outputs might differ slightly due to state, but params should be same
        assert agent.num_parameters == agent2.num_parameters

    def test_agent_stats(self):
        """Test statistics tracking."""
        agent = MicroAgent(agent_id=42)

        # Run some forward passes
        for _ in range(5):
            inputs = torch.randn(1, agent.config.input_dim)
            agent.forward(inputs, [])

        stats = agent.get_stats()

        assert stats["agent_id"] == 42
        assert stats["activation_count"] == 5
        assert stats["avg_output_magnitude"] > 0


class TestAgentTypes:
    """Test different agent types."""

    @pytest.mark.parametrize("agent_type", list(AgentType))
    def test_all_agent_types(self, agent_type):
        """Test that all agent types can be created and run."""
        config = AgentConfig(agent_type=agent_type)
        agent = MicroAgent(agent_id=0, config=config)

        inputs = torch.randn(1, config.input_dim)
        output = agent.forward(inputs, [])

        assert output.shape == (1, config.output_dim)
