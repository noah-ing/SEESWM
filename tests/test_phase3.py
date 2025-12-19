"""
Tests for Phase 3: Specialization.

Tests specialized agent architectures, typed messaging, and role emergence.
"""

import pytest
import torch

from src.agents.micro_agent import AgentConfig, AgentType
from src.agents.specializations import (
    PerceptionNetwork,
    ReasoningNetwork,
    MemoryNetwork,
    PlanningNetwork,
    SpecializedAgent,
    create_specialized_swarm,
)
from src.swarm.typed_messaging import (
    TypedMessage,
    TypedMessageBus,
    MessageType,
    MessageFilter,
    MessageEncoder,
    MessageDecoder,
    TypeAwareAggregator,
    AGENT_MESSAGE_TYPES,
)
from src.swarm.specialized_graph import (
    SpecializedSwarmGraph,
    SpecializedSwarmConfig,
    RoleEmergenceTracker,
)
from src.swarm.graph import TopologyType


class TestSpecializedNetworks:
    """Test specialized agent architectures."""

    def test_perception_network(self):
        """Test PerceptionNetwork forward pass."""
        config = AgentConfig(
            agent_type=AgentType.PERCEPTION,
            input_dim=64,
            hidden_dim=32,
            output_dim=16,
            message_dim=16,
        )
        network = PerceptionNetwork(config)

        batch_size = 2
        inputs = torch.randn(batch_size, config.input_dim)
        messages = torch.randn(batch_size, config.message_dim)
        state = torch.zeros(batch_size, config.state_dim)

        output, new_state = network(inputs, messages, state)

        assert output.shape == (batch_size, config.output_dim)
        assert new_state.shape == (batch_size, config.state_dim)
        assert not torch.isnan(output).any()

    def test_reasoning_network(self):
        """Test ReasoningNetwork with working memory."""
        config = AgentConfig(
            agent_type=AgentType.REASONING,
            input_dim=64,
            hidden_dim=32,
            output_dim=16,
            message_dim=16,
        )
        network = ReasoningNetwork(config)

        batch_size = 2
        inputs = torch.randn(batch_size, config.input_dim)
        messages = torch.randn(batch_size, config.message_dim)
        state = torch.zeros(batch_size, config.state_dim)

        output, new_state = network(inputs, messages, state)

        assert output.shape == (batch_size, config.output_dim)
        assert new_state.shape == (batch_size, config.state_dim)
        assert not torch.isnan(output).any()

    def test_memory_network(self):
        """Test MemoryNetwork with key-value memory."""
        config = AgentConfig(
            agent_type=AgentType.MEMORY,
            input_dim=64,
            hidden_dim=32,
            output_dim=16,
            message_dim=16,
        )
        network = MemoryNetwork(config)

        batch_size = 2
        inputs = torch.randn(batch_size, config.input_dim)
        messages = torch.randn(batch_size, config.message_dim)
        state = torch.zeros(batch_size, config.state_dim)

        output, new_state = network(inputs, messages, state)

        assert output.shape == (batch_size, config.output_dim)
        assert new_state.shape == (batch_size, config.state_dim)
        assert not torch.isnan(output).any()

    def test_planning_network(self):
        """Test PlanningNetwork with goal conditioning."""
        config = AgentConfig(
            agent_type=AgentType.PLANNING,
            input_dim=64,
            hidden_dim=32,
            output_dim=16,
            message_dim=16,
        )
        network = PlanningNetwork(config)

        batch_size = 2
        inputs = torch.randn(batch_size, config.input_dim)
        messages = torch.randn(batch_size, config.message_dim)
        goal = torch.randn(batch_size, config.state_dim)

        output, new_goal = network(inputs, messages, goal)

        assert output.shape == (batch_size, config.output_dim)
        assert new_goal.shape == (batch_size, config.state_dim)
        assert not torch.isnan(output).any()

    def test_specialized_agent_creation(self):
        """Test SpecializedAgent factory creates correct networks."""
        for agent_type in [AgentType.PERCEPTION, AgentType.REASONING,
                          AgentType.MEMORY, AgentType.PLANNING]:
            config = AgentConfig(agent_type=agent_type)
            agent = SpecializedAgent(agent_id=0, config=config)

            # Check correct network type
            if agent_type == AgentType.PERCEPTION:
                assert isinstance(agent.network, PerceptionNetwork)
            elif agent_type == AgentType.REASONING:
                assert isinstance(agent.network, ReasoningNetwork)
            elif agent_type == AgentType.MEMORY:
                assert isinstance(agent.network, MemoryNetwork)
            elif agent_type == AgentType.PLANNING:
                assert isinstance(agent.network, PlanningNetwork)

    def test_create_specialized_swarm_helper(self):
        """Test create_specialized_swarm helper function."""
        agents = create_specialized_swarm(
            num_perception=2,
            num_reasoning=2,
            num_memory=2,
            num_planning=2,
            hidden_dim=32,
        )

        assert len(agents) == 8
        assert sum(1 for a in agents if a.config.agent_type == AgentType.PERCEPTION) == 2
        assert sum(1 for a in agents if a.config.agent_type == AgentType.REASONING) == 2
        assert sum(1 for a in agents if a.config.agent_type == AgentType.MEMORY) == 2
        assert sum(1 for a in agents if a.config.agent_type == AgentType.PLANNING) == 2


class TestTypedMessaging:
    """Test typed message passing system."""

    def test_typed_message_creation(self):
        """Test TypedMessage dataclass."""
        content = torch.randn(16)
        msg = TypedMessage(
            content=content,
            message_type=MessageType.PERCEPT,
            source_id=0,
            source_type=AgentType.PERCEPTION,
            target_id=1,
            priority=0.8,
        )

        assert msg.message_type == MessageType.PERCEPT
        assert msg.source_type == AgentType.PERCEPTION
        assert msg.priority == 0.8
        assert not msg.is_control_message

    def test_control_message_detection(self):
        """Test control message detection."""
        control_msg = TypedMessage(
            content=torch.randn(16),
            message_type=MessageType.REWARD_SIGNAL,
            source_id=0,
            source_type=AgentType.GENERAL,
        )
        assert control_msg.is_control_message

        content_msg = TypedMessage(
            content=torch.randn(16),
            message_type=MessageType.PERCEPT,
            source_id=0,
            source_type=AgentType.PERCEPTION,
        )
        assert not content_msg.is_control_message

    def test_message_filter(self):
        """Test MessageFilter functionality."""
        filter_percepts = MessageFilter(allowed_types=[MessageType.PERCEPT])

        percept_msg = TypedMessage(
            content=torch.randn(16),
            message_type=MessageType.PERCEPT,
            source_id=0,
            source_type=AgentType.PERCEPTION,
        )
        inference_msg = TypedMessage(
            content=torch.randn(16),
            message_type=MessageType.INFERENCE,
            source_id=1,
            source_type=AgentType.REASONING,
        )

        assert filter_percepts.accepts(percept_msg)
        assert not filter_percepts.accepts(inference_msg)

    def test_typed_message_bus_registration(self):
        """Test TypedMessageBus agent registration."""
        bus = TypedMessageBus(num_agents=4, message_dim=16)

        bus.register_agent(0, AgentType.PERCEPTION)
        bus.register_agent(1, AgentType.REASONING)
        bus.register_agent(2, AgentType.MEMORY)
        bus.register_agent(3, AgentType.PLANNING)

        assert bus.agent_types[0] == AgentType.PERCEPTION
        assert bus.agent_types[3] == AgentType.PLANNING

    def test_typed_message_bus_send_receive(self):
        """Test sending and receiving typed messages."""
        bus = TypedMessageBus(num_agents=3, message_dim=8)
        bus.register_agent(0, AgentType.PERCEPTION)
        bus.register_agent(1, AgentType.REASONING)
        bus.register_agent(2, AgentType.PLANNING)

        # Send from perception to reasoning
        bus.send(
            content=torch.randn(8),
            source_id=0,
            message_type=MessageType.PERCEPT,
            target_id=1,
        )
        bus.step()

        # Reasoning should receive
        messages = bus.receive(1)
        assert len(messages) == 1
        assert messages[0].message_type == MessageType.PERCEPT

    def test_typed_broadcast(self):
        """Test broadcast with type filtering."""
        bus = TypedMessageBus(num_agents=4, message_dim=8)
        bus.register_agent(0, AgentType.PERCEPTION)
        bus.register_agent(1, AgentType.REASONING)
        bus.register_agent(2, AgentType.MEMORY)
        bus.register_agent(3, AgentType.PLANNING)

        # Broadcast percept
        bus.send(
            content=torch.randn(8),
            source_id=0,
            message_type=MessageType.PERCEPT,
        )
        bus.step()

        # Check which agents received (based on default filters)
        stats = bus.get_stats()
        assert stats["total_messages"] == 1

    def test_message_encoder(self):
        """Test MessageEncoder adds type information."""
        encoder = MessageEncoder(message_dim=16)

        content = torch.randn(2, 16)
        encoded = encoder(content, MessageType.PERCEPT)

        assert encoded.shape == (2, 16)
        assert not torch.equal(encoded, content)

    def test_message_decoder(self):
        """Test MessageDecoder predicts types."""
        decoder = MessageDecoder(message_dim=16)

        content = torch.randn(1, 16)
        logits = decoder(content)

        assert logits.shape == (1, len(MessageType))

    def test_type_aware_aggregator(self):
        """Test TypeAwareAggregator processes messages by type."""
        aggregator = TypeAwareAggregator(message_dim=16, output_dim=8)

        messages = [
            TypedMessage(
                content=torch.randn(16),
                message_type=MessageType.PERCEPT,
                source_id=0,
                source_type=AgentType.PERCEPTION,
            ),
            TypedMessage(
                content=torch.randn(16),
                message_type=MessageType.REWARD_SIGNAL,
                source_id=1,
                source_type=AgentType.GENERAL,
            ),
        ]

        output = aggregator(messages)
        assert output.shape == (1, 8)
        assert not torch.isnan(output).any()


class TestSpecializedSwarmGraph:
    """Test SpecializedSwarmGraph integration."""

    def test_specialized_swarm_creation(self):
        """Test creating a specialized swarm."""
        config = SpecializedSwarmConfig(
            num_agents=12,
            num_perception=3,
            num_reasoning=3,
            num_memory=3,
            num_planning=3,
            hidden_dim=32,
        )
        swarm = SpecializedSwarmGraph(config=config)

        assert len(swarm.agents) == 12
        assert len(swarm.input_agents) == 3  # Perception agents
        assert len(swarm.output_agents) == 3  # Planning agents

    def test_specialized_swarm_forward(self):
        """Test forward pass through specialized swarm."""
        config = SpecializedSwarmConfig(
            num_agents=8,
            num_perception=2,
            num_reasoning=2,
            num_memory=2,
            num_planning=2,
            input_dim=32,
            output_dim=16,
            hidden_dim=32,
        )
        swarm = SpecializedSwarmGraph(config=config)

        batch_size = 2
        inputs = torch.randn(batch_size, 32)

        output = swarm.step(inputs)

        assert output.shape == (batch_size, 16)
        assert not torch.isnan(output).any()

    @pytest.mark.parametrize("topology", [
        TopologyType.HIERARCHICAL,
        TopologyType.MODULAR,
        TopologyType.SMALL_WORLD,
    ])
    def test_specialized_swarm_topologies(self, topology):
        """Test specialized swarm with different topologies."""
        config = SpecializedSwarmConfig(
            num_agents=16,
            num_perception=4,
            num_reasoning=4,
            num_memory=4,
            num_planning=4,
            topology=topology,
        )
        swarm = SpecializedSwarmGraph(config=config)

        inputs = torch.randn(1, config.input_dim)
        output = swarm.step(inputs)

        assert output.shape == (1, config.output_dim)
        assert not torch.isnan(output).any()

    def test_message_stats_tracking(self):
        """Test that message statistics are tracked."""
        config = SpecializedSwarmConfig(
            num_agents=8,
            num_perception=2,
            num_reasoning=2,
            num_memory=2,
            num_planning=2,
        )
        swarm = SpecializedSwarmGraph(config=config)

        inputs = torch.randn(1, config.input_dim)
        swarm.step(inputs)

        stats = swarm.get_message_stats()
        assert "total_messages" in stats
        assert stats["total_messages"] > 0

    def test_role_emergence_tracking(self):
        """Test role emergence metrics."""
        config = SpecializedSwarmConfig(
            num_agents=8,
            num_perception=2,
            num_reasoning=2,
            num_memory=2,
            num_planning=2,
            track_role_emergence=True,
        )
        swarm = SpecializedSwarmGraph(config=config)

        # Run a few steps
        for _ in range(3):
            inputs = torch.randn(1, config.input_dim)
            swarm.step(inputs)

        metrics = swarm.get_role_emergence_metrics()
        assert "avg_specialization_entropy" in metrics
        assert "type_concentrations" in metrics

    def test_specialized_swarm_parameters(self):
        """Test parameter collection."""
        config = SpecializedSwarmConfig(
            num_agents=4,
            num_perception=1,
            num_reasoning=1,
            num_memory=1,
            num_planning=1,
            hidden_dim=32,
        )
        swarm = SpecializedSwarmGraph(config=config)

        params = list(swarm.parameters())
        total = swarm.total_parameters

        assert len(params) > 0
        assert total > 0

    def test_specialized_swarm_state_dict(self):
        """Test saving and loading state."""
        config = SpecializedSwarmConfig(
            num_agents=4,
            num_perception=1,
            num_reasoning=1,
            num_memory=1,
            num_planning=1,
        )
        swarm = SpecializedSwarmGraph(config=config)

        # Run forward to initialize states
        inputs = torch.randn(1, config.input_dim)
        output1 = swarm.step(inputs)

        # Save state
        state = swarm.state_dict()

        # Create new swarm and load
        swarm2 = SpecializedSwarmGraph(config=config)
        swarm2.load_state_dict(state)

        # Outputs should match after loading
        swarm.reset(1)
        swarm2.reset(1)

        # Set to eval mode for deterministic outputs
        for agent in swarm.agents.values():
            agent.network.eval()
        for agent in swarm2.agents.values():
            agent.network.eval()

        output1 = swarm.step(inputs)
        output2 = swarm2.step(inputs)

        assert torch.allclose(output1, output2, atol=1e-5)


class TestRoleEmergenceTracker:
    """Test RoleEmergenceTracker metrics."""

    def test_tracker_initialization(self):
        """Test tracker initializes correctly."""
        agents = create_specialized_swarm(
            num_perception=2,
            num_reasoning=2,
            num_memory=2,
            num_planning=2,
        )
        agents_dict = {a.agent_id: a for a in agents}
        tracker = RoleEmergenceTracker(agents_dict)

        assert tracker.total_updates == 0
        assert len(tracker.message_counts) == 8

    def test_tracker_update(self):
        """Test tracker updates with message stats."""
        agents = create_specialized_swarm(
            num_perception=2,
            num_reasoning=2,
            num_memory=2,
            num_planning=2,
        )
        agents_dict = {a.agent_id: a for a in agents}
        tracker = RoleEmergenceTracker(agents_dict)

        # Simulate message stats
        stats = {
            "by_type": {
                "PERCEPT": 10,
                "INFERENCE": 5,
            }
        }
        tracker.update(stats)

        assert tracker.total_updates == 1

    def test_tracker_metrics(self):
        """Test tracker computes metrics."""
        agents = create_specialized_swarm(
            num_perception=2,
            num_reasoning=2,
            num_memory=2,
            num_planning=2,
        )
        agents_dict = {a.agent_id: a for a in agents}
        tracker = RoleEmergenceTracker(agents_dict)

        # Update a few times
        for _ in range(5):
            stats = {"by_type": {"PERCEPT": 10, "INFERENCE": 5}}
            tracker.update(stats)

        metrics = tracker.get_metrics()

        assert "avg_specialization_entropy" in metrics
        assert "type_concentrations" in metrics
        assert metrics["total_updates"] == 5


class TestAgentMessageTypes:
    """Test agent-message type mappings."""

    def test_agent_message_types_defined(self):
        """Test AGENT_MESSAGE_TYPES has entries for all agent types."""
        for agent_type in [AgentType.PERCEPTION, AgentType.REASONING,
                          AgentType.MEMORY, AgentType.PLANNING]:
            assert agent_type in AGENT_MESSAGE_TYPES
            assert len(AGENT_MESSAGE_TYPES[agent_type]) > 0

    def test_perception_sends_percepts(self):
        """Test perception agents send perceptual message types."""
        types = AGENT_MESSAGE_TYPES[AgentType.PERCEPTION]
        assert MessageType.PERCEPT in types

    def test_reasoning_sends_inferences(self):
        """Test reasoning agents send inference types."""
        types = AGENT_MESSAGE_TYPES[AgentType.REASONING]
        assert MessageType.INFERENCE in types

    def test_memory_sends_recalls(self):
        """Test memory agents send recall types."""
        types = AGENT_MESSAGE_TYPES[AgentType.MEMORY]
        assert MessageType.MEMORY_RECALL in types

    def test_planning_sends_goals(self):
        """Test planning agents send goal types."""
        types = AGENT_MESSAGE_TYPES[AgentType.PLANNING]
        assert MessageType.GOAL in types or MessageType.ACTION_PROPOSAL in types
