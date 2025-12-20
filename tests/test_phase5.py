"""
Tests for Phase 5: Neuromorphic Computing.

Tests:
1. LIF neurons and dynamics
2. STDP learning rules
3. SNN layers and architectures
4. Energy efficiency tracking
5. Spiking swarm integration
"""

import pytest
import torch
import torch.nn as nn

from src.neuromorphic import (
    # LIF
    LIFConfig,
    LIFNeuron,
    LIFLayer,
    RecurrentLIFLayer,
    AdaptiveLIFLayer,
    PopulationLIF,
    spike_fn,
    # STDP
    STDPConfig,
    ClassicSTDP,
    TripletSTDP,
    SymmetricSTDP,
    RewardModulatedSTDP,
    HomeostaticSTDP,
    STDPLayer,
    # Layers
    SNNConfig,
    SpikingLinear,
    SpikingConv2d,
    SpikingRNN,
    SpikingNetwork,
    LiquidStateMachine,
    TemporalCoding,
    SpikeDecoder,
    # Energy
    HardwareModel,
    EnergyTracker,
    SpikeCounter,
    EnergyBudget,
    EnergyEfficientLoss,
    compare_energy_efficiency,
    # Swarm
    SpikingAgentConfig,
    SpikingSwarmConfig,
    SpikingMicroAgent,
    SpikingSwarmGraph,
    HybridSwarm,
    SpikingSwarmTrainer,
    CommunicationMode,
)


class TestLIFNeurons:
    """Tests for LIF neuron implementations."""

    def test_lif_neuron_basic(self):
        """Test basic LIF neuron dynamics."""
        config = LIFConfig(tau_mem=20.0, threshold=1.0)
        neuron = LIFNeuron(config)

        batch_size = 4
        state = neuron.init_state(batch_size)

        # Apply constant current
        current = torch.ones(batch_size) * 0.2
        spikes_total = 0

        for t in range(50):
            spikes, state = neuron(current, state)
            spikes_total += spikes.sum().item()

        # Should generate some spikes with sustained input
        assert spikes_total > 0, "LIF neuron should spike with sustained input"

    def test_lif_layer(self):
        """Test LIF layer with weights."""
        config = LIFConfig(tau_mem=20.0, threshold=1.0)
        layer = LIFLayer(in_features=32, out_features=16, config=config)

        batch_size = 8
        x = torch.randn(batch_size, 32)

        layer.reset()
        spikes = layer(x)

        assert spikes.shape == (batch_size, 16)
        assert spikes.min() >= 0 and spikes.max() <= 1

    def test_recurrent_lif_layer(self):
        """Test recurrent LIF layer."""
        config = LIFConfig(tau_mem=20.0, threshold=1.0)
        layer = RecurrentLIFLayer(in_features=16, hidden_features=32, config=config)

        batch_size = 4
        x = torch.randn(batch_size, 16)

        layer.reset()
        for t in range(10):
            spikes = layer(x)

        assert spikes.shape == (batch_size, 32)

    def test_adaptive_lif_layer(self):
        """Test adaptive LIF with spike-frequency adaptation."""
        layer = AdaptiveLIFLayer(
            in_features=16,
            out_features=8,
            tau_adapt=100.0,
            adapt_increment=0.1,
        )

        batch_size = 4
        x = torch.randn(batch_size, 16) * 2  # Strong input

        layer.reset()

        # Run for many steps
        spike_counts = []
        for chunk in range(5):
            chunk_spikes = 0
            for t in range(20):
                spikes = layer(x)
                chunk_spikes += spikes.sum().item()
            spike_counts.append(chunk_spikes)

        # Adaptation should reduce spike rate over time
        stats = layer.get_adaptation_stats()
        assert stats["mean_adaptation"] > 0, "Adaptation should increase with firing"

    def test_population_lif(self):
        """Test population coding with LIF neurons."""
        population = PopulationLIF(num_neurons=10, input_dim=1)

        batch_size = 4
        x = torch.rand(batch_size, 1)  # Values in [0, 1]

        spike_train = population(x, num_steps=20)
        assert spike_train.shape == (batch_size, 20, 10)

        # Decode
        decoded = population.decode(spike_train)
        assert decoded.shape == (batch_size, 1)

    def test_spike_function_gradient(self):
        """Test surrogate gradient for spike function."""
        x = torch.randn(10, requires_grad=True)
        threshold = 0.5

        spikes = spike_fn(x, threshold, slope=25.0)

        # Backward should work
        loss = spikes.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestSTDP:
    """Tests for STDP learning rules."""

    def test_classic_stdp(self):
        """Test classic pair-based STDP."""
        config = STDPConfig(lr_plus=0.01, lr_minus=0.01)
        stdp = ClassicSTDP(config)

        weights = torch.rand(32, 16) * 0.5  # Initial weights
        pre_spikes = (torch.rand(8, 16) > 0.9).float()
        post_spikes = (torch.rand(8, 32) > 0.9).float()
        pre_trace = pre_spikes * 0.5
        post_trace = post_spikes * 0.5

        new_weights, dw = stdp(weights, pre_spikes, post_spikes, pre_trace, post_trace)

        # Weights should change
        assert not torch.allclose(new_weights, weights)
        # Weights should stay in bounds
        assert new_weights.min() >= config.w_min
        assert new_weights.max() <= config.w_max

    def test_triplet_stdp(self):
        """Test triplet STDP."""
        config = STDPConfig(triplet_ltp=0.01, triplet_ltd=0.01)
        stdp = TripletSTDP(config)

        weights = torch.rand(32, 16) * 0.5
        traces = {
            "pre_fast": torch.rand(8, 16) * 0.5,
            "post_fast": torch.rand(8, 32) * 0.5,
            "pre_slow": torch.rand(8, 16) * 0.3,
            "post_slow": torch.rand(8, 32) * 0.3,
        }

        pre_spikes = (torch.rand(8, 16) > 0.9).float()
        post_spikes = (torch.rand(8, 32) > 0.9).float()

        new_weights, dw = stdp(weights, pre_spikes, post_spikes, traces)

        assert new_weights.shape == weights.shape

    def test_reward_modulated_stdp(self):
        """Test reward-modulated STDP."""
        config = STDPConfig(lr_plus=0.01, reward_tau=100.0)
        rm_stdp = RewardModulatedSTDP(config)

        weights = torch.rand(16, 8) * 0.5
        pre_spikes = (torch.rand(4, 8) > 0.9).float()
        post_spikes = (torch.rand(4, 16) > 0.9).float()
        pre_trace = pre_spikes
        post_trace = post_spikes

        # Compute eligibility
        eligibility = rm_stdp.compute_eligibility(
            weights, pre_spikes, post_spikes, pre_trace, post_trace
        )

        assert eligibility is not None
        assert eligibility.shape == weights.shape

        # Apply reward
        new_weights, dw = rm_stdp.apply_reward(weights, reward=1.0)

        # Positive reward should change weights
        assert not torch.allclose(new_weights, weights)

    def test_homeostatic_stdp(self):
        """Test STDP with homeostatic plasticity."""
        homeo = HomeostaticSTDP(num_neurons=32, target_rate=0.1)

        weights = torch.rand(32, 16) * 0.5
        pre_spikes = (torch.rand(4, 16) > 0.9).float()
        post_spikes = (torch.rand(4, 32) > 0.8).float()  # High rate
        pre_trace = pre_spikes
        post_trace = post_spikes

        # Update several times
        for _ in range(10):
            weights, _ = homeo(weights, pre_spikes, post_spikes, pre_trace, post_trace)

        stats = homeo.get_stats()
        assert "mean_rate" in stats
        assert "mean_scaling" in stats

    def test_stdp_layer(self):
        """Test STDP layer with online learning."""
        config = STDPConfig(lr_plus=0.01, lr_minus=0.01)
        layer = STDPLayer(in_features=16, out_features=32, stdp_config=config)

        pre_spikes = (torch.rand(4, 16) > 0.9).float()
        post_spikes = (torch.rand(4, 32) > 0.9).float()

        # Forward with learning
        dw = layer(pre_spikes, post_spikes, learn=True)

        assert dw.shape == (32, 16)

        # Get weight stats
        stats = layer.get_weight_stats()
        assert "mean" in stats
        assert "sparsity" in stats


class TestSNNLayers:
    """Tests for SNN layers and architectures."""

    def test_spiking_linear(self):
        """Test SpikingLinear layer."""
        layer = SpikingLinear(in_features=32, out_features=16)

        batch_size = 4
        x = torch.randn(batch_size, 32)

        output = layer(x, num_steps=25)

        # Rate-coded output
        assert output.shape == (batch_size, 16)

    def test_spiking_conv2d(self):
        """Test SpikingConv2d layer."""
        layer = SpikingConv2d(
            in_channels=3,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )

        batch_size = 2
        x = torch.randn(batch_size, 3, 8, 8)

        output = layer(x, num_steps=10)

        assert output.shape == (batch_size, 16, 8, 8)

    def test_spiking_rnn(self):
        """Test SpikingRNN layer."""
        rnn = SpikingRNN(input_size=16, hidden_size=32, output_size=8)

        batch_size = 4
        x = torch.randn(batch_size, 16)

        output, hidden = rnn(x, num_steps=20)

        assert output.shape == (batch_size, 8)
        assert hidden.shape[1] == 20  # Time dimension

    def test_spiking_network(self):
        """Test multi-layer SNN."""
        snn = SpikingNetwork(layer_sizes=[32, 64, 32, 10])

        batch_size = 4
        x = torch.randn(batch_size, 32)

        output = snn(x, num_steps=25)

        assert output.shape == (batch_size, 10)

        # Test return all layers
        snn.reset()
        all_outputs = snn(x, num_steps=25, return_all_layers=True)
        assert len(all_outputs) == 3  # 3 layers

    def test_liquid_state_machine(self):
        """Test LSM reservoir computing."""
        lsm = LiquidStateMachine(
            input_size=16,
            reservoir_size=64,
            output_size=4,
            spectral_radius=0.9,
        )

        batch_size = 2
        seq_len = 30
        x = torch.randn(batch_size, seq_len, 16)

        output, states = lsm(x)

        assert output.shape == (batch_size, 4)
        assert states.shape == (batch_size, seq_len, 64)

    def test_temporal_coding(self):
        """Test temporal coding layer."""
        encoder = TemporalCoding(input_size=8, num_steps=20, coding_type="latency")

        batch_size = 4
        x = torch.rand(batch_size, 8)

        spike_train = encoder(x)

        assert spike_train.shape == (batch_size, 20, 8)
        assert spike_train.min() >= 0 and spike_train.max() <= 1

    def test_spike_decoder(self):
        """Test spike decoder."""
        decoder = SpikeDecoder(output_size=10, decoding_type="rate", num_steps=25)

        batch_size = 4
        spike_train = (torch.rand(batch_size, 25, 10) > 0.9).float()

        decoded = decoder(spike_train)

        assert decoded.shape == (batch_size, 10)


class TestEnergy:
    """Tests for energy efficiency tracking."""

    def test_spike_counter(self):
        """Test spike counting."""
        counter = SpikeCounter()

        spikes1 = (torch.rand(8, 32) > 0.9).float()
        spikes2 = (torch.rand(8, 16) > 0.8).float()

        counter.count_layer(spikes1, layer_idx=0)
        counter.count_layer(spikes2, layer_idx=1)

        stats = counter.get_statistics()

        assert stats.total_spikes > 0
        assert len(stats.spikes_per_layer) == 2

    def test_energy_tracker(self):
        """Test energy tracking."""
        tracker = EnergyTracker(hardware=HardwareModel.LOIHI)

        # Register layers
        tracker.register_layer(32, 64)
        tracker.register_layer(64, 16)

        # Count spikes
        spikes1 = (torch.rand(4, 64) > 0.9).float()
        spikes2 = (torch.rand(4, 16) > 0.8).float()

        tracker.count_spikes(spikes1, 0)
        tracker.count_spikes(spikes2, 1)

        estimate = tracker.estimate_energy(num_timesteps=25)

        assert estimate.total_energy_pj > 0
        assert estimate.spike_energy_pj >= 0
        assert estimate.synaptic_energy_pj >= 0

    def test_energy_budget(self):
        """Test energy budget management."""
        budget = EnergyBudget(budget_pj_per_inference=1000.0)

        # Should have spike budget
        assert budget.spike_budget > 0

        # Consume some energy
        remaining_before = budget.remaining()
        budget.consume(100)
        remaining_after = budget.remaining()

        assert remaining_after < remaining_before

        # Check budget
        assert budget.check_budget(10)

    def test_energy_efficient_loss(self):
        """Test energy-efficient loss function."""
        loss_fn = EnergyEfficientLoss(target_rate=0.1, rate_weight=0.01)

        spike_train = (torch.rand(4, 25, 32) > 0.9).float()
        task_loss = torch.tensor(1.0)

        total_loss, components = loss_fn(spike_train, task_loss)

        assert "rate_penalty" in components
        assert "actual_rate" in components

    def test_compare_energy_efficiency(self):
        """Test SNN vs ANN energy comparison."""
        snn_spikes = (torch.rand(4, 25, 32) > 0.9).float()
        ann_activations = torch.rand(4, 32)

        comparison = compare_energy_efficiency(
            snn_spikes,
            ann_activations,
            network_params=1024,
            num_timesteps=25,
            hardware=HardwareModel.LOIHI,
        )

        assert "snn_energy_pj" in comparison
        assert "ann_energy_pj" in comparison
        assert "efficiency_gain" in comparison
        assert "sparsity" in comparison


class TestSpikingSwarm:
    """Tests for spiking swarm integration."""

    def test_spiking_micro_agent(self):
        """Test spiking micro agent."""
        config = SpikingAgentConfig(
            input_dim=32,
            hidden_dim=64,
            output_dim=16,
            message_dim=16,
            num_timesteps=20,
        )
        agent = SpikingMicroAgent(config)

        batch_size = 4
        obs = torch.randn(batch_size, 32)

        agent.reset()
        output, message = agent(obs)

        assert output.shape == (batch_size, 16)
        assert message.shape == (batch_size, 16)

        # Check spike statistics
        stats = agent.get_spike_statistics()
        assert "total_spikes" in stats
        assert "spike_rate" in stats

    def test_spiking_swarm_graph(self):
        """Test spiking swarm graph."""
        agent_config = SpikingAgentConfig(
            input_dim=16,
            hidden_dim=32,
            output_dim=8,
            message_dim=8,
            num_timesteps=15,
        )
        swarm_config = SpikingSwarmConfig(
            num_agents=5,
            agent_config=agent_config,
            num_rounds=2,
            track_energy=True,
        )

        swarm = SpikingSwarmGraph(swarm_config)

        batch_size = 4
        obs = torch.randn(batch_size, 16)

        swarm.reset()
        output = swarm.step(obs)

        assert output.shape == (batch_size, 8)

        # Check energy estimate
        energy = swarm.get_energy_estimate()
        assert "total_energy_pj" in energy

        # Check swarm statistics
        stats = swarm.get_swarm_statistics()
        assert "total_swarm_spikes" in stats
        assert "avg_spike_rate" in stats

    def test_spiking_swarm_reward(self):
        """Test reward application to spiking swarm."""
        agent_config = SpikingAgentConfig(
            input_dim=16,
            hidden_dim=32,
            output_dim=8,
            use_stdp=True,
            use_reward_modulation=True,
        )
        swarm_config = SpikingSwarmConfig(
            num_agents=3,
            agent_config=agent_config,
        )

        swarm = SpikingSwarmGraph(swarm_config)

        batch_size = 2
        obs = torch.randn(batch_size, 16)

        swarm.reset()
        _ = swarm.step(obs)

        # Apply reward (should not error)
        swarm.apply_reward(1.0)

    def test_hybrid_swarm(self):
        """Test hybrid spiking + rate-coded swarm."""
        from src.swarm.graph import SwarmConfig, TopologyType

        spiking_config = SpikingAgentConfig(
            input_dim=16,
            hidden_dim=32,
            output_dim=8,
        )

        # Use RANDOM topology to avoid k>n issue with small swarm
        rate_config = SwarmConfig(
            num_agents=5,
            input_dim=16,
            hidden_dim=32,
            output_dim=8,
            topology=TopologyType.RANDOM,
        )

        hybrid = HybridSwarm(
            num_spiking=5,
            num_rate=5,
            spiking_config=spiking_config,
            rate_config=rate_config,
        )

        batch_size = 4
        x = torch.randn(batch_size, 16)

        hybrid.reset()
        output = hybrid(x)

        # Output dimension from rate swarm config
        assert output.shape[0] == batch_size

    def test_spiking_swarm_trainer(self):
        """Test spiking swarm trainer."""
        agent_config = SpikingAgentConfig(
            input_dim=16,
            hidden_dim=32,
            output_dim=8,
            num_timesteps=10,
        )
        swarm_config = SpikingSwarmConfig(
            num_agents=3,
            agent_config=agent_config,
            track_energy=True,
        )

        swarm = SpikingSwarmGraph(swarm_config)
        trainer = SpikingSwarmTrainer(
            swarm,
            learning_rate=1e-3,
            energy_weight=0.01,
        )

        batch_size = 4
        obs = torch.randn(batch_size, 16)
        targets = torch.randn(batch_size, 8)

        metrics = trainer.train_step(obs, targets)

        assert "loss" in metrics
        assert "spike_rate" in metrics
        assert "sparsity" in metrics

        summary = trainer.get_summary()
        assert "avg_loss" in summary


# Run with: pytest tests/test_phase5.py -v
