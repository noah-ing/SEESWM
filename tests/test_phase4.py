"""
Tests for Phase 4: Self-Model and Meta-Cognition.

Tests uncertainty, calibration, meta-learning, and theory of mind.
"""

import pytest
import torch
import torch.nn as nn

from src.self_model.uncertainty import (
    UncertaintyEstimate,
    EnsembleUncertainty,
    MCDropoutUncertainty,
    EvidentialNetwork,
    UncertaintyAggregator,
    UncertaintyHead,
)
from src.self_model.calibration import (
    compute_calibration_metrics,
    TemperatureScaling,
    PlattScaling,
    FocalLoss,
    LabelSmoothing,
    CalibrationLoss,
    CalibrationTracker,
)
from src.self_model.meta_learning import (
    Task,
    MetaLearningConfig,
    MAML,
    MetaSGD,
    Reptile,
    TaskEmbedding,
)
from src.self_model.theory_of_mind import (
    IntentType,
    BeliefState,
    BeliefEncoder,
    IntentPredictor,
    TheoryOfMind,
    CollectiveBeliefAggregator,
)
from src.self_model.swarm_integration import (
    MetaCognitiveSwarm,
    MetaCognitionTrainer,
)
from src.swarm.specialized_graph import SpecializedSwarmGraph, SpecializedSwarmConfig


class TestUncertainty:
    """Test uncertainty quantification methods."""

    def test_uncertainty_estimate(self):
        """Test UncertaintyEstimate dataclass."""
        estimate = UncertaintyEstimate(
            prediction=torch.randn(10),
            aleatoric=0.1,
            epistemic=0.2,
            total=0.3,
            confidence=0.7,
        )
        assert estimate.total == 0.3
        assert not estimate.should_abstain(threshold=0.5)
        assert estimate.should_abstain(threshold=0.8)

    def test_ensemble_uncertainty(self):
        """Test EnsembleUncertainty."""
        def make_network():
            return nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 5))

        ensemble = EnsembleUncertainty(make_network, num_members=3)

        inputs = torch.randn(4, 32)
        output, estimate = ensemble(inputs)

        assert output.shape == (4, 5)
        assert estimate.epistemic >= 0
        assert 0 <= estimate.confidence <= 1

    def test_mc_dropout_uncertainty(self):
        """Test Monte Carlo Dropout uncertainty."""
        network = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 5))
        mc = MCDropoutUncertainty(network, num_samples=5)

        inputs = torch.randn(4, 32)
        output, estimate = mc(inputs)

        assert output.shape == (4, 5)
        assert estimate.epistemic >= 0

    def test_evidential_network(self):
        """Test EvidentialNetwork."""
        network = EvidentialNetwork(input_dim=32, num_classes=5)

        inputs = torch.randn(4, 32)
        probs, estimate = network(inputs)

        assert probs.shape == (4, 5)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)
        assert estimate.aleatoric >= 0
        assert estimate.epistemic >= 0

    def test_evidential_loss(self):
        """Test EvidentialNetwork loss computation."""
        network = EvidentialNetwork(input_dim=32, num_classes=5)

        inputs = torch.randn(4, 32)
        targets = torch.randint(0, 5, (4,))

        loss = network.compute_loss(inputs, targets)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_uncertainty_aggregator(self):
        """Test UncertaintyAggregator."""
        aggregator = UncertaintyAggregator()

        estimates = {
            "ensemble": UncertaintyEstimate(
                prediction=torch.randn(10),
                epistemic=0.2,
                aleatoric=0.1,
            ),
            "mcdropout": UncertaintyEstimate(
                prediction=torch.randn(10),
                epistemic=0.3,
                aleatoric=0.05,
            ),
        }

        combined = aggregator.aggregate(estimates)
        assert combined.total >= 0

    def test_uncertainty_head(self):
        """Test UncertaintyHead."""
        head = UncertaintyHead(feature_dim=64)

        features = torch.randn(4, 64)
        aleatoric, epistemic = head(features)

        assert aleatoric >= 0
        assert epistemic >= 0


class TestCalibration:
    """Test calibration methods."""

    def test_calibration_metrics(self):
        """Test compute_calibration_metrics."""
        logits = torch.randn(100, 5)
        targets = torch.randint(0, 5, (100,))

        metrics = compute_calibration_metrics(logits, targets)

        assert 0 <= metrics.ece <= 1
        assert 0 <= metrics.mce <= 1
        assert metrics.brier >= 0
        assert len(metrics.bin_accuracies) == 15

    def test_temperature_scaling(self):
        """Test TemperatureScaling."""
        temp_scaler = TemperatureScaling()

        logits = torch.randn(50, 5)
        targets = torch.randint(0, 5, (50,))

        # Fit temperature
        temp = temp_scaler.fit(logits, targets)
        assert temp > 0

        # Apply scaling
        scaled = temp_scaler(logits)
        assert scaled.shape == logits.shape

    def test_platt_scaling(self):
        """Test PlattScaling."""
        scaler = PlattScaling(num_classes=5)

        logits = torch.randn(50, 5)
        targets = torch.randint(0, 5, (50,))

        result = scaler.fit(logits, targets)
        assert "final_nll" in result

        scaled = scaler(logits)
        assert scaled.shape == logits.shape

    def test_focal_loss(self):
        """Test FocalLoss."""
        focal = FocalLoss(gamma=2.0)

        logits = torch.randn(16, 5)
        targets = torch.randint(0, 5, (16,))

        loss = focal(logits, targets)
        assert loss.item() >= 0

    def test_label_smoothing(self):
        """Test LabelSmoothing."""
        smoother = LabelSmoothing(smoothing=0.1, num_classes=5)

        logits = torch.randn(16, 5)
        targets = torch.randint(0, 5, (16,))

        loss = smoother(logits, targets)
        assert loss.item() >= 0

    def test_calibration_loss(self):
        """Test CalibrationLoss."""
        cal_loss = CalibrationLoss(calibration_weight=0.1)

        logits = torch.randn(16, 5)
        targets = torch.randint(0, 5, (16,))

        total_loss, details = cal_loss(logits, targets)
        assert "task_loss" in details
        assert "calibration_loss" in details

    def test_calibration_tracker(self):
        """Test CalibrationTracker."""
        tracker = CalibrationTracker()

        for _ in range(10):
            logits = torch.randn(20, 5)
            targets = torch.randint(0, 5, (20,))
            tracker.update(logits, targets)

        summary = tracker.get_summary()
        assert "avg_ece" in summary
        assert tracker.total_updates == 10


class TestMetaLearning:
    """Test meta-learning methods."""

    def test_task_creation(self):
        """Test Task dataclass."""
        task = Task(
            support_x=torch.randn(5, 32),
            support_y=torch.randint(0, 5, (5,)),
            query_x=torch.randn(10, 32),
            query_y=torch.randint(0, 5, (10,)),
        )
        assert task.support_x.shape[0] == 5
        assert task.query_x.shape[0] == 10

    def test_maml_adapt(self):
        """Test MAML adaptation."""
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 5))
        maml = MAML(model, MetaLearningConfig(inner_steps=2))

        support_x = torch.randn(5, 32)
        support_y = torch.randint(0, 5, (5,))

        adapted = maml.adapt(support_x, support_y)
        assert adapted is not model  # Should be a copy

        # Check adapted makes different predictions
        inputs = torch.randn(3, 32)
        with torch.no_grad():
            orig_out = model(inputs)
            adapted_out = adapted(inputs)
        # Outputs should differ after adaptation
        assert not torch.allclose(orig_out, adapted_out)

    def test_maml_meta_train(self):
        """Test MAML meta-training step."""
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 5))
        config = MetaLearningConfig(inner_steps=1, first_order=True)
        maml = MAML(model, config)

        tasks = [
            Task(
                support_x=torch.randn(5, 32),
                support_y=torch.randint(0, 5, (5,)),
                query_x=torch.randn(5, 32),
                query_y=torch.randint(0, 5, (5,)),
            )
            for _ in range(2)
        ]

        stats = maml.meta_train_step(tasks)
        assert "query_loss" in stats
        assert "query_accuracy" in stats

    def test_reptile(self):
        """Test Reptile meta-learning."""
        model = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 5))
        reptile = Reptile(model, inner_steps=2)

        tasks = [
            Task(
                support_x=torch.randn(5, 32),
                support_y=torch.randint(0, 5, (5,)),
                query_x=torch.randn(5, 32),
                query_y=torch.randint(0, 5, (5,)),
            )
            for _ in range(2)
        ]

        stats = reptile.meta_train_step(tasks)
        assert "query_loss" in stats

    def test_task_embedding(self):
        """Test TaskEmbedding."""
        embedder = TaskEmbedding(input_dim=32, embedding_dim=16)

        support = torch.randn(5, 32)
        embedding = embedder(support)

        assert embedding.shape == (16,)

        # Add to bank and find similar
        embedder.add_to_bank(embedding, task_id=0)
        embedder.add_to_bank(torch.randn(16), task_id=1)

        similar = embedder.find_similar_tasks(embedding, top_k=2)
        assert len(similar) == 2
        assert similar[0][0] == 0  # Most similar should be itself


class TestTheoryOfMind:
    """Test Theory of Mind components."""

    def test_belief_encoder(self):
        """Test BeliefEncoder."""
        encoder = BeliefEncoder(observation_dim=32, belief_dim=16)

        obs = torch.randn(32)
        belief = encoder(obs)

        assert belief.world_belief.shape[-1] == 16
        assert belief.self_belief.shape[-1] == 16
        assert 0 <= belief.confidence <= 1

    def test_intent_predictor(self):
        """Test IntentPredictor."""
        predictor = IntentPredictor(action_dim=5)

        actions = torch.randint(0, 5, (10,))
        intent, confidence, logits = predictor(actions)

        assert isinstance(intent, IntentType)
        assert 0 <= confidence <= 1
        assert logits.shape[-1] == len(IntentType)

    def test_theory_of_mind_beliefs(self):
        """Test TheoryOfMind belief updates."""
        tom = TheoryOfMind(observation_dim=32, action_dim=5)

        obs = torch.randn(32)
        belief = tom.update_belief(agent_id=0, observation=obs, timestamp=1)

        assert belief.world_belief is not None
        assert tom.agent_models[0].belief_state is not None

    def test_theory_of_mind_intents(self):
        """Test TheoryOfMind intent prediction."""
        tom = TheoryOfMind(observation_dim=32, action_dim=5)

        # Add some actions
        for i in range(10):
            tom.update_intent(agent_id=0, action=i % 5)

        model = tom.agent_models[0]
        assert len(model.action_history) == 10
        assert model.predicted_intent != IntentType.UNKNOWN

    def test_perspective_taking(self):
        """Test perspective taking."""
        tom = TheoryOfMind(observation_dim=32, action_dim=5)

        # Setup belief
        tom.update_belief(agent_id=0, observation=torch.randn(32))

        our_obs = torch.randn(32)
        their_view = tom.take_perspective(agent_id=0, our_observation=our_obs)

        assert their_view.shape == our_obs.shape

    def test_trust_update(self):
        """Test trust score updates."""
        tom = TheoryOfMind(observation_dim=32, action_dim=5)

        # Initially 0.5
        assert tom.get_or_create_model(0).trust_score == 0.5

        # Update with correct predictions
        for _ in range(10):
            tom.update_trust(agent_id=0, prediction_correct=True)

        assert tom.agent_models[0].trust_score > 0.5

    def test_collective_belief_aggregator(self):
        """Test CollectiveBeliefAggregator."""
        aggregator = CollectiveBeliefAggregator(belief_dim=16)

        beliefs = [
            BeliefState(
                world_belief=torch.randn(16),
                self_belief=torch.randn(16),
                confidence=0.8,
            )
            for _ in range(5)
        ]

        collective = aggregator(beliefs)
        assert collective.shape == (16,)


class TestSwarmIntegration:
    """Test self-model integration with swarm."""

    def test_metacognitive_swarm_creation(self):
        """Test MetaCognitiveSwarm creation."""
        swarm_config = SpecializedSwarmConfig(
            num_agents=8,
            num_perception=2,
            num_reasoning=2,
            num_memory=2,
            num_planning=2,
            hidden_dim=32,
            input_dim=32,
            output_dim=16,
        )
        swarm = SpecializedSwarmGraph(config=swarm_config)
        metacog = MetaCognitiveSwarm(swarm, enable_tom=True)

        assert len(metacog.agent_metacog) == 8
        assert metacog.enable_tom

    def test_metacognitive_forward(self):
        """Test forward pass with meta-cognition."""
        swarm_config = SpecializedSwarmConfig(
            num_agents=8,
            num_perception=2,
            num_reasoning=2,
            num_memory=2,
            num_planning=2,
            hidden_dim=32,
            input_dim=32,
            output_dim=16,
        )
        swarm = SpecializedSwarmGraph(config=swarm_config)
        metacog = MetaCognitiveSwarm(swarm)

        inputs = torch.randn(4, 32)
        output, meta_info = metacog(inputs, return_uncertainty=True)

        assert output.shape == (4, 16)
        assert "agent_uncertainties" in meta_info
        assert "collective_uncertainty" in meta_info

    def test_should_abstain(self):
        """Test abstention decision."""
        swarm_config = SpecializedSwarmConfig(
            num_agents=4,
            num_perception=1,
            num_reasoning=1,
            num_memory=1,
            num_planning=1,
            hidden_dim=32,
            input_dim=32,
            output_dim=16,
        )
        swarm = SpecializedSwarmGraph(config=swarm_config)
        metacog = MetaCognitiveSwarm(swarm)

        # Run forward to get uncertainty
        inputs = torch.randn(2, 32)
        metacog(inputs, return_uncertainty=True)

        should_abstain, reasons = metacog.should_abstain()
        assert isinstance(should_abstain, bool)
        assert isinstance(reasons, dict)

    def test_domain_experts(self):
        """Test finding domain experts."""
        swarm_config = SpecializedSwarmConfig(
            num_agents=8,
            num_perception=2,
            num_reasoning=2,
            num_memory=2,
            num_planning=2,
        )
        swarm = SpecializedSwarmGraph(config=swarm_config)
        metacog = MetaCognitiveSwarm(swarm, num_domains=5)

        # Update some beliefs
        for agent_id in range(8):
            metacog.agent_metacog[agent_id].self_model.update_beliefs(
                domain=0, outcome=0.5 + 0.1 * agent_id
            )

        experts = metacog.get_domain_experts(domain=0, top_k=3)
        assert len(experts) == 3
        # Higher agent IDs should be experts (they have higher outcomes)
        assert experts[0] >= 5

    def test_metacog_summary(self):
        """Test meta-cognitive summary."""
        swarm_config = SpecializedSwarmConfig(
            num_agents=4,
            num_perception=1,
            num_reasoning=1,
            num_memory=1,
            num_planning=1,
        )
        swarm = SpecializedSwarmGraph(config=swarm_config)
        metacog = MetaCognitiveSwarm(swarm, enable_tom=True)

        summary = metacog.get_swarm_metacog_summary()

        assert "collective_uncertainty" in summary
        assert "type_stats" in summary
        assert "tom_enabled" in summary

    def test_metacognition_trainer(self):
        """Test MetaCognitionTrainer."""
        swarm_config = SpecializedSwarmConfig(
            num_agents=4,
            num_perception=1,
            num_reasoning=1,
            num_memory=1,
            num_planning=1,
            hidden_dim=32,
            input_dim=32,
            output_dim=16,
        )
        swarm = SpecializedSwarmGraph(config=swarm_config)
        metacog = MetaCognitiveSwarm(swarm)
        trainer = MetaCognitionTrainer(metacog)

        inputs = torch.randn(8, 32)
        targets = torch.randint(0, 10, (8,))

        loss = trainer.train_uncertainty(inputs, targets)
        assert loss >= 0

        stats = trainer.get_stats()
        assert "avg_uncertainty_error" in stats
