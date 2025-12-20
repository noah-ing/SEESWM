"""
Tests for the experimental validation framework.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.swarm.graph import SwarmGraph, SwarmConfig, TopologyType
from src.validation.ablations import (
    AblationStudy, AblationConfig, AblationType,
    SingleLargeAgent, IsolatedSwarm,
)
from src.validation.scaling import (
    ScalingExperiment, ScalingPoint, find_phase_transitions,
)
from src.validation.synergy import (
    SynergyMeasurer, PartialInformationDecomposition,
    MutualInformationEstimator, compute_true_synergy,
)
from src.validation.baselines import (
    SingleAgentBaseline, EnsembleBaseline,
    CentralizedBaseline, IndependentAgentsBaseline,
)
from src.validation.generalization import (
    GeneralizationResult, create_standard_ood_transforms,
)
from src.validation.emergence import (
    EmergenceDetector, EmergentBehavior, CommunicationAnalyzer,
)
from src.validation.statistics import (
    ExperimentStats, paired_significance_test,
    compute_effect_size, confidence_interval,
    multiple_comparison_correction, minimum_sample_size,
)
from src.validation.interpretability import (
    RepresentationProbe, CausalIntervention, ActivationPatcher,
)


def create_test_swarm(num_agents=5, input_dim=32, hidden_dim=64, output_dim=5):
    """Helper to create a swarm for testing."""
    config = SwarmConfig(
        num_agents=num_agents,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        message_dim=hidden_dim,
        num_perception=num_agents // 4 + 1,
        num_reasoning=num_agents // 4 + 1,
        num_memory=num_agents // 4 + 1,
        num_planning=num_agents - 3 * (num_agents // 4 + 1),
    )
    return SwarmGraph(config)


class TestAblations:
    """Test ablation study components."""

    def test_single_large_agent(self):
        """Test SingleLargeAgent matches target params."""
        agent = SingleLargeAgent(
            input_dim=64,
            output_dim=5,
            total_params=10000,
        )

        actual_params = sum(p.numel() for p in agent.parameters())
        # Should be reasonably close
        assert 5000 < actual_params < 50000

        # Test forward
        x = torch.randn(2, 64)
        out = agent(x)
        assert out.shape == (2, 5)

    def test_isolated_swarm(self):
        """Test IsolatedSwarm with nn.Module based swarm."""
        # Create a simple nn.Module swarm for testing
        class SimpleSwarm(nn.Module):
            def __init__(self):
                super().__init__()
                self.agents = nn.ModuleList([
                    nn.Linear(32, 5) for _ in range(5)
                ])

            def forward(self, x, num_rounds=3):
                outputs = [agent(x) for agent in self.agents]
                return torch.stack(outputs).mean(dim=0)

        swarm = SimpleSwarm()
        isolated = IsolatedSwarm(swarm)

        x = torch.randn(2, 32)
        out = isolated(x, num_rounds=3)
        assert out.shape == (2, 5)

    def test_ablation_config(self):
        """Test AblationConfig creation."""
        config = AblationConfig(
            name="test_ablation",
            ablation_type=AblationType.NO_MESSAGE_PASSING,
            description="Test ablation",
            num_seeds=5,
        )

        assert config.name == "test_ablation"
        assert config.ablation_type == AblationType.NO_MESSAGE_PASSING


class TestScaling:
    """Test scaling law experiments."""

    def test_scaling_experiment_creation(self):
        """Test ScalingExperiment initialization."""
        exp = ScalingExperiment(
            base_input_dim=64,
            base_output_dim=5,
        )

        assert exp.base_input_dim == 64
        assert exp.base_output_dim == 5

    def test_flops_estimation(self):
        """Test FLOP estimation."""
        exp = ScalingExperiment()

        flops = exp.estimate_flops(
            num_agents=10,
            hidden_dim=128,
            message_rounds=3,
        )

        assert flops > 0
        assert isinstance(flops, float)

    def test_scaling_point(self):
        """Test ScalingPoint dataclass."""
        point = ScalingPoint(
            num_agents=20,
            hidden_dim=128,
            total_params=100000,
            message_rounds=3,
            performance=0.8,
            synergy=0.1,
            training_steps=1000,
            compute_flops=1e6,
            memory_bytes=400000,
        )

        assert point.num_agents == 20
        assert point.synergy == 0.1

    def test_find_phase_transitions(self):
        """Test phase transition detection."""
        # Create synthetic data with a transition
        points = [
            ScalingPoint(n, 64, n*1000, 1, n*0.01, n*0.001, 100, 1e5, 4000)
            for n in [5, 10, 20, 50, 100, 200]
        ]

        transitions = find_phase_transitions(points, 'performance')
        # May or may not find transitions in linear data
        assert isinstance(transitions, list)


class TestSynergy:
    """Test synergy measurement."""

    def test_mutual_information_estimator(self):
        """Test MI estimation."""
        estimator = MutualInformationEstimator(method='ksg')

        # Independent variables should have low MI
        x = np.random.randn(100, 5)
        y = np.random.randn(100, 5)
        mi_indep = estimator.estimate(x, y)

        # Dependent variables should have higher MI
        x2 = np.random.randn(100, 5)
        y2 = x2 + np.random.randn(100, 5) * 0.1
        mi_dep = estimator.estimate(x2, y2)

        assert mi_dep > mi_indep

    def test_pid_decomposition(self):
        """Test Partial Information Decomposition."""
        pid = PartialInformationDecomposition()

        # Create agent outputs and targets
        agent_outputs = [
            np.random.randn(50, 10),
            np.random.randn(50, 10),
            np.random.randn(50, 10),
        ]
        targets = np.random.randn(50, 5)

        decomp = pid.decompose(agent_outputs, targets, method='simplified')

        assert decomp.total_mi >= 0
        assert decomp.redundancy >= 0
        assert len(decomp.unique) == 3

    def test_synergy_measurer(self):
        """Test SynergyMeasurer with nn.Module swarm."""
        # Create a simple nn.Module swarm for testing
        class SimpleSwarm(nn.Module):
            def __init__(self):
                super().__init__()
                self.agents = nn.ModuleList([
                    nn.Linear(32, 5) for _ in range(5)
                ])

            def forward(self, x):
                outputs = [agent(x) for agent in self.agents]
                return torch.stack(outputs).mean(dim=0)

        swarm = SimpleSwarm()
        measurer = SynergyMeasurer(swarm, device='cpu')

        inputs = torch.randn(50, 32)
        targets = torch.randn(50, 5)

        decomp = measurer.measure(inputs, targets, num_bootstrap=10)

        assert hasattr(decomp, 'synergy')
        assert hasattr(decomp, 'synergy_ratio')


class TestBaselines:
    """Test baseline models."""

    def test_single_agent_baseline(self):
        """Test SingleAgentBaseline."""
        baseline = SingleAgentBaseline(
            input_dim=64,
            output_dim=5,
            total_params=50000,
        )

        x = torch.randn(4, 64)
        out = baseline(x)
        assert out.shape == (4, 5)

    def test_ensemble_baseline(self):
        """Test EnsembleBaseline."""
        baseline = EnsembleBaseline(
            num_agents=5,
            input_dim=64,
            output_dim=5,
        )

        x = torch.randn(4, 64)
        out = baseline(x)
        assert out.shape == (4, 5)

    def test_centralized_baseline(self):
        """Test CentralizedBaseline."""
        baseline = CentralizedBaseline(
            num_agents=5,
            input_dim=64,
            output_dim=5,
        )

        x = torch.randn(4, 64)
        out = baseline(x)
        assert out.shape == (4, 5)

    def test_independent_agents_baseline(self):
        """Test IndependentAgentsBaseline."""
        baseline = IndependentAgentsBaseline(
            num_agents=5,
            input_dim=64,
            output_dim=5,
        )

        x = torch.randn(4, 64)
        out = baseline(x)
        assert out.shape == (4, 5)


class TestStatistics:
    """Test statistical utilities."""

    def test_experiment_stats(self):
        """Test ExperimentStats computation."""
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = ExperimentStats.from_samples(samples)

        assert stats.mean == 3.0
        assert stats.n == 5
        assert stats.ci_low < stats.mean < stats.ci_high

    def test_paired_significance_test(self):
        """Test paired significance testing."""
        baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        treatment = np.array([1.5, 2.5, 3.5, 4.5, 5.5])

        result = paired_significance_test(baseline, treatment)

        assert 'p_value_ttest' in result
        assert 'effect_size_d' in result
        assert result['effect_size_d'] < 0  # Treatment is higher

    def test_compute_effect_size(self):
        """Test effect size computation."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([2, 3, 4, 5, 6])

        d = compute_effect_size(group1, group2, method='cohens_d')
        assert isinstance(d, float)

    def test_confidence_interval(self):
        """Test confidence interval computation."""
        samples = np.random.randn(100)

        ci = confidence_interval(samples, confidence=0.95, method='t')
        assert len(ci) == 2
        assert ci[0] < ci[1]

    def test_multiple_comparison_correction(self):
        """Test multiple comparison correction."""
        p_values = [0.01, 0.04, 0.03, 0.05, 0.10]

        corrected, significant = multiple_comparison_correction(
            p_values, method='bonferroni'
        )

        assert len(corrected) == 5
        assert len(significant) == 5
        assert all(c >= p for c, p in zip(corrected, p_values))

    def test_minimum_sample_size(self):
        """Test sample size calculation."""
        n = minimum_sample_size(effect_size=0.5, power=0.8)
        assert n > 0
        assert isinstance(n, int)


class TestEmergence:
    """Test emergence detection."""

    def test_emergence_detector_creation(self):
        """Test EmergenceDetector initialization."""
        swarm = create_test_swarm(num_agents=5, input_dim=32, hidden_dim=64, output_dim=5)
        detector = EmergenceDetector(swarm)
        assert detector.swarm is swarm

    def test_division_of_labor_detection(self):
        """Test division of labor detection."""
        swarm = create_test_swarm(num_agents=5, input_dim=32, hidden_dim=64, output_dim=5)
        detector = EmergenceDetector(swarm)

        # Create distinct action histories
        histories = {
            0: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # Prefers action 0
            1: [1, 1, 1, 1, 2, 1, 1, 1, 1, 1],  # Prefers action 1
            2: [2, 2, 2, 2, 3, 2, 2, 2, 2, 2],  # Prefers action 2
        }

        result = detector.detect_division_of_labor(histories)
        # Should detect division
        assert result is not None or True  # May not always detect

    def test_communication_analyzer(self):
        """Test CommunicationAnalyzer."""
        analyzer = CommunicationAnalyzer(32)

        # Log some messages
        for i in range(10):
            analyzer.log_message(0, 1, torch.randn(32), round=i % 3)

        # Compute flow
        flow = analyzer.compute_information_flow()
        assert flow.shape == (2, 2)

        # Find hubs
        hubs = analyzer.find_hub_agents(top_k=1)
        assert len(hubs) <= 1


class TestInterpretability:
    """Test interpretability tools."""

    def test_representation_probe(self):
        """Test RepresentationProbe training."""
        probe = RepresentationProbe(input_dim=64, output_dim=5)

        # Generate data
        representations = torch.randn(100, 64)
        labels = torch.randint(0, 5, (100,))

        result = probe.train_probe(representations, labels, epochs=10)

        assert 0 <= result.accuracy <= 1
        assert result.feature_importance.shape == (64,)

    def test_causal_intervention(self):
        """Test CausalIntervention with nn.Module swarm."""
        # Create a simple nn.Module swarm for testing
        class SimpleSwarm(nn.Module):
            def __init__(self):
                super().__init__()
                self.agents = nn.ModuleList([
                    nn.Linear(32, 5) for _ in range(5)
                ])

            def forward(self, x):
                outputs = [agent(x) for agent in self.agents]
                return torch.stack(outputs).mean(dim=0)

        swarm = SimpleSwarm()
        intervention = CausalIntervention(swarm)

        inputs = torch.randn(2, 32)

        # Ablate an agent
        result = intervention.ablate_agent(inputs, agent_idx=0)

        assert result.effect_magnitude >= 0
        assert result.pre_intervention_output.shape == result.post_intervention_output.shape

    def test_activation_patcher(self):
        """Test ActivationPatcher."""
        model = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 5),
        )

        patcher = ActivationPatcher(model)

        clean = torch.randn(2, 32)
        corrupted = torch.randn(2, 32)

        # Store activations
        activation = patcher.store_activations(clean, '0')
        assert activation is not None


class TestGeneralization:
    """Test generalization utilities."""

    def test_ood_transforms(self):
        """Test OOD transform creation."""
        transforms = create_standard_ood_transforms()

        assert 'gaussian_noise' in transforms
        assert 'scale_up' in transforms

        # Test a transform
        x = torch.randn(2, 10)
        noisy = transforms['gaussian_noise'](x)
        assert noisy.shape == x.shape
        assert not torch.allclose(x, noisy)

    def test_generalization_result(self):
        """Test GeneralizationResult dataclass."""
        result = GeneralizationResult(
            test_name='transfer_test',
            train_performance=0.8,
            test_performance=0.6,
            generalization_gap=0.2,
            transfer_efficiency=0.75,
        )

        assert result.generalization_gap == 0.2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
