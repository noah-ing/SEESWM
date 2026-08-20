"""
Generalization Test Suite.

Tests whether the swarm generalizes to:
- Zero-shot transfer (new environments)
- Compositional tasks (novel combinations)
- Out-of-distribution scenarios
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Any
from abc import ABC, abstractmethod

from ..environment.cosmos import CosmosEnvironment


@dataclass
class GeneralizationResult:
    """Result of a generalization test."""
    test_name: str
    train_performance: float
    test_performance: float
    generalization_gap: float  # train - test
    transfer_efficiency: float  # test / train


class GeneralizationTest(ABC):
    """Base class for generalization tests."""

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        num_episodes: int = 10,
    ) -> GeneralizationResult:
        """Run the generalization test."""
        pass


class ZeroShotTransfer(GeneralizationTest):
    """
    Test transfer to novel environments without fine-tuning.

    Train: Environment A
    Test: Environment B (different physics/layout)
    """

    def __init__(
        self,
        train_env_config: Dict[str, Any],
        test_env_configs: List[Dict[str, Any]],
        episode_length: int = 100,
        device: str = "cpu",
    ):
        self.train_env_config = train_env_config
        self.test_env_configs = test_env_configs
        self.episode_length = episode_length
        self.device = device

    def evaluate(
        self,
        model: nn.Module,
        num_episodes: int = 10,
    ) -> Dict[str, GeneralizationResult]:
        """Evaluate zero-shot transfer."""
        if hasattr(model, 'to'):
            model = model.to(self.device)
        if hasattr(model, 'eval'):
            model.eval()

        results = {}

        # Evaluate on training environment
        train_env = CosmosEnvironment(**self.train_env_config)
        train_perf = self._run_episodes(model, train_env, num_episodes)

        # Evaluate on each test environment
        for i, test_config in enumerate(self.test_env_configs):
            test_env = CosmosEnvironment(**test_config)
            test_perf = self._run_episodes(model, test_env, num_episodes)

            gap = train_perf - test_perf
            efficiency = test_perf / (train_perf + 1e-10)

            results[f'transfer_{i}'] = GeneralizationResult(
                test_name=f'transfer_to_env_{i}',
                train_performance=train_perf,
                test_performance=test_perf,
                generalization_gap=gap,
                transfer_efficiency=efficiency,
            )

        return results

    def _run_episodes(
        self,
        model,
        env: CosmosEnvironment,
        num_episodes: int,
    ) -> float:
        """Run episodes and return average reward."""
        total_reward = 0.0
        is_swarm_like = hasattr(model, 'step') and hasattr(model, 'reset')

        for _ in range(num_episodes):
            obs = env.reset()
            episode_reward = 0.0

            if is_swarm_like:
                model.reset(batch_size=1)

            for _ in range(self.episode_length):
                obs_tensor = obs[0].to_tensor(self.device).unsqueeze(0)

                with torch.no_grad():
                    if is_swarm_like:
                        action_logits = model.step(obs_tensor)
                    else:
                        action_logits = model(obs_tensor)
                    if action_logits.ndim != 2 or action_logits.shape[-1] != 5:
                        raise ValueError(
                            "generalization evaluation requires a complete five-action "
                            "policy, not a latent swarm representation"
                        )
                    action = action_logits.argmax(dim=-1).item()

                # Environment returns 3 values: observations, rewards, dones
                obs, rewards, dones = env.step([action])
                reward = rewards[0] if rewards else 0.0
                done = dones[0] if dones else False
                episode_reward += reward

                if done:
                    break

            total_reward += episode_reward

        return total_reward / num_episodes


class CompositionalTest(GeneralizationTest):
    """
    Test compositional generalization.

    Train: Tasks A, B separately
    Test: Task A+B combined
    """

    def __init__(
        self,
        task_a_fn: Callable,
        task_b_fn: Callable,
        combined_fn: Callable,
        device: str = "cpu",
    ):
        """
        Args:
            task_a_fn: Function that returns (input, target) for task A
            task_b_fn: Function that returns (input, target) for task B
            combined_fn: Function that returns (input, target) for A+B
        """
        self.task_a_fn = task_a_fn
        self.task_b_fn = task_b_fn
        self.combined_fn = combined_fn
        self.device = device

    def evaluate(
        self,
        model: nn.Module,
        num_episodes: int = 100,
    ) -> GeneralizationResult:
        """Evaluate compositional generalization."""
        if hasattr(model, 'to'):
            model = model.to(self.device)
        if hasattr(model, 'eval'):
            model.eval()

        # Performance on individual tasks
        perf_a = self._evaluate_task(model, self.task_a_fn, num_episodes)
        perf_b = self._evaluate_task(model, self.task_b_fn, num_episodes)

        # Performance on combined task
        perf_combined = self._evaluate_task(model, self.combined_fn, num_episodes)

        # Expected performance if fully compositional
        expected = (perf_a + perf_b) / 2

        return GeneralizationResult(
            test_name='compositional',
            train_performance=expected,
            test_performance=perf_combined,
            generalization_gap=expected - perf_combined,
            transfer_efficiency=perf_combined / (expected + 1e-10),
        )

    def _evaluate_task(
        self,
        model: nn.Module,
        task_fn: Callable,
        num_samples: int,
    ) -> float:
        """Evaluate on a task."""
        total_correct = 0

        for _ in range(num_samples):
            inputs, targets = task_fn()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                outputs = model(inputs)

                # Assume classification
                predicted = outputs.argmax(dim=-1)
                correct = (predicted == targets).sum().item()
                total_correct += correct / targets.numel()

        return total_correct / num_samples


class OODTest(GeneralizationTest):
    """
    Out-of-distribution generalization test.

    Tests robustness to distribution shift.
    """

    def __init__(
        self,
        in_distribution_fn: Callable,
        ood_transforms: Dict[str, Callable],
        device: str = "cpu",
    ):
        """
        Args:
            in_distribution_fn: Returns (input, target) from training dist
            ood_transforms: Dict of transform name -> transform function
        """
        self.in_dist_fn = in_distribution_fn
        self.ood_transforms = ood_transforms
        self.device = device

    def evaluate(
        self,
        model: nn.Module,
        num_episodes: int = 100,
    ) -> Dict[str, GeneralizationResult]:
        """Evaluate OOD robustness."""
        if hasattr(model, 'to'):
            model = model.to(self.device)
        if hasattr(model, 'eval'):
            model.eval()

        results = {}

        # In-distribution performance
        in_dist_perf = self._evaluate_samples(model, self.in_dist_fn, num_episodes)

        # OOD performance for each transform
        for name, transform in self.ood_transforms.items():
            ood_fn = lambda: self._apply_transform(transform)
            ood_perf = self._evaluate_samples(model, ood_fn, num_episodes)

            results[name] = GeneralizationResult(
                test_name=f'ood_{name}',
                train_performance=in_dist_perf,
                test_performance=ood_perf,
                generalization_gap=in_dist_perf - ood_perf,
                transfer_efficiency=ood_perf / (in_dist_perf + 1e-10),
            )

        return results

    def _apply_transform(self, transform: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sample and apply OOD transform."""
        inputs, targets = self.in_dist_fn()
        transformed_inputs = transform(inputs)
        return transformed_inputs, targets

    def _evaluate_samples(
        self,
        model: nn.Module,
        sample_fn: Callable,
        num_samples: int,
    ) -> float:
        """Evaluate on samples."""
        total_loss = 0.0

        for _ in range(num_samples):
            inputs, targets = sample_fn()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                outputs = model(inputs)
                loss = ((outputs - targets) ** 2).mean().item()
                total_loss += loss

        return 1.0 / (1.0 + total_loss / num_samples)  # Convert to accuracy-like metric


class NoveltyTest(GeneralizationTest):
    """
    Test handling of novel inputs.

    Measures whether model appropriately signals uncertainty on new data.
    """

    def __init__(
        self,
        familiar_fn: Callable,
        novel_fn: Callable,
        device: str = "cpu",
    ):
        self.familiar_fn = familiar_fn
        self.novel_fn = novel_fn
        self.device = device

    def evaluate(
        self,
        model: nn.Module,
        num_episodes: int = 100,
    ) -> GeneralizationResult:
        """
        Evaluate novelty detection.

        Good model should have:
        - Low uncertainty on familiar inputs
        - High uncertainty on novel inputs
        """
        if hasattr(model, 'to'):
            model = model.to(self.device)
        if hasattr(model, 'eval'):
            model.eval()

        # Get uncertainty estimates
        familiar_uncertainty = self._measure_uncertainty(
            model, self.familiar_fn, num_episodes
        )
        novel_uncertainty = self._measure_uncertainty(
            model, self.novel_fn, num_episodes
        )

        # Novelty detection score: how much higher is uncertainty on novel?
        detection_score = novel_uncertainty / (familiar_uncertainty + 1e-10)

        return GeneralizationResult(
            test_name='novelty_detection',
            train_performance=1.0 / (1.0 + familiar_uncertainty),
            test_performance=detection_score,
            generalization_gap=0.0,  # Not applicable
            transfer_efficiency=detection_score,
        )

    def _measure_uncertainty(
        self,
        model: nn.Module,
        sample_fn: Callable,
        num_samples: int,
    ) -> float:
        """Measure model uncertainty on samples."""
        total_uncertainty = 0.0

        for _ in range(num_samples):
            inputs, _ = sample_fn()
            inputs = inputs.to(self.device)

            with torch.no_grad():
                outputs = model(inputs)

                # Use output variance as uncertainty proxy
                if outputs.dim() > 1:
                    uncertainty = outputs.var().item()
                else:
                    # For single outputs, use multiple forward passes
                    multiple_outputs = [model(inputs) for _ in range(5)]
                    stacked = torch.stack(multiple_outputs)
                    uncertainty = stacked.var(dim=0).mean().item()

                total_uncertainty += uncertainty

        return total_uncertainty / num_samples


def create_standard_ood_transforms() -> Dict[str, Callable]:
    """Create standard OOD transforms for testing."""
    return {
        'gaussian_noise': lambda x: x + torch.randn_like(x) * 0.5,
        'uniform_noise': lambda x: x + (torch.rand_like(x) - 0.5) * 0.5,
        'scale_up': lambda x: x * 2.0,
        'scale_down': lambda x: x * 0.5,
        'offset': lambda x: x + 1.0,
        'negate': lambda x: -x,
        'clip_high': lambda x: torch.clamp(x, max=0.5),
        'clip_low': lambda x: torch.clamp(x, min=-0.5),
        'shuffle_features': lambda x: x[:, torch.randperm(x.shape[-1])] if x.dim() > 1 else x,
        'dropout_features': lambda x: x * (torch.rand_like(x) > 0.2).float(),
    }


def run_generalization_suite(
    model: nn.Module,
    env_config: Dict[str, Any],
    device: str = "cpu",
    num_episodes: int = 10,
) -> Dict[str, GeneralizationResult]:
    """
    Run complete generalization test suite.

    Returns:
        Dictionary of test name -> result
    """
    all_results = {}

    # Zero-shot transfer tests
    test_configs = [
        {**env_config, 'grid_size': env_config.get('grid_size', 32) * 2},  # Larger
        {**env_config, 'num_resources': env_config.get('num_resources', 20) * 2},  # More resources
        {**env_config, 'num_hazards': env_config.get('num_hazards', 10) * 2},  # More hazards
    ]

    transfer_test = ZeroShotTransfer(
        train_env_config=env_config,
        test_env_configs=test_configs,
        device=device,
    )

    transfer_results = transfer_test.evaluate(model, num_episodes)
    all_results.update(transfer_results)

    return all_results
