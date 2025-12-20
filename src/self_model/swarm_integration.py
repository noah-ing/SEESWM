"""
Integration of self-model with specialized swarm.

Gives the swarm meta-cognitive abilities:
- Per-agent uncertainty estimation
- Swarm-level confidence calibration
- Collective belief aggregation
- Meta-learning for fast adaptation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..agents.micro_agent import AgentType
from ..swarm.specialized_graph import SpecializedSwarmGraph, SpecializedSwarmConfig
from .capabilities import SelfModel, CapabilityBelief
from .uncertainty import (
    UncertaintyEstimate,
    UncertaintyHead,
    EnsembleUncertainty,
    MCDropoutUncertainty,
)
from .calibration import (
    TemperatureScaling,
    CalibrationTracker,
    compute_calibration_metrics,
)
from .theory_of_mind import TheoryOfMind, CollectiveBeliefAggregator
from .meta_learning import MAML, MetaLearningConfig, Task


@dataclass
class AgentMetaCognition:
    """Meta-cognitive state for a single agent."""

    agent_id: int
    agent_type: AgentType

    # Self-model
    self_model: SelfModel

    # Uncertainty estimation
    uncertainty_head: Optional[UncertaintyHead] = None
    last_uncertainty: Optional[UncertaintyEstimate] = None

    # Domain-specific capabilities
    domain_capabilities: Dict[str, float] = field(default_factory=dict)

    # Recent performance
    recent_accuracies: List[float] = field(default_factory=list)
    abstention_rate: float = 0.0


class MetaCognitiveSwarm(nn.Module):
    """
    Swarm enhanced with meta-cognitive abilities.

    Wraps a SpecializedSwarmGraph and adds:
    - Per-agent self-models
    - Uncertainty estimation
    - Calibrated confidence
    - Theory of mind between agents
    - Meta-learning for adaptation
    """

    def __init__(
        self,
        swarm: SpecializedSwarmGraph,
        num_domains: int = 10,
        enable_tom: bool = True,
        enable_meta_learning: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        self.swarm = swarm
        self.device = device
        self.num_domains = num_domains

        # Per-agent meta-cognition
        self.agent_metacog: Dict[int, AgentMetaCognition] = {}
        self._init_agent_metacognition()

        # Uncertainty heads per agent type
        self.uncertainty_heads = nn.ModuleDict()
        self._init_uncertainty_heads()

        # Swarm-level calibration
        self.calibrator = TemperatureScaling()
        self.calibration_tracker = CalibrationTracker()

        # Theory of Mind (if enabled)
        self.enable_tom = enable_tom
        if enable_tom:
            self.tom = TheoryOfMind(
                observation_dim=swarm.config.input_dim,
                action_dim=5,
                belief_dim=64,
                device=device,
            )
            self.belief_aggregator = CollectiveBeliefAggregator()

        # Meta-learning (if enabled)
        self.enable_meta_learning = enable_meta_learning
        if enable_meta_learning:
            # Create a simple task network for meta-learning demo
            self.task_network = nn.Sequential(
                nn.Linear(swarm.config.input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_domains),
            )
            meta_config = MetaLearningConfig(inner_lr=0.01, inner_steps=3)
            self.meta_learner = MAML(self.task_network, meta_config)

        # Collective uncertainty
        self.collective_uncertainty = 0.0

    def _init_agent_metacognition(self) -> None:
        """Initialize meta-cognition for each agent."""
        for agent_id, agent in self.swarm.agents.items():
            self.agent_metacog[agent_id] = AgentMetaCognition(
                agent_id=agent_id,
                agent_type=agent.config.agent_type,
                self_model=SelfModel(num_domains=self.num_domains),
            )

    def _init_uncertainty_heads(self) -> None:
        """Create uncertainty heads for each agent type."""
        for agent_type in AgentType:
            self.uncertainty_heads[agent_type.name] = UncertaintyHead(
                feature_dim=self.swarm.config.hidden_dim,
            ).to(self.device)

    def forward(
        self,
        inputs: torch.Tensor,
        return_uncertainty: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with meta-cognitive augmentation.

        Returns:
            (output, meta_info) where meta_info contains uncertainty, etc.
        """
        batch_size = inputs.shape[0]

        # Standard swarm forward
        output = self.swarm.step(inputs)

        meta_info = {}

        if return_uncertainty:
            # Compute per-agent uncertainties
            agent_uncertainties = {}
            for agent_id, agent in self.swarm.agents.items():
                if agent.local_state is not None:
                    head = self.uncertainty_heads[agent.config.agent_type.name]
                    aleatoric, epistemic = head(agent.local_state)
                    agent_uncertainties[agent_id] = {
                        "aleatoric": aleatoric,
                        "epistemic": epistemic,
                        "total": aleatoric + epistemic,
                    }

                    # Update agent's metacog
                    self.agent_metacog[agent_id].last_uncertainty = UncertaintyEstimate(
                        prediction=output,
                        aleatoric=aleatoric,
                        epistemic=epistemic,
                        total=aleatoric + epistemic,
                    )

            meta_info["agent_uncertainties"] = agent_uncertainties

            # Collective uncertainty
            if agent_uncertainties:
                total_uncertainties = [u["total"] for u in agent_uncertainties.values()]
                self.collective_uncertainty = sum(total_uncertainties) / len(total_uncertainties)
                meta_info["collective_uncertainty"] = self.collective_uncertainty

        return output, meta_info

    def should_abstain(self, threshold: float = 0.5) -> Tuple[bool, Dict]:
        """
        Determine if the swarm should abstain from answering.

        Returns (should_abstain, reasons).
        """
        reasons = {}

        # Check collective uncertainty
        if self.collective_uncertainty > threshold:
            reasons["high_collective_uncertainty"] = self.collective_uncertainty

        # Check if many agents are uncertain
        uncertain_agents = 0
        for agent_id, metacog in self.agent_metacog.items():
            if metacog.last_uncertainty and metacog.last_uncertainty.confidence < 0.5:
                uncertain_agents += 1

        uncertain_ratio = uncertain_agents / len(self.agent_metacog)
        if uncertain_ratio > 0.5:
            reasons["many_uncertain_agents"] = uncertain_ratio

        return len(reasons) > 0, reasons

    def update_after_outcome(
        self,
        domain: int,
        outcome: float,
        predicted_confidence: float,
    ) -> None:
        """
        Update meta-cognitive state after observing outcome.

        Args:
            domain: Task domain
            outcome: Actual outcome (0-1)
            predicted_confidence: What we predicted
        """
        # Update each agent's self-model
        for agent_id, metacog in self.agent_metacog.items():
            metacog.self_model.record_outcome(domain, outcome, predicted_confidence)
            metacog.recent_accuracies.append(outcome)
            if len(metacog.recent_accuracies) > 100:
                metacog.recent_accuracies = metacog.recent_accuracies[-100:]

        # Update calibration tracking
        # (In real usage, we'd have logits to track)

    def get_domain_experts(self, domain: int, top_k: int = 5) -> List[int]:
        """
        Find agents best suited for a domain.

        Returns list of agent IDs with highest capability in domain.
        """
        capabilities = []
        for agent_id, metacog in self.agent_metacog.items():
            belief = metacog.self_model.capabilities.get(
                domain, CapabilityBelief()
            )
            capabilities.append((agent_id, belief.mean))

        capabilities.sort(key=lambda x: x[1], reverse=True)
        return [agent_id for agent_id, _ in capabilities[:top_k]]

    def update_tom(
        self,
        observations: Dict[int, torch.Tensor],
        actions: Dict[int, int],
    ) -> None:
        """
        Update Theory of Mind models based on observations.

        Args:
            observations: What each agent observed
            actions: What each agent did
        """
        if not self.enable_tom:
            return

        for agent_id, obs in observations.items():
            self.tom.update_belief(agent_id, obs)

        for agent_id, action in actions.items():
            self.tom.update_intent(agent_id, action)

    def get_collective_belief(self) -> Optional[torch.Tensor]:
        """Get aggregated belief across all modeled agents."""
        if not self.enable_tom:
            return None

        beliefs = [
            model.belief_state
            for model in self.tom.agent_models.values()
            if model.belief_state is not None
        ]

        if not beliefs:
            return None

        return self.belief_aggregator(beliefs)

    def meta_adapt(
        self,
        support_data: Tuple[torch.Tensor, torch.Tensor],
        num_steps: int = 5,
    ) -> nn.Module:
        """
        Quickly adapt to a new task.

        Args:
            support_data: (inputs, targets) for adaptation
            num_steps: Gradient steps for adaptation

        Returns:
            Adapted network
        """
        if not self.enable_meta_learning:
            return self.task_network

        support_x, support_y = support_data
        return self.meta_learner.adapt(support_x, support_y, num_steps)

    def get_swarm_metacog_summary(self) -> Dict:
        """Get summary of swarm meta-cognitive state."""
        # Per-type statistics
        type_stats = {}
        for agent_type in AgentType:
            agents_of_type = [
                mc for mc in self.agent_metacog.values()
                if mc.agent_type == agent_type
            ]
            if agents_of_type:
                avg_calibration = sum(
                    mc.self_model.get_calibration_error()
                    for mc in agents_of_type
                ) / len(agents_of_type)

                avg_capability = sum(
                    mc.self_model.get_stats()["avg_capability"]
                    for mc in agents_of_type
                ) / len(agents_of_type)

                type_stats[agent_type.name] = {
                    "num_agents": len(agents_of_type),
                    "avg_calibration_error": avg_calibration,
                    "avg_capability": avg_capability,
                }

        # Overall stats
        all_calibrations = [
            mc.self_model.get_calibration_error()
            for mc in self.agent_metacog.values()
        ]

        return {
            "collective_uncertainty": self.collective_uncertainty,
            "avg_calibration_error": sum(all_calibrations) / max(1, len(all_calibrations)),
            "type_stats": type_stats,
            "tom_enabled": self.enable_tom,
            "meta_learning_enabled": self.enable_meta_learning,
            "num_agents_modeled": len(self.tom.agent_models) if self.enable_tom else 0,
        }

    def parameters(self):
        """Get all trainable parameters."""
        params = list(self.swarm.parameters())
        for head in self.uncertainty_heads.values():
            params.extend(head.parameters())
        params.extend(self.calibrator.parameters())
        if self.enable_tom:
            params.extend(self.tom.parameters())
            params.extend(self.belief_aggregator.parameters())
        if self.enable_meta_learning:
            params.extend(self.task_network.parameters())
        return params


class MetaCognitionTrainer:
    """
    Trainer for meta-cognitive abilities.

    Trains:
    - Uncertainty estimation
    - Calibration
    - Theory of mind
    - Meta-learning
    """

    def __init__(
        self,
        metacog_swarm: MetaCognitiveSwarm,
        learning_rate: float = 1e-4,
    ):
        self.swarm = metacog_swarm
        self.optimizer = torch.optim.Adam(
            metacog_swarm.parameters(),
            lr=learning_rate,
        )

        # Tracking
        self.uncertainty_errors: List[float] = []
        self.calibration_errors: List[float] = []

    def train_uncertainty(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """
        Train uncertainty estimation.

        Uncertainty should be high when predictions are wrong.
        """
        self.optimizer.zero_grad()

        # Forward pass through swarm
        batch_size = inputs.shape[0]
        output = self.swarm.swarm.step(inputs)

        # Compute prediction error
        if targets.dim() == 1:
            pred_correct = (output.argmax(-1) == targets).float()
        else:
            pred_correct = 1 - (output - targets).abs().mean(dim=-1)

        # Target: high uncertainty where predictions are wrong
        target_uncertainty = 1 - pred_correct.mean()

        # Get uncertainty from heads (differentiable)
        loss = torch.tensor(0.0, device=self.swarm.device, requires_grad=True)

        for agent_id, agent in self.swarm.swarm.agents.items():
            if agent.local_state is not None:
                head = self.swarm.uncertainty_heads[agent.config.agent_type.name]
                uncertainties = head.uncertainty_net(agent.local_state)
                predicted_total = uncertainties[..., 0].mean() + uncertainties[..., 1].mean()

                # Loss: uncertainty should match error rate
                agent_loss = (predicted_total - target_uncertainty) ** 2
                loss = loss + agent_loss

        num_agents = len(self.swarm.swarm.agents)
        if num_agents > 0:
            loss = loss / num_agents

        if loss.requires_grad:
            loss.backward()
            self.optimizer.step()

        loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
        self.uncertainty_errors.append(loss_value)
        return loss_value

    def train_calibration(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Train calibration using temperature scaling."""
        return self.swarm.calibrator.fit(logits, targets)

    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            "avg_uncertainty_error": (
                sum(self.uncertainty_errors[-100:]) / max(1, len(self.uncertainty_errors[-100:]))
            ),
            "avg_calibration_error": (
                sum(self.calibration_errors[-100:]) / max(1, len(self.calibration_errors[-100:]))
            ),
            "total_updates": len(self.uncertainty_errors),
        }
