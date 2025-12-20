"""
Theory of Mind - Modeling other agents' beliefs and intentions.

Enables agents to:
1. Model what other agents believe (belief modeling)
2. Predict what other agents will do (intent prediction)
3. Reason about how actions affect other agents (strategic reasoning)
4. Communicate more effectively (perspective taking)

Essential for coordination in multi-agent swarms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum, auto

import torch
import torch.nn as nn
import torch.nn.functional as F


class IntentType(Enum):
    """Types of agent intentions."""

    EXPLORE = auto()  # Seeking novelty
    EXPLOIT = auto()  # Maximizing known rewards
    COMMUNICATE = auto()  # Sharing information
    COOPERATE = auto()  # Working toward shared goal
    COMPETE = auto()  # Pursuing individual goal
    AVOID = auto()  # Escaping threat
    UNKNOWN = auto()  # Cannot determine


@dataclass
class BeliefState:
    """Representation of an agent's believed state."""

    # What does the agent believe about world state?
    world_belief: torch.Tensor  # Latent representation

    # What does the agent believe about its own state?
    self_belief: torch.Tensor

    # Confidence in these beliefs
    confidence: float = 0.5

    # When was this belief formed?
    timestamp: int = 0


@dataclass
class AgentModel:
    """Model of another agent."""

    agent_id: int
    agent_type: Optional[str] = None

    # Beliefs
    belief_state: Optional[BeliefState] = None

    # Predicted intent
    predicted_intent: IntentType = IntentType.UNKNOWN
    intent_confidence: float = 0.5

    # Behavioral statistics
    action_history: List[int] = field(default_factory=list)
    message_history: List[torch.Tensor] = field(default_factory=list)

    # Trust/reliability
    trust_score: float = 0.5
    prediction_accuracy: float = 0.5


class BeliefEncoder(nn.Module):
    """
    Encode observations into belief representations.

    Maps what an agent can observe into a representation
    of what it likely believes.
    """

    def __init__(
        self,
        observation_dim: int,
        belief_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.belief_dim = belief_dim

        # Observation -> belief mapping
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Separate heads for world and self beliefs
        self.world_head = nn.Linear(hidden_dim, belief_dim)
        self.self_head = nn.Linear(hidden_dim, belief_dim)

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        observation: torch.Tensor,
    ) -> BeliefState:
        """
        Infer belief state from observation.

        Args:
            observation: What an agent observes

        Returns:
            Inferred BeliefState
        """
        hidden = self.encoder(observation)

        world_belief = self.world_head(hidden)
        self_belief = self.self_head(hidden)
        confidence = self.confidence_head(hidden).item()

        return BeliefState(
            world_belief=world_belief,
            self_belief=self_belief,
            confidence=confidence,
        )


class IntentPredictor(nn.Module):
    """
    Predict agent intentions from behavior.

    Uses action history and context to predict
    what an agent is trying to achieve.
    """

    def __init__(
        self,
        action_dim: int,
        context_dim: int = 64,
        hidden_dim: int = 128,
        num_intents: int = len(IntentType),
    ):
        super().__init__()
        self.num_intents = num_intents

        # Action sequence encoder (RNN)
        self.action_embed = nn.Embedding(action_dim, 32)
        self.action_rnn = nn.GRU(32, hidden_dim, batch_first=True)

        # Context integration
        self.context_proj = nn.Linear(context_dim, hidden_dim)

        # Intent classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_intents),
        )

        # Confidence head
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        action_history: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[IntentType, float, torch.Tensor]:
        """
        Predict intent from action history.

        Args:
            action_history: Sequence of past actions [seq_len]
            context: Optional context vector [context_dim]

        Returns:
            (predicted_intent, confidence, logits)
        """
        # Embed actions
        if action_history.dim() == 1:
            action_history = action_history.unsqueeze(0)

        embedded = self.action_embed(action_history)

        # Encode sequence
        _, hidden = self.action_rnn(embedded)
        hidden = hidden.squeeze(0)  # [batch, hidden]

        # Integrate context
        if context is not None:
            if context.dim() == 1:
                context = context.unsqueeze(0)
            context_hidden = self.context_proj(context)
            combined = torch.cat([hidden, context_hidden], dim=-1)
        else:
            combined = torch.cat([hidden, torch.zeros_like(hidden)], dim=-1)

        # Classify intent
        logits = self.classifier(combined)
        intent_idx = logits.argmax(dim=-1).item()
        intent = list(IntentType)[intent_idx]

        # Confidence
        conf = self.confidence(combined).item()

        return intent, conf, logits


class TheoryOfMind(nn.Module):
    """
    Full Theory of Mind module.

    Maintains models of other agents and enables:
    - Belief inference
    - Intent prediction
    - Perspective taking
    - Strategic reasoning
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int = 5,
        belief_dim: int = 64,
        max_agents: int = 100,
        device: str = "cpu",
    ):
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.belief_dim = belief_dim
        self.device = device

        # Core modules
        self.belief_encoder = BeliefEncoder(observation_dim, belief_dim)
        self.intent_predictor = IntentPredictor(action_dim, belief_dim)

        # Agent models
        self.agent_models: Dict[int, AgentModel] = {}

        # Perspective taking: simulate other agent's view
        self.perspective_transform = nn.Sequential(
            nn.Linear(observation_dim + belief_dim, 128),
            nn.ReLU(),
            nn.Linear(128, observation_dim),
        )

        # Strategic reasoning: predict response to our action
        self.response_predictor = nn.Sequential(
            nn.Linear(belief_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def get_or_create_model(self, agent_id: int) -> AgentModel:
        """Get or create model for an agent."""
        if agent_id not in self.agent_models:
            self.agent_models[agent_id] = AgentModel(agent_id=agent_id)
        return self.agent_models[agent_id]

    def update_belief(
        self,
        agent_id: int,
        observation: torch.Tensor,
        timestamp: int = 0,
    ) -> BeliefState:
        """
        Update belief model for an agent.

        Args:
            agent_id: Which agent we're modeling
            observation: What that agent can observe
            timestamp: Current time step
        """
        model = self.get_or_create_model(agent_id)

        belief_state = self.belief_encoder(observation)
        belief_state.timestamp = timestamp

        model.belief_state = belief_state
        return belief_state

    def update_intent(
        self,
        agent_id: int,
        action: int,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[IntentType, float]:
        """
        Update intent prediction after observing action.

        Args:
            agent_id: Which agent we're modeling
            action: Action they just took
            context: Current context
        """
        model = self.get_or_create_model(agent_id)

        # Add to history
        model.action_history.append(action)
        if len(model.action_history) > 50:
            model.action_history = model.action_history[-50:]

        # Predict intent
        if len(model.action_history) >= 3:
            action_tensor = torch.tensor(
                model.action_history[-20:],
                device=self.device,
            )
            intent, confidence, _ = self.intent_predictor(
                action_tensor, context
            )
            model.predicted_intent = intent
            model.intent_confidence = confidence

        return model.predicted_intent, model.intent_confidence

    def take_perspective(
        self,
        agent_id: int,
        our_observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        Simulate what an agent observes from their perspective.

        Args:
            agent_id: Agent whose perspective to take
            our_observation: Our own observation

        Returns:
            Estimated observation from their perspective
        """
        model = self.get_or_create_model(agent_id)

        if model.belief_state is None:
            # No belief model yet - return identity
            return our_observation

        # Transform our observation based on their beliefs
        combined = torch.cat([
            our_observation,
            model.belief_state.world_belief,
        ], dim=-1)

        their_view = self.perspective_transform(combined)
        return their_view

    def predict_response(
        self,
        agent_id: int,
        our_action: int,
    ) -> torch.Tensor:
        """
        Predict how an agent will respond to our action.

        Args:
            agent_id: Agent whose response to predict
            our_action: Action we're considering

        Returns:
            Distribution over their likely responses
        """
        model = self.get_or_create_model(agent_id)

        if model.belief_state is None:
            # Uniform prediction
            return torch.ones(self.action_dim) / self.action_dim

        # Encode our action
        action_onehot = F.one_hot(
            torch.tensor(our_action),
            self.action_dim,
        ).float()

        combined = torch.cat([
            model.belief_state.self_belief.squeeze(),
            action_onehot,
        ])

        response_logits = self.response_predictor(combined)
        return F.softmax(response_logits, dim=-1)

    def update_trust(
        self,
        agent_id: int,
        prediction_correct: bool,
        decay: float = 0.95,
    ) -> float:
        """
        Update trust score based on prediction accuracy.

        Args:
            agent_id: Agent to update
            prediction_correct: Was our prediction correct?
            decay: Decay rate for old observations
        """
        model = self.get_or_create_model(agent_id)

        # Exponential moving average
        model.prediction_accuracy = (
            decay * model.prediction_accuracy +
            (1 - decay) * float(prediction_correct)
        )

        # Trust is based on prediction accuracy
        model.trust_score = model.prediction_accuracy

        return model.trust_score

    def get_most_trusted(self, top_k: int = 5) -> List[Tuple[int, float]]:
        """Get most trusted agents."""
        trust_scores = [
            (agent_id, model.trust_score)
            for agent_id, model in self.agent_models.items()
        ]
        trust_scores.sort(key=lambda x: x[1], reverse=True)
        return trust_scores[:top_k]

    def get_agent_summary(self, agent_id: int) -> Dict:
        """Get summary of our model of an agent."""
        if agent_id not in self.agent_models:
            return {"error": "No model for this agent"}

        model = self.agent_models[agent_id]
        return {
            "agent_id": agent_id,
            "agent_type": model.agent_type,
            "predicted_intent": model.predicted_intent.name,
            "intent_confidence": model.intent_confidence,
            "trust_score": model.trust_score,
            "action_history_length": len(model.action_history),
            "has_belief_model": model.belief_state is not None,
        }


class CollectiveBeliefAggregator(nn.Module):
    """
    Aggregate beliefs across multiple agents.

    Creates a shared understanding from individual perspectives.
    """

    def __init__(
        self,
        belief_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()

        # Attention-based aggregation
        self.attention = nn.Sequential(
            nn.Linear(belief_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Combine individual beliefs
        self.combiner = nn.Sequential(
            nn.Linear(belief_dim, belief_dim),
            nn.ReLU(),
            nn.Linear(belief_dim, belief_dim),
        )

    def forward(
        self,
        beliefs: List[BeliefState],
    ) -> torch.Tensor:
        """
        Aggregate multiple beliefs into collective belief.

        Args:
            beliefs: List of individual belief states

        Returns:
            Collective belief representation
        """
        if not beliefs:
            return None

        # Stack world beliefs
        world_beliefs = torch.stack([b.world_belief for b in beliefs])
        confidences = torch.tensor([b.confidence for b in beliefs])

        # Attention weights (incorporate confidence)
        attn_scores = self.attention(world_beliefs).squeeze(-1)
        attn_scores = attn_scores * confidences  # Weight by confidence

        attn_weights = F.softmax(attn_scores, dim=0)

        # Weighted combination
        aggregated = (attn_weights.unsqueeze(-1) * world_beliefs).sum(dim=0)

        # Process
        collective = self.combiner(aggregated)

        return collective
