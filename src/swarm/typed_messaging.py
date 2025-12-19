"""
Typed message passing for specialized agent communication.

Messages carry type information that indicates their semantic meaning:
- Perception messages: Feature embeddings, saliency maps
- Reasoning messages: Logical conclusions, inferences
- Memory messages: Retrieved memories, storage confirmations
- Planning messages: Goals, subgoals, action proposals
- Control messages: Neuromodulation signals, attention cues

Type-aware routing allows agents to process messages differently
based on the sender's specialization and message semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Callable, Tuple
import torch
import torch.nn as nn

from ..agents.micro_agent import AgentType


class MessageType(Enum):
    """Semantic types for messages in the swarm."""

    # Content types (from specialized agents)
    PERCEPT = auto()          # Raw perceptual features from Perception agents
    SALIENCY = auto()         # Attention/importance signals from Perception
    INFERENCE = auto()        # Logical conclusions from Reasoning agents
    QUERY = auto()            # Information requests (to Memory agents)
    MEMORY_RECALL = auto()    # Retrieved memories from Memory agents
    GOAL = auto()             # Goals from Planning agents
    SUBGOAL = auto()          # Intermediate goals from Planning
    ACTION_PROPOSAL = auto()  # Proposed actions from Planning

    # Control types (neuromodulation)
    REWARD_SIGNAL = auto()    # Dopamine-like reward
    NOVELTY_SIGNAL = auto()   # Curiosity signal
    UNCERTAINTY = auto()      # Confidence/uncertainty
    ATTENTION_CUE = auto()    # "Look at this" signal

    # Generic fallback
    GENERIC = auto()


# Map agent types to the message types they typically send
AGENT_MESSAGE_TYPES: Dict[AgentType, List[MessageType]] = {
    AgentType.PERCEPTION: [MessageType.PERCEPT, MessageType.SALIENCY],
    AgentType.REASONING: [MessageType.INFERENCE, MessageType.QUERY],
    AgentType.MEMORY: [MessageType.MEMORY_RECALL],
    AgentType.PLANNING: [MessageType.GOAL, MessageType.SUBGOAL, MessageType.ACTION_PROPOSAL],
    AgentType.GENERAL: [MessageType.GENERIC],
}


@dataclass
class TypedMessage:
    """
    A typed message passed between agents.

    Contains both content (tensor) and rich type information
    for semantic routing and processing.
    """
    content: torch.Tensor
    message_type: MessageType
    source_id: int
    source_type: AgentType  # Type of agent that sent this
    target_id: Optional[int] = None  # None = broadcast
    priority: float = 1.0  # Higher priority messages processed first
    timestamp: int = 0
    metadata: Dict = field(default_factory=dict)

    def clone(self) -> TypedMessage:
        """Create a deep copy of this message."""
        return TypedMessage(
            content=self.content.clone(),
            message_type=self.message_type,
            source_id=self.source_id,
            source_type=self.source_type,
            target_id=self.target_id,
            priority=self.priority,
            timestamp=self.timestamp,
            metadata=dict(self.metadata),
        )

    @property
    def is_control_message(self) -> bool:
        """Check if this is a control/neuromodulation message."""
        return self.message_type in {
            MessageType.REWARD_SIGNAL,
            MessageType.NOVELTY_SIGNAL,
            MessageType.UNCERTAINTY,
            MessageType.ATTENTION_CUE,
        }


class MessageFilter:
    """Filter messages by various criteria."""

    def __init__(
        self,
        allowed_types: Optional[List[MessageType]] = None,
        allowed_sources: Optional[List[AgentType]] = None,
        min_priority: float = 0.0,
        max_age: Optional[int] = None,
    ):
        self.allowed_types = set(allowed_types) if allowed_types else None
        self.allowed_sources = set(allowed_sources) if allowed_sources else None
        self.min_priority = min_priority
        self.max_age = max_age

    def accepts(self, msg: TypedMessage, current_time: int = 0) -> bool:
        """Check if message passes filter criteria."""
        if self.allowed_types and msg.message_type not in self.allowed_types:
            return False
        if self.allowed_sources and msg.source_type not in self.allowed_sources:
            return False
        if msg.priority < self.min_priority:
            return False
        if self.max_age is not None and (current_time - msg.timestamp) > self.max_age:
            return False
        return True


class TypedMessageBus:
    """
    Type-aware message routing system.

    Features:
    - Type-based filtering and routing
    - Priority queuing
    - Agent-specific message acceptance rules
    - Statistics tracking by message type
    """

    def __init__(
        self,
        num_agents: int,
        message_dim: int,
        device: str = "cpu",
    ):
        self.num_agents = num_agents
        self.message_dim = message_dim
        self.device = device

        # Message buffers
        self.current_messages: Dict[int, List[TypedMessage]] = {
            i: [] for i in range(num_agents)
        }
        self.next_messages: Dict[int, List[TypedMessage]] = {
            i: [] for i in range(num_agents)
        }

        # Per-agent filters (agents can subscribe to specific message types)
        self.agent_filters: Dict[int, MessageFilter] = {}

        # Agent type registry
        self.agent_types: Dict[int, AgentType] = {}

        # Statistics
        self.stats_by_type: Dict[MessageType, int] = {t: 0 for t in MessageType}
        self.total_messages = 0
        self.round_count = 0

    def register_agent(self, agent_id: int, agent_type: AgentType) -> None:
        """Register an agent's type for routing."""
        self.agent_types[agent_id] = agent_type

        # Set default filter based on agent type
        self.agent_filters[agent_id] = self._default_filter_for_type(agent_type)

    def _default_filter_for_type(self, agent_type: AgentType) -> MessageFilter:
        """Create default message filter based on agent specialization."""
        if agent_type == AgentType.PERCEPTION:
            # Perception agents listen for attention cues and queries
            return MessageFilter(allowed_types=[
                MessageType.ATTENTION_CUE,
                MessageType.QUERY,
                MessageType.GOAL,  # Know what to look for
            ])
        elif agent_type == AgentType.REASONING:
            # Reasoning agents need percepts and memories
            return MessageFilter(allowed_types=[
                MessageType.PERCEPT,
                MessageType.SALIENCY,
                MessageType.MEMORY_RECALL,
                MessageType.GOAL,
            ])
        elif agent_type == AgentType.MEMORY:
            # Memory agents respond to queries and store percepts/inferences
            return MessageFilter(allowed_types=[
                MessageType.QUERY,
                MessageType.PERCEPT,
                MessageType.INFERENCE,
            ])
        elif agent_type == AgentType.PLANNING:
            # Planning agents need inferences and current percepts
            return MessageFilter(allowed_types=[
                MessageType.INFERENCE,
                MessageType.MEMORY_RECALL,
                MessageType.PERCEPT,
                MessageType.SALIENCY,
            ])
        else:
            # General agents accept everything
            return MessageFilter()

    def set_filter(self, agent_id: int, msg_filter: MessageFilter) -> None:
        """Set custom message filter for an agent."""
        self.agent_filters[agent_id] = msg_filter

    def send(
        self,
        content: torch.Tensor,
        source_id: int,
        message_type: MessageType,
        target_id: Optional[int] = None,
        priority: float = 1.0,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Send a typed message.

        Args:
            content: Tensor content
            source_id: Sending agent ID
            message_type: Semantic type of message
            target_id: Specific recipient (None = broadcast)
            priority: Message priority (higher = more important)
            metadata: Optional additional data
        """
        source_type = self.agent_types.get(source_id, AgentType.GENERAL)

        msg = TypedMessage(
            content=content,
            message_type=message_type,
            source_id=source_id,
            source_type=source_type,
            target_id=target_id,
            priority=priority,
            timestamp=self.round_count,
            metadata=metadata or {},
        )

        if target_id is None:
            # Broadcast to all (respecting filters)
            for agent_id in range(self.num_agents):
                if agent_id != source_id:
                    agent_filter = self.agent_filters.get(agent_id, MessageFilter())
                    if agent_filter.accepts(msg, self.round_count):
                        self.next_messages[agent_id].append(msg.clone())
        else:
            # Direct send (always delivered, filter checked on receive)
            self.next_messages[target_id].append(msg.clone())

        self.total_messages += 1
        self.stats_by_type[message_type] += 1

    def receive(
        self,
        agent_id: int,
        filter_by_type: Optional[List[MessageType]] = None,
    ) -> List[TypedMessage]:
        """
        Get messages for an agent, optionally filtered by type.

        Returns messages sorted by priority (highest first).
        """
        messages = self.current_messages[agent_id]

        if filter_by_type:
            type_set = set(filter_by_type)
            messages = [m for m in messages if m.message_type in type_set]

        # Sort by priority (descending)
        return sorted(messages, key=lambda m: m.priority, reverse=True)

    def receive_as_tensors(
        self,
        agent_id: int,
        filter_by_type: Optional[List[MessageType]] = None,
    ) -> Tuple[torch.Tensor, List[MessageType]]:
        """
        Get messages as stacked tensors with type info.

        Returns:
            (stacked_tensors, message_types)
        """
        messages = self.receive(agent_id, filter_by_type)

        if not messages:
            return torch.zeros(0, self.message_dim, device=self.device), []

        tensors = torch.stack([m.content for m in messages])
        types = [m.message_type for m in messages]

        return tensors, types

    def step(self) -> None:
        """Advance to next round, swap buffers."""
        self.current_messages = self.next_messages
        self.next_messages = {i: [] for i in range(self.num_agents)}
        self.round_count += 1

    def reset(self) -> None:
        """Clear all messages and reset state."""
        self.current_messages = {i: [] for i in range(self.num_agents)}
        self.next_messages = {i: [] for i in range(self.num_agents)}
        self.round_count = 0

    def get_stats(self) -> Dict:
        """Get message statistics."""
        return {
            "total_messages": self.total_messages,
            "rounds": self.round_count,
            "by_type": {t.name: count for t, count in self.stats_by_type.items() if count > 0},
        }


class MessageEncoder(nn.Module):
    """
    Encode message type information into the message content.

    This allows agents to be aware of message types even when
    processing raw tensor content.
    """

    def __init__(self, message_dim: int, num_types: int = len(MessageType)):
        super().__init__()
        self.message_dim = message_dim
        self.num_types = num_types

        # Type embeddings
        self.type_embedding = nn.Embedding(num_types, message_dim)

        # Combine content with type info
        self.combiner = nn.Sequential(
            nn.Linear(message_dim * 2, message_dim),
            nn.GELU(),
            nn.Linear(message_dim, message_dim),
        )

    def forward(
        self,
        content: torch.Tensor,
        message_type: MessageType,
    ) -> torch.Tensor:
        """Add type information to message content."""
        type_idx = torch.tensor([message_type.value - 1], device=content.device)
        type_emb = self.type_embedding(type_idx)

        if content.dim() == 1:
            content = content.unsqueeze(0)
            type_emb = type_emb.expand(content.shape[0], -1)

        combined = torch.cat([content, type_emb.expand(content.shape[0], -1)], dim=-1)
        return self.combiner(combined)


class MessageDecoder(nn.Module):
    """
    Decode message type from content.

    Agents can use this to understand what kind of information
    they're receiving without explicit type labels.
    """

    def __init__(self, message_dim: int, num_types: int = len(MessageType)):
        super().__init__()
        self.message_dim = message_dim
        self.num_types = num_types

        self.classifier = nn.Sequential(
            nn.Linear(message_dim, message_dim),
            nn.GELU(),
            nn.Linear(message_dim, num_types),
        )

    def forward(self, content: torch.Tensor) -> torch.Tensor:
        """Predict message type from content. Returns logits."""
        return self.classifier(content)

    def predict_type(self, content: torch.Tensor) -> MessageType:
        """Get most likely message type."""
        logits = self.forward(content)
        type_idx = logits.argmax(dim=-1).item() + 1  # +1 because enum starts at 1
        return MessageType(type_idx)


class TypeAwareAggregator(nn.Module):
    """
    Aggregate messages with type-aware processing.

    Different message types are processed through different
    pathways before being combined.
    """

    def __init__(self, message_dim: int, output_dim: int):
        super().__init__()
        self.message_dim = message_dim
        self.output_dim = output_dim

        # Type-specific processors
        self.type_processors = nn.ModuleDict({
            "content": nn.Sequential(
                nn.Linear(message_dim, message_dim),
                nn.GELU(),
            ),
            "control": nn.Sequential(
                nn.Linear(message_dim, message_dim),
                nn.Sigmoid(),  # Control signals are gating
            ),
        })

        # Final aggregation
        self.output_proj = nn.Linear(message_dim * 2, output_dim)

    def forward(
        self,
        messages: List[TypedMessage],
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Aggregate messages with type awareness.

        Returns aggregated representation.
        """
        if not messages:
            return torch.zeros(1, self.output_dim, device=device)

        # Separate content and control messages
        content_msgs = [m for m in messages if not m.is_control_message]
        control_msgs = [m for m in messages if m.is_control_message]

        # Process content messages
        if content_msgs:
            content_stack = torch.stack([m.content for m in content_msgs])
            content_processed = self.type_processors["content"](content_stack)
            content_agg = content_processed.mean(dim=0, keepdim=True)
        else:
            content_agg = torch.zeros(1, self.message_dim, device=device)

        # Process control messages
        if control_msgs:
            control_stack = torch.stack([m.content for m in control_msgs])
            control_processed = self.type_processors["control"](control_stack)
            control_agg = control_processed.mean(dim=0, keepdim=True)
        else:
            control_agg = torch.ones(1, self.message_dim, device=device)  # Neutral gate

        # Combine: content modulated by control
        combined = torch.cat([content_agg * control_agg, content_agg], dim=-1)

        return self.output_proj(combined)
