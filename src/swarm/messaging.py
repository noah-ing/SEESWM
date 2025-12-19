"""
Message passing infrastructure for the swarm.

Handles routing messages between agents and tracking message statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class Message:
    """
    A message passed between agents in the swarm.

    Messages contain:
    - Content: The actual tensor data
    - Metadata: Source, destination, type info
    """

    content: torch.Tensor
    source_id: int
    target_id: Optional[int] = None  # None = broadcast
    message_type: str = "default"
    timestamp: int = 0

    def clone(self) -> Message:
        """Create a copy of this message."""
        return Message(
            content=self.content.clone(),
            source_id=self.source_id,
            target_id=self.target_id,
            message_type=self.message_type,
            timestamp=self.timestamp,
        )


@dataclass
class MessageStats:
    """Statistics about message passing in the swarm."""

    total_messages: int = 0
    messages_per_round: list[int] = field(default_factory=list)
    avg_message_magnitude: float = 0.0
    active_channels: int = 0


class MessageBus:
    """
    Central message routing system for the swarm.

    Handles:
    - Collecting messages from agents
    - Routing to appropriate recipients
    - Tracking message statistics
    """

    def __init__(self, num_agents: int, message_dim: int, device: str = "cpu"):
        self.num_agents = num_agents
        self.message_dim = message_dim
        self.device = device

        # Message buffers: current round and next round
        self.current_messages: dict[int, list[Message]] = {i: [] for i in range(num_agents)}
        self.next_messages: dict[int, list[Message]] = {i: [] for i in range(num_agents)}

        # Statistics
        self.stats = MessageStats()
        self.round_count = 0

    def send(self, message: Message) -> None:
        """
        Queue a message for delivery in the next round.

        If target_id is None, broadcasts to all agents.
        """
        if message.target_id is None:
            # Broadcast to all except sender
            for agent_id in range(self.num_agents):
                if agent_id != message.source_id:
                    self.next_messages[agent_id].append(message.clone())
        else:
            self.next_messages[message.target_id].append(message.clone())

        self.stats.total_messages += 1

    def receive(self, agent_id: int) -> list[Message]:
        """Get all messages for an agent in the current round."""
        return self.current_messages[agent_id]

    def get_neighbor_tensors(self, agent_id: int) -> list[torch.Tensor]:
        """Get message contents as tensors for an agent."""
        return [msg.content for msg in self.current_messages[agent_id]]

    def step(self) -> None:
        """
        Advance to next round.

        Swaps message buffers and clears next round buffer.
        """
        # Record stats
        messages_this_round = sum(len(msgs) for msgs in self.current_messages.values())
        self.stats.messages_per_round.append(messages_this_round)

        # Update average magnitude
        if messages_this_round > 0:
            all_magnitudes = []
            for msgs in self.current_messages.values():
                for msg in msgs:
                    all_magnitudes.append(msg.content.abs().mean().item())
            if all_magnitudes:
                new_avg = sum(all_magnitudes) / len(all_magnitudes)
                # Exponential moving average
                alpha = 0.1
                self.stats.avg_message_magnitude = (
                    alpha * new_avg + (1 - alpha) * self.stats.avg_message_magnitude
                )

        # Swap buffers
        self.current_messages = self.next_messages
        self.next_messages = {i: [] for i in range(self.num_agents)}
        self.round_count += 1

        # Count active channels
        self.stats.active_channels = sum(
            1 for msgs in self.current_messages.values() if msgs
        )

    def reset(self) -> None:
        """Clear all messages and reset state."""
        self.current_messages = {i: [] for i in range(self.num_agents)}
        self.next_messages = {i: [] for i in range(self.num_agents)}
        self.round_count = 0

    def get_stats(self) -> dict:
        """Get message statistics."""
        return {
            "total_messages": self.stats.total_messages,
            "rounds": self.round_count,
            "avg_messages_per_round": (
                sum(self.stats.messages_per_round) / max(1, len(self.stats.messages_per_round))
            ),
            "avg_message_magnitude": self.stats.avg_message_magnitude,
            "active_channels": self.stats.active_channels,
        }
