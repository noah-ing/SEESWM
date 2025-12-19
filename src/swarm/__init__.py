"""
Swarm module - Graph topology and message passing.
"""

from .graph import SwarmGraph, SwarmConfig, TopologyType
from .messaging import Message, MessageBus
from .typed_messaging import (
    TypedMessage,
    TypedMessageBus,
    MessageType,
    MessageFilter,
    MessageEncoder,
    MessageDecoder,
    TypeAwareAggregator,
)
from .specialized_graph import (
    SpecializedSwarmGraph,
    SpecializedSwarmConfig,
    RoleEmergenceTracker,
)

__all__ = [
    # Original
    "SwarmGraph",
    "SwarmConfig",
    "TopologyType",
    "Message",
    "MessageBus",
    # Typed messaging
    "TypedMessage",
    "TypedMessageBus",
    "MessageType",
    "MessageFilter",
    "MessageEncoder",
    "MessageDecoder",
    "TypeAwareAggregator",
    # Specialized swarm
    "SpecializedSwarmGraph",
    "SpecializedSwarmConfig",
    "RoleEmergenceTracker",
]
