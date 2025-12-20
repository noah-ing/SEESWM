"""
Integration of neuromorphic computing with swarm architecture.

Creates spiking agents that communicate via discrete spike events,
enabling event-driven swarm computation.

Features:
1. SpikingMicroAgent: Agent using LIF neurons
2. SpikingSwarmGraph: Swarm with spike-based message passing
3. Hybrid agents (mix of spiking and rate-coded)
4. STDP-based swarm learning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..agents.micro_agent import AgentType, AgentConfig
from ..swarm.graph import SwarmGraph, SwarmConfig, TopologyType
from ..swarm.messaging import MessageBus

from .lif import LIFConfig, LIFLayer, RecurrentLIFLayer, AdaptiveLIFLayer, spike_fn
from .stdp import STDPConfig, STDPLayer, RewardModulatedSTDP, HomeostaticSTDP
from .layers import SpikingNetwork, LiquidStateMachine, TemporalCoding, SpikeDecoder
from .energy import EnergyTracker, SpikeCounter, HardwareModel, EnergyEfficientLoss


class CommunicationMode(Enum):
    """How agents communicate in spiking swarm."""
    SPIKE = "spike"           # Pure spike communication
    RATE = "rate"             # Rate-coded (spike counts)
    HYBRID = "hybrid"         # Both spikes and rates
    TEMPORAL = "temporal"     # Precise spike timing


@dataclass
class SpikingAgentConfig:
    """Configuration for spiking agents."""

    # Architecture
    input_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 32
    message_dim: int = 32

    # LIF parameters
    lif_config: Optional[LIFConfig] = None

    # Temporal parameters
    num_timesteps: int = 25
    dt: float = 1.0

    # Learning
    use_stdp: bool = True
    stdp_config: Optional[STDPConfig] = None
    use_reward_modulation: bool = True

    # Communication
    comm_mode: CommunicationMode = CommunicationMode.RATE

    # Agent type
    agent_type: AgentType = AgentType.PERCEPTION

    def __post_init__(self):
        if self.lif_config is None:
            self.lif_config = LIFConfig()
        if self.stdp_config is None:
            self.stdp_config = STDPConfig()


class SpikingMicroAgent(nn.Module):
    """
    Spiking neural network agent.

    Uses LIF neurons for processing and communicates via spikes.
    """

    def __init__(self, config: SpikingAgentConfig):
        super().__init__()
        self.config = config

        # Temporal coding for input
        self.input_encoder = TemporalCoding(
            config.input_dim,
            config.num_timesteps,
            coding_type="latency",
        )

        # Spiking processing layers
        self.hidden = RecurrentLIFLayer(
            config.input_dim,
            config.hidden_dim,
            config.lif_config,
        )

        # Message processing (receives from neighbors)
        self.message_processor = LIFLayer(
            config.message_dim,
            config.hidden_dim,
            config.lif_config,
        )

        # Output layer
        self.output = LIFLayer(
            config.hidden_dim * 2,  # Hidden + messages
            config.output_dim,
            config.lif_config,
        )

        # Message generation
        self.message_generator = LIFLayer(
            config.hidden_dim,
            config.message_dim,
            config.lif_config,
        )

        # Output decoder
        self.decoder = SpikeDecoder(
            config.output_dim,
            decoding_type="rate",
            num_steps=config.num_timesteps,
        )

        # STDP learning (if enabled)
        if config.use_stdp:
            if config.use_reward_modulation:
                self.stdp = RewardModulatedSTDP(config.stdp_config)
            else:
                self.stdp = HomeostaticSTDP(
                    config.hidden_dim,
                    target_rate=0.1,
                    config=config.stdp_config,
                )
        else:
            self.stdp = None

        # Spike traces for STDP
        self.pre_trace: Optional[torch.Tensor] = None
        self.post_trace: Optional[torch.Tensor] = None
        self.trace_decay = 0.95

        # State
        self.local_state: Optional[torch.Tensor] = None
        self.spike_history: List[torch.Tensor] = []

        # Energy tracking
        self.spike_counter = SpikeCounter()

    def reset(self) -> None:
        """Reset agent state."""
        self.hidden.reset()
        self.message_processor.reset()
        self.output.reset()
        self.message_generator.reset()
        self.local_state = None
        self.spike_history = []
        self.pre_trace = None
        self.post_trace = None
        self.spike_counter.reset()

    def forward(
        self,
        observation: torch.Tensor,
        incoming_messages: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process observation and messages over time.

        Args:
            observation: Input observation [batch, input_dim]
            incoming_messages: Messages from neighbors [batch, message_dim] or None

        Returns:
            (output, outgoing_message)
        """
        batch_size = observation.shape[0]
        device = observation.device

        # Encode observation as spike train
        obs_spikes = self.input_encoder(observation)  # [batch, time, input_dim]

        # Process over timesteps
        hidden_spikes_all = []
        output_spikes_all = []
        message_spikes_all = []

        for t in range(self.config.num_timesteps):
            # Hidden layer processing
            hidden_spikes = self.hidden(obs_spikes[:, t])
            hidden_spikes_all.append(hidden_spikes)

            # Process incoming messages
            if incoming_messages is not None:
                msg_spikes = self.message_processor(incoming_messages)
            else:
                msg_spikes = torch.zeros(batch_size, self.config.hidden_dim, device=device)

            # Combine hidden and message states
            combined = torch.cat([hidden_spikes, msg_spikes], dim=-1)

            # Output
            out_spikes = self.output(combined)
            output_spikes_all.append(out_spikes)

            # Generate outgoing message
            msg_out = self.message_generator(hidden_spikes)
            message_spikes_all.append(msg_out)

            # STDP update (if enabled and training)
            if self.training and self.stdp is not None:
                self._update_stdp(obs_spikes[:, t], hidden_spikes)

            # Track spikes
            self.spike_counter.count_timestep(hidden_spikes)

        # Stack over time
        hidden_train = torch.stack(hidden_spikes_all, dim=1)
        output_train = torch.stack(output_spikes_all, dim=1)
        message_train = torch.stack(message_spikes_all, dim=1)

        # Store state
        self.local_state = hidden_train[:, -1]
        self.spike_history = hidden_spikes_all

        # Decode output
        output = self.decoder(output_train)

        # Decode message (rate-coded for communication)
        if self.config.comm_mode == CommunicationMode.RATE:
            message = message_train.mean(dim=1)
        else:
            message = message_train[:, -1]  # Last spike pattern

        return output, message

    def _update_stdp(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
    ) -> None:
        """Update STDP traces and weights."""
        # Initialize traces
        if self.pre_trace is None:
            self.pre_trace = torch.zeros_like(pre_spikes)
            self.post_trace = torch.zeros_like(post_spikes)

        # Update traces
        self.pre_trace = self.trace_decay * self.pre_trace + pre_spikes
        self.post_trace = self.trace_decay * self.post_trace + post_spikes

        # Apply STDP to hidden layer weights
        if hasattr(self.hidden, 'input_weight'):
            with torch.no_grad():
                # Compute eligibility
                if hasattr(self.stdp, 'compute_eligibility'):
                    self.stdp.compute_eligibility(
                        self.hidden.input_weight,
                        pre_spikes,
                        post_spikes,
                        self.pre_trace,
                        self.post_trace,
                    )

    def apply_reward(self, reward: float) -> None:
        """Apply reward signal for reward-modulated STDP."""
        if self.stdp is not None and hasattr(self.stdp, 'apply_reward'):
            if hasattr(self.hidden, 'input_weight'):
                new_weight, _ = self.stdp.apply_reward(
                    self.hidden.input_weight,
                    reward,
                )
                self.hidden.input_weight.data = new_weight

    def get_spike_statistics(self) -> Dict[str, float]:
        """Get spike statistics."""
        stats = self.spike_counter.get_statistics()
        return {
            "total_spikes": stats.total_spikes,
            "spike_rate": stats.overall_spike_rate,
            "sparsity": 1.0 - stats.overall_spike_rate,
        }


@dataclass
class SpikingSwarmConfig:
    """Configuration for spiking swarm."""

    num_agents: int = 20
    agent_config: Optional[SpikingAgentConfig] = None

    # Topology
    topology: TopologyType = TopologyType.SMALL_WORLD

    # Message passing
    num_rounds: int = 3
    spike_threshold: float = 0.5  # Threshold for spike-based communication

    # Energy
    track_energy: bool = True
    hardware: HardwareModel = HardwareModel.LOIHI

    # Learning
    global_stdp: bool = True  # Apply STDP across swarm connections

    def __post_init__(self):
        if self.agent_config is None:
            self.agent_config = SpikingAgentConfig()


class SpikingSwarmGraph(nn.Module):
    """
    Swarm of spiking agents.

    Agents communicate via discrete spike events, enabling
    event-driven distributed computation.
    """

    def __init__(self, config: SpikingSwarmConfig):
        super().__init__()
        self.config = config

        # Create agents
        self.agents = nn.ModuleDict()
        for i in range(config.num_agents):
            agent_cfg = SpikingAgentConfig(
                input_dim=config.agent_config.input_dim,
                hidden_dim=config.agent_config.hidden_dim,
                output_dim=config.agent_config.output_dim,
                message_dim=config.agent_config.message_dim,
                lif_config=config.agent_config.lif_config,
                num_timesteps=config.agent_config.num_timesteps,
                use_stdp=config.agent_config.use_stdp,
                stdp_config=config.agent_config.stdp_config,
                comm_mode=config.agent_config.comm_mode,
            )
            self.agents[str(i)] = SpikingMicroAgent(agent_cfg)

        # Build topology
        self.adjacency = self._build_topology()

        # Message buffer (rate-coded)
        self.message_buffer: Dict[int, torch.Tensor] = {}

        # Energy tracking
        if config.track_energy:
            self.energy_tracker = EnergyTracker(hardware=config.hardware)
            # Register layers for energy estimation
            for i in range(config.num_agents):
                self.energy_tracker.register_layer(
                    config.agent_config.input_dim,
                    config.agent_config.hidden_dim,
                )
        else:
            self.energy_tracker = None

        # STDP for inter-agent connections (if enabled)
        if config.global_stdp:
            self.global_stdp_layer = STDPLayer(
                config.agent_config.message_dim,
                config.agent_config.message_dim,
                config.agent_config.stdp_config,
            )
        else:
            self.global_stdp_layer = None

    def _build_topology(self) -> Dict[int, List[int]]:
        """Build swarm topology."""
        n = self.config.num_agents
        adjacency = {i: [] for i in range(n)}

        if self.config.topology == TopologyType.SMALL_WORLD:
            # Watts-Strogatz small-world
            k = 4  # Each node connects to k nearest neighbors
            p = 0.3  # Rewiring probability

            # Ring lattice
            for i in range(n):
                for j in range(1, k // 2 + 1):
                    adjacency[i].append((i + j) % n)
                    adjacency[i].append((i - j) % n)

        elif self.config.topology == TopologyType.RANDOM:
            # Random graph
            import random
            p = 0.2
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        adjacency[i].append(j)
                        adjacency[j].append(i)

        else:
            # Fully connected
            for i in range(n):
                for j in range(n):
                    if i != j:
                        adjacency[i].append(j)

        return adjacency

    def reset(self) -> None:
        """Reset all agents."""
        for agent in self.agents.values():
            agent.reset()
        self.message_buffer = {}
        if self.energy_tracker:
            self.energy_tracker.reset()

    def step(
        self,
        observations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single step of spiking swarm.

        Args:
            observations: Observations for all agents [batch, num_agents, input_dim]
                         or shared [batch, input_dim]

        Returns:
            Swarm output [batch, output_dim]
        """
        batch_size = observations.shape[0]
        device = observations.device

        # Handle shared vs per-agent observations
        if observations.dim() == 2:
            per_agent_obs = observations.unsqueeze(1).expand(
                -1, self.config.num_agents, -1
            )
        else:
            per_agent_obs = observations

        # Initialize message buffer
        message_dim = self.config.agent_config.message_dim
        for i in range(self.config.num_agents):
            self.message_buffer[i] = torch.zeros(batch_size, message_dim, device=device)

        # Message passing rounds
        outputs = []
        for round_idx in range(self.config.num_rounds):
            round_messages = {}
            round_outputs = []

            for i, agent in self.agents.items():
                agent_idx = int(i)

                # Gather incoming messages from neighbors
                neighbors = self.adjacency[agent_idx]
                if neighbors:
                    neighbor_msgs = [self.message_buffer[n] for n in neighbors]
                    incoming = torch.stack(neighbor_msgs, dim=1).mean(dim=1)
                else:
                    incoming = None

                # Agent forward pass
                output, message = agent(per_agent_obs[:, agent_idx], incoming)

                round_messages[agent_idx] = message
                round_outputs.append(output)

                # Track energy
                if self.energy_tracker:
                    stats = agent.get_spike_statistics()
                    self.energy_tracker.spike_counter.total_spikes += int(stats["total_spikes"])

            # Update message buffer
            self.message_buffer = round_messages
            outputs = round_outputs

        # Aggregate outputs
        output_stack = torch.stack(outputs, dim=1)  # [batch, num_agents, output_dim]
        swarm_output = output_stack.mean(dim=1)  # [batch, output_dim]

        return swarm_output

    def apply_reward(self, reward: float) -> None:
        """Apply reward signal to all agents."""
        for agent in self.agents.values():
            agent.apply_reward(reward)

    def get_energy_estimate(self) -> Dict[str, float]:
        """Get energy consumption estimate."""
        if self.energy_tracker:
            estimate = self.energy_tracker.estimate_energy()
            return {
                "total_energy_pj": estimate.total_energy_pj,
                "spike_energy_pj": estimate.spike_energy_pj,
                "synaptic_energy_pj": estimate.synaptic_energy_pj,
                "energy_ratio": estimate.energy_ratio,
            }
        return {}

    def get_swarm_statistics(self) -> Dict[str, float]:
        """Get swarm-wide statistics."""
        total_spikes = 0
        total_rate = 0.0

        for agent in self.agents.values():
            stats = agent.get_spike_statistics()
            total_spikes += stats["total_spikes"]
            total_rate += stats["spike_rate"]

        num_agents = len(self.agents)

        return {
            "total_swarm_spikes": total_spikes,
            "avg_spike_rate": total_rate / num_agents if num_agents > 0 else 0.0,
            "avg_sparsity": 1.0 - (total_rate / num_agents) if num_agents > 0 else 1.0,
        }


class HybridSwarm(nn.Module):
    """
    Hybrid swarm with both spiking and rate-coded agents.

    Allows gradual transition from rate-coded to fully spiking.
    """

    def __init__(
        self,
        num_spiking: int = 10,
        num_rate: int = 10,
        spiking_config: Optional[SpikingAgentConfig] = None,
        rate_config: Optional[SwarmConfig] = None,
    ):
        super().__init__()

        # Spiking agents
        spiking_swarm_config = SpikingSwarmConfig(
            num_agents=num_spiking,
            agent_config=spiking_config,
        )
        self.spiking_swarm = SpikingSwarmGraph(spiking_swarm_config)

        # Rate-coded agents (from existing swarm)
        if rate_config is None:
            rate_config = SwarmConfig(
                num_agents=num_rate,
                input_dim=spiking_config.input_dim if spiking_config else 64,
                hidden_dim=spiking_config.hidden_dim if spiking_config else 128,
                output_dim=spiking_config.output_dim if spiking_config else 32,
            )
        self.rate_swarm = SwarmGraph(rate_config)

        # Bridge between swarms
        bridge_dim = spiking_config.message_dim if spiking_config else 32
        self.spike_to_rate = nn.Linear(bridge_dim, rate_config.hidden_dim)
        self.rate_to_spike = nn.Linear(rate_config.hidden_dim, bridge_dim)

        # Output combination
        self.combiner = nn.Linear(
            (spiking_config.output_dim if spiking_config else 32) + rate_config.output_dim,
            rate_config.output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through hybrid swarm."""
        # Spiking swarm
        spiking_out = self.spiking_swarm.step(x)

        # Rate swarm
        rate_out = self.rate_swarm.step(x)

        # Combine outputs
        combined = torch.cat([spiking_out, rate_out], dim=-1)
        output = self.combiner(combined)

        return output

    def reset(self) -> None:
        """Reset both swarms."""
        self.spiking_swarm.reset()
        self.rate_swarm.reset()


class SpikingSwarmTrainer:
    """
    Trainer for spiking swarms.

    Supports:
    - Surrogate gradient training
    - STDP-based learning
    - Energy-constrained training
    """

    def __init__(
        self,
        swarm: SpikingSwarmGraph,
        learning_rate: float = 1e-3,
        energy_weight: float = 0.01,
        target_spike_rate: float = 0.1,
    ):
        self.swarm = swarm
        self.energy_weight = energy_weight
        self.target_rate = target_spike_rate

        # Optimizer (for surrogate gradient training)
        self.optimizer = torch.optim.Adam(swarm.parameters(), lr=learning_rate)

        # Energy-efficient loss
        self.energy_loss = EnergyEfficientLoss(
            target_rate=target_spike_rate,
            rate_weight=energy_weight,
        )

        # History
        self.loss_history: List[float] = []
        self.energy_history: List[Dict] = []

    def train_step(
        self,
        observations: torch.Tensor,
        targets: torch.Tensor,
        reward: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            observations: Input observations
            targets: Target outputs
            reward: Optional reward for STDP modulation

        Returns:
            Training metrics
        """
        self.swarm.train()
        self.swarm.reset()
        self.optimizer.zero_grad()

        # Forward pass
        output = self.swarm.step(observations)

        # Task loss
        task_loss = F.mse_loss(output, targets)

        # Energy penalty
        stats = self.swarm.get_swarm_statistics()
        rate_penalty = (stats["avg_spike_rate"] - self.target_rate) ** 2
        total_loss = task_loss + self.energy_weight * rate_penalty

        # Backward (surrogate gradients)
        total_loss.backward()
        self.optimizer.step()

        # Apply reward for STDP (if provided)
        if reward is not None:
            self.swarm.apply_reward(reward)

        # Track
        self.loss_history.append(total_loss.item())
        if self.swarm.energy_tracker:
            energy = self.swarm.get_energy_estimate()
            self.energy_history.append(energy)

        return {
            "loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "rate_penalty": rate_penalty,
            "spike_rate": stats["avg_spike_rate"],
            "sparsity": stats["avg_sparsity"],
        }

    def get_summary(self) -> Dict[str, float]:
        """Get training summary."""
        recent_losses = self.loss_history[-100:] if self.loss_history else [0]

        summary = {
            "avg_loss": sum(recent_losses) / len(recent_losses),
            "total_steps": len(self.loss_history),
        }

        if self.energy_history:
            recent_energy = self.energy_history[-100:]
            summary["avg_energy_pj"] = sum(
                e.get("total_energy_pj", 0) for e in recent_energy
            ) / len(recent_energy)

        return summary
