"""
Emergent Behavior Detection.

Detect behaviors that weren't explicitly rewarded but emerged from training.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict


@dataclass
class EmergentBehavior:
    """Detected emergent behavior."""
    name: str
    description: str
    frequency: float  # How often observed
    strength: float  # How strongly exhibited
    evidence: List[str] = field(default_factory=list)


@dataclass
class CommunicationPattern:
    """Detected communication pattern."""
    source_agents: List[int]
    target_agents: List[int]
    message_type: str
    frequency: float
    semantic_content: Optional[str] = None


class BehaviorProbe(nn.Module):
    """Linear probe to detect specific behaviors from representations."""

    def __init__(
        self,
        input_dim: int,
        behavior_classes: int,
    ):
        super().__init__()
        self.probe = nn.Linear(input_dim, behavior_classes)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        return self.probe(representations)


class EmergenceDetector:
    """Detect emergent behaviors in trained swarms."""

    def __init__(
        self,
        swarm: nn.Module,
        device: str = "cpu",
    ):
        self.swarm = swarm
        self.device = device

        # Behavior detection hooks
        self.activations = {}
        self.message_log = []

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach()
            return hook

        for name, module in self.swarm.named_modules():
            if 'agent' in name.lower():
                module.register_forward_hook(hook_fn(name))

    def detect_tool_use(
        self,
        trajectories: List[Dict],
    ) -> Optional[EmergentBehavior]:
        """
        Detect if agents learn to use environmental objects as tools.

        Looks for patterns like:
        - Collecting materials before building
        - Using resources strategically
        """
        tool_use_count = 0
        total_episodes = len(trajectories)

        for traj in trajectories:
            observations = traj.get('observations', [])
            actions = traj.get('actions', [])
            rewards = traj.get('rewards', [])

            # Detect material collection followed by high reward
            material_collected = False
            for i, obs in enumerate(observations):
                if hasattr(obs, 'inventory'):
                    if obs.inventory.get('material', 0) > 0:
                        material_collected = True

                        # Check if subsequent rewards are higher
                        if i + 5 < len(rewards):
                            subsequent_reward = sum(rewards[i:i+5])
                            if subsequent_reward > sum(rewards[:5]):
                                tool_use_count += 1
                                break

        frequency = tool_use_count / (total_episodes + 1e-10)

        if frequency > 0.1:  # At least 10% of episodes
            return EmergentBehavior(
                name="tool_use",
                description="Agents collect materials before high-reward actions",
                frequency=frequency,
                strength=min(1.0, frequency * 2),
                evidence=[f"Observed in {tool_use_count}/{total_episodes} episodes"],
            )
        return None

    def detect_communication_protocols(
        self,
        message_history: List[Dict],
    ) -> List[CommunicationPattern]:
        """
        Detect if stable communication protocols emerge.

        Looks for:
        - Consistent message patterns between agent pairs
        - Semantic clustering of messages
        """
        patterns = []

        # Group messages by source-target pairs
        pair_messages = defaultdict(list)
        for msg in message_history:
            source = msg.get('source', -1)
            target = msg.get('target', -1)
            content = msg.get('content', None)

            if content is not None:
                pair_messages[(source, target)].append(content)

        # Analyze each pair
        for (source, target), messages in pair_messages.items():
            if len(messages) < 10:
                continue

            # Check for consistency (low variance in message content)
            if isinstance(messages[0], torch.Tensor):
                stacked = torch.stack(messages)
                variance = stacked.var(dim=0).mean().item()

                if variance < 0.5:  # Low variance = consistent pattern
                    patterns.append(CommunicationPattern(
                        source_agents=[source],
                        target_agents=[target],
                        message_type='consistent',
                        frequency=len(messages) / len(message_history),
                    ))

        return patterns

    def detect_division_of_labor(
        self,
        agent_action_histories: Dict[int, List[int]],
    ) -> Optional[EmergentBehavior]:
        """
        Detect if agents specialize in different roles.

        Looks for:
        - Agents preferring different action distributions
        - Stable role assignments over time
        """
        from scipy.stats import entropy

        num_agents = len(agent_action_histories)
        if num_agents < 2:
            return None

        # Compute action distributions per agent
        action_distributions = []
        for agent_id, actions in agent_action_histories.items():
            if len(actions) < 10:
                continue

            counts = np.bincount(actions, minlength=5)
            probs = counts / (counts.sum() + 1e-10)
            action_distributions.append(probs)

        if len(action_distributions) < 2:
            return None

        # Measure diversity of roles
        # High diversity = different agents do different things
        role_diversity = 0
        for i, dist_i in enumerate(action_distributions):
            for j, dist_j in enumerate(action_distributions):
                if i < j:
                    # KL divergence between distributions
                    kl = entropy(dist_i + 1e-10, dist_j + 1e-10)
                    role_diversity += kl

        role_diversity /= (len(action_distributions) * (len(action_distributions) - 1) / 2)

        if role_diversity > 0.5:  # Significant diversity
            return EmergentBehavior(
                name="division_of_labor",
                description="Agents show distinct behavioral preferences",
                frequency=1.0,  # Always present if detected
                strength=min(1.0, role_diversity),
                evidence=[f"Role diversity score: {role_diversity:.3f}"],
            )
        return None

    def detect_anticipatory_behavior(
        self,
        trajectories: List[Dict],
    ) -> Optional[EmergentBehavior]:
        """
        Detect if agents show planning/anticipation.

        Looks for:
        - Actions that don't give immediate reward but lead to better outcomes
        - Goal-directed movement patterns
        """
        anticipatory_count = 0
        total_opportunities = 0

        for traj in trajectories:
            actions = traj.get('actions', [])
            rewards = traj.get('rewards', [])

            for i in range(len(rewards) - 5):
                immediate = rewards[i]
                future = sum(rewards[i+1:i+5])

                # Anticipatory: low immediate, high future
                if immediate < 0.1 and future > immediate * 5:
                    anticipatory_count += 1
                total_opportunities += 1

        frequency = anticipatory_count / (total_opportunities + 1e-10)

        if frequency > 0.05:  # At least 5% anticipatory actions
            return EmergentBehavior(
                name="anticipatory_behavior",
                description="Agents take suboptimal immediate actions for future gain",
                frequency=frequency,
                strength=min(1.0, frequency * 5),
                evidence=[f"Observed {anticipatory_count} anticipatory actions"],
            )
        return None

    def detect_cooperation(
        self,
        multi_agent_trajectories: List[Dict],
    ) -> Optional[EmergentBehavior]:
        """
        Detect cooperative behaviors between agents.

        Looks for:
        - Coordinated actions
        - Resource sharing patterns
        """
        cooperation_events = 0
        total_steps = 0

        for traj in multi_agent_trajectories:
            agent_actions = traj.get('agent_actions', {})

            if len(agent_actions) < 2:
                continue

            for step in range(len(list(agent_actions.values())[0])):
                total_steps += 1

                # Check for coordinated movement (same direction)
                step_actions = [actions[step] for actions in agent_actions.values() if step < len(actions)]

                if len(set(step_actions)) == 1 and len(step_actions) > 1:
                    # All agents took same action - coordination
                    cooperation_events += 1

        frequency = cooperation_events / (total_steps + 1e-10)

        if frequency > 0.1:  # 10% coordinated actions
            return EmergentBehavior(
                name="cooperation",
                description="Agents coordinate their actions",
                frequency=frequency,
                strength=min(1.0, frequency * 3),
                evidence=[f"Observed {cooperation_events} coordinated steps"],
            )
        return None

    def analyze_all(
        self,
        trajectories: List[Dict],
        message_history: Optional[List[Dict]] = None,
    ) -> Dict[str, EmergentBehavior]:
        """Run all emergence detection analyses."""
        behaviors = {}

        # Tool use
        tool_use = self.detect_tool_use(trajectories)
        if tool_use:
            behaviors['tool_use'] = tool_use

        # Division of labor
        agent_actions = defaultdict(list)
        for traj in trajectories:
            for agent_id, actions in traj.get('agent_actions', {}).items():
                agent_actions[agent_id].extend(actions)

        division = self.detect_division_of_labor(dict(agent_actions))
        if division:
            behaviors['division_of_labor'] = division

        # Anticipatory behavior
        anticipation = self.detect_anticipatory_behavior(trajectories)
        if anticipation:
            behaviors['anticipatory'] = anticipation

        # Cooperation
        cooperation = self.detect_cooperation(trajectories)
        if cooperation:
            behaviors['cooperation'] = cooperation

        # Communication protocols
        if message_history:
            protocols = self.detect_communication_protocols(message_history)
            if protocols:
                behaviors['communication'] = EmergentBehavior(
                    name="communication_protocol",
                    description=f"Detected {len(protocols)} stable communication patterns",
                    frequency=1.0,
                    strength=min(1.0, len(protocols) / 10),
                    evidence=[f"{len(protocols)} patterns detected"],
                )

        return behaviors


class CommunicationAnalyzer:
    """Analyze message passing patterns in the swarm."""

    def __init__(self, swarm: nn.Module):
        self.swarm = swarm
        self.message_buffer = []

    def log_message(
        self,
        source: int,
        target: int,
        content: torch.Tensor,
        round: int,
    ):
        """Log a message for analysis."""
        self.message_buffer.append({
            'source': source,
            'target': target,
            'content': content.detach().cpu(),
            'round': round,
        })

    def compute_information_flow(self) -> np.ndarray:
        """
        Compute information flow matrix between agents.

        Returns adjacency matrix where entry (i,j) is information flow i->j.
        """
        if not self.message_buffer:
            return np.array([])

        # Find number of agents
        max_agent = max(
            max(m['source'] for m in self.message_buffer),
            max(m['target'] for m in self.message_buffer)
        ) + 1

        flow_matrix = np.zeros((max_agent, max_agent))

        for msg in self.message_buffer:
            source = msg['source']
            target = msg['target']
            content = msg['content']

            # Information = entropy of message
            variance = content.var().item()
            info = np.log(variance + 1e-10) + 1  # Normalize

            flow_matrix[source, target] += max(0, info)

        # Normalize by count
        count_matrix = np.zeros((max_agent, max_agent))
        for msg in self.message_buffer:
            count_matrix[msg['source'], msg['target']] += 1

        flow_matrix = flow_matrix / (count_matrix + 1e-10)

        return flow_matrix

    def find_hub_agents(self, top_k: int = 3) -> List[int]:
        """Find agents that are communication hubs."""
        flow = self.compute_information_flow()
        if flow.size == 0:
            return []

        # Hub = high total incoming + outgoing
        in_flow = flow.sum(axis=0)
        out_flow = flow.sum(axis=1)
        total_flow = in_flow + out_flow

        return list(np.argsort(total_flow)[-top_k:][::-1])

    def clear(self):
        """Clear message buffer."""
        self.message_buffer = []
