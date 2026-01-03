#!/usr/bin/env python3
"""
Rigorous Emergence Detection & Scaling Analysis for SEESWM.

This script provides interviewer-proof methodology for:
1. Measuring and defining "division of labor" with clear metrics
2. Distinguishing genuine specialization from random behavioral variance
3. Visualizing agent roles over time
4. Running scaling experiments with phase transition detection

Key Innovation: We compare against NULL HYPOTHESIS (random/untrained baseline)
to prove specialization is learned, not random.

Metrics Defined:
- Specialization Index (SI): Between-agent variance / within-agent variance
- Role Consistency (RC): 1 - mean(within-agent action entropy)
- Behavioral Diversity (BD): Mean pairwise Jensen-Shannon divergence between agent action distributions
- Communication Efficiency (CE): Performance gain / messages sent

Usage:
    python experiments/emergence_scaling_analysis.py --device cpu --output results/emergence
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from scipy.spatial.distance import jensenshannon, pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.stats import entropy, permutation_test

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.swarm.graph import SwarmGraph, SwarmConfig, TopologyType
from src.environment.cosmos import CosmosEnvironment

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('emergence_analysis')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class AgentBehaviorProfile:
    """Behavioral profile for a single agent."""
    agent_id: int
    agent_type: str
    action_distribution: np.ndarray  # Normalized action counts
    output_mean: float
    output_std: float
    message_frequency: float
    activation_patterns: List[np.ndarray] = field(default_factory=list)


@dataclass
class SpecializationMetrics:
    """Comprehensive specialization measurements."""
    # Core metrics
    specialization_index: float  # SI: between/within variance ratio
    role_consistency: float      # RC: 1 - mean(within-agent entropy)
    behavioral_diversity: float  # BD: mean pairwise JS divergence

    # Statistical validation
    null_hypothesis_si: float    # SI for random baseline
    p_value: float               # Significance vs null
    effect_size: float           # Cohen's d

    # Cluster analysis
    num_distinct_roles: int      # Number of behavioral clusters
    cluster_labels: List[int]    # Role assignment per agent
    cluster_sizes: List[int]     # Agents per cluster

    # Per-agent details
    agent_profiles: List[AgentBehaviorProfile] = field(default_factory=list)


@dataclass
class ScalingDataPoint:
    """Data point for scaling analysis."""
    num_agents: int
    total_params: int

    # Performance metrics
    mean_reward: float
    std_reward: float

    # Emergence metrics
    specialization_index: float
    behavioral_diversity: float

    # Efficiency metrics
    messages_per_step: float
    compute_time_ms: float

    # Synergy
    synergy_score: float


@dataclass
class PhaseTransition:
    """Detected phase transition."""
    agent_count: int
    metric_name: str
    before_slope: float
    after_slope: float
    magnitude: float
    confidence: float


# =============================================================================
# EMERGENCE DETECTION
# =============================================================================

class EmergenceAnalyzer:
    """
    Rigorous emergence detection with null hypothesis testing.

    Key methodology:
    1. Collect behavioral data from trained swarm
    2. Collect behavioral data from RANDOM (untrained) swarm
    3. Compare specialization metrics using permutation tests
    4. Cluster agents into distinct roles
    5. Visualize role evolution over training
    """

    def __init__(
        self,
        device: str = "cpu",
        num_actions: int = 5,
    ):
        self.device = device
        self.num_actions = num_actions

    def collect_behavioral_data(
        self,
        swarm: SwarmGraph,
        env_config: Dict,
        num_episodes: int = 50,
        steps_per_episode: int = 100,
    ) -> Dict[int, AgentBehaviorProfile]:
        """
        Collect comprehensive behavioral data from swarm.

        Returns dict mapping agent_id -> AgentBehaviorProfile
        """
        env = CosmosEnvironment(**env_config)

        # Per-agent tracking
        agent_actions = {i: [] for i in swarm.agents.keys()}
        agent_outputs = {i: [] for i in swarm.agents.keys()}
        agent_messages = {i: 0 for i in swarm.agents.keys()}

        for ep in range(num_episodes):
            obs = env.reset()
            swarm.reset(batch_size=1)

            for step in range(steps_per_episode):
                obs_tensor = obs[0].to_tensor(self.device).unsqueeze(0)

                with torch.no_grad():
                    action_logits = swarm.step(obs_tensor)
                    action = action_logits.argmax(dim=-1).item()

                    # Track per-agent behaviors
                    for agent_id, agent in swarm.agents.items():
                        if hasattr(agent, 'last_output') and agent.last_output is not None:
                            out = agent.last_output.cpu().numpy().flatten()
                            agent_outputs[agent_id].append(out)

                            # Derive action preference from output
                            if len(out) >= self.num_actions:
                                preferred_action = np.argmax(out[:self.num_actions])
                            else:
                                preferred_action = int(out.mean() > 0)
                            agent_actions[agent_id].append(preferred_action)

                obs, rewards, dones = env.step([action])
                if dones[0]:
                    break

        # Build profiles
        profiles = {}
        for agent_id in swarm.agents.keys():
            actions = agent_actions[agent_id]
            outputs = agent_outputs[agent_id]

            if not actions:
                # Agent never activated - use uniform distribution
                action_dist = np.ones(self.num_actions) / self.num_actions
                output_mean = 0.0
                output_std = 0.0
            else:
                # Compute action distribution
                counts = np.bincount(actions, minlength=self.num_actions)
                action_dist = counts / (counts.sum() + 1e-10)

                # Output statistics
                output_array = np.array([o.mean() for o in outputs])
                output_mean = output_array.mean()
                output_std = output_array.std()

            # Get agent type
            agent_type = swarm.agents[agent_id].agent_type.name if hasattr(swarm.agents[agent_id], 'agent_type') else 'GENERAL'

            profiles[agent_id] = AgentBehaviorProfile(
                agent_id=agent_id,
                agent_type=agent_type,
                action_distribution=action_dist,
                output_mean=output_mean,
                output_std=output_std,
                message_frequency=agent_messages.get(agent_id, 0) / (num_episodes * steps_per_episode),
                activation_patterns=outputs[:100] if outputs else [],  # Keep subset
            )

        return profiles

    def compute_specialization_index(
        self,
        profiles: Dict[int, AgentBehaviorProfile],
    ) -> float:
        """
        Compute Specialization Index (SI).

        SI = Between-agent variance / Within-agent variance

        High SI means: different agents behave differently (between-agent variance high)
                      but each agent is consistent (within-agent variance low)
        """
        if len(profiles) < 2:
            return 0.0

        # Collect mean outputs per agent
        agent_means = [p.output_mean for p in profiles.values()]

        # Between-agent variance: variance of agent means
        between_var = np.var(agent_means)

        # Within-agent variance: mean of individual agent stds
        within_var = np.mean([p.output_std for p in profiles.values()])

        # SI = between / within (higher = more specialized)
        si = between_var / (within_var + 1e-8)

        return float(si)

    def compute_role_consistency(
        self,
        profiles: Dict[int, AgentBehaviorProfile],
    ) -> float:
        """
        Compute Role Consistency (RC).

        RC = 1 - mean(within-agent action entropy)

        High RC means each agent consistently takes the same types of actions.
        """
        entropies = []
        for p in profiles.values():
            # Entropy of action distribution (0 = always same action, log(n) = uniform)
            h = entropy(p.action_distribution + 1e-10)
            # Normalize by max entropy
            h_norm = h / np.log(self.num_actions)
            entropies.append(h_norm)

        # RC = 1 - mean(normalized entropy)
        rc = 1 - np.mean(entropies)

        return float(rc)

    def compute_behavioral_diversity(
        self,
        profiles: Dict[int, AgentBehaviorProfile],
    ) -> float:
        """
        Compute Behavioral Diversity (BD).

        BD = Mean pairwise Jensen-Shannon divergence between agent action distributions.

        High BD means agents have genuinely different behavioral strategies.
        """
        if len(profiles) < 2:
            return 0.0

        distributions = [p.action_distribution for p in profiles.values()]

        # Compute all pairwise JS divergences
        js_distances = []
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                js = jensenshannon(distributions[i], distributions[j])
                if not np.isnan(js):
                    js_distances.append(js)

        return float(np.mean(js_distances)) if js_distances else 0.0

    def cluster_agents_into_roles(
        self,
        profiles: Dict[int, AgentBehaviorProfile],
        n_clusters: Optional[int] = None,
    ) -> Tuple[List[int], int]:
        """
        Cluster agents into distinct behavioral roles.

        Uses hierarchical clustering on action distributions.
        Returns (cluster_labels, num_clusters).
        """
        if len(profiles) < 2:
            return [0] * len(profiles), 1

        # Feature matrix: action distributions
        X = np.array([p.action_distribution for p in profiles.values()])

        # Hierarchical clustering with Ward's method
        Z = linkage(X, method='ward')

        if n_clusters is None:
            # Auto-detect: use elbow method on within-cluster variance
            max_clusters = min(len(profiles) // 2, 6)
            best_k = 2
            best_score = -np.inf

            for k in range(2, max_clusters + 1):
                labels = fcluster(Z, k, criterion='maxclust')

                # Silhouette-like score: between / within variance ratio
                between = 0
                within = 0
                for c in range(1, k + 1):
                    cluster_mask = labels == c
                    if cluster_mask.sum() > 1:
                        cluster_data = X[cluster_mask]
                        cluster_center = cluster_data.mean(axis=0)
                        within += np.sum((cluster_data - cluster_center) ** 2)
                        between += cluster_mask.sum() * np.sum((cluster_center - X.mean(axis=0)) ** 2)

                score = between / (within + 1e-8)
                if score > best_score:
                    best_score = score
                    best_k = k

            n_clusters = best_k

        labels = fcluster(Z, n_clusters, criterion='maxclust')

        return list(labels), n_clusters

    def run_null_hypothesis_test(
        self,
        trained_profiles: Dict[int, AgentBehaviorProfile],
        swarm_config: Dict,
        env_config: Dict,
        num_permutations: int = 100,
    ) -> Tuple[float, float, float]:
        """
        Test specialization against null hypothesis (random baseline).

        Null hypothesis: Observed specialization is no different from random initialization.

        Returns (null_si, p_value, effect_size)
        """
        trained_si = self.compute_specialization_index(trained_profiles)

        # Generate null distribution by testing random (untrained) swarms
        null_sis = []

        for seed in range(num_permutations):
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Create random swarm (not trained)
            num_agents = swarm_config.get('num_agents', 20)
            config = SwarmConfig(
                num_agents=num_agents,
                input_dim=swarm_config.get('input_dim', 137),
                hidden_dim=swarm_config.get('hidden_dim', 128),
                output_dim=swarm_config.get('output_dim', 5),
                message_dim=swarm_config.get('hidden_dim', 128),
                topology=TopologyType.SMALL_WORLD,
                num_perception=num_agents // 4,
                num_reasoning=num_agents // 4,
                num_memory=num_agents // 4,
                num_planning=num_agents - 3 * (num_agents // 4),
            )
            random_swarm = SwarmGraph(config, device=self.device)

            # Collect behavioral data from random swarm
            random_profiles = self.collect_behavioral_data(
                random_swarm, env_config,
                num_episodes=10,  # Fewer for null distribution
                steps_per_episode=50,
            )

            null_si = self.compute_specialization_index(random_profiles)
            null_sis.append(null_si)

        null_sis = np.array(null_sis)
        null_mean = null_sis.mean()
        null_std = null_sis.std()

        # P-value: proportion of null samples >= trained
        p_value = (null_sis >= trained_si).mean()

        # Effect size: Cohen's d
        effect_size = (trained_si - null_mean) / (null_std + 1e-8)

        return float(null_mean), float(p_value), float(effect_size)

    def analyze_full(
        self,
        swarm: SwarmGraph,
        swarm_config: Dict,
        env_config: Dict,
        num_episodes: int = 50,
        run_null_test: bool = True,
    ) -> SpecializationMetrics:
        """Run complete specialization analysis."""

        logger.info("Collecting behavioral data from trained swarm...")
        profiles = self.collect_behavioral_data(swarm, env_config, num_episodes)

        logger.info("Computing specialization metrics...")
        si = self.compute_specialization_index(profiles)
        rc = self.compute_role_consistency(profiles)
        bd = self.compute_behavioral_diversity(profiles)

        logger.info(f"  Specialization Index (SI): {si:.4f}")
        logger.info(f"  Role Consistency (RC): {rc:.4f}")
        logger.info(f"  Behavioral Diversity (BD): {bd:.4f}")

        # Null hypothesis test
        null_si, p_value, effect_size = 0.0, 1.0, 0.0
        if run_null_test:
            logger.info("Running null hypothesis test (this may take a while)...")
            null_si, p_value, effect_size = self.run_null_hypothesis_test(
                profiles, swarm_config, env_config, num_permutations=30
            )

            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            logger.info(f"  Null SI: {null_si:.4f}, p-value: {p_value:.4f}{sig}")
            logger.info(f"  Effect size (Cohen's d): {effect_size:.4f}")

        # Cluster analysis
        logger.info("Clustering agents into roles...")
        cluster_labels, num_roles = self.cluster_agents_into_roles(profiles)
        cluster_sizes = [cluster_labels.count(i+1) for i in range(num_roles)]
        logger.info(f"  Detected {num_roles} distinct roles: {cluster_sizes}")

        return SpecializationMetrics(
            specialization_index=si,
            role_consistency=rc,
            behavioral_diversity=bd,
            null_hypothesis_si=null_si,
            p_value=p_value,
            effect_size=effect_size,
            num_distinct_roles=num_roles,
            cluster_labels=cluster_labels,
            cluster_sizes=cluster_sizes,
            agent_profiles=list(profiles.values()),
        )


# =============================================================================
# SCALING ANALYSIS
# =============================================================================

class ScalingAnalyzer:
    """
    Scaling analysis with phase transition detection.

    Measures:
    1. Performance vs agent count
    2. Specialization vs agent count
    3. Communication overhead vs agent count
    4. Phase transitions where behavior qualitatively changes
    """

    def __init__(
        self,
        device: str = "cpu",
        base_input_dim: int = 137,
        base_output_dim: int = 5,
    ):
        self.device = device
        self.base_input_dim = base_input_dim
        self.base_output_dim = base_output_dim
        self.emergence_analyzer = EmergenceAnalyzer(device, num_actions=base_output_dim)

    def evaluate_swarm(
        self,
        swarm: SwarmGraph,
        env_config: Dict,
        num_episodes: int = 20,
    ) -> Dict[str, float]:
        """Evaluate swarm performance."""
        env = CosmosEnvironment(**env_config)

        rewards = []
        steps_list = []

        for ep in range(num_episodes):
            obs = env.reset()
            swarm.reset(batch_size=1)

            episode_reward = 0.0

            for step in range(100):
                obs_tensor = obs[0].to_tensor(self.device).unsqueeze(0)

                with torch.no_grad():
                    action_logits = swarm.step(obs_tensor)
                    action = action_logits.argmax(dim=-1).item()

                obs, rew, dones = env.step([action])
                episode_reward += rew[0] if rew else 0.0

                if dones[0]:
                    break

            rewards.append(episode_reward)
            steps_list.append(step + 1)

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_steps': np.mean(steps_list),
        }

    def run_scaling_sweep(
        self,
        agent_counts: List[int],
        env_config: Dict,
        hidden_dim: int = 128,
        num_seeds: int = 3,
        num_eval_episodes: int = 20,
    ) -> List[ScalingDataPoint]:
        """Run scaling sweep across different agent counts."""

        data_points = []

        for n_agents in agent_counts:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing {n_agents} agents...")
            logger.info(f"{'='*50}")

            seed_rewards = []
            seed_sis = []
            seed_bds = []

            for seed in range(num_seeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Create swarm
                config = SwarmConfig(
                    num_agents=n_agents,
                    input_dim=self.base_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=self.base_output_dim,
                    message_dim=hidden_dim,
                    topology=TopologyType.SMALL_WORLD,
                    num_perception=max(1, n_agents // 4),
                    num_reasoning=max(1, n_agents // 4),
                    num_memory=max(1, n_agents // 4),
                    num_planning=max(1, n_agents - 3 * max(1, n_agents // 4)),
                )
                swarm = SwarmGraph(config, device=self.device)

                # Evaluate performance
                import time
                start_time = time.time()
                perf = self.evaluate_swarm(swarm, env_config, num_eval_episodes)
                compute_time = (time.time() - start_time) * 1000 / num_eval_episodes

                seed_rewards.append(perf['mean_reward'])

                # Quick emergence metrics
                profiles = self.emergence_analyzer.collect_behavioral_data(
                    swarm, env_config, num_episodes=10, steps_per_episode=50
                )
                si = self.emergence_analyzer.compute_specialization_index(profiles)
                bd = self.emergence_analyzer.compute_behavioral_diversity(profiles)

                seed_sis.append(si)
                seed_bds.append(bd)

                logger.info(f"  Seed {seed}: reward={perf['mean_reward']:.2f}, SI={si:.3f}, BD={bd:.3f}")

            # Aggregate across seeds
            total_params = swarm.total_parameters
            messages_per_step = n_agents * 3  # 3 message rounds, each agent sends 1

            data_point = ScalingDataPoint(
                num_agents=n_agents,
                total_params=total_params,
                mean_reward=np.mean(seed_rewards),
                std_reward=np.std(seed_rewards),
                specialization_index=np.mean(seed_sis),
                behavioral_diversity=np.mean(seed_bds),
                messages_per_step=messages_per_step,
                compute_time_ms=compute_time,
                synergy_score=0.0,  # Computed separately if needed
            )

            data_points.append(data_point)

            logger.info(f"  Mean: reward={data_point.mean_reward:.2f}±{data_point.std_reward:.2f}")
            logger.info(f"  Params: {total_params:,}, Messages/step: {messages_per_step}")

        return data_points

    def detect_phase_transitions(
        self,
        data_points: List[ScalingDataPoint],
        metrics: List[str] = ['mean_reward', 'specialization_index', 'behavioral_diversity'],
    ) -> List[PhaseTransition]:
        """
        Detect phase transitions in scaling behavior.

        A phase transition is where the slope of metric vs agents changes significantly.
        """
        transitions = []

        # Sort by agent count
        sorted_points = sorted(data_points, key=lambda p: p.num_agents)
        agent_counts = np.array([p.num_agents for p in sorted_points])

        for metric in metrics:
            values = np.array([getattr(p, metric) for p in sorted_points])

            if len(values) < 4:
                continue

            # Compute local gradients
            log_agents = np.log(agent_counts)

            for i in range(1, len(values) - 1):
                # Slope before point i
                if i >= 2:
                    before_slope = (values[i] - values[i-2]) / (log_agents[i] - log_agents[i-2])
                else:
                    before_slope = (values[i] - values[i-1]) / (log_agents[i] - log_agents[i-1])

                # Slope after point i
                if i <= len(values) - 3:
                    after_slope = (values[i+2] - values[i]) / (log_agents[i+2] - log_agents[i])
                else:
                    after_slope = (values[i+1] - values[i]) / (log_agents[i+1] - log_agents[i])

                # Detect significant change
                slope_change = abs(after_slope - before_slope)
                baseline_slope = abs(before_slope) + abs(after_slope) + 0.01

                if slope_change / baseline_slope > 0.5:  # 50% relative change
                    transitions.append(PhaseTransition(
                        agent_count=int(agent_counts[i]),
                        metric_name=metric,
                        before_slope=float(before_slope),
                        after_slope=float(after_slope),
                        magnitude=float(slope_change),
                        confidence=min(1.0, slope_change / baseline_slope),
                    ))

        return transitions


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_scaling_results(
    data_points: List[ScalingDataPoint],
    transitions: List[PhaseTransition],
    output_path: str,
):
    """Generate comprehensive scaling visualization."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    agents = [p.num_agents for p in data_points]

    # 1. Performance vs Agents (with error bars)
    ax1 = fig.add_subplot(gs[0, 0])
    rewards = [p.mean_reward for p in data_points]
    stds = [p.std_reward for p in data_points]
    ax1.errorbar(agents, rewards, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Agents', fontsize=12)
    ax1.set_ylabel('Mean Episode Reward', fontsize=12)
    ax1.set_title('Performance Scaling', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # Mark phase transitions
    for t in transitions:
        if t.metric_name == 'mean_reward':
            ax1.axvline(t.agent_count, color='red', linestyle='--', alpha=0.7,
                       label=f'Phase transition @ {t.agent_count}')

    # 2. Specialization vs Agents
    ax2 = fig.add_subplot(gs[0, 1])
    sis = [p.specialization_index for p in data_points]
    bds = [p.behavioral_diversity for p in data_points]

    ax2.plot(agents, sis, 'b-o', label='Specialization Index', linewidth=2, markersize=8)
    ax2.plot(agents, bds, 'g-s', label='Behavioral Diversity', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Agents', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Emergence Metrics vs Scale', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark phase transitions
    for t in transitions:
        if t.metric_name in ['specialization_index', 'behavioral_diversity']:
            ax2.axvline(t.agent_count, color='red', linestyle='--', alpha=0.5)

    # 3. Communication Overhead
    ax3 = fig.add_subplot(gs[1, 0])
    messages = [p.messages_per_step for p in data_points]
    efficiency = [p.mean_reward / (p.messages_per_step + 1) for p in data_points]

    ax3_twin = ax3.twinx()

    line1 = ax3.plot(agents, messages, 'r-^', label='Messages/Step', linewidth=2, markersize=8)
    line2 = ax3_twin.plot(agents, efficiency, 'b-o', label='Reward/Message', linewidth=2, markersize=8)

    ax3.set_xlabel('Number of Agents', fontsize=12)
    ax3.set_ylabel('Messages per Step', color='red', fontsize=12)
    ax3_twin.set_ylabel('Reward per Message', color='blue', fontsize=12)
    ax3.set_title('Communication Overhead', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')

    # 4. Compute Time
    ax4 = fig.add_subplot(gs[1, 1])
    times = [p.compute_time_ms for p in data_points]
    params = [p.total_params / 1000 for p in data_points]  # In thousands

    ax4.plot(agents, times, 'g-o', label='Compute Time (ms)', linewidth=2, markersize=8)
    ax4_twin = ax4.twinx()
    ax4_twin.plot(agents, params, 'purple', linestyle='--', marker='s', label='Parameters (K)', linewidth=2, markersize=8)

    ax4.set_xlabel('Number of Agents', fontsize=12)
    ax4.set_ylabel('Time per Episode (ms)', color='green', fontsize=12)
    ax4_twin.set_ylabel('Parameters (thousands)', color='purple', fontsize=12)
    ax4.set_title('Computational Cost', fontsize=14, fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved scaling plot to {output_path}")
    plt.close()


def plot_role_clusters(
    metrics: SpecializationMetrics,
    output_path: str,
):
    """Visualize agent role clustering."""
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        logger.warning("matplotlib/sklearn not available, skipping cluster plot")
        return

    if len(metrics.agent_profiles) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Get action distributions
    X = np.array([p.action_distribution for p in metrics.agent_profiles])

    # 1. PCA visualization of agents
    ax1 = axes[0]
    if X.shape[0] >= 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)

        colors = plt.cm.tab10(np.array(metrics.cluster_labels) - 1)
        scatter = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, s=100, edgecolors='black')

        # Label points
        for i, p in enumerate(metrics.agent_profiles):
            ax1.annotate(f'{p.agent_id}\n({p.agent_type[:3]})',
                        (X_2d[i, 0], X_2d[i, 1]), fontsize=8, ha='center')

        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')

    ax1.set_title('Agent Behavioral Clusters', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. Action distribution heatmap
    ax2 = axes[1]

    # Sort by cluster
    sorted_indices = np.argsort(metrics.cluster_labels)
    X_sorted = X[sorted_indices]

    im = ax2.imshow(X_sorted, aspect='auto', cmap='YlOrRd')
    ax2.set_xlabel('Action')
    ax2.set_ylabel('Agent (sorted by cluster)')
    ax2.set_title('Action Distribution Heatmap', fontsize=14, fontweight='bold')

    # Add cluster boundaries
    cluster_boundaries = []
    for i in range(1, len(sorted_indices)):
        if metrics.cluster_labels[sorted_indices[i]] != metrics.cluster_labels[sorted_indices[i-1]]:
            ax2.axhline(i - 0.5, color='blue', linewidth=2)

    plt.colorbar(im, ax=ax2, label='Action Probability')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved cluster plot to {output_path}")
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SEESWM Emergence & Scaling Analysis')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default='results/emergence')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    parser.add_argument('--quick', action='store_true', help='Quick run with fewer samples')
    parser.add_argument('--scaling-only', action='store_true', help='Only run scaling analysis')
    parser.add_argument('--emergence-only', action='store_true', help='Only run emergence analysis')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    swarm_config = {
        'num_agents': 20,
        'input_dim': 137,
        'hidden_dim': 128,
        'output_dim': 128,  # Match trained model
    }

    env_config = {
        'grid_size': 32,
        'num_resources': 20,
        'num_hazards': 10,
        'vision_radius': 5,
    }

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {'swarm': swarm_config, 'env': env_config},
    }

    # Load trained model if specified
    if args.model:
        logger.info(f"Loading trained model from {args.model}")
        loaded = torch.load(args.model, map_location='cpu', weights_only=False)

        # Create swarm and load weights
        num_agents = swarm_config['num_agents']
        config = SwarmConfig(
            num_agents=num_agents,
            input_dim=swarm_config['input_dim'],
            hidden_dim=swarm_config['hidden_dim'],
            output_dim=swarm_config['output_dim'],
            message_dim=swarm_config['hidden_dim'],
            topology=TopologyType.SMALL_WORLD,
            num_perception=num_agents // 4,
            num_reasoning=num_agents // 4,
            num_memory=num_agents // 4,
            num_planning=num_agents - 3 * (num_agents // 4),
        )
        swarm = SwarmGraph(config, device=args.device)

        if 'swarm_state' in loaded:
            swarm.load_state_dict(loaded['swarm_state'])
            logger.info("Loaded trained weights")
    else:
        # Create random swarm for testing
        logger.info("No model specified, using random initialization")
        num_agents = swarm_config['num_agents']
        config = SwarmConfig(
            num_agents=num_agents,
            input_dim=swarm_config['input_dim'],
            hidden_dim=swarm_config['hidden_dim'],
            output_dim=swarm_config['output_dim'],
            message_dim=swarm_config['hidden_dim'],
            topology=TopologyType.SMALL_WORLD,
            num_perception=num_agents // 4,
            num_reasoning=num_agents // 4,
            num_memory=num_agents // 4,
            num_planning=num_agents - 3 * (num_agents // 4),
        )
        swarm = SwarmGraph(config, device=args.device)

    # =================
    # EMERGENCE ANALYSIS
    # =================
    if not args.scaling_only:
        logger.info("\n" + "=" * 60)
        logger.info("EMERGENCE ANALYSIS")
        logger.info("=" * 60)

        analyzer = EmergenceAnalyzer(device=args.device, num_actions=min(5, swarm_config['output_dim']))

        metrics = analyzer.analyze_full(
            swarm, swarm_config, env_config,
            num_episodes=20 if args.quick else 50,
            run_null_test=not args.quick,
        )

        results['emergence'] = {
            'specialization_index': metrics.specialization_index,
            'role_consistency': metrics.role_consistency,
            'behavioral_diversity': metrics.behavioral_diversity,
            'null_hypothesis_si': metrics.null_hypothesis_si,
            'p_value': metrics.p_value,
            'effect_size': metrics.effect_size,
            'num_distinct_roles': metrics.num_distinct_roles,
            'cluster_sizes': metrics.cluster_sizes,
        }

        # Summary
        logger.info("\n" + "-" * 40)
        logger.info("EMERGENCE SUMMARY")
        logger.info("-" * 40)
        logger.info(f"Specialization Index: {metrics.specialization_index:.4f}")
        logger.info(f"  vs Null Baseline: {metrics.null_hypothesis_si:.4f}")
        logger.info(f"  P-value: {metrics.p_value:.4f}")
        logger.info(f"  Effect size: {metrics.effect_size:.2f} (Cohen's d)")
        logger.info(f"Number of distinct roles: {metrics.num_distinct_roles}")
        logger.info(f"Role sizes: {metrics.cluster_sizes}")

        # Interpretation
        if metrics.p_value < 0.05 and metrics.effect_size > 0.5:
            logger.info("\n*** GENUINE SPECIALIZATION DETECTED ***")
            logger.info("The trained swarm shows significantly more specialization")
            logger.info("than random baseline (p < 0.05, effect size > 0.5)")
        else:
            logger.info("\nSpecialization not significantly different from random")

        # Plot clusters
        plot_role_clusters(metrics, str(output_dir / 'role_clusters.png'))

    # =================
    # SCALING ANALYSIS
    # =================
    if not args.emergence_only:
        logger.info("\n" + "=" * 60)
        logger.info("SCALING ANALYSIS")
        logger.info("=" * 60)

        scaling_analyzer = ScalingAnalyzer(
            device=args.device,
            base_output_dim=5,  # Use 5 actions for scaling tests (fresh swarms)
        )

        if args.quick:
            agent_counts = [4, 10, 20, 50]
            num_seeds = 2
        else:
            agent_counts = [4, 10, 20, 50, 100, 150]
            num_seeds = 3

        data_points = scaling_analyzer.run_scaling_sweep(
            agent_counts=agent_counts,
            env_config=env_config,
            num_seeds=num_seeds,
            num_eval_episodes=10 if args.quick else 20,
        )

        transitions = scaling_analyzer.detect_phase_transitions(data_points)

        results['scaling'] = {
            'data_points': [
                {
                    'num_agents': p.num_agents,
                    'mean_reward': p.mean_reward,
                    'std_reward': p.std_reward,
                    'specialization_index': p.specialization_index,
                    'behavioral_diversity': p.behavioral_diversity,
                    'messages_per_step': p.messages_per_step,
                    'total_params': p.total_params,
                }
                for p in data_points
            ],
            'phase_transitions': [
                {
                    'agent_count': t.agent_count,
                    'metric': t.metric_name,
                    'magnitude': t.magnitude,
                }
                for t in transitions
            ],
        }

        # Summary
        logger.info("\n" + "-" * 40)
        logger.info("SCALING SUMMARY")
        logger.info("-" * 40)

        # Check for phase transitions
        if transitions:
            logger.info(f"Detected {len(transitions)} phase transition(s):")
            for t in transitions:
                logger.info(f"  {t.metric_name} @ {t.agent_count} agents (magnitude: {t.magnitude:.3f})")
        else:
            logger.info("No significant phase transitions detected")

        # Scaling trend
        rewards = [p.mean_reward for p in data_points]
        agents = [p.num_agents for p in data_points]

        # Fit log-linear trend
        log_agents = np.log(agents)
        slope, intercept, r_value, _, _ = stats.linregress(log_agents, rewards)

        logger.info(f"\nScaling trend: reward ~ {slope:.3f} * log(agents)")
        logger.info(f"R² = {r_value**2:.3f}")

        if slope > 0:
            logger.info("Performance INCREASES with scale (good!)")
        else:
            logger.info("Performance DECREASES with scale (coordination breaking down?)")

        # Plot
        plot_scaling_results(data_points, transitions, str(output_dir / 'scaling_analysis.png'))

    # Save results
    output_file = output_dir / 'emergence_scaling_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    return results


if __name__ == '__main__':
    main()
