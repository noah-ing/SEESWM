#!/usr/bin/env python3
"""
Exploratory behavioral-pattern and scaling analysis for SEESWM.

This script provides utilities for:
1. Measuring candidate "division of labor" with explicit metrics
2. Comparing candidate specialization with random behavioral variance
3. Visualizing agent roles over time
4. Running scaling experiments with heuristic slope-change flags

When a trained checkpoint is supplied, the candidate model can be compared with
randomly initialized swarms. Without `--model`, all results describe random
initialization and cannot demonstrate learned specialization.

Metrics Defined:
- Specialization Index (SI): Between-agent variance / within-agent variance
- Role Consistency (RC): 1 - mean(within-agent action entropy)
- Behavioral Diversity (BD): Mean pairwise Jensen-Shannon divergence between agent action distributions

Usage:
    python experiments/emergence_scaling_analysis.py --device cpu --output results/behavioral-local
"""

import argparse
import json
import logging
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.swarm.graph import (
    SwarmGraph,
    SwarmConfig,
    TopologyType,
    swarm_config_from_dict,
    swarm_config_to_dict,
)
from src.environment.cosmos import EnvironmentConfig
from experiments.validate_rigorously import (
    build_candidate,
    build_environment,
    environment_config_to_dict,
    load_checkpoint_with_digest,
    set_swarm_eval,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('behavioral_analysis')


def _analysis_source_provenance() -> Dict[str, Any]:
    """Return the current revision and dirty flag without requiring Git."""
    repository = Path(__file__).resolve().parent.parent
    try:
        revision = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        status = subprocess.run(
            ['git', 'status', '--porcelain', '--untracked-files=normal'],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return {'revision': None, 'dirty': None}
    return {
        'revision': revision.stdout.strip() if revision.returncode == 0 else None,
        'dirty': bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _random_control_seeds(candidate_seed: int, count: int) -> List[int]:
    """Choose declared nonnegative control seeds excluding the candidate seed."""
    seeds: List[int] = []
    value = 0
    while len(seeds) < count:
        if value != candidate_seed:
            seeds.append(value)
        value += 1
    return seeds


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _balanced_role_counts(num_agents: int) -> Tuple[int, int, int, int]:
    """Assign every agent to exactly one of the four declared roles."""
    if num_agents < 1:
        raise ValueError("num_agents must be positive")
    base, remainder = divmod(num_agents, 4)
    counts = [base + int(index < remainder) for index in range(4)]
    return counts[0], counts[1], counts[2], counts[3]


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
    activation_patterns: List[np.ndarray] = field(default_factory=list)


@dataclass
class SpecializationMetrics:
    """Comprehensive specialization measurements."""
    # Core metrics
    specialization_index: float  # SI: between/within variance ratio
    role_consistency: float      # RC: 1 - mean(within-agent entropy)
    behavioral_diversity: float  # BD: mean pairwise JS divergence

    # Fresh-random comparison (descriptive; not independent validation)
    random_control_si: Optional[float]
    monte_carlo_p_value: Optional[float]
    standardized_null_difference: Optional[float]
    null_sample_count: int
    random_comparison_performed: bool

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

    # Behavioral-pattern metrics
    specialization_index: float
    behavioral_diversity: float

    # Architectural work proxy
    agent_forward_calls_per_step: int
    compute_time_ms: float


@dataclass
class SlopeChangeFlag:
    """Heuristic local slope-change flag; not evidence of a phase transition."""
    agent_count: int
    metric_name: str
    before_slope: float
    after_slope: float
    magnitude: float
    relative_change_score: float


# =============================================================================
# BEHAVIORAL-PATTERN ANALYSIS
# =============================================================================

class EmergenceAnalyzer:
    """
    Exploratory behavioral-pattern analysis with a random-init comparison.

    Methodology:
    1. Collect behavioral data from a candidate swarm
    2. Collect behavioral data from randomly initialized controls
    3. Compare descriptive specialization metrics
    4. Cluster agents into distinct roles

    These diagnostics do not establish emergence. A trained candidate must use
    the exact policy head restored from its checkpoint; representation vectors
    are never treated as environment actions directly.
    """

    def __init__(
        self,
        device: str = "cpu",
        num_actions: int = 5,
        policy_head: Optional[nn.Module] = None,
        policy_head_spec: Optional[Dict[str, Any]] = None,
        candidate_seed: int = 0,
    ):
        self.device = device
        self.num_actions = num_actions
        self.policy_head = policy_head
        self.policy_head_spec = policy_head_spec
        self.candidate_seed = candidate_seed

    def _policy_logits(self, representation: torch.Tensor) -> torch.Tensor:
        """Map a swarm representation to exactly ``num_actions`` logits."""
        logits = (
            self.policy_head(representation)
            if self.policy_head is not None
            else representation
        )
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(f"invalid policy-logit shape: {tuple(logits.shape)}")
        if logits.shape[-1] != self.num_actions:
            raise ValueError(
                "a policy head is required when swarm output_dim differs from "
                f"the {self.num_actions}-action environment"
            )
        return logits

    def _fresh_policy_head(self) -> Optional[nn.Module]:
        """Create a randomly initialized control head matching the candidate."""
        if self.policy_head_spec is None:
            return None
        policy_type = self.policy_head_spec.get("type")
        input_dim = int(self.policy_head_spec["input_dim"])
        hidden_dim = int(self.policy_head_spec["hidden_dim"])
        num_actions = int(self.policy_head_spec["num_actions"])
        if num_actions != self.num_actions:
            raise ValueError("control policy action count does not match the analyzer")
        if policy_type == "mlp_relu":
            policy = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions),
            )
        elif policy_type == "policy_head_tanh":
            from src.training import PolicyHead

            policy = PolicyHead(input_dim, num_actions, hidden_dim=hidden_dim)
        else:
            raise ValueError(f"unsupported policy head type: {policy_type!r}")
        return policy.to(self.device).eval()

    def collect_behavioral_data(
        self,
        swarm: SwarmGraph,
        env_config: Dict,
        num_episodes: int = 50,
        steps_per_episode: int = 100,
        evaluation_seed: Optional[int] = None,
    ) -> Dict[int, AgentBehaviorProfile]:
        """
        Collect comprehensive behavioral data from swarm.

        Returns dict mapping agent_id -> AgentBehaviorProfile
        """
        if num_episodes < 1 or steps_per_episode < 1:
            raise ValueError("behavioral sampling counts must be positive")
        if evaluation_seed is not None:
            _seed_everything(evaluation_seed)
        env = build_environment(env_config)

        # Per-agent tracking
        agent_actions = {i: [] for i in swarm.agents.keys()}
        agent_outputs = {i: [] for i in swarm.agents.keys()}

        for ep in range(num_episodes):
            obs = env.reset()
            swarm.reset(batch_size=1)

            for step in range(steps_per_episode):
                obs_tensor = obs[0].to_tensor(self.device).unsqueeze(0)

                with torch.no_grad():
                    representation = swarm.step(obs_tensor)
                    action_logits = self._policy_logits(representation)
                    action = action_logits.argmax(dim=-1).item()

                    # Track per-agent behaviors
                    for agent_id, agent in swarm.agents.items():
                        if hasattr(agent, 'last_output') and agent.last_output is not None:
                            agent_output = agent.last_output
                            out = agent_output.cpu().numpy().flatten()
                            agent_outputs[agent_id].append(out)

                            # Map each agent representation through the same
                            # candidate policy head before deriving a preference.
                            agent_logits = self._policy_logits(agent_output)
                            preferred_action = int(agent_logits.argmax(dim=-1).item())
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

        # Within-agent variance: mean of squared per-agent standard deviations.
        within_var = np.mean([p.output_std**2 for p in profiles.values()])

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

        High BD means the observed action distributions differ more.
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
        # ``maxclust`` is an upper bound and can return fewer clusters. Report
        # only labels that actually occurred, normalized to 1..k.
        unique_labels = sorted(int(label) for label in np.unique(labels))
        normalized = {
            label: index + 1 for index, label in enumerate(unique_labels)
        }
        normalized_labels = [normalized[int(label)] for label in labels]

        return normalized_labels, len(unique_labels)

    def run_random_init_comparison(
        self,
        trained_profiles: Dict[int, AgentBehaviorProfile],
        swarm_config: Dict,
        env_config: Dict,
        num_permutations: int = 100,
        num_episodes: int = 50,
        steps_per_episode: int = 100,
    ) -> Tuple[float, float, Optional[float], int]:
        """
        Compare the candidate SI with fresh random-initialization controls.

        This is an exploratory Monte Carlo comparison, not an independently
        validated hypothesis test.

        Returns (null_si, plus-one Monte Carlo p, standardized difference, n).
        """
        trained_si = self.compute_specialization_index(trained_profiles)

        # Generate null distribution by testing random (untrained) swarms
        null_sis = []

        for seed in _random_control_seeds(self.candidate_seed, num_permutations):
            _seed_everything(seed)

            # Match the candidate architecture and topology while using fresh,
            # untrained parameters for both the swarm and policy head.
            if self.policy_head_spec and self.policy_head_spec.get("type") == "policy_head_tanh":
                from src.swarm.specialized_graph import (
                    SpecializedSwarmGraph,
                    specialized_swarm_config_from_dict,
                )

                config = specialized_swarm_config_from_dict(swarm_config)
                random_swarm = SpecializedSwarmGraph(config, device=self.device)
            else:
                config = swarm_config_from_dict(swarm_config)
                random_swarm = SwarmGraph(config, device=self.device)
            set_swarm_eval(random_swarm)
            random_analyzer = EmergenceAnalyzer(
                device=self.device,
                num_actions=self.num_actions,
                policy_head=self._fresh_policy_head(),
                policy_head_spec=self.policy_head_spec,
                candidate_seed=seed,
            )

            # Collect behavioral data from random swarm
            random_profiles = random_analyzer.collect_behavioral_data(
                random_swarm, env_config,
                num_episodes=num_episodes,
                steps_per_episode=steps_per_episode,
                evaluation_seed=self.candidate_seed,
            )

            null_si = self.compute_specialization_index(random_profiles)
            null_sis.append(null_si)

        null_sis = np.array(null_sis)
        null_mean = null_sis.mean()
        null_std = null_sis.std()

        # Plus-one correction prevents an impossible zero p-value and gives the
        # finite Monte Carlo resolution explicitly.
        exceedances = int(np.sum(null_sis >= trained_si))
        p_value = (exceedances + 1) / (len(null_sis) + 1)

        # This is not Cohen's d: there is one candidate aggregate and a random
        # control distribution, so report only a standardized null difference.
        standardized_difference = (
            None
            if null_std <= np.finfo(float).eps
            else float((trained_si - null_mean) / null_std)
        )

        return (
            float(null_mean),
            float(p_value),
            standardized_difference,
            int(len(null_sis)),
        )

    def analyze_full(
        self,
        swarm: SwarmGraph,
        swarm_config: Dict,
        env_config: Dict,
        num_episodes: int = 50,
        steps_per_episode: int = 100,
        run_null_test: bool = True,
    ) -> SpecializationMetrics:
        """Run complete specialization analysis."""

        logger.info("Collecting behavioral data from candidate swarm...")
        profiles = self.collect_behavioral_data(
            swarm,
            env_config,
            num_episodes=num_episodes,
            steps_per_episode=steps_per_episode,
            evaluation_seed=self.candidate_seed,
        )

        logger.info("Computing specialization metrics...")
        si = self.compute_specialization_index(profiles)
        rc = self.compute_role_consistency(profiles)
        bd = self.compute_behavioral_diversity(profiles)

        logger.info(f"  Specialization Index (SI): {si:.4f}")
        logger.info(f"  Role Consistency (RC): {rc:.4f}")
        logger.info(f"  Behavioral Diversity (BD): {bd:.4f}")

        # Null hypothesis test
        random_control_si = None
        monte_carlo_p_value = None
        standardized_null_difference = None
        null_sample_count = 0
        random_comparison_performed = False
        if run_null_test:
            logger.info("Running fresh-random comparison (this may take a while)...")
            (
                random_control_si,
                monte_carlo_p_value,
                standardized_null_difference,
                null_sample_count,
            ) = self.run_random_init_comparison(
                profiles,
                swarm_config,
                env_config,
                num_permutations=30,
                num_episodes=num_episodes,
                steps_per_episode=steps_per_episode,
            )
            random_comparison_performed = True

            logger.info(
                "  Random-control SI: %.4f, plus-one Monte Carlo p=%.4f (n=%d)",
                random_control_si,
                monte_carlo_p_value,
                null_sample_count,
            )
            if standardized_null_difference is None:
                logger.info("  Standardized null difference: undefined (zero null SD)")
            else:
                logger.info(
                    "  Standardized null difference: %.4f",
                    standardized_null_difference,
                )

        # Cluster analysis
        logger.info("Clustering observed action distributions...")
        cluster_labels, num_roles = self.cluster_agents_into_roles(profiles)
        cluster_sizes = [cluster_labels.count(i+1) for i in range(num_roles)]
        logger.info(f"  Returned {num_roles} behavioral cluster(s): {cluster_sizes}")

        return SpecializationMetrics(
            specialization_index=si,
            role_consistency=rc,
            behavioral_diversity=bd,
            random_control_si=random_control_si,
            monte_carlo_p_value=monte_carlo_p_value,
            standardized_null_difference=standardized_null_difference,
            null_sample_count=null_sample_count,
            random_comparison_performed=random_comparison_performed,
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
    Random-initialization scaling analysis with heuristic slope-change flags.

    Measures:
    1. Performance vs agent count
    2. Specialization vs agent count
    3. Agent-forward-call work proxy vs agent count
    4. Local slope changes for follow-up analysis
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
        evaluation_seed: int = 0,
    ) -> Dict[str, float]:
        """Evaluate a fresh policy under a declared environment seed."""
        if num_episodes < 1:
            raise ValueError("num_episodes must be positive")
        _seed_everything(evaluation_seed)
        env = build_environment(env_config)

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
                    if (
                        action_logits.ndim != 2
                        or action_logits.shape != (1, self.base_output_dim)
                    ):
                        raise ValueError(
                            "scaling policy returned an invalid action-logit shape: "
                            f"{tuple(action_logits.shape)}"
                        )
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

        if num_seeds < 1 or num_eval_episodes < 1:
            raise ValueError("num_seeds and num_eval_episodes must be positive")
        if not agent_counts or any(count < 1 for count in agent_counts):
            raise ValueError("agent_counts must contain positive values")

        data_points = []

        for n_agents in agent_counts:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing {n_agents} agents...")
            logger.info(f"{'='*50}")

            seed_rewards = []
            seed_sis = []
            seed_bds = []
            seed_compute_times = []

            for seed in range(num_seeds):
                _seed_everything(seed)

                # Create swarm
                role_counts = _balanced_role_counts(n_agents)
                config = SwarmConfig(
                    num_agents=n_agents,
                    input_dim=self.base_input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=self.base_output_dim,
                    message_dim=hidden_dim,
                    topology=TopologyType.SMALL_WORLD,
                    num_perception=role_counts[0],
                    num_reasoning=role_counts[1],
                    num_memory=role_counts[2],
                    num_planning=role_counts[3],
                )
                swarm = SwarmGraph(config, device=self.device)
                set_swarm_eval(swarm)

                # Evaluate performance
                import time
                start_time = time.time()
                perf = self.evaluate_swarm(
                    swarm,
                    env_config,
                    num_eval_episodes,
                    evaluation_seed=seed,
                )
                compute_time = (time.time() - start_time) * 1000 / num_eval_episodes
                seed_compute_times.append(compute_time)

                seed_rewards.append(perf['mean_reward'])

                # Quick descriptive behavioral metrics
                profiles = self.emergence_analyzer.collect_behavioral_data(
                    swarm,
                    env_config,
                    num_episodes=10,
                    steps_per_episode=50,
                    evaluation_seed=seed,
                )
                si = self.emergence_analyzer.compute_specialization_index(profiles)
                bd = self.emergence_analyzer.compute_behavioral_diversity(profiles)

                seed_sis.append(si)
                seed_bds.append(bd)

                logger.info(f"  Seed {seed}: reward={perf['mean_reward']:.2f}, SI={si:.3f}, BD={bd:.3f}")

            # Aggregate across seeds
            total_params = swarm.total_parameters
            agent_forward_calls_per_step = (
                len(swarm.input_agents)
                + n_agents * swarm.config.message_passing_rounds
            )

            data_point = ScalingDataPoint(
                num_agents=n_agents,
                total_params=total_params,
                mean_reward=np.mean(seed_rewards),
                std_reward=np.std(seed_rewards),
                specialization_index=np.mean(seed_sis),
                behavioral_diversity=np.mean(seed_bds),
                agent_forward_calls_per_step=agent_forward_calls_per_step,
                compute_time_ms=np.mean(seed_compute_times),
            )

            data_points.append(data_point)

            logger.info(f"  Mean: reward={data_point.mean_reward:.2f}±{data_point.std_reward:.2f}")
            logger.info(
                "  Params: %s, agent forward calls/step: %d",
                f"{total_params:,}",
                agent_forward_calls_per_step,
            )

        return data_points

    def flag_slope_changes(
        self,
        data_points: List[ScalingDataPoint],
        metrics: List[str] = ['mean_reward', 'specialization_index', 'behavioral_diversity'],
    ) -> List[SlopeChangeFlag]:
        """
        Flag large local slope changes in scaling behavior.

        This heuristic is descriptive and does not establish a phase transition.
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
                    transitions.append(SlopeChangeFlag(
                        agent_count=int(agent_counts[i]),
                        metric_name=metric,
                        before_slope=float(before_slope),
                        after_slope=float(after_slope),
                        magnitude=float(slope_change),
                        relative_change_score=min(
                            1.0,
                            slope_change / baseline_slope,
                        ),
                    ))

        return transitions


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_scaling_results(
    data_points: List[ScalingDataPoint],
    transitions: List[SlopeChangeFlag],
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

    # Mark heuristic slope-change flags
    for t in transitions:
        if t.metric_name == 'mean_reward':
            ax1.axvline(t.agent_count, color='red', linestyle='--', alpha=0.7,
                       label=f'Slope-change flag @ {t.agent_count}')

    # 2. Specialization vs Agents
    ax2 = fig.add_subplot(gs[0, 1])
    sis = [p.specialization_index for p in data_points]
    bds = [p.behavioral_diversity for p in data_points]

    ax2.plot(agents, sis, 'b-o', label='Specialization Index', linewidth=2, markersize=8)
    ax2.plot(agents, bds, 'g-s', label='Behavioral Diversity', linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Agents', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Behavioral Metrics vs Scale', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Mark heuristic slope-change flags
    for t in transitions:
        if t.metric_name in ['specialization_index', 'behavioral_diversity']:
            ax2.axvline(t.agent_count, color='red', linestyle='--', alpha=0.5)

    # 3. Architectural work proxy
    ax3 = fig.add_subplot(gs[1, 0])
    forward_calls = [p.agent_forward_calls_per_step for p in data_points]
    ax3.plot(
        agents,
        forward_calls,
        'r-^',
        label='Agent forward calls/step',
        linewidth=2,
        markersize=8,
    )

    ax3.set_xlabel('Number of Agents', fontsize=12)
    ax3.set_ylabel('Agent forward calls per environment step', fontsize=12)
    ax3.set_title('Architectural Work Proxy', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper left')

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
        ax1.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=colors,
            s=100,
            edgecolors='black',
        )

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
    parser = argparse.ArgumentParser(
        description='SEESWM behavioral-pattern and random-init scaling diagnostics'
    )
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=0, help='Candidate evaluation seed')
    parser.add_argument(
        '--episodes',
        type=int,
        default=None,
        help='Candidate/control episodes (default: 20 quick, otherwise 50)',
    )
    parser.add_argument(
        '--steps-per-episode',
        type=int,
        default=100,
        help='Maximum candidate/control steps per episode',
    )
    parser.add_argument('--output', type=str, default='results/behavioral-local')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to a schema-v2 trained-policy checkpoint',
    )
    parser.add_argument('--quick', action='store_true', help='Quick run with fewer samples')
    parser.add_argument('--scaling-only', action='store_true', help='Only run scaling analysis')
    parser.add_argument(
        '--behavior-only',
        action='store_true',
        help='Only run behavioral-pattern analysis',
    )
    parser.add_argument(
        '--emergence-only',
        dest='behavior_only',
        action='store_true',
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    candidate_episodes = (
        args.episodes if args.episodes is not None else (20 if args.quick else 50)
    )
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    if args.scaling_only and args.behavior_only:
        parser.error('--scaling-only and --behavior-only are mutually exclusive')
    if args.scaling_only and args.model:
        parser.error('--model applies only to behavioral-pattern analysis')
    if candidate_episodes <= 0 or args.steps_per_episode <= 0:
        parser.error('--episodes and --steps-per-episode must be positive')

    _seed_everything(args.seed)
    analysis_source = _analysis_source_provenance()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the exact trained policy, or construct an explicitly random smoke
    # test whose swarm output is already five-dimensional.
    trained_checkpoint_loaded = False
    policy_head = None
    policy_head_spec = None
    checkpoint_provenance = None
    if args.model:
        checkpoint_path = Path(args.model).expanduser().resolve(strict=True)
        logger.info("Loading trained policy from %s", checkpoint_path)
        loaded, checkpoint_digest = load_checkpoint_with_digest(checkpoint_path)
        swarm, policy_head, env_config = build_candidate(loaded, args.device)
        policy_head_spec = dict(loaded['policy_head'])
        if policy_head_spec.get('type') == 'policy_head_tanh':
            from src.swarm.specialized_graph import specialized_swarm_config_to_dict

            swarm_config = specialized_swarm_config_to_dict(swarm.config)
        else:
            swarm_config = swarm_config_to_dict(swarm.config)
        trained_checkpoint_loaded = True
        checkpoint_provenance = {
            'file': checkpoint_path.name,
            'sha256': checkpoint_digest,
            'schema_version': int(loaded['schema_version']),
            'source_revision': loaded.get('source_revision'),
            'source_dirty': loaded.get('source_dirty'),
            'training_seed': loaded.get('seed'),
            'dependency_versions': loaded.get('dependency_versions'),
            'loaded_successfully': True,
        }
        logger.info("Loaded trained swarm and policy head")
    else:
        logger.info("No model specified; running a random-initialization smoke test")
        num_agents = 20
        config = SwarmConfig(
            num_agents=num_agents,
            input_dim=137,
            hidden_dim=128,
            output_dim=5,
            message_dim=128,
            topology=TopologyType.SMALL_WORLD,
            num_perception=num_agents // 4,
            num_reasoning=num_agents // 4,
            num_memory=num_agents // 4,
            num_planning=num_agents - 3 * (num_agents // 4),
        )
        swarm = SwarmGraph(config, device=args.device)
        set_swarm_eval(swarm)
        swarm_config = swarm_config_to_dict(config)
        env_config = environment_config_to_dict(
            EnvironmentConfig(
                grid_size=32,
                num_resources=20,
                num_hazards=10,
                vision_radius=5,
            )
        )

    results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'config': {
            'swarm': swarm_config,
            'env': env_config,
            'device': args.device,
        },
    }

    results['provenance'] = {
        'behavioral_analysis': {
            'initialization': (
                'trained_checkpoint' if trained_checkpoint_loaded else 'fresh_random'
            ),
            'checkpoint': checkpoint_provenance,
            'candidate_seed': int(args.seed),
            'random_control_seeds': (
                []
                if args.quick or args.scaling_only
                else _random_control_seeds(args.seed, 30)
            ),
            'episodes_per_candidate_and_control': int(candidate_episodes),
            'steps_per_episode': int(args.steps_per_episode),
            'random_control_count': 0 if args.quick or args.scaling_only else 30,
        },
        'scaling_analysis': {
            'initialization': 'fresh_random',
            'note': 'Scaling never reuses the candidate checkpoint.',
        },
        'quick_mode': args.quick,
        'analysis_source': analysis_source,
        'analysis_runtime': {
            'python': sys.version.split()[0],
            'torch': str(torch.__version__),
            'numpy': str(np.__version__),
        },
    }

    # =================
    # BEHAVIORAL-PATTERN ANALYSIS
    # =================
    if not args.scaling_only:
        logger.info("\n" + "=" * 60)
        logger.info("BEHAVIORAL-PATTERN ANALYSIS")
        logger.info("=" * 60)

        analyzer = EmergenceAnalyzer(
            device=args.device,
            num_actions=5,
            policy_head=policy_head,
            policy_head_spec=policy_head_spec,
            candidate_seed=args.seed,
        )

        # Re-seed because checkpoint/environment reconstruction may consume RNG.
        _seed_everything(args.seed)

        metrics = analyzer.analyze_full(
            swarm, swarm_config, env_config,
            num_episodes=candidate_episodes,
            steps_per_episode=args.steps_per_episode,
            run_null_test=not args.quick,
        )

        results['behavioral_patterns'] = {
            'specialization_index': metrics.specialization_index,
            'role_consistency': metrics.role_consistency,
            'behavioral_diversity': metrics.behavioral_diversity,
            'random_control_si': metrics.random_control_si,
            'monte_carlo_p_value': metrics.monte_carlo_p_value,
            'standardized_null_difference': metrics.standardized_null_difference,
            'null_sample_count': metrics.null_sample_count,
            'random_comparison_performed': metrics.random_comparison_performed,
            'num_distinct_roles': metrics.num_distinct_roles,
            'cluster_sizes': metrics.cluster_sizes,
        }

        # Summary
        logger.info("\n" + "-" * 40)
        logger.info("BEHAVIORAL-PATTERN SUMMARY")
        logger.info("-" * 40)
        logger.info(f"Specialization Index: {metrics.specialization_index:.4f}")
        if metrics.random_comparison_performed:
            logger.info(f"  vs Random Controls: {metrics.random_control_si:.4f}")
            logger.info(
                "  Plus-one Monte Carlo p: %.4f (n=%d random controls)",
                metrics.monte_carlo_p_value,
                metrics.null_sample_count,
            )
            if metrics.standardized_null_difference is None:
                logger.info("  Standardized null difference: undefined (zero null SD)")
            else:
                logger.info(
                    "  Standardized null difference: %.2f",
                    metrics.standardized_null_difference,
                )
        else:
            logger.info("  Fresh-random comparison: skipped")
        logger.info(f"Number of distinct roles: {metrics.num_distinct_roles}")
        logger.info(f"Role sizes: {metrics.cluster_sizes}")

        # Interpretation
        if not metrics.random_comparison_performed:
            logger.info("\nRandom-init comparison skipped; no comparative conclusion")
        elif (
            trained_checkpoint_loaded
            and metrics.monte_carlo_p_value is not None
            and metrics.monte_carlo_p_value < 0.05
        ):
            logger.info("\nCandidate differs from the fresh-random comparison")
            logger.info("under this 30-control exploratory comparison (p < 0.05)")
            logger.info("This is not evidence of emergence without trained controls")
        elif not trained_checkpoint_loaded:
            logger.info("\nRandom-initialization smoke test only; no learned specialization claim")
        else:
            logger.info("\nCandidate SI was not unusual in this random-init comparison")

        # Plot clusters
        plot_role_clusters(metrics, str(output_dir / 'role_clusters.png'))

    # =================
    # SCALING ANALYSIS
    # =================
    if not args.behavior_only:
        logger.info("\n" + "=" * 60)
        logger.info("SCALING ANALYSIS")
        logger.info("=" * 60)

        scaling_analyzer = ScalingAnalyzer(
            device=args.device,
            base_input_dim=int(swarm_config['input_dim']),
            base_output_dim=5,  # Use 5 actions for scaling tests (fresh swarms)
        )

        if args.quick:
            agent_counts = [4, 10, 20, 50]
            num_seeds = 2
            scaling_eval_episodes = 10
        else:
            agent_counts = [4, 10, 20, 50, 100, 150]
            num_seeds = 3
            scaling_eval_episodes = 20

        results['provenance']['scaling_analysis'].update({
            'seeds': list(range(num_seeds)),
            'evaluation_episodes_per_seed': scaling_eval_episodes,
            'evaluation_steps_per_episode': 100,
            'behavioral_episodes_per_seed': 10,
            'behavioral_steps_per_episode': 50,
            'agent_counts': agent_counts,
        })

        data_points = scaling_analyzer.run_scaling_sweep(
            agent_counts=agent_counts,
            env_config=env_config,
            num_seeds=num_seeds,
            num_eval_episodes=scaling_eval_episodes,
        )

        transitions = scaling_analyzer.flag_slope_changes(data_points)

        results['scaling'] = {
            'seeds': list(range(num_seeds)),
            'data_points': [
                {
                    'num_agents': int(p.num_agents),
                    'mean_reward': float(p.mean_reward),
                    'std_reward': float(p.std_reward),
                    'specialization_index': float(p.specialization_index),
                    'behavioral_diversity': float(p.behavioral_diversity),
                    'agent_forward_calls_per_step': int(
                        p.agent_forward_calls_per_step
                    ),
                    'compute_time_ms': float(p.compute_time_ms),
                    'total_params': int(p.total_params),
                }
                for p in data_points
            ],
            'slope_change_flags': [
                {
                    'agent_count': int(t.agent_count),
                    'metric': t.metric_name,
                    'magnitude': float(t.magnitude),
                }
                for t in transitions
            ],
        }

        # Summary
        logger.info("\n" + "-" * 40)
        logger.info("SCALING SUMMARY")
        logger.info("-" * 40)

        # Report heuristic slope-change flags
        if transitions:
            logger.info(f"Flagged {len(transitions)} local slope change(s):")
            for t in transitions:
                logger.info(f"  {t.metric_name} @ {t.agent_count} agents (magnitude: {t.magnitude:.3f})")
        else:
            logger.info("No large local slope changes flagged")

        # Scaling trend
        rewards = [p.mean_reward for p in data_points]
        agents = [p.num_agents for p in data_points]

        # Fit log-linear trend
        log_agents = np.log(agents)
        slope, _, r_value, _, _ = stats.linregress(log_agents, rewards)

        logger.info(f"\nScaling trend: reward ~ {slope:.3f} * log(agents)")
        logger.info(f"R² = {r_value**2:.3f}")

        logger.info(
            "Observed reward trend direction: %s",
            "positive" if slope > 0 else "non-positive",
        )

        # Plot
        plot_scaling_results(data_points, transitions, str(output_dir / 'scaling_analysis.png'))

    # Save results
    output_file = output_dir / 'behavioral_scaling_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    return results


if __name__ == '__main__':
    main()
