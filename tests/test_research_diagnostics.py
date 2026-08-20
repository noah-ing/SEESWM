"""Focused invariants for exploratory diagnostic utilities."""

import numpy as np

from experiments.emergence_scaling_analysis import (
    AgentBehaviorProfile,
    EmergenceAnalyzer,
    _balanced_role_counts,
)


def test_balanced_role_counts_assign_every_agent_once():
    for num_agents in range(1, 25):
        counts = _balanced_role_counts(num_agents)
        assert sum(counts) == num_agents
        assert max(counts) - min(counts) <= 1


def test_cluster_count_reports_only_clusters_that_occurred():
    profiles = {
        agent_id: AgentBehaviorProfile(
            agent_id=agent_id,
            agent_type="GENERAL",
            action_distribution=np.full(5, 0.2),
            output_mean=0.0,
            output_std=0.0,
        )
        for agent_id in range(4)
    }

    labels, cluster_count = EmergenceAnalyzer().cluster_agents_into_roles(
        profiles,
        n_clusters=4,
    )

    assert cluster_count == len(set(labels))
    assert sorted(set(labels)) == list(range(1, cluster_count + 1))
