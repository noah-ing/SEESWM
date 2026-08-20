"""
Exploratory information-decomposition approximations.

The public names are retained for compatibility, but the implementations use
finite-sample mutual-information estimators and simplified or pairwise
approximations. They are not a formal multivariate PID implementation and a
positive score is not, by itself, evidence of emergence.

Key concepts:
- Redundancy: Information shared by multiple agents
- Unique: Information only one agent provides
- Synergy: Information that emerges from combining agents

Synergy = I(X1,...,Xn; Y) - Σ I(Xi; Y)
(simplified; true PID is more nuanced)
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from scipy import stats


@dataclass
class SynergyDecomposition:
    """Result of Partial Information Decomposition."""
    # Total mutual information
    total_mi: float

    # Decomposition components
    redundancy: float  # Shared information
    unique: List[float]  # Per-agent unique information
    synergy: float  # Residual under the selected approximation

    # Derived metrics
    synergy_ratio: float  # synergy / total_mi
    redundancy_ratio: float  # redundancy / total_mi

    # Confidence
    std_synergy: float = 0.0
    num_samples: int = 0


class MutualInformationEstimator:
    """Estimate mutual information from samples."""

    def __init__(self, method: str = 'ksg'):
        """
        Args:
            method: Estimation method ('ksg', 'kde', 'binning')
        """
        self.method = method

    def estimate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        k: int = 3,
    ) -> float:
        """
        Estimate I(X; Y).

        Args:
            x: Samples of X, shape (n_samples, dim_x)
            y: Samples of Y, shape (n_samples, dim_y)
            k: Number of neighbors for KSG estimator
        """
        if self.method == 'ksg':
            return self._ksg_estimator(x, y, k)
        elif self.method == 'kde':
            return self._kde_estimator(x, y)
        elif self.method == 'binning':
            return self._binning_estimator(x, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _ksg_estimator(self, x: np.ndarray, y: np.ndarray, k: int = 3) -> float:
        """
        KSG-inspired nearest-neighbor approximation.

        This implementation does not reproduce every metric and boundary
        convention of the reference KSG estimator.
        """
        from scipy.spatial import cKDTree

        n = len(x)
        if n < k + 1:
            return 0.0

        # Ensure 2D
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Joint space
        xy = np.concatenate([x, y], axis=1)

        # Build trees
        tree_xy = cKDTree(xy)
        tree_x = cKDTree(x)
        tree_y = cKDTree(y)

        # Find k-th neighbor distances in joint space
        # Use k+1 because query point is included
        distances, _ = tree_xy.query(xy, k=k+1)
        eps = distances[:, k]  # Distance to k-th neighbor

        # Count points within eps in marginals
        nx = np.array([
            len(tree_x.query_ball_point(x[i], eps[i] - 1e-10))
            for i in range(n)
        ])
        ny = np.array([
            len(tree_y.query_ball_point(y[i], eps[i] - 1e-10))
            for i in range(n)
        ])

        # Avoid log(0)
        nx = np.maximum(nx, 1)
        ny = np.maximum(ny, 1)

        # KSG formula
        from scipy.special import digamma
        mi = digamma(k) + digamma(n) - np.mean(digamma(nx) + digamma(ny))

        return max(0.0, mi)  # MI is non-negative

    def _kde_estimator(self, x: np.ndarray, y: np.ndarray) -> float:
        """Kernel density estimation based MI."""
        from scipy.stats import gaussian_kde

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        try:
            # Estimate densities
            kde_x = gaussian_kde(x.T)
            kde_y = gaussian_kde(y.T)
            kde_xy = gaussian_kde(np.concatenate([x, y], axis=1).T)

            # Sample points for integration
            n = len(x)
            log_pxy = kde_xy.logpdf(np.concatenate([x, y], axis=1).T)
            log_px = kde_x.logpdf(x.T)
            log_py = kde_y.logpdf(y.T)

            mi = np.mean(log_pxy - log_px - log_py)
            return max(0.0, mi)

        except Exception:
            return 0.0

    def _binning_estimator(self, x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        """Simple first-component histogram MI estimate."""
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Discretize
        x_binned = np.floor(bins * (x - x.min()) / (x.max() - x.min() + 1e-10)).astype(int)
        y_binned = np.floor(bins * (y - y.min()) / (y.max() - y.min() + 1e-10)).astype(int)

        # Joint histogram
        x_flat = x_binned[:, 0] if x_binned.ndim > 1 else x_binned
        y_flat = y_binned[:, 0] if y_binned.ndim > 1 else y_binned

        joint_hist, _, _ = np.histogram2d(x_flat, y_flat, bins=bins)
        joint_prob = joint_hist / joint_hist.sum()

        # Marginals
        px = joint_prob.sum(axis=1)
        py = joint_prob.sum(axis=0)

        # MI
        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if joint_prob[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += joint_prob[i, j] * np.log(joint_prob[i, j] / (px[i] * py[j]))

        return max(0.0, mi)


class PartialInformationDecomposition:
    """
    Compute exploratory information-decomposition approximations.

    Decomposes the total mutual information I(X1, ..., Xn; Y) into:
    - Redundancy: Information all agents share
    - Unique: Information only specific agents provide
    - Synergy: Information only available from combining agents
    """

    def __init__(
        self,
        mi_estimator: Optional[MutualInformationEstimator] = None,
    ):
        self.mi_estimator = mi_estimator or MutualInformationEstimator('ksg')

    def decompose(
        self,
        agent_outputs: List[np.ndarray],
        targets: np.ndarray,
        method: str = 'broja',
    ) -> SynergyDecomposition:
        """
        Decompose information from multiple agents.

        Args:
            agent_outputs: List of arrays, each (n_samples, output_dim)
            targets: Ground truth, shape (n_samples, target_dim)
            method: PID method ('broja', 'simplified', 'williams')

        Returns:
            SynergyDecomposition with all components.
        """
        n_agents = len(agent_outputs)

        if method == 'simplified':
            return self._simplified_pid(agent_outputs, targets)
        elif method == 'broja':
            return self._broja_pid(agent_outputs, targets)
        else:
            return self._williams_beer_pid(agent_outputs, targets)

    def _simplified_pid(
        self,
        agent_outputs: List[np.ndarray],
        targets: np.ndarray,
    ) -> SynergyDecomposition:
        """
        Simplified PID using co-information.

        Synergy = I(X1,...,Xn; Y) - Σ I(Xi; Y)

        This is a heuristic residual, not a formal multivariate PID estimate.
        """
        n_agents = len(agent_outputs)

        # Total information from combined outputs
        combined = np.concatenate(agent_outputs, axis=1)
        total_mi = self.mi_estimator.estimate(combined, targets)

        # Individual agent information
        individual_mi = []
        for i, output in enumerate(agent_outputs):
            mi = self.mi_estimator.estimate(output, targets)
            individual_mi.append(mi)

        sum_individual = sum(individual_mi)

        # Simplified synergy
        synergy = total_mi - sum_individual

        # Redundancy (minimum individual MI as lower bound)
        redundancy = min(individual_mi) if individual_mi else 0.0

        # Unique information
        unique = [mi - redundancy for mi in individual_mi]

        return SynergyDecomposition(
            total_mi=total_mi,
            redundancy=redundancy,
            unique=unique,
            synergy=synergy,
            synergy_ratio=synergy / (total_mi + 1e-10),
            redundancy_ratio=redundancy / (total_mi + 1e-10),
        )

    def _broja_pid(
        self,
        agent_outputs: List[np.ndarray],
        targets: np.ndarray,
    ) -> SynergyDecomposition:
        """
        Legacy ``broja`` option using a pairwise residual approximation.

        This does not implement the BROJA optimization procedure.
        """
        # For computational efficiency, use pairwise approximation
        n_agents = len(agent_outputs)

        if n_agents < 2:
            return self._simplified_pid(agent_outputs, targets)

        # Combined MI
        combined = np.concatenate(agent_outputs, axis=1)
        total_mi = self.mi_estimator.estimate(combined, targets)

        # Individual MIs
        individual_mi = [
            self.mi_estimator.estimate(out, targets)
            for out in agent_outputs
        ]

        # Pairwise redundancy (minimum specific information)
        pair_redundancies = []
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                # Redundancy is min(I(Xi;Y), I(Xj;Y)) adjusted for dependence
                pair_combined = np.concatenate([agent_outputs[i], agent_outputs[j]], axis=1)
                pair_mi = self.mi_estimator.estimate(pair_combined, targets)

                # Redundancy between pair
                red = individual_mi[i] + individual_mi[j] - pair_mi
                pair_redundancies.append(max(0, red))

        redundancy = np.mean(pair_redundancies) if pair_redundancies else 0.0

        # Unique information (MI minus redundant part)
        unique = [max(0, mi - redundancy) for mi in individual_mi]

        # Synergy = Total - (Redundancy + Unique)
        synergy = total_mi - redundancy - sum(unique)

        return SynergyDecomposition(
            total_mi=total_mi,
            redundancy=redundancy,
            unique=unique,
            synergy=synergy,
            synergy_ratio=synergy / (total_mi + 1e-10),
            redundancy_ratio=redundancy / (total_mi + 1e-10),
        )

    def _williams_beer_pid(
        self,
        agent_outputs: List[np.ndarray],
        targets: np.ndarray,
    ) -> SynergyDecomposition:
        """
        Legacy ``williams`` option using minimum marginal MI as redundancy.

        This is a simplified residual calculation, not a full PID lattice.
        """
        # Use minimum mutual information as redundancy
        individual_mi = [
            self.mi_estimator.estimate(out, targets)
            for out in agent_outputs
        ]

        # Combined MI
        combined = np.concatenate(agent_outputs, axis=1)
        total_mi = self.mi_estimator.estimate(combined, targets)

        # I_min redundancy
        redundancy = min(individual_mi) if individual_mi else 0.0

        # Unique
        unique = [mi - redundancy for mi in individual_mi]

        # Synergy
        synergy = total_mi - sum(individual_mi)

        return SynergyDecomposition(
            total_mi=total_mi,
            redundancy=redundancy,
            unique=unique,
            synergy=synergy,
            synergy_ratio=synergy / (total_mi + 1e-10),
            redundancy_ratio=redundancy / (total_mi + 1e-10),
        )


class SynergyMeasurer:
    """Measure synergy in a swarm over many samples."""

    def __init__(
        self,
        swarm: nn.Module,
        pid_method: str = 'broja',
        device: str = 'cpu',
    ):
        self.swarm = swarm
        self.pid = PartialInformationDecomposition()
        self.pid_method = pid_method
        self.device = device

    def measure(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_bootstrap: int = 100,
    ) -> SynergyDecomposition:
        """
        Measure synergy on given data.

        Args:
            inputs: Input observations
            targets: Ground truth targets
            num_bootstrap: Bootstrap samples for confidence

        Returns:
            SynergyDecomposition with statistics.
        """
        # Set to eval mode if available (nn.Module)
        if hasattr(self.swarm, 'eval'):
            self.swarm.eval()

        with torch.no_grad():
            # Get individual agent outputs
            agent_outputs = []

            if hasattr(self.swarm, 'agents'):
                # Handle SwarmGraph (dict of MicroAgents) and nn.Module (ModuleList)
                agents = self.swarm.agents
                if isinstance(agents, dict):
                    agents = agents.values()

                for agent in agents:
                    # For MicroAgent, use network directly with zero messages
                    if hasattr(agent, 'network'):
                        inp = inputs.to(self.device)
                        # MicroAgent.network expects (inputs, messages, state)
                        batch_size = inp.shape[0]
                        msg_dim = agent.config.message_dim if hasattr(agent, 'config') else 128
                        state_dim = agent.config.state_dim if hasattr(agent, 'config') else 128
                        zero_msg = torch.zeros(batch_size, msg_dim, device=self.device)
                        zero_state = torch.zeros(batch_size, state_dim, device=self.device)
                        out, _ = agent.network(inp, zero_msg, zero_state)
                    elif callable(agent):
                        out = agent(inputs.to(self.device))
                    else:
                        continue
                    agent_outputs.append(out.cpu().numpy())
            else:
                # Single model - use step() for SwarmGraph-like, __call__ for nn.Module
                if hasattr(self.swarm, 'step'):
                    out = self.swarm.step(inputs.to(self.device))
                else:
                    out = self.swarm(inputs.to(self.device))
                agent_outputs = [out.cpu().numpy()]

        targets_np = targets.cpu().numpy()

        # Compute PID
        decomp = self.pid.decompose(
            agent_outputs, targets_np, method=self.pid_method
        )

        # Bootstrap for confidence interval
        synergies = []
        n = len(inputs)

        for _ in range(num_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            boot_outputs = [out[idx] for out in agent_outputs]
            boot_targets = targets_np[idx]

            boot_decomp = self.pid.decompose(
                boot_outputs, boot_targets, method=self.pid_method
            )
            synergies.append(boot_decomp.synergy)

        decomp.std_synergy = np.std(synergies)
        decomp.num_samples = n

        return decomp


def compute_true_synergy(
    swarm: nn.Module,
    task_data: Tuple[torch.Tensor, torch.Tensor],
    method: str = 'broja',
    device: str = 'cpu',
) -> float:
    """
    Compatibility wrapper for the selected approximate decomposition score.

    Args:
        swarm: The swarm model
        task_data: (inputs, targets) tuple
        method: PID method
        device: Device

    Returns:
        Approximate residual score; interpret only under the declared estimator.
    """
    inputs, targets = task_data
    measurer = SynergyMeasurer(swarm, pid_method=method, device=device)
    decomp = measurer.measure(inputs, targets)
    return decomp.synergy
