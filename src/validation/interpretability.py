"""
Interpretability Suite.

Tools for understanding what the swarm is learning and how it processes information.
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict


@dataclass
class ProbeResult:
    """Result of a representation probe."""
    target_variable: str
    accuracy: float
    r_squared: float  # For regression
    feature_importance: np.ndarray


@dataclass
class InterventionResult:
    """Result of a causal intervention."""
    intervention_type: str
    pre_intervention_output: torch.Tensor
    post_intervention_output: torch.Tensor
    effect_magnitude: float


class RepresentationProbe(nn.Module):
    """
    Linear probe for analyzing what information is encoded in representations.

    Based on the probing methodology from:
    "A Structural Probe for Finding Syntax in Word Representations"
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        probe_type: str = 'linear',
    ):
        super().__init__()

        if probe_type == 'linear':
            self.probe = nn.Linear(input_dim, output_dim)
        elif probe_type == 'mlp':
            self.probe = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, output_dim),
            )
        else:
            self.probe = nn.Linear(input_dim, output_dim)

        self.probe_type = probe_type

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        return self.probe(representations)

    def train_probe(
        self,
        representations: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.01,
    ) -> ProbeResult:
        """Train the probe on labeled data."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Determine task type
        is_regression = labels.dtype == torch.float32 and labels.dim() > 1

        if is_regression:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(representations)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluate
        with torch.no_grad():
            predictions = self(representations)

            if is_regression:
                mse = ((predictions - labels) ** 2).mean().item()
                ss_res = ((predictions - labels) ** 2).sum().item()
                ss_tot = ((labels - labels.mean()) ** 2).sum().item()
                r_squared = 1 - ss_res / (ss_tot + 1e-10)
                accuracy = 1.0 / (1.0 + mse)  # Convert to accuracy-like
            else:
                predicted_classes = predictions.argmax(dim=-1)
                accuracy = (predicted_classes == labels).float().mean().item()
                r_squared = 0.0

        # Feature importance (from first layer weights)
        if hasattr(self.probe, 'weight'):
            weights = self.probe.weight.detach().cpu().numpy()
        else:
            weights = self.probe[0].weight.detach().cpu().numpy()

        importance = np.abs(weights).mean(axis=0)

        return ProbeResult(
            target_variable='unknown',
            accuracy=accuracy,
            r_squared=r_squared,
            feature_importance=importance,
        )


class MessageAnalyzer:
    """Analyze message content and semantics."""

    def __init__(self, message_dim: int):
        self.message_dim = message_dim
        self.message_history = []

    def log_message(
        self,
        source: int,
        target: int,
        content: torch.Tensor,
        metadata: Optional[Dict] = None,
    ):
        """Log a message."""
        self.message_history.append({
            'source': source,
            'target': target,
            'content': content.detach().cpu(),
            'metadata': metadata or {},
        })

    def cluster_messages(self, n_clusters: int = 5) -> Dict[int, List[int]]:
        """Cluster messages by content similarity."""
        if len(self.message_history) < n_clusters:
            return {}

        contents = torch.stack([m['content'].flatten() for m in self.message_history])
        contents = contents.numpy()

        # Simple k-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(contents)

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[int(label)].append(i)

        return dict(clusters)

    def find_semantic_roles(self) -> Dict[str, List[int]]:
        """Attempt to identify semantic roles of messages."""
        if len(self.message_history) < 10:
            return {}

        contents = torch.stack([m['content'].flatten() for m in self.message_history])

        # Analyze statistical properties
        roles = {}

        # High magnitude = emphasis/importance
        magnitudes = contents.norm(dim=-1)
        high_mag_idx = (magnitudes > magnitudes.mean() + magnitudes.std()).nonzero().flatten()
        roles['high_emphasis'] = high_mag_idx.tolist()

        # Low variance = factual/stable
        variances = contents.var(dim=-1)
        low_var_idx = (variances < variances.mean() - variances.std()).nonzero().flatten()
        roles['stable_content'] = low_var_idx.tolist()

        # Sparse messages = selective
        sparsity = (contents.abs() < 0.1).float().mean(dim=-1)
        sparse_idx = (sparsity > 0.5).nonzero().flatten()
        roles['sparse_selective'] = sparse_idx.tolist()

        return roles

    def compute_message_entropy(self) -> float:
        """Compute entropy of message distribution."""
        if len(self.message_history) < 2:
            return 0.0

        contents = torch.stack([m['content'].flatten() for m in self.message_history])

        # Discretize to bins
        bins = 20
        binned = ((contents + 1) / 2 * bins).long().clamp(0, bins-1)

        # Compute histogram
        hist = torch.zeros(bins)
        for val in binned.flatten():
            hist[val] += 1

        probs = hist / hist.sum()
        entropy = -(probs * (probs + 1e-10).log()).sum().item()

        return entropy


class CausalIntervention:
    """Perform causal interventions to understand model behavior."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.activation_cache = {}

    def _register_hooks(self, layer_names: List[str]):
        """Register hooks on specified layers."""
        for name, module in self.model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: self._cache_activation(n, o)
                )
                self.hooks.append(hook)

    def _cache_activation(self, name: str, output: torch.Tensor):
        """Cache activation for intervention."""
        self.activation_cache[name] = output.clone()

    def ablate_agent(
        self,
        inputs: torch.Tensor,
        agent_idx: int,
    ) -> InterventionResult:
        """
        Ablate a specific agent and measure effect.

        Sets agent output to zero.
        """
        # Get baseline output - handle SwarmGraph (step) vs nn.Module (call)
        is_swarm_like = hasattr(self.model, 'step') and hasattr(self.model, 'reset')
        with torch.no_grad():
            if is_swarm_like:
                self.model.reset(batch_size=inputs.shape[0])
                baseline_output = self.model.step(inputs)
            else:
                baseline_output = self.model(inputs)

        # Ablate agent
        if hasattr(self.model, 'agents'):
            agents = self.model.agents
            # Handle dict (SwarmGraph) or list
            if isinstance(agents, dict):
                agent = agents.get(agent_idx)
            else:
                agent = agents[agent_idx] if agent_idx < len(agents) else None

            if agent is not None:
                original_forward = agent.forward

                def ablated_forward(x, *args, **kwargs):
                    return torch.zeros_like(original_forward(x, *args, **kwargs))

                agent.forward = ablated_forward

                with torch.no_grad():
                    if is_swarm_like:
                        self.model.reset(batch_size=inputs.shape[0])
                        ablated_output = self.model.step(inputs)
                    else:
                        ablated_output = self.model(inputs)

                agent.forward = original_forward
            else:
                ablated_output = baseline_output
        else:
            ablated_output = baseline_output

        effect = (baseline_output - ablated_output).abs().mean().item()

        return InterventionResult(
            intervention_type=f'ablate_agent_{agent_idx}',
            pre_intervention_output=baseline_output,
            post_intervention_output=ablated_output,
            effect_magnitude=effect,
        )

    def corrupt_messages(
        self,
        inputs: torch.Tensor,
        corruption_type: str = 'zero',
    ) -> InterventionResult:
        """
        Corrupt inter-agent messages and measure effect.

        corruption_type: 'zero', 'noise', 'shuffle', 'random'
        """
        # Get baseline output - handle SwarmGraph (step) vs nn.Module (call)
        is_swarm_like = hasattr(self.model, 'step') and hasattr(self.model, 'reset')
        with torch.no_grad():
            if is_swarm_like:
                self.model.reset(batch_size=inputs.shape[0])
                baseline_output = self.model.step(inputs)
            else:
                baseline_output = self.model(inputs)

        # Apply corruption
        if hasattr(self.model, 'aggregate_messages'):
            original_aggregate = self.model.aggregate_messages

            def corrupted_aggregate(messages):
                aggregated = original_aggregate(messages)

                if corruption_type == 'zero':
                    return torch.zeros_like(aggregated)
                elif corruption_type == 'noise':
                    return aggregated + torch.randn_like(aggregated)
                elif corruption_type == 'shuffle':
                    idx = torch.randperm(aggregated.shape[0])
                    return aggregated[idx]
                elif corruption_type == 'random':
                    return torch.randn_like(aggregated)
                else:
                    return aggregated

            self.model.aggregate_messages = corrupted_aggregate

            with torch.no_grad():
                if is_swarm_like:
                    self.model.reset(batch_size=inputs.shape[0])
                    corrupted_output = self.model.step(inputs)
                else:
                    corrupted_output = self.model(inputs)

            self.model.aggregate_messages = original_aggregate
        else:
            corrupted_output = baseline_output

        effect = (baseline_output - corrupted_output).abs().mean().item()

        return InterventionResult(
            intervention_type=f'corrupt_messages_{corruption_type}',
            pre_intervention_output=baseline_output,
            post_intervention_output=corrupted_output,
            effect_magnitude=effect,
        )

    def clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activation_cache = {}


class ActivationPatcher:
    """
    Activation patching for identifying task-relevant features.

    Based on: "Interpretability in the Wild" (Wang et al., 2022)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.stored_activations = {}

    def store_activations(
        self,
        inputs: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        """Store activations from a forward pass."""
        activation = None

        def hook_fn(module, inp, out):
            nonlocal activation
            activation = out.clone()

        # Find the layer
        for name, module in self.model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(hook_fn)

                with torch.no_grad():
                    self.model(inputs)

                handle.remove()
                break

        if activation is not None:
            self.stored_activations[layer_name] = activation

        return activation

    def patch_activation(
        self,
        clean_inputs: torch.Tensor,
        corrupted_inputs: torch.Tensor,
        layer_name: str,
        patch_indices: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Patch activations from clean run into corrupted run.

        Returns:
            (output, effect_size)
        """
        # Get clean activations
        clean_activation = self.store_activations(clean_inputs, layer_name)

        if clean_activation is None:
            return None, 0.0

        # Hook to patch activations
        def patch_hook(module, inp, out):
            patched = out.clone()
            if patch_indices is not None:
                patched[:, patch_indices] = clean_activation[:, patch_indices]
            else:
                patched = clean_activation
            return patched

        # Get corrupted output
        with torch.no_grad():
            corrupted_output = self.model(corrupted_inputs)

        # Get patched output
        for name, module in self.model.named_modules():
            if name == layer_name:
                handle = module.register_forward_hook(patch_hook)

                with torch.no_grad():
                    patched_output = self.model(corrupted_inputs)

                handle.remove()
                break

        effect = (patched_output - corrupted_output).abs().mean().item()

        return patched_output, effect

    def find_important_features(
        self,
        clean_inputs: torch.Tensor,
        corrupted_inputs: torch.Tensor,
        layer_name: str,
        top_k: int = 10,
    ) -> List[int]:
        """Find most important features by individual patching."""
        clean_activation = self.store_activations(clean_inputs, layer_name)

        if clean_activation is None:
            return []

        feature_dim = clean_activation.shape[-1]
        effects = []

        for i in range(feature_dim):
            _, effect = self.patch_activation(
                clean_inputs, corrupted_inputs, layer_name, [i]
            )
            effects.append((i, effect))

        # Sort by effect
        effects.sort(key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in effects[:top_k]]


def run_interpretability_suite(
    swarm: nn.Module,
    test_inputs: torch.Tensor,
    test_labels: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> Dict:
    """
    Run complete interpretability analysis.

    Returns:
        Dictionary with all analysis results.
    """
    results = {}
    if hasattr(swarm, 'to'):
        swarm = swarm.to(device)
    test_inputs = test_inputs.to(device)

    # 1. Representation probing
    if hasattr(swarm, 'agents') and len(swarm.agents) > 0:
        # Get first agent (handle dict or list)
        agents = swarm.agents
        if isinstance(agents, dict):
            agent = list(agents.values())[0]
        else:
            agent = agents[0]

        with torch.no_grad():
            # For MicroAgent, use network directly
            if hasattr(agent, 'network'):
                batch_size = test_inputs.shape[0]
                msg_dim = agent.config.message_dim if hasattr(agent, 'config') else 128
                state_dim = agent.config.state_dim if hasattr(agent, 'config') else 128
                zero_msg = torch.zeros(batch_size, msg_dim, device=device)
                zero_state = torch.zeros(batch_size, state_dim, device=device)
                representations, _ = agent.network(test_inputs, zero_msg, zero_state)
            elif callable(agent):
                representations = agent(test_inputs)
            else:
                representations = test_inputs  # Fallback

        if test_labels is not None and representations is not None:
            probe = RepresentationProbe(
                representations.shape[-1],
                test_labels.max().item() + 1 if test_labels.dtype == torch.long else test_labels.shape[-1],
            ).to(device)

            probe_result = probe.train_probe(representations, test_labels.to(device))
            results['probe'] = probe_result

    # 2. Causal interventions
    intervention = CausalIntervention(swarm)

    # Ablate each agent
    if hasattr(swarm, 'agents'):
        ablation_effects = []
        agent_ids = list(swarm.agents.keys()) if isinstance(swarm.agents, dict) else range(len(swarm.agents))
        for i in agent_ids:
            result = intervention.ablate_agent(test_inputs, i)
            ablation_effects.append(result.effect_magnitude)
        results['agent_importance'] = ablation_effects

    # Corrupt messages
    msg_result = intervention.corrupt_messages(test_inputs, 'zero')
    results['message_importance'] = msg_result.effect_magnitude

    return results
