"""
Neural Architecture Search (NAS) for agent networks.

Implements:
1. DARTS: Differentiable Architecture Search
2. ENAS: Efficient NAS with weight sharing
3. Random Search baseline
4. Search spaces for agent components
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Callable, Any
from enum import Enum
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class OperationType(Enum):
    """Primitive operations for NAS."""
    NONE = "none"           # Zero operation
    SKIP = "skip"           # Skip connection (identity)
    LINEAR = "linear"       # Linear layer
    RELU_LINEAR = "relu_linear"  # Linear + ReLU
    GELU_LINEAR = "gelu_linear"  # Linear + GELU
    ATTENTION = "attention"      # Self-attention
    CONV1D = "conv1d"       # 1D convolution
    GRU = "gru"             # GRU cell
    LSTM = "lstm"           # LSTM cell


@dataclass
class SearchSpace:
    """Defines the search space for architecture."""

    # Operations to search over
    operations: List[OperationType] = field(default_factory=lambda: [
        OperationType.SKIP,
        OperationType.LINEAR,
        OperationType.RELU_LINEAR,
        OperationType.ATTENTION,
    ])

    # Dimension ranges
    min_hidden_dim: int = 32
    max_hidden_dim: int = 256
    dim_step: int = 32

    # Depth ranges
    min_layers: int = 1
    max_layers: int = 6

    # Attention parameters
    min_heads: int = 1
    max_heads: int = 8

    # Other
    dropout_range: Tuple[float, float] = (0.0, 0.5)


class Operation(nn.Module):
    """
    Differentiable operation for DARTS.

    Each operation can be weighted during architecture search.
    """

    def __init__(
        self,
        op_type: OperationType,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
    ):
        super().__init__()
        self.op_type = op_type
        self.in_dim = in_dim
        self.out_dim = out_dim

        if op_type == OperationType.NONE:
            self.op = None

        elif op_type == OperationType.SKIP:
            if in_dim != out_dim:
                self.op = nn.Linear(in_dim, out_dim)
            else:
                self.op = nn.Identity()

        elif op_type == OperationType.LINEAR:
            self.op = nn.Linear(in_dim, out_dim)

        elif op_type == OperationType.RELU_LINEAR:
            self.op = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
            )

        elif op_type == OperationType.GELU_LINEAR:
            self.op = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
            )

        elif op_type == OperationType.ATTENTION:
            self.op = nn.MultiheadAttention(
                embed_dim=in_dim,
                num_heads=num_heads,
                batch_first=True,
            )
            self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        elif op_type == OperationType.CONV1D:
            self.op = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=1)

        elif op_type == OperationType.GRU:
            self.op = nn.GRUCell(in_dim, out_dim)
            self.hidden = None

        elif op_type == OperationType.LSTM:
            self.op = nn.LSTMCell(in_dim, out_dim)
            self.hidden = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply operation."""
        if self.op_type == OperationType.NONE:
            return torch.zeros(x.shape[0], self.out_dim, device=x.device)

        elif self.op_type == OperationType.ATTENTION:
            # Add sequence dimension if needed
            if x.dim() == 2:
                x = x.unsqueeze(1)
            attn_out, _ = self.op(x, x, x)
            return self.proj(attn_out.squeeze(1))

        elif self.op_type == OperationType.CONV1D:
            # Add channel dimension
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            x = x.transpose(1, 2)
            out = self.op(x)
            return out.transpose(1, 2).squeeze(-1)

        elif self.op_type in [OperationType.GRU, OperationType.LSTM]:
            if self.hidden is None:
                self.hidden = torch.zeros(x.shape[0], self.out_dim, device=x.device)

            if self.op_type == OperationType.GRU:
                self.hidden = self.op(x, self.hidden)
                return self.hidden
            else:
                if isinstance(self.hidden, tuple):
                    h, c = self.hidden
                else:
                    h = self.hidden
                    c = torch.zeros_like(h)
                self.hidden = self.op(x, (h, c))
                return self.hidden[0]

        else:
            return self.op(x)

    def reset(self) -> None:
        """Reset state (for recurrent ops)."""
        self.hidden = None


class MixedOperation(nn.Module):
    """
    Mixed operation that combines all candidates.

    Weights are learned during architecture search.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        search_space: SearchSpace,
    ):
        super().__init__()
        self.operations = nn.ModuleList([
            Operation(op_type, in_dim, out_dim)
            for op_type in search_space.operations
        ])

        # Architecture weights (to be learned)
        self.alpha = nn.Parameter(torch.zeros(len(search_space.operations)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with softmax-weighted operations."""
        weights = F.softmax(self.alpha, dim=0)

        output = torch.zeros(x.shape[0], self.operations[0].out_dim, device=x.device)

        for weight, op in zip(weights, self.operations):
            output = output + weight * op(x)

        return output

    def get_selected_operation(self) -> OperationType:
        """Get the operation with highest weight."""
        idx = self.alpha.argmax().item()
        return self.operations[idx].op_type


class DARTSCell(nn.Module):
    """
    DARTS cell with searchable operations.

    Forms the building block of the supernet.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_nodes: int = 4,
        search_space: Optional[SearchSpace] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.search_space = search_space or SearchSpace()

        # Input projection
        self.input_proj = nn.Linear(in_dim, out_dim)

        # Mixed operations between all node pairs
        self.ops = nn.ModuleDict()

        for i in range(num_nodes):
            for j in range(i):
                key = f"{j}_to_{i}"
                self.ops[key] = MixedOperation(out_dim, out_dim, self.search_space)

        # Output projection
        self.output_proj = nn.Linear(out_dim * num_nodes, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through DARTS cell."""
        # Project input
        x = self.input_proj(x)

        # Node states
        states = [x]

        for i in range(1, self.num_nodes):
            # Aggregate from all previous nodes
            s = torch.zeros_like(x)
            for j in range(i):
                key = f"{j}_to_{i}"
                s = s + self.ops[key](states[j])
            states.append(s)

        # Concatenate all states
        concat = torch.cat(states, dim=-1)

        return self.output_proj(concat)

    def get_architecture(self) -> Dict[str, OperationType]:
        """Get selected architecture."""
        arch = {}
        for key, op in self.ops.items():
            arch[key] = op.get_selected_operation()
        return arch


class DARTSAgent(nn.Module):
    """
    Agent with DARTS-searchable architecture.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_cells: int = 3,
        nodes_per_cell: int = 4,
        search_space: Optional[SearchSpace] = None,
    ):
        super().__init__()
        self.search_space = search_space or SearchSpace()

        # Stack of DARTS cells
        self.cells = nn.ModuleList()

        for i in range(num_cells):
            in_d = input_dim if i == 0 else hidden_dim
            self.cells.append(DARTSCell(in_d, hidden_dim, nodes_per_cell, self.search_space))

        # Output head
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through DARTS agent."""
        for cell in self.cells:
            x = cell(x)
        return self.output(x)

    def get_architecture_weights(self) -> List[torch.Tensor]:
        """Get all architecture weights (for optimization)."""
        weights = []
        for cell in self.cells:
            for op in cell.ops.values():
                weights.append(op.alpha)
        return weights

    def get_architecture(self) -> List[Dict[str, OperationType]]:
        """Get selected architecture for all cells."""
        return [cell.get_architecture() for cell in self.cells]


class DARTSSearcher:
    """
    DARTS architecture search algorithm.

    Alternates between:
    1. Training network weights (inner loop)
    2. Updating architecture weights (outer loop)
    """

    def __init__(
        self,
        model: DARTSAgent,
        arch_lr: float = 3e-4,
        weight_lr: float = 0.025,
        weight_decay: float = 3e-4,
    ):
        self.model = model

        # Separate optimizers for weights and architecture
        self.weight_optimizer = torch.optim.SGD(
            [p for n, p in model.named_parameters() if 'alpha' not in n],
            lr=weight_lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )

        self.arch_optimizer = torch.optim.Adam(
            model.get_architecture_weights(),
            lr=arch_lr,
            betas=(0.5, 0.999),
            weight_decay=1e-3,
        )

    def search_step(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        val_data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable,
    ) -> Dict[str, float]:
        """One step of architecture search."""
        train_x, train_y = train_data
        val_x, val_y = val_data

        # Update architecture on validation data
        self.arch_optimizer.zero_grad()
        val_out = self.model(val_x)
        arch_loss = loss_fn(val_out, val_y)
        arch_loss.backward()
        self.arch_optimizer.step()

        # Update weights on training data
        self.weight_optimizer.zero_grad()
        train_out = self.model(train_x)
        weight_loss = loss_fn(train_out, train_y)
        weight_loss.backward()
        self.weight_optimizer.step()

        return {
            "arch_loss": arch_loss.item(),
            "weight_loss": weight_loss.item(),
        }

    def derive_architecture(self) -> List[Dict[str, OperationType]]:
        """Get final architecture after search."""
        return self.model.get_architecture()


class ENASController(nn.Module):
    """
    ENAS controller for sampling architectures.

    Uses an LSTM to generate architecture decisions sequentially.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        num_layers: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.search_space = search_space
        self.num_layers = num_layers
        self.num_operations = len(search_space.operations)

        # LSTM controller
        self.lstm = nn.LSTMCell(hidden_dim, hidden_dim)

        # Embedding for previous decisions
        self.embedding = nn.Embedding(self.num_operations, hidden_dim)

        # Heads for different decisions
        self.op_head = nn.Linear(hidden_dim, self.num_operations)
        self.skip_head = nn.Linear(hidden_dim, num_layers)  # Which layers to skip from

        # Initial input
        self.init_input = nn.Parameter(torch.zeros(1, hidden_dim))
        self.init_hidden = nn.Parameter(torch.zeros(1, hidden_dim))
        self.init_cell = nn.Parameter(torch.zeros(1, hidden_dim))

    def forward(
        self,
        batch_size: int = 1,
    ) -> Tuple[List[int], List[List[int]], torch.Tensor]:
        """
        Sample an architecture.

        Returns:
            (operations, skip_connections, log_probs)
        """
        device = self.init_input.device

        # Initialize
        inputs = self.init_input.expand(batch_size, -1)
        hidden = self.init_hidden.expand(batch_size, -1).contiguous()
        cell = self.init_cell.expand(batch_size, -1).contiguous()

        operations = []
        skip_connections = []
        log_probs = []
        entropies = []

        for layer_idx in range(self.num_layers):
            # LSTM step
            hidden, cell = self.lstm(inputs, (hidden, cell))

            # Sample operation
            op_logits = self.op_head(hidden)
            op_probs = F.softmax(op_logits, dim=-1)
            op_dist = torch.distributions.Categorical(op_probs)
            op = op_dist.sample()

            operations.append(op.item())
            log_probs.append(op_dist.log_prob(op))
            entropies.append(op_dist.entropy())

            # Sample skip connections (from previous layers)
            if layer_idx > 0:
                skip_logits = self.skip_head(hidden)[:, :layer_idx]
                skip_probs = torch.sigmoid(skip_logits)
                skip_dist = torch.distributions.Bernoulli(skip_probs)
                skips = skip_dist.sample()

                skip_connections.append(skips.squeeze(0).tolist())
                skip_log_prob = skip_dist.log_prob(skips).sum()
                log_probs.append(skip_log_prob.unsqueeze(0))
            else:
                skip_connections.append([])

            # Update input
            inputs = self.embedding(op)

        # Ensure all log_probs are same shape before stacking
        log_probs = [lp.view(-1) for lp in log_probs]
        total_log_prob = torch.cat(log_probs).sum()

        return operations, skip_connections, total_log_prob

    def sample_architecture(self) -> Dict[str, Any]:
        """Sample a complete architecture."""
        with torch.no_grad():
            ops, skips, _ = self.forward(batch_size=1)

        return {
            "operations": [self.search_space.operations[i] for i in ops],
            "skip_connections": skips,
        }


class ENASSharedNetwork(nn.Module):
    """
    Shared network for ENAS with weight sharing.

    All architectures share the same weight matrices.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        search_space: Optional[SearchSpace] = None,
    ):
        super().__init__()
        self.search_space = search_space or SearchSpace()
        self.num_layers = num_layers

        # Shared operations for each layer
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layer_ops = nn.ModuleDict({
                op.name: Operation(op, in_d, hidden_dim)
                for op in self.search_space.operations
            })
            self.layers.append(layer_ops)

        # Skip connection projections
        self.skip_projs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Output
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        architecture: Dict[str, Any],
    ) -> torch.Tensor:
        """Forward with specific architecture."""
        operations = architecture["operations"]
        skip_connections = architecture["skip_connections"]

        layer_outputs = []

        for i, op_type in enumerate(operations):
            # Get operation
            op = self.layers[i][op_type.name]

            # Apply operation
            if i == 0:
                out = op(x)
            else:
                # Add skip connections
                skip_input = layer_outputs[-1]
                for j, should_skip in enumerate(skip_connections[i]):
                    if should_skip:
                        skip_input = skip_input + self.skip_projs[j](layer_outputs[j])
                out = op(skip_input)

            layer_outputs.append(out)

        return self.output(layer_outputs[-1])


class ENASSearcher:
    """
    ENAS search algorithm.

    Alternates between:
    1. Training shared weights
    2. Training controller with REINFORCE
    """

    def __init__(
        self,
        controller: ENASController,
        shared_network: ENASSharedNetwork,
        controller_lr: float = 3.5e-4,
        shared_lr: float = 0.05,
        entropy_weight: float = 0.0001,
    ):
        self.controller = controller
        self.shared_network = shared_network
        self.entropy_weight = entropy_weight

        self.controller_optimizer = torch.optim.Adam(
            controller.parameters(),
            lr=controller_lr,
        )

        self.shared_optimizer = torch.optim.SGD(
            shared_network.parameters(),
            lr=shared_lr,
            momentum=0.9,
            weight_decay=1e-4,
        )

        self.baseline = None

    def train_shared(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable,
        num_steps: int = 1,
    ) -> float:
        """Train shared weights with sampled architecture."""
        total_loss = 0.0

        for _ in range(num_steps):
            # Sample architecture
            arch = self.controller.sample_architecture()

            # Forward
            x, y = data
            out = self.shared_network(x, arch)
            loss = loss_fn(out, y)

            # Backward
            self.shared_optimizer.zero_grad()
            loss.backward()
            self.shared_optimizer.step()

            total_loss += loss.item()

        return total_loss / num_steps

    def train_controller(
        self,
        val_data: Tuple[torch.Tensor, torch.Tensor],
        loss_fn: Callable,
        num_samples: int = 10,
    ) -> Dict[str, float]:
        """Train controller with REINFORCE."""
        x, y = val_data

        rewards = []
        log_probs = []

        for _ in range(num_samples):
            # Sample architecture
            ops, skips, log_prob = self.controller.forward()

            arch = {
                "operations": [self.controller.search_space.operations[i] for i in ops],
                "skip_connections": skips,
            }

            # Evaluate
            with torch.no_grad():
                out = self.shared_network(x, arch)
                loss = loss_fn(out, y)
                reward = -loss.item()  # Negative loss as reward

            rewards.append(reward)
            log_probs.append(log_prob)

        # Update baseline
        avg_reward = sum(rewards) / len(rewards)
        if self.baseline is None:
            self.baseline = avg_reward
        else:
            self.baseline = 0.95 * self.baseline + 0.05 * avg_reward

        # REINFORCE update
        self.controller_optimizer.zero_grad()

        policy_loss = 0
        for reward, log_prob in zip(rewards, log_probs):
            advantage = reward - self.baseline
            policy_loss = policy_loss - advantage * log_prob

        policy_loss = policy_loss / num_samples
        policy_loss.backward()

        self.controller_optimizer.step()

        return {
            "avg_reward": avg_reward,
            "policy_loss": policy_loss.item(),
        }


class RandomSearchNAS:
    """
    Random search baseline for NAS.

    Simple but surprisingly effective.
    """

    def __init__(
        self,
        search_space: SearchSpace,
        build_fn: Callable[[Dict], nn.Module],
        eval_fn: Callable[[nn.Module], float],
    ):
        self.search_space = search_space
        self.build_fn = build_fn
        self.eval_fn = eval_fn

        self.history: List[Tuple[Dict, float]] = []
        self.best_arch: Optional[Dict] = None
        self.best_score: float = float('-inf')

    def sample_architecture(self) -> Dict[str, Any]:
        """Sample random architecture."""
        # Random number of layers
        num_layers = random.randint(
            self.search_space.min_layers,
            self.search_space.max_layers,
        )

        # Random operations per layer
        operations = [
            random.choice(self.search_space.operations)
            for _ in range(num_layers)
        ]

        # Random hidden dimension
        hidden_dim = random.choice(range(
            self.search_space.min_hidden_dim,
            self.search_space.max_hidden_dim + 1,
            self.search_space.dim_step,
        ))

        # Random dropout
        dropout = random.uniform(*self.search_space.dropout_range)

        return {
            "operations": operations,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "num_heads": random.randint(
                self.search_space.min_heads,
                self.search_space.max_heads,
            ),
        }

    def search(self, num_samples: int = 100) -> Dict[str, Any]:
        """Run random search."""
        for _ in range(num_samples):
            arch = self.sample_architecture()

            # Build and evaluate
            model = self.build_fn(arch)
            score = self.eval_fn(model)

            self.history.append((arch, score))

            if score > self.best_score:
                self.best_score = score
                self.best_arch = arch

        return self.best_arch

    def get_top_k(self, k: int = 5) -> List[Tuple[Dict, float]]:
        """Get top k architectures."""
        sorted_history = sorted(self.history, key=lambda x: x[1], reverse=True)
        return sorted_history[:k]
