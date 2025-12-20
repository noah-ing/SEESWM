"""
Meta-learning for fast adaptation.

"Learning to learn" - agents that can quickly adapt to new tasks
by leveraging experience from previous tasks.

Approaches:
1. MAML: Learn initialization that enables fast fine-tuning
2. Meta-SGD: Learn per-parameter learning rates
3. Task Embedding: Learn to recognize task similarity
4. Reptile: Simplified meta-learning via averaging
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Callable
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD


@dataclass
class Task:
    """A task for meta-learning."""

    support_x: torch.Tensor  # Examples for adaptation
    support_y: torch.Tensor  # Labels for adaptation
    query_x: torch.Tensor  # Examples for evaluation
    query_y: torch.Tensor  # Labels for evaluation
    task_id: Optional[int] = None


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning."""

    # MAML settings
    inner_lr: float = 0.01  # Learning rate for task adaptation
    inner_steps: int = 5  # Gradient steps for adaptation
    outer_lr: float = 0.001  # Meta-learning rate

    # Meta-SGD settings
    learn_lr: bool = True  # Learn per-parameter learning rates
    lr_init: float = 0.01

    # Task sampling
    tasks_per_batch: int = 4
    shots: int = 5  # Examples per class for few-shot

    # First-order approximation
    first_order: bool = False  # Use first-order MAML (faster)


class MAML(nn.Module):
    """
    Model-Agnostic Meta-Learning.

    Learns an initialization from which the model can quickly
    adapt to new tasks with just a few gradient steps.

    Reference: "Model-Agnostic Meta-Learning for Fast Adaptation"
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[MetaLearningConfig] = None,
    ):
        super().__init__()
        self.model = model
        self.config = config or MetaLearningConfig()

        # Meta-optimizer
        self.meta_optimizer = Adam(
            self.model.parameters(),
            lr=self.config.outer_lr,
        )

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> nn.Module:
        """
        Adapt model to a task using support set.

        Returns a new model with adapted parameters.
        """
        num_steps = num_steps or self.config.inner_steps

        # Clone model for adaptation
        adapted_model = deepcopy(self.model)

        # Inner loop optimization
        inner_optimizer = SGD(
            adapted_model.parameters(),
            lr=self.config.inner_lr,
        )

        for _ in range(num_steps):
            inner_optimizer.zero_grad()
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)
            loss.backward()
            inner_optimizer.step()

        return adapted_model

    def meta_train_step(self, tasks: List[Task]) -> Dict[str, float]:
        """
        One meta-training step on a batch of tasks.

        For each task:
        1. Adapt to support set
        2. Evaluate on query set
        3. Accumulate gradients
        """
        self.meta_optimizer.zero_grad()

        total_query_loss = 0.0
        total_query_acc = 0.0

        for task in tasks:
            # Adapt to task
            adapted_model = self.adapt(task.support_x, task.support_y)

            # Evaluate on query set
            query_logits = adapted_model(task.query_x)
            query_loss = F.cross_entropy(query_logits, task.query_y)

            # For first-order MAML, we don't backprop through adaptation
            if self.config.first_order:
                # Copy gradients to original model
                for p_orig, p_adapt in zip(
                    self.model.parameters(),
                    adapted_model.parameters(),
                ):
                    if p_adapt.grad is not None:
                        if p_orig.grad is None:
                            p_orig.grad = p_adapt.grad.clone()
                        else:
                            p_orig.grad += p_adapt.grad.clone()
            else:
                # Full second-order MAML
                query_loss.backward()

            total_query_loss += query_loss.item()

            # Accuracy
            preds = query_logits.argmax(dim=-1)
            acc = (preds == task.query_y).float().mean().item()
            total_query_acc += acc

        # Average and update
        num_tasks = len(tasks)
        self.meta_optimizer.step()

        return {
            "query_loss": total_query_loss / num_tasks,
            "query_accuracy": total_query_acc / num_tasks,
        }

    def evaluate(self, task: Task) -> Dict[str, float]:
        """Evaluate on a task without updating meta-parameters."""
        with torch.no_grad():
            # Adapt (creates new model, doesn't affect self.model)
            adapted = self.adapt(task.support_x, task.support_y)

            # Evaluate
            logits = adapted(task.query_x)
            loss = F.cross_entropy(logits, task.query_y).item()

            preds = logits.argmax(dim=-1)
            acc = (preds == task.query_y).float().mean().item()

        return {"loss": loss, "accuracy": acc}


class MetaSGD(nn.Module):
    """
    Meta-SGD: Learned per-parameter learning rates.

    Instead of a single learning rate, learns an optimal
    learning rate for each parameter.

    Reference: "Meta-SGD: Learning to Learn Quickly for Few-Shot Learning"
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[MetaLearningConfig] = None,
    ):
        super().__init__()
        self.model = model
        self.config = config or MetaLearningConfig()

        # Learnable per-parameter learning rates
        self.lr_params = nn.ParameterList([
            nn.Parameter(torch.ones_like(p) * self.config.lr_init)
            for p in model.parameters()
        ])

        # Meta optimizer
        all_params = list(model.parameters()) + list(self.lr_params)
        self.meta_optimizer = Adam(all_params, lr=self.config.outer_lr)

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> nn.Module:
        """Adapt with learned learning rates."""
        num_steps = num_steps or self.config.inner_steps

        # Clone model
        adapted_model = deepcopy(self.model)
        adapted_params = list(adapted_model.parameters())

        for _ in range(num_steps):
            # Forward pass
            logits = adapted_model(support_x)
            loss = F.cross_entropy(logits, support_y)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_params,
                create_graph=not self.config.first_order,
            )

            # Update with learned learning rates
            for p, g, lr in zip(adapted_params, grads, self.lr_params):
                p.data = p.data - lr.abs() * g  # abs() ensures positive lr

        return adapted_model

    def meta_train_step(self, tasks: List[Task]) -> Dict[str, float]:
        """Meta-training step."""
        self.meta_optimizer.zero_grad()

        total_loss = 0.0
        total_acc = 0.0

        for task in tasks:
            adapted = self.adapt(task.support_x, task.support_y)

            logits = adapted(task.query_x)
            loss = F.cross_entropy(logits, task.query_y)
            loss.backward()

            total_loss += loss.item()
            acc = (logits.argmax(-1) == task.query_y).float().mean().item()
            total_acc += acc

        self.meta_optimizer.step()

        return {
            "query_loss": total_loss / len(tasks),
            "query_accuracy": total_acc / len(tasks),
            "avg_lr": sum(lr.abs().mean().item() for lr in self.lr_params) / len(self.lr_params),
        }


class Reptile(nn.Module):
    """
    Reptile: Simplified meta-learning.

    Instead of differentiating through adaptation,
    simply moves initialization toward adapted parameters.

    Much simpler and often competitive with MAML.

    Reference: "On First-Order Meta-Learning Algorithms"
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 1.0,
        inner_steps: int = 5,
    ):
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        """Adapt to task (same as standard fine-tuning)."""
        adapted = deepcopy(self.model)
        optimizer = SGD(adapted.parameters(), lr=self.inner_lr)

        for _ in range(self.inner_steps):
            optimizer.zero_grad()
            logits = adapted(support_x)
            loss = F.cross_entropy(logits, support_y)
            loss.backward()
            optimizer.step()

        return adapted

    def meta_train_step(self, tasks: List[Task]) -> Dict[str, float]:
        """
        Reptile update: average adapted parameters.
        """
        # Store original parameters
        old_params = [p.clone() for p in self.model.parameters()]

        # Accumulate adapted parameters
        new_params = [torch.zeros_like(p) for p in self.model.parameters()]
        total_loss = 0.0

        for task in tasks:
            adapted = self.adapt(task.support_x, task.support_y)

            for new_p, adapted_p in zip(new_params, adapted.parameters()):
                new_p += adapted_p.data

            # Evaluate
            with torch.no_grad():
                logits = adapted(task.query_x)
                total_loss += F.cross_entropy(logits, task.query_y).item()

        # Average
        for new_p in new_params:
            new_p /= len(tasks)

        # Reptile update: move toward average
        with torch.no_grad():
            for p, old_p, new_p in zip(
                self.model.parameters(), old_params, new_params
            ):
                p.data = old_p + self.outer_lr * (new_p - old_p)

        return {"query_loss": total_loss / len(tasks)}


class TaskEmbedding(nn.Module):
    """
    Learn task embeddings for transfer.

    Maps task support sets to embeddings that capture task identity.
    Similar tasks should have similar embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Encode each example
        self.example_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Aggregate examples into task embedding
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Task embedding bank (for similarity lookup)
        self.embedding_bank: List[Tuple[torch.Tensor, int]] = []

    def forward(self, support_x: torch.Tensor) -> torch.Tensor:
        """
        Compute task embedding from support set.

        Args:
            support_x: Support examples [num_examples, input_dim]

        Returns:
            Task embedding [embedding_dim]
        """
        # Encode each example
        encoded = self.example_encoder(support_x)  # [N, hidden]

        # Aggregate (mean pooling)
        aggregated = encoded.mean(dim=0)  # [hidden]

        # Final embedding
        embedding = self.aggregator(aggregated)  # [embedding_dim]

        return embedding

    def add_to_bank(
        self,
        task_embedding: torch.Tensor,
        task_id: int,
    ) -> None:
        """Store task embedding for future similarity lookup."""
        self.embedding_bank.append((task_embedding.detach(), task_id))

    def find_similar_tasks(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Find most similar tasks in the bank."""
        if not self.embedding_bank:
            return []

        similarities = []
        for stored_emb, task_id in self.embedding_bank:
            sim = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                stored_emb.unsqueeze(0),
            ).item()
            similarities.append((task_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


class AdaptiveMetaLearner(nn.Module):
    """
    Adaptive meta-learner that chooses strategy based on task.

    Combines multiple meta-learning approaches and selects
    the best one for each task.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[MetaLearningConfig] = None,
    ):
        super().__init__()
        self.config = config or MetaLearningConfig()

        # Multiple strategies
        self.maml = MAML(deepcopy(model), config)
        self.reptile = Reptile(
            deepcopy(model),
            inner_lr=self.config.inner_lr,
            inner_steps=self.config.inner_steps,
        )

        # Task embedding for strategy selection
        # Assume model has .input_dim attribute or we default
        input_dim = getattr(model, 'input_dim', 64)
        self.task_embedder = TaskEmbedding(input_dim)

        # Strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # 2 strategies
        )

        # Track performance per strategy
        self.strategy_performance: Dict[str, List[float]] = {
            "maml": [],
            "reptile": [],
        }

    def select_strategy(self, task: Task) -> str:
        """Select best strategy for task."""
        # Get task embedding
        with torch.no_grad():
            embedding = self.task_embedder(task.support_x)
            logits = self.strategy_selector(embedding)
            strategy_idx = logits.argmax().item()

        return "maml" if strategy_idx == 0 else "reptile"

    def adapt(self, task: Task) -> Tuple[nn.Module, str]:
        """Adapt using selected strategy."""
        strategy = self.select_strategy(task)

        if strategy == "maml":
            adapted = self.maml.adapt(task.support_x, task.support_y)
        else:
            adapted = self.reptile.adapt(task.support_x, task.support_y)

        return adapted, strategy

    def update_performance(
        self,
        strategy: str,
        accuracy: float,
    ) -> None:
        """Track strategy performance for future selection."""
        self.strategy_performance[strategy].append(accuracy)
        # Keep recent history
        if len(self.strategy_performance[strategy]) > 100:
            self.strategy_performance[strategy] = \
                self.strategy_performance[strategy][-100:]

    def get_strategy_stats(self) -> Dict[str, float]:
        """Get average performance per strategy."""
        stats = {}
        for strategy, accs in self.strategy_performance.items():
            if accs:
                stats[f"{strategy}_avg_acc"] = sum(accs) / len(accs)
        return stats
