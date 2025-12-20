"""
Distributed training infrastructure for large-scale swarms.

Supports:
1. Data parallelism: Same model, different data shards
2. Model parallelism: Split agents across devices
3. Pipeline parallelism: Agents in sequence across devices
4. Asynchronous training: Non-blocking gradient updates
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Callable
from enum import Enum
import os
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler


class ParallelismMode(Enum):
    """Type of parallelism for distributed training."""
    DATA = "data"           # Same model, different data
    MODEL = "model"         # Different model parts
    PIPELINE = "pipeline"   # Sequential model stages
    HYBRID = "hybrid"       # Combination


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    # Parallelism
    mode: ParallelismMode = ParallelismMode.DATA
    world_size: int = 1       # Total number of processes
    rank: int = 0             # This process's rank
    local_rank: int = 0       # Rank on this node

    # Communication
    backend: str = "nccl"     # nccl for GPU, gloo for CPU
    init_method: str = "env://"

    # Synchronization
    sync_gradients: bool = True
    gradient_accumulation_steps: int = 1
    sync_batch_norm: bool = True

    # Efficiency
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every_n_steps: int = 1000


class DistributedManager:
    """
    Manager for distributed training setup.

    Handles initialization, device assignment, and cleanup.
    """

    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        self.initialized = False
        self.device: Optional[torch.device] = None

    def initialize(self) -> None:
        """Initialize distributed training."""
        if self.initialized:
            return

        # Check environment variables
        if "WORLD_SIZE" in os.environ:
            self.config.world_size = int(os.environ["WORLD_SIZE"])
        if "RANK" in os.environ:
            self.config.rank = int(os.environ["RANK"])
        if "LOCAL_RANK" in os.environ:
            self.config.local_rank = int(os.environ["LOCAL_RANK"])

        # Initialize process group
        if self.config.world_size > 1:
            dist.init_process_group(
                backend=self.config.backend,
                init_method=self.config.init_method,
                world_size=self.config.world_size,
                rank=self.config.rank,
            )

        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device(f"cuda:{self.config.local_rank}")
        else:
            self.device = torch.device("cpu")

        self.initialized = True

    def cleanup(self) -> None:
        """Clean up distributed training."""
        if self.initialized and self.config.world_size > 1:
            dist.destroy_process_group()
        self.initialized = False

    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.config.rank == 0

    def barrier(self) -> None:
        """Synchronize all processes."""
        if self.config.world_size > 1:
            dist.barrier()

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        """Reduce tensor across all processes."""
        if self.config.world_size > 1:
            dist.all_reduce(tensor, op=op)
        return tensor

    def all_gather(
        self,
        tensor: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Gather tensors from all processes."""
        if self.config.world_size > 1:
            gathered = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
            dist.all_gather(gathered, tensor)
            return gathered
        return [tensor]

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
    ) -> torch.Tensor:
        """Broadcast tensor from source to all processes."""
        if self.config.world_size > 1:
            dist.broadcast(tensor, src=src)
        return tensor


class DistributedSwarm(nn.Module):
    """
    Wrapper for distributed swarm training.

    Distributes agents across devices and handles communication.
    """

    def __init__(
        self,
        swarm: nn.Module,
        config: DistributedConfig,
        manager: DistributedManager,
    ):
        super().__init__()
        self.config = config
        self.manager = manager

        # Move to device
        self.swarm = swarm.to(manager.device)

        # Wrap with DDP for data parallelism
        if config.mode == ParallelismMode.DATA and config.world_size > 1:
            if config.sync_batch_norm:
                self.swarm = nn.SyncBatchNorm.convert_sync_batchnorm(self.swarm)

            self.swarm = DDP(
                self.swarm,
                device_ids=[config.local_rank] if torch.cuda.is_available() else None,
                output_device=config.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=config.find_unused_parameters,
                bucket_cap_mb=config.bucket_cap_mb,
            )

        self.gradient_accumulation_count = 0

    def forward(self, *args, **kwargs):
        """Forward pass through distributed swarm."""
        return self.swarm(*args, **kwargs)

    def backward_step(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        """
        Backward step with gradient accumulation.

        Returns True if optimizer step was taken.
        """
        # Scale loss for gradient accumulation
        scaled_loss = loss / self.config.gradient_accumulation_steps
        scaled_loss.backward()

        self.gradient_accumulation_count += 1

        if self.gradient_accumulation_count >= self.config.gradient_accumulation_steps:
            # Sync gradients if needed
            if self.config.sync_gradients and self.config.world_size > 1:
                self._sync_gradients()

            optimizer.step()
            optimizer.zero_grad()
            self.gradient_accumulation_count = 0
            return True

        return False

    def _sync_gradients(self) -> None:
        """Synchronize gradients across processes."""
        # DDP handles this automatically, but we can add custom logic
        pass

    @property
    def module(self) -> nn.Module:
        """Get the underlying module (unwrap DDP if needed)."""
        if isinstance(self.swarm, DDP):
            return self.swarm.module
        return self.swarm


class ModelParallelSwarm(nn.Module):
    """
    Swarm with model parallelism.

    Splits agents across multiple devices.
    """

    def __init__(
        self,
        agents: nn.ModuleDict,
        device_map: Dict[str, torch.device],
    ):
        super().__init__()
        self.agents = agents
        self.device_map = device_map

        # Move each agent to its assigned device
        for agent_id, device in device_map.items():
            if agent_id in agents:
                agents[agent_id] = agents[agent_id].to(device)

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with cross-device communication."""
        outputs = {}

        for agent_id, agent in self.agents.items():
            device = self.device_map.get(agent_id, torch.device("cpu"))

            # Move input to agent's device
            agent_input = inputs.get(agent_id)
            if agent_input is not None:
                agent_input = agent_input.to(device)

            # Forward
            output = agent(agent_input)
            outputs[agent_id] = output

        return outputs

    def gather_outputs(
        self,
        outputs: Dict[str, torch.Tensor],
        target_device: torch.device,
    ) -> torch.Tensor:
        """Gather outputs from all agents to target device."""
        gathered = []
        for agent_id in sorted(outputs.keys()):
            output = outputs[agent_id].to(target_device)
            gathered.append(output)
        return torch.stack(gathered, dim=1)


class PipelineParallelSwarm(nn.Module):
    """
    Swarm with pipeline parallelism.

    Agents are organized in stages that execute in sequence.
    """

    def __init__(
        self,
        stages: List[nn.Module],
        devices: List[torch.device],
        micro_batch_size: int = 4,
    ):
        super().__init__()
        self.stages = nn.ModuleList(stages)
        self.devices = devices
        self.micro_batch_size = micro_batch_size
        self.num_stages = len(stages)

        # Move stages to devices
        for stage, device in zip(self.stages, devices):
            stage.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with pipeline parallelism.

        Uses micro-batching for efficiency.
        """
        batch_size = x.shape[0]
        num_micro_batches = math.ceil(batch_size / self.micro_batch_size)

        # Split into micro-batches
        micro_batches = torch.chunk(x, num_micro_batches, dim=0)

        # Pipeline execution
        outputs = []

        for mb in micro_batches:
            current = mb

            for stage_idx, stage in enumerate(self.stages):
                device = self.devices[stage_idx]
                current = current.to(device)
                current = stage(current)

            outputs.append(current)

        # Gather outputs
        return torch.cat(outputs, dim=0)


class AsyncGradientAggregator:
    """
    Asynchronous gradient aggregation for large-scale training.

    Uses parameter server style updates.
    """

    def __init__(
        self,
        model: nn.Module,
        num_workers: int,
        staleness_tolerance: int = 3,
    ):
        self.model = model
        self.num_workers = num_workers
        self.staleness_tolerance = staleness_tolerance

        # Gradient buffers
        self.gradient_buffers: Dict[int, Dict[str, torch.Tensor]] = {}
        self.versions: Dict[int, int] = {}
        self.current_version = 0

        # Initialize buffers
        for worker_id in range(num_workers):
            self.gradient_buffers[worker_id] = {}
            self.versions[worker_id] = 0

    def submit_gradients(
        self,
        worker_id: int,
        gradients: Dict[str, torch.Tensor],
    ) -> bool:
        """
        Submit gradients from a worker.

        Returns True if gradients were accepted.
        """
        # Check staleness
        staleness = self.current_version - self.versions[worker_id]
        if staleness > self.staleness_tolerance:
            return False

        self.gradient_buffers[worker_id] = gradients
        self.versions[worker_id] = self.current_version
        return True

    def aggregate(self) -> Dict[str, torch.Tensor]:
        """Aggregate gradients from all workers."""
        aggregated = {}

        # Get all non-stale gradients
        valid_buffers = []
        for worker_id, grads in self.gradient_buffers.items():
            if self.current_version - self.versions[worker_id] <= self.staleness_tolerance:
                valid_buffers.append(grads)

        if not valid_buffers:
            return {}

        # Average gradients
        for name in valid_buffers[0].keys():
            stacked = torch.stack([buf[name] for buf in valid_buffers if name in buf])
            aggregated[name] = stacked.mean(dim=0)

        self.current_version += 1
        return aggregated


class DistributedTrainer:
    """
    Trainer for distributed swarm training.

    Handles data distribution, gradient synchronization, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        learning_rate: float = 1e-3,
    ):
        self.config = config
        self.manager = DistributedManager(config)
        self.manager.initialize()

        # Wrap model
        self.model = DistributedSwarm(model, config, self.manager)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        # Training state
        self.global_step = 0
        self.epoch = 0

    def train_epoch(
        self,
        dataloader: DataLoader,
        loss_fn: Callable,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Move to device
            if isinstance(batch, (tuple, list)):
                batch = [b.to(self.manager.device) for b in batch]
            else:
                batch = batch.to(self.manager.device)

            # Forward
            if isinstance(batch, (tuple, list)):
                output = self.model(batch[0])
                loss = loss_fn(output, batch[1])
            else:
                output = self.model(batch)
                loss = loss_fn(output)

            # Backward
            stepped = self.model.backward_step(loss, self.optimizer)

            if stepped:
                self.global_step += 1

                # Checkpoint
                if (
                    self.manager.is_main_process()
                    and self.global_step % self.config.save_every_n_steps == 0
                ):
                    self.save_checkpoint()

            total_loss += loss.item()
            num_batches += 1

        # Reduce loss across processes
        avg_loss = total_loss / max(1, num_batches)
        if self.config.world_size > 1:
            loss_tensor = torch.tensor([avg_loss], device=self.manager.device)
            self.manager.all_reduce(loss_tensor)
            avg_loss = loss_tensor.item() / self.config.world_size

        self.epoch += 1

        return {"loss": avg_loss, "epoch": self.epoch}

    def save_checkpoint(self, path: Optional[str] = None) -> None:
        """Save training checkpoint."""
        if path is None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            path = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_{self.global_step}.pt"
            )

        checkpoint = {
            "model_state_dict": self.model.module.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.manager.device)
        self.model.module.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]

    def cleanup(self) -> None:
        """Clean up distributed training."""
        self.manager.cleanup()


def create_distributed_dataloader(
    dataset,
    batch_size: int,
    config: DistributedConfig,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a dataloader for distributed training."""
    sampler = None
    if config.world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=shuffle,
        )
        shuffle = False  # Sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
