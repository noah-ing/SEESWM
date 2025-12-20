"""
Evolution and Scaling module for SEESWM.

Provides infrastructure for:
1. Distributed training across devices
2. Evolutionary optimization of swarm topology
3. Neural Architecture Search for agents
4. Scaling to 1000+ agent swarms
"""

# Distributed training
from .distributed import (
    # Configuration
    DistributedConfig,
    ParallelismMode,
    # Management
    DistributedManager,
    DistributedSwarm,
    DistributedTrainer,
    # Parallelism strategies
    ModelParallelSwarm,
    PipelineParallelSwarm,
    AsyncGradientAggregator,
    # Utilities
    create_distributed_dataloader,
)

# Genetic/Evolutionary optimization
from .genetic import (
    # Configuration
    EvolutionConfig,
    MutationType,
    # Genes
    Gene,
    AgentGene,
    ConnectionGene,
    SwarmGenome,
    # Evolution
    InnovationTracker,
    SwarmMutator,
    SwarmCrossover,
    Species,
    genome_distance,
    GeneticOptimizer,
    # CMA-ES
    CMAES,
)

# Neural Architecture Search
from .nas import (
    # Search space
    SearchSpace,
    OperationType,
    # Operations
    Operation,
    MixedOperation,
    # DARTS
    DARTSCell,
    DARTSAgent,
    DARTSSearcher,
    # ENAS
    ENASController,
    ENASSharedNetwork,
    ENASSearcher,
    # Random search
    RandomSearchNAS,
)

# Scaling utilities
from .scaling import (
    # Configuration
    ScalingConfig,
    # Core components
    AgentPool,
    AgentGroup,
    HierarchicalSwarm,
    SparseMessageGraph,
    # Memory optimization
    CheckpointedSwarm,
    BatchedAgentProcessor,
    OffloadedSwarm,
    StreamingSwarm,
    # Utilities
    estimate_memory_usage,
)

__all__ = [
    # Distributed
    "DistributedConfig",
    "ParallelismMode",
    "DistributedManager",
    "DistributedSwarm",
    "DistributedTrainer",
    "ModelParallelSwarm",
    "PipelineParallelSwarm",
    "AsyncGradientAggregator",
    "create_distributed_dataloader",
    # Genetic
    "EvolutionConfig",
    "MutationType",
    "Gene",
    "AgentGene",
    "ConnectionGene",
    "SwarmGenome",
    "InnovationTracker",
    "SwarmMutator",
    "SwarmCrossover",
    "Species",
    "genome_distance",
    "GeneticOptimizer",
    "CMAES",
    # NAS
    "SearchSpace",
    "OperationType",
    "Operation",
    "MixedOperation",
    "DARTSCell",
    "DARTSAgent",
    "DARTSSearcher",
    "ENASController",
    "ENASSharedNetwork",
    "ENASSearcher",
    "RandomSearchNAS",
    # Scaling
    "ScalingConfig",
    "AgentPool",
    "AgentGroup",
    "HierarchicalSwarm",
    "SparseMessageGraph",
    "CheckpointedSwarm",
    "BatchedAgentProcessor",
    "OffloadedSwarm",
    "StreamingSwarm",
    "estimate_memory_usage",
]
