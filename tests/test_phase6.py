"""
Tests for Phase 6: Evolution & Scaling.

Tests:
1. Distributed training infrastructure
2. Genetic/evolutionary optimization
3. Neural Architecture Search
4. Scaling utilities
"""

import pytest
import torch
import torch.nn as nn

from src.evolution import (
    # Distributed
    DistributedConfig,
    ParallelismMode,
    DistributedManager,
    DistributedSwarm,
    ModelParallelSwarm,
    PipelineParallelSwarm,
    # Genetic
    EvolutionConfig,
    SwarmGenome,
    AgentGene,
    ConnectionGene,
    InnovationTracker,
    SwarmMutator,
    SwarmCrossover,
    Species,
    genome_distance,
    GeneticOptimizer,
    CMAES,
    # NAS
    SearchSpace,
    OperationType,
    Operation,
    MixedOperation,
    DARTSCell,
    DARTSAgent,
    DARTSSearcher,
    ENASController,
    ENASSharedNetwork,
    ENASSearcher,
    RandomSearchNAS,
    # Scaling
    ScalingConfig,
    AgentPool,
    AgentGroup,
    HierarchicalSwarm,
    SparseMessageGraph,
    CheckpointedSwarm,
    BatchedAgentProcessor,
    estimate_memory_usage,
)


class TestDistributed:
    """Tests for distributed training infrastructure."""

    def test_distributed_config(self):
        """Test distributed configuration."""
        config = DistributedConfig(
            mode=ParallelismMode.DATA,
            world_size=4,
            rank=0,
        )
        assert config.world_size == 4
        assert config.rank == 0
        assert config.mode == ParallelismMode.DATA

    def test_distributed_manager(self):
        """Test distributed manager (single process)."""
        config = DistributedConfig(world_size=1, rank=0)
        manager = DistributedManager(config)
        manager.initialize()

        assert manager.initialized
        assert manager.is_main_process()
        assert manager.device is not None

        manager.cleanup()

    def test_distributed_swarm_wrapper(self):
        """Test DistributedSwarm wrapper."""
        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 16))
        config = DistributedConfig(world_size=1, rank=0)
        manager = DistributedManager(config)
        manager.initialize()

        wrapped = DistributedSwarm(model, config, manager)

        x = torch.randn(4, 32)
        output = wrapped(x)

        assert output.shape == (4, 16)
        manager.cleanup()

    def test_model_parallel_swarm(self):
        """Test model parallel swarm."""
        agents = nn.ModuleDict({
            "0": nn.Linear(32, 64),
            "1": nn.Linear(32, 64),
        })
        device_map = {
            "0": torch.device("cpu"),
            "1": torch.device("cpu"),
        }

        swarm = ModelParallelSwarm(agents, device_map)

        inputs = {
            "0": torch.randn(4, 32),
            "1": torch.randn(4, 32),
        }

        outputs = swarm(inputs)

        assert "0" in outputs
        assert "1" in outputs
        assert outputs["0"].shape == (4, 64)

    def test_pipeline_parallel_swarm(self):
        """Test pipeline parallel swarm."""
        stages = [
            nn.Linear(32, 64),
            nn.Linear(64, 64),
            nn.Linear(64, 16),
        ]
        devices = [torch.device("cpu")] * 3

        swarm = PipelineParallelSwarm(stages, devices, micro_batch_size=2)

        x = torch.randn(8, 32)
        output = swarm(x)

        assert output.shape == (8, 16)


class TestGenetic:
    """Tests for genetic/evolutionary optimization."""

    def test_swarm_genome(self):
        """Test SwarmGenome creation and copy."""
        genome = SwarmGenome(input_dim=64, output_dim=32)

        genome.agents[0] = AgentGene(
            innovation_number=0,
            agent_id=0,
            agent_type=1,
        )

        genome.connections[0] = ConnectionGene(
            innovation_number=0,
            from_agent=0,
            to_agent=1,
            weight=0.5,
        )

        # Test copy
        copy = genome.copy()
        assert len(copy.agents) == 1
        assert len(copy.connections) == 1
        assert copy.agents[0].agent_type == 1

    def test_innovation_tracker(self):
        """Test innovation tracking."""
        tracker = InnovationTracker()

        inn1 = tracker.get_agent_innovation(0)
        inn2 = tracker.get_agent_innovation(1)
        inn3 = tracker.get_agent_innovation(0)  # Same agent

        assert inn1 == inn3  # Same agent, same innovation
        assert inn1 != inn2

    def test_swarm_mutator(self):
        """Test genome mutation."""
        config = EvolutionConfig(
            weight_mutation_rate=1.0,
            add_agent_rate=0.5,
            add_connection_rate=0.5,
        )
        tracker = InnovationTracker()
        mutator = SwarmMutator(config, tracker)

        genome = SwarmGenome()
        for i in range(5):
            genome.agents[i] = AgentGene(
                innovation_number=i,
                agent_id=i,
                agent_type=0,
            )

        mutated = mutator.mutate(genome)

        # Should have same or different number of agents/connections
        assert len(mutated.agents) >= 1

    def test_swarm_crossover(self):
        """Test genome crossover."""
        crossover = SwarmCrossover()

        parent1 = SwarmGenome()
        parent2 = SwarmGenome()

        for i in range(3):
            parent1.agents[i] = AgentGene(i, i, 0)
            parent2.agents[i] = AgentGene(i, i, 1)

        parent1.fitness = 0.8
        parent2.fitness = 0.6

        offspring = crossover.crossover(parent1, parent2)

        assert len(offspring.agents) > 0

    def test_species(self):
        """Test species management."""
        genome = SwarmGenome()
        genome.fitness = 0.5

        species = Species(species_id=0, representative=genome)

        genome2 = SwarmGenome()
        genome2.fitness = 0.7

        species.add_member(genome2)

        assert len(species.members) == 2
        assert species.best_fitness == 0.7

    def test_genome_distance(self):
        """Test distance computation between genomes."""
        config = EvolutionConfig()

        g1 = SwarmGenome()
        g2 = SwarmGenome()

        for i in range(5):
            g1.agents[i] = AgentGene(i, i, 0)

        for i in range(3):
            g2.agents[i] = AgentGene(i, i, 1)

        distance = genome_distance(g1, g2, config)

        assert distance >= 0

    def test_genetic_optimizer(self):
        """Test genetic optimizer."""
        config = EvolutionConfig(
            population_size=10,
            elite_size=2,
        )

        def fitness_fn(genome):
            return len(genome.agents) / 10.0

        optimizer = GeneticOptimizer(config, fitness_fn)

        # Create initial genome
        initial = SwarmGenome()
        for i in range(5):
            initial.agents[i] = AgentGene(i, i, 0)

        optimizer.initialize_population(initial)
        optimizer.evolve()

        assert optimizer.generation == 1
        assert optimizer.best_genome is not None

    def test_cmaes(self):
        """Test CMA-ES optimizer."""
        dim = 5
        cmaes = CMAES(dim=dim, sigma=0.5)

        # Simple sphere function
        def fitness_fn(x):
            return -sum(x ** 2)

        best, fitness = cmaes.optimize(fitness_fn, max_generations=20)

        assert best.shape == (dim,)
        assert cmaes.generation > 0


class TestNAS:
    """Tests for Neural Architecture Search."""

    def test_search_space(self):
        """Test search space configuration."""
        space = SearchSpace(
            operations=[OperationType.LINEAR, OperationType.SKIP],
            min_layers=2,
            max_layers=4,
        )

        assert len(space.operations) == 2
        assert space.min_layers == 2

    def test_operation(self):
        """Test individual operations."""
        ops = [
            OperationType.SKIP,
            OperationType.LINEAR,
            OperationType.RELU_LINEAR,
        ]

        for op_type in ops:
            op = Operation(op_type, in_dim=32, out_dim=64)
            x = torch.randn(4, 32)
            output = op(x)
            assert output.shape == (4, 64)

    def test_mixed_operation(self):
        """Test mixed operation for DARTS."""
        space = SearchSpace()
        mixed = MixedOperation(in_dim=32, out_dim=64, search_space=space)

        x = torch.randn(4, 32)
        output = mixed(x)

        assert output.shape == (4, 64)

        # Check architecture selection
        selected = mixed.get_selected_operation()
        assert isinstance(selected, OperationType)

    def test_darts_cell(self):
        """Test DARTS cell."""
        cell = DARTSCell(in_dim=32, out_dim=64, num_nodes=3)

        x = torch.randn(4, 32)
        output = cell(x)

        assert output.shape == (4, 64)

        # Get architecture
        arch = cell.get_architecture()
        assert len(arch) > 0

    def test_darts_agent(self):
        """Test DARTS agent."""
        agent = DARTSAgent(
            input_dim=32,
            hidden_dim=64,
            output_dim=16,
            num_cells=2,
        )

        x = torch.randn(4, 32)
        output = agent(x)

        assert output.shape == (4, 16)

        # Get architecture
        arch = agent.get_architecture()
        assert len(arch) == 2  # 2 cells

    def test_darts_searcher(self):
        """Test DARTS search step."""
        agent = DARTSAgent(input_dim=32, hidden_dim=32, output_dim=16, num_cells=1)
        searcher = DARTSSearcher(agent)

        train_data = (torch.randn(4, 32), torch.randn(4, 16))
        val_data = (torch.randn(4, 32), torch.randn(4, 16))

        metrics = searcher.search_step(train_data, val_data, nn.MSELoss())

        assert "arch_loss" in metrics
        assert "weight_loss" in metrics

    def test_enas_controller(self):
        """Test ENAS controller."""
        space = SearchSpace()
        controller = ENASController(space, num_layers=3, hidden_dim=32)

        ops, skips, log_prob = controller.forward()

        assert len(ops) == 3
        assert len(skips) == 3

    def test_enas_shared_network(self):
        """Test ENAS shared network."""
        space = SearchSpace()
        network = ENASSharedNetwork(
            input_dim=32,
            hidden_dim=64,
            output_dim=16,
            num_layers=3,
            search_space=space,
        )

        arch = {
            "operations": [OperationType.LINEAR] * 3,
            "skip_connections": [[], [False], [False, True]],
        }

        x = torch.randn(4, 32)
        output = network(x, arch)

        assert output.shape == (4, 16)

    def test_random_search_nas(self):
        """Test random search baseline."""
        space = SearchSpace()

        def build_fn(arch):
            return nn.Linear(32, 16)

        def eval_fn(model):
            return 0.5 + 0.1 * (hash(str(model)) % 10) / 10

        searcher = RandomSearchNAS(space, build_fn, eval_fn)
        best = searcher.search(num_samples=10)

        assert best is not None
        assert "operations" in best


class TestScaling:
    """Tests for scaling utilities."""

    def test_scaling_config(self):
        """Test scaling configuration."""
        config = ScalingConfig(
            num_levels=3,
            agents_per_group=10,
        )

        assert config.num_levels == 3
        assert config.agents_per_group == 10

    def test_agent_pool(self):
        """Test lazy agent pool."""
        def factory(agent_id):
            return nn.Linear(32, 64)

        pool = AgentPool(num_agents=100, agent_factory=factory, cache_size=10)

        # Get agents
        agent0 = pool.get(0)
        agent1 = pool.get(1)

        assert agent0 is not None
        assert len(pool.cache) == 2

        # Get same agent again
        agent0_again = pool.get(0)
        assert agent0 is agent0_again

    def test_agent_group(self):
        """Test agent group."""
        def factory(agent_id):
            return nn.Linear(32, 64)

        pool = AgentPool(10, factory, cache_size=10)
        group = AgentGroup(
            group_id=0,
            agent_ids=[0, 1, 2],
            agent_pool=pool,
            hidden_dim=64,
        )

        inputs = {
            0: torch.randn(1, 32),
            1: torch.randn(1, 32),
            2: torch.randn(1, 32),
        }

        outputs, repr = group(inputs)

        assert len(outputs) == 3
        assert repr.shape[-1] == 64

    def test_hierarchical_swarm(self):
        """Test hierarchical swarm."""
        def factory(agent_id):
            return nn.Linear(32, 64)

        config = ScalingConfig(
            num_levels=2,
            agents_per_group=5,
            groups_per_supergroup=5,
        )

        swarm = HierarchicalSwarm(
            num_agents=25,
            agent_factory=factory,
            config=config,
            hidden_dim=64,
        )

        inputs = {i: torch.randn(1, 32) for i in range(10)}
        outputs = swarm(inputs)

        assert len(outputs) > 0

        stats = swarm.get_statistics()
        assert "num_agents" in stats
        assert "num_groups" in stats

    def test_sparse_message_graph(self):
        """Test sparse message graph."""
        config = ScalingConfig(message_sparsity=0.1)
        positions = torch.rand(50, 2)

        graph = SparseMessageGraph(50, config, positions)

        neighbors = graph.get_neighbors(0)
        assert len(neighbors) > 0

        targets = graph.get_message_targets(0, stochastic=True)
        assert len(targets) > 0

    def test_checkpointed_swarm(self):
        """Test gradient checkpointing."""
        agents = nn.ModuleDict({
            str(i): nn.Linear(32, 64) for i in range(5)
        })

        swarm = CheckpointedSwarm(agents, checkpoint_every=2)

        inputs = {str(i): torch.randn(4, 32) for i in range(5)}
        messages = {str(i): torch.randn(4, 64) for i in range(5)}

        outputs = swarm(inputs, messages)

        assert len(outputs) == 5

    def test_batched_processor(self):
        """Test batched agent processing."""
        template = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
        )

        processor = BatchedAgentProcessor(
            agent_template=template,
            num_agents=10,
            batch_size=4,
            hidden_dim=64,
        )

        inputs = {i: torch.randn(1, 32) for i in range(10)}
        outputs = processor.process_all(inputs)

        assert len(outputs) == 10

    def test_memory_estimation(self):
        """Test memory estimation."""
        estimate = estimate_memory_usage(
            num_agents=100,
            hidden_dim=64,
            num_layers=2,
        )

        assert "total_params" in estimate
        assert "total_params_mb" in estimate
        assert estimate["total_params"] > 0


# Run with: pytest tests/test_phase6.py -v
