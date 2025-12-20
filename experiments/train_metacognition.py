"""
Meta-cognition training for SEESWM.

Phase 4: Self-model with uncertainty, calibration, and meta-learning.
Trains agents to know what they know and don't know.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from src.swarm.specialized_graph import SpecializedSwarmGraph, SpecializedSwarmConfig
from src.swarm.graph import TopologyType
from src.self_model import (
    MetaCognitiveSwarm,
    MetaCognitionTrainer,
    UncertaintyEstimate,
    TemperatureScaling,
    CalibrationTracker,
    compute_calibration_metrics,
    MAML,
    MetaLearningConfig,
    Task,
    TheoryOfMind,
)
from src.environment.cosmos import CosmosEnvironment, EnvironmentConfig
from src.world_model.jepa import WorldModel
from src.training import PPOConfig
from src.utils.logging import setup_logger, MetricsLogger


def create_classification_task(
    input_dim: int,
    num_classes: int,
    num_examples: int,
    device: str = "cpu",
) -> Task:
    """Create a random classification task for meta-learning."""
    # Random linear classifier
    weights = torch.randn(num_classes, input_dim, device=device)

    # Generate examples
    support_x = torch.randn(num_examples, input_dim, device=device)
    support_y = (support_x @ weights.T).argmax(dim=-1)

    query_x = torch.randn(num_examples, input_dim, device=device)
    query_y = (query_x @ weights.T).argmax(dim=-1)

    return Task(
        support_x=support_x,
        support_y=support_y,
        query_x=query_x,
        query_y=query_y,
    )


def train_metacognition(
    num_iterations: int = 500,
    device: str = "cpu",
    log_dir: str = "logs/metacognition",
    seed: int = 42,
):
    """
    Train meta-cognitive abilities.

    Includes:
    - Uncertainty estimation
    - Calibration
    - Meta-learning for fast adaptation
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_dir) / f"metacog_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("seeswm_metacog", log_dir=str(run_dir))
    metrics = MetricsLogger(str(run_dir))

    logger.info("=" * 60)
    logger.info("SEESWM Phase 4: Meta-Cognition Training")
    logger.info("=" * 60)

    # Create specialized swarm
    swarm_config = SpecializedSwarmConfig(
        num_agents=16,
        num_perception=4,
        num_reasoning=4,
        num_memory=4,
        num_planning=4,
        topology=TopologyType.HIERARCHICAL,
        hidden_dim=64,
        input_dim=64,
        output_dim=32,
    )
    swarm = SpecializedSwarmGraph(config=swarm_config, device=device)
    logger.info(f"Swarm: {swarm_config.num_agents} agents, {swarm.total_parameters:,} params")

    # Wrap with meta-cognitive abilities
    metacog_swarm = MetaCognitiveSwarm(
        swarm=swarm,
        num_domains=10,
        enable_tom=True,
        enable_meta_learning=True,
        device=device,
    )
    logger.info("Meta-cognitive swarm initialized")
    logger.info(f"  - Theory of Mind: {metacog_swarm.enable_tom}")
    logger.info(f"  - Meta-learning: {metacog_swarm.enable_meta_learning}")

    # Trainer
    trainer = MetaCognitionTrainer(metacog_swarm, learning_rate=1e-4)

    # Training loop
    logger.info("\nStarting training...")

    for iteration in tqdm(range(num_iterations), desc="Training"):
        # Create random classification task
        inputs = torch.randn(32, swarm_config.input_dim, device=device)
        targets = torch.randint(0, 10, (32,), device=device)

        # Forward pass with uncertainty
        output, meta_info = metacog_swarm(inputs, return_uncertainty=True)

        # Train uncertainty estimation
        uncertainty_loss = trainer.train_uncertainty(inputs, targets)

        # Check if should abstain
        should_abstain, reasons = metacog_swarm.should_abstain()

        # Log periodically
        if iteration % 50 == 0:
            summary = metacog_swarm.get_swarm_metacog_summary()
            logger.info(
                f"Iter {iteration}: "
                f"uncertainty_loss={uncertainty_loss:.4f}, "
                f"collective_uncertainty={summary['collective_uncertainty']:.3f}, "
                f"abstain={should_abstain}"
            )

            metrics.log(
                iteration,
                uncertainty_loss=uncertainty_loss,
                collective_uncertainty=summary["collective_uncertainty"],
                calibration_error=summary["avg_calibration_error"],
            )

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    final_summary = metacog_swarm.get_swarm_metacog_summary()
    logger.info(f"Final collective uncertainty: {final_summary['collective_uncertainty']:.4f}")
    logger.info(f"Final calibration error: {final_summary['avg_calibration_error']:.4f}")

    # Save
    torch.save({
        "swarm": swarm.state_dict(),
        "final_summary": final_summary,
    }, run_dir / "final_model.pt")

    metrics.save()
    logger.info(f"\nResults saved to: {run_dir}")

    return metacog_swarm


def train_meta_learning(
    num_iterations: int = 500,
    tasks_per_batch: int = 4,
    device: str = "cpu",
    log_dir: str = "logs/meta_learning",
    seed: int = 42,
):
    """
    Train meta-learning (MAML) for fast adaptation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(log_dir) / f"maml_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("seeswm_maml", log_dir=str(run_dir))

    logger.info("=" * 60)
    logger.info("SEESWM Phase 4: Meta-Learning (MAML) Training")
    logger.info("=" * 60)

    # Simple task network
    input_dim = 64
    num_classes = 5

    task_network = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    ).to(device)

    config = MetaLearningConfig(
        inner_lr=0.01,
        inner_steps=5,
        outer_lr=0.001,
        tasks_per_batch=tasks_per_batch,
    )

    maml = MAML(task_network, config)
    logger.info(f"MAML initialized: inner_lr={config.inner_lr}, inner_steps={config.inner_steps}")

    # Training
    logger.info("\nStarting meta-training...")

    for iteration in tqdm(range(num_iterations), desc="Meta-Training"):
        # Sample tasks
        tasks = [
            create_classification_task(input_dim, num_classes, 10, device)
            for _ in range(tasks_per_batch)
        ]

        # Meta-train step
        stats = maml.meta_train_step(tasks)

        if iteration % 50 == 0:
            logger.info(
                f"Iter {iteration}: "
                f"query_loss={stats['query_loss']:.4f}, "
                f"query_acc={stats['query_accuracy']:.3f}"
            )

    # Evaluate adaptation speed
    logger.info("\nEvaluating adaptation speed...")

    adaptation_results = []
    for num_steps in [1, 3, 5, 10]:
        test_task = create_classification_task(input_dim, num_classes, 20, device)
        maml.config.inner_steps = num_steps
        result = maml.evaluate(test_task)
        adaptation_results.append((num_steps, result["accuracy"]))
        logger.info(f"  {num_steps} steps: accuracy={result['accuracy']:.3f}")

    # Save
    torch.save({
        "task_network": task_network.state_dict(),
        "adaptation_results": adaptation_results,
    }, run_dir / "final_model.pt")

    logger.info(f"\nResults saved to: {run_dir}")

    return maml


def demo_theory_of_mind(
    num_agents: int = 10,
    num_steps: int = 100,
    device: str = "cpu",
):
    """
    Demonstrate Theory of Mind capabilities.
    """
    print("=" * 60)
    print("Theory of Mind Demo")
    print("=" * 60)

    obs_dim = 32
    action_dim = 5

    tom = TheoryOfMind(
        observation_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )

    # Simulate agents
    print(f"\nSimulating {num_agents} agents for {num_steps} steps...")

    for step in range(num_steps):
        for agent_id in range(num_agents):
            # Random observation
            obs = torch.randn(obs_dim, device=device)

            # Update belief
            tom.update_belief(agent_id, obs, timestamp=step)

            # Random action
            action = np.random.randint(action_dim)
            tom.update_intent(agent_id, action)

    # Print summaries
    print("\nAgent Models:")
    for agent_id in range(min(5, num_agents)):
        summary = tom.get_agent_summary(agent_id)
        print(f"  Agent {agent_id}: intent={summary['predicted_intent']}, "
              f"confidence={summary['intent_confidence']:.2f}")

    # Most trusted agents
    print("\nMost Trusted Agents:")
    trusted = tom.get_most_trusted(5)
    for agent_id, trust in trusted:
        print(f"  Agent {agent_id}: trust={trust:.3f}")


def compare_with_without_metacognition(
    num_iterations: int = 200,
    device: str = "cpu",
):
    """
    Compare performance with and without meta-cognitive abilities.
    """
    print("=" * 60)
    print("Comparison: With vs Without Meta-Cognition")
    print("=" * 60)

    results = {}

    for use_metacog in [False, True]:
        label = "with_metacog" if use_metacog else "without_metacog"
        print(f"\nTraining {label}...")

        # Create swarm
        swarm_config = SpecializedSwarmConfig(
            num_agents=12,
            num_perception=3,
            num_reasoning=3,
            num_memory=3,
            num_planning=3,
            hidden_dim=32,
            input_dim=32,
            output_dim=16,
        )
        swarm = SpecializedSwarmGraph(config=swarm_config, device=device)

        if use_metacog:
            model = MetaCognitiveSwarm(swarm, enable_tom=True, device=device)
        else:
            model = swarm

        # Simple training loop
        optimizer = torch.optim.Adam(
            model.parameters() if use_metacog else swarm.parameters(),
            lr=1e-3,
        )

        losses = []
        abstention_rates = []

        for i in tqdm(range(num_iterations), desc=label):
            inputs = torch.randn(16, 32, device=device)
            targets = torch.randn(16, 16, device=device)

            optimizer.zero_grad()

            if use_metacog:
                output, meta_info = model(inputs)
                should_abstain, _ = model.should_abstain()
                abstention_rates.append(float(should_abstain))
            else:
                output = model.step(inputs)

            loss = F.mse_loss(output, targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        results[label] = {
            "final_loss": np.mean(losses[-50:]),
            "losses": losses,
        }

        if use_metacog:
            results[label]["avg_abstention"] = np.mean(abstention_rates[-50:])
            results[label]["final_summary"] = model.get_swarm_metacog_summary()

    # Print comparison
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    for label, data in results.items():
        print(f"\n{label}:")
        print(f"  Final loss: {data['final_loss']:.4f}")
        if "avg_abstention" in data:
            print(f"  Avg abstention rate: {data['avg_abstention']:.3f}")
            print(f"  Collective uncertainty: {data['final_summary']['collective_uncertainty']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 4: Meta-Cognition Training")
    parser.add_argument("--mode", type=str, default="metacog",
                       choices=["metacog", "maml", "tom", "compare"],
                       help="Training mode")
    parser.add_argument("--iterations", type=int, default=500, help="Training iterations")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--log-dir", type=str, default="logs/metacognition", help="Log dir")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    if args.mode == "metacog":
        train_metacognition(
            num_iterations=args.iterations,
            device=args.device,
            log_dir=args.log_dir,
            seed=args.seed,
        )
    elif args.mode == "maml":
        train_meta_learning(
            num_iterations=args.iterations,
            device=args.device,
            log_dir=args.log_dir,
            seed=args.seed,
        )
    elif args.mode == "tom":
        demo_theory_of_mind(device=args.device)
    elif args.mode == "compare":
        compare_with_without_metacognition(
            num_iterations=args.iterations,
            device=args.device,
        )


if __name__ == "__main__":
    main()
