#!/usr/bin/env python3
"""
Training script for neuromorphic (spiking) swarm.

Explores:
1. Spiking neural network training with surrogate gradients
2. STDP-based unsupervised learning
3. Sparse-activity training objectives
4. Model-based SNN/ANN operation-energy estimates
5. Spiking swarm on environment tasks
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.neuromorphic import (
    # LIF
    LIFConfig,
    LIFLayer,
    SpikingNetwork,
    LiquidStateMachine,
    # STDP
    STDPConfig,
    STDPLayer,
    ClassicSTDP,
    RewardModulatedSTDP,
    # Energy
    HardwareModel,
    EnergyTracker,
    EnergyEfficientLoss,
    compare_energy_efficiency,
    # Swarm
    SpikingAgentConfig,
    SpikingSwarmConfig,
    SpikingSwarmGraph,
    SpikingSwarmTrainer,
    CommunicationMode,
)


def train_spiking_network(
    num_epochs: int = 100,
    batch_size: int = 32,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """
    Train a basic spiking neural network with surrogate gradients.

    Uses a simple classification task to demonstrate SNN training.
    """
    print("\n=== Training Spiking Neural Network ===")
    print(f"Device: {device}, Epochs: {num_epochs}")

    # Create network
    layer_sizes = [64, 128, 64, 10]
    lif_config = LIFConfig(
        tau_mem=20.0,
        threshold=1.0,
        surrogate_slope=25.0,
    )

    snn = SpikingNetwork(
        layer_sizes=layer_sizes,
        config=None,  # Will use default SNNConfig
        use_adaptive=True,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(snn.parameters(), lr=1e-3)

    # Energy tracking
    energy_tracker = EnergyTracker(hardware=HardwareModel.LOIHI)
    for i in range(len(layer_sizes) - 1):
        energy_tracker.register_layer(layer_sizes[i], layer_sizes[i + 1])

    # Training loop
    history = {"loss": [], "accuracy": [], "spike_rate": [], "energy_ratio": []}

    for epoch in range(num_epochs):
        # Generate random data
        x = torch.randn(batch_size, layer_sizes[0], device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)

        # Forward pass
        snn.reset()
        output = snn(x, num_steps=25)

        # Loss
        loss = F.cross_entropy(output, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            predictions = output.argmax(dim=-1)
            accuracy = (predictions == targets).float().mean().item()

            # Get spike trains for energy estimation
            snn.reset()
            all_spikes = snn(x, num_steps=25, return_all_layers=True)

            # Count spikes
            total_spikes = sum(s.sum().item() for s in all_spikes)
            total_neurons = sum(layer_sizes[1:]) * 25  # neurons * timesteps
            spike_rate = total_spikes / (total_neurons * batch_size)

            # Energy comparison
            ann_output = torch.randn(batch_size, layer_sizes[-1], device=device)
            energy_stats = compare_energy_efficiency(
                all_spikes[-1],  # Use last layer spikes
                ann_output,
                sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1)),
                num_timesteps=25,
                hardware=HardwareModel.LOIHI,
            )

        history["loss"].append(loss.item())
        history["accuracy"].append(accuracy)
        history["spike_rate"].append(spike_rate)
        history["energy_ratio"].append(energy_stats["efficiency_gain"])

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Loss: {loss.item():.4f}, Accuracy: {accuracy:.2%}")
            print(f"  Spike Rate: {spike_rate:.4f}, Energy Gain: {energy_stats['efficiency_gain']:.2f}x")

    return history


def train_stdp_unsupervised(
    num_iterations: int = 500,
    batch_size: int = 16,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """
    Unsupervised learning with STDP.

    Demonstrates weight evolution through spike-timing plasticity.
    """
    print("\n=== STDP Unsupervised Learning ===")
    print(f"Iterations: {num_iterations}")

    # Create STDP layer
    input_size = 64
    output_size = 32

    stdp_config = STDPConfig(
        lr_plus=0.01,
        lr_minus=0.012,  # Slightly higher for stability
        tau_plus=20.0,
        tau_minus=20.0,
        w_min=0.0,
        w_max=1.0,
    )

    lif_config = LIFConfig(tau_mem=20.0, threshold=1.0)

    # Input layer (rate-to-spike)
    input_layer = LIFLayer(input_size, output_size, lif_config).to(device)

    # STDP for plastic synapses
    stdp = ClassicSTDP(stdp_config)

    # Traces
    pre_trace = torch.zeros(batch_size, input_size, device=device)
    post_trace = torch.zeros(batch_size, output_size, device=device)
    trace_decay = 0.95

    # History
    history = {
        "weight_mean": [],
        "weight_std": [],
        "sparsity": [],
        "spike_rate": [],
    }

    for iteration in range(num_iterations):
        # Generate structured input (correlated patterns)
        pattern_id = iteration % 4
        base = torch.zeros(batch_size, input_size, device=device)

        # Create different patterns
        start_idx = pattern_id * (input_size // 4)
        end_idx = start_idx + input_size // 4
        base[:, start_idx:end_idx] = 1.0

        # Add noise
        noise = torch.rand_like(base) * 0.3
        input_current = base + noise

        # Reset layer
        input_layer.reset()

        # Simulate multiple timesteps
        num_steps = 25
        all_pre_spikes = []
        all_post_spikes = []

        for t in range(num_steps):
            # Convert to spikes (rate coding)
            pre_spikes = (torch.rand_like(input_current) < input_current * 0.3).float()
            post_spikes = input_layer(pre_spikes)

            all_pre_spikes.append(pre_spikes)
            all_post_spikes.append(post_spikes)

            # Update traces
            pre_trace = trace_decay * pre_trace + pre_spikes
            post_trace = trace_decay * post_trace + post_spikes

            # Apply STDP
            with torch.no_grad():
                new_weight, _ = stdp(
                    input_layer.weight,
                    pre_spikes,
                    post_spikes,
                    pre_trace,
                    post_trace,
                )
                input_layer.weight.data = new_weight

        # Compute statistics
        with torch.no_grad():
            weight = input_layer.weight.data
            history["weight_mean"].append(weight.mean().item())
            history["weight_std"].append(weight.std().item())
            history["sparsity"].append((weight < 0.1).float().mean().item())

            total_post_spikes = sum(s.sum().item() for s in all_post_spikes)
            spike_rate = total_post_spikes / (batch_size * output_size * num_steps)
            history["spike_rate"].append(spike_rate)

        if (iteration + 1) % 100 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"  Weight: mean={weight.mean():.4f}, std={weight.std():.4f}")
            print(f"  Sparsity: {history['sparsity'][-1]:.2%}, Spike Rate: {spike_rate:.4f}")

    return history


def train_reward_modulated_stdp(
    num_episodes: int = 200,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """
    Reinforcement learning with reward-modulated STDP.

    Demonstrates how reward signals can modulate STDP for goal-directed learning.
    """
    print("\n=== Reward-Modulated STDP ===")
    print(f"Episodes: {num_episodes}")

    # Setup
    input_size = 32
    hidden_size = 64
    output_size = 4  # 4 actions

    # Spiking layers
    lif_config = LIFConfig(tau_mem=20.0, threshold=1.0)
    hidden_layer = LIFLayer(input_size, hidden_size, lif_config).to(device)
    output_layer = LIFLayer(hidden_size, output_size, lif_config).to(device)

    # Reward-modulated STDP
    stdp_config = STDPConfig(lr_plus=0.01, lr_minus=0.01, reward_tau=100.0)
    rm_stdp_hidden = RewardModulatedSTDP(stdp_config)
    rm_stdp_output = RewardModulatedSTDP(stdp_config)

    # Traces
    trace_decay = 0.95
    pre_trace_h = torch.zeros(1, input_size, device=device)
    post_trace_h = torch.zeros(1, hidden_size, device=device)
    pre_trace_o = torch.zeros(1, hidden_size, device=device)
    post_trace_o = torch.zeros(1, output_size, device=device)

    history = {"reward": [], "correct_actions": []}

    for episode in range(num_episodes):
        # Reset
        hidden_layer.reset()
        output_layer.reset()
        rm_stdp_hidden.reset()
        rm_stdp_output.reset()

        pre_trace_h.zero_()
        post_trace_h.zero_()
        pre_trace_o.zero_()
        post_trace_o.zero_()

        episode_reward = 0.0
        correct_count = 0

        # Simple task: input pattern determines correct action
        num_trials = 20

        for trial in range(num_trials):
            # Generate input pattern
            pattern_id = trial % output_size
            input_pattern = torch.zeros(1, input_size, device=device)
            start = pattern_id * (input_size // output_size)
            end = start + input_size // output_size
            input_pattern[:, start:end] = 1.0
            input_pattern += torch.rand_like(input_pattern) * 0.2

            # Simulate timesteps
            action_votes = torch.zeros(1, output_size, device=device)

            for t in range(15):
                # Rate-to-spike
                pre_spikes = (torch.rand_like(input_pattern) < input_pattern * 0.4).float()

                # Hidden layer
                hidden_spikes = hidden_layer(pre_spikes)

                # Output layer
                output_spikes = output_layer(hidden_spikes)
                action_votes += output_spikes

                # Update traces
                pre_trace_h = trace_decay * pre_trace_h + pre_spikes
                post_trace_h = trace_decay * post_trace_h + hidden_spikes
                pre_trace_o = trace_decay * pre_trace_o + hidden_spikes
                post_trace_o = trace_decay * post_trace_o + output_spikes

                # Compute eligibility (STDP without reward)
                rm_stdp_hidden.compute_eligibility(
                    hidden_layer.weight,
                    pre_spikes, hidden_spikes,
                    pre_trace_h, post_trace_h,
                )
                rm_stdp_output.compute_eligibility(
                    output_layer.weight,
                    hidden_spikes, output_spikes,
                    pre_trace_o, post_trace_o,
                )

            # Select action
            action = action_votes.argmax(dim=-1).item()
            correct_action = pattern_id

            # Compute reward
            if action == correct_action:
                reward = 1.0
                correct_count += 1
            else:
                reward = -0.5

            episode_reward += reward

            # Apply reward to STDP
            with torch.no_grad():
                new_h, _ = rm_stdp_hidden.apply_reward(hidden_layer.weight, reward)
                hidden_layer.weight.data = new_h

                new_o, _ = rm_stdp_output.apply_reward(output_layer.weight, reward)
                output_layer.weight.data = new_o

        history["reward"].append(episode_reward)
        history["correct_actions"].append(correct_count / num_trials)

        if (episode + 1) % 40 == 0:
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f}, Accuracy: {correct_count/num_trials:.2%}")

    return history


def train_spiking_swarm(
    num_iterations: int = 300,
    num_agents: int = 10,
    device: str = "cpu",
) -> Dict[str, List[float]]:
    """
    Train a spiking swarm on a simple task.
    """
    print("\n=== Spiking Swarm Training ===")
    print(f"Agents: {num_agents}, Iterations: {num_iterations}")

    # Create spiking swarm
    agent_config = SpikingAgentConfig(
        input_dim=32,
        hidden_dim=64,
        output_dim=16,
        message_dim=16,
        num_timesteps=20,
        use_stdp=True,
        comm_mode=CommunicationMode.RATE,
    )

    swarm_config = SpikingSwarmConfig(
        num_agents=num_agents,
        agent_config=agent_config,
        num_rounds=2,
        track_energy=True,
        hardware=HardwareModel.LOIHI,
    )

    swarm = SpikingSwarmGraph(swarm_config).to(device)

    # Trainer
    trainer = SpikingSwarmTrainer(
        swarm,
        learning_rate=1e-3,
        energy_weight=0.01,
        target_spike_rate=0.1,
    )

    history = {"loss": [], "spike_rate": [], "energy_pj": [], "task_loss": []}

    for iteration in range(num_iterations):
        # Generate data
        batch_size = 16
        x = torch.randn(batch_size, agent_config.input_dim, device=device)
        targets = torch.randn(batch_size, agent_config.output_dim, device=device)

        # Train step
        metrics = trainer.train_step(x, targets, reward=None)

        history["loss"].append(metrics["loss"])
        history["task_loss"].append(metrics["task_loss"])
        history["spike_rate"].append(metrics["spike_rate"])

        # Energy
        energy = swarm.get_energy_estimate()
        history["energy_pj"].append(energy.get("total_energy_pj", 0))

        if (iteration + 1) % 60 == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"  Loss: {metrics['loss']:.4f}, Task Loss: {metrics['task_loss']:.4f}")
            print(f"  Spike Rate: {metrics['spike_rate']:.4f}, Sparsity: {metrics['sparsity']:.2%}")
            if energy:
                print(f"  Energy: {energy.get('total_energy_pj', 0):.2f} pJ")

    return history


def compare_snn_vs_ann(
    num_iterations: int = 100,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """
    Compare energy efficiency of SNN vs equivalent ANN.
    """
    print("\n=== SNN vs ANN Energy Comparison ===")

    layer_sizes = [64, 128, 64, 10]
    batch_size = 32
    num_timesteps = 25

    # SNN
    lif_config = LIFConfig(tau_mem=20.0, threshold=1.0)
    snn = SpikingNetwork(layer_sizes=layer_sizes).to(device)

    # Equivalent ANN
    ann = nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10),
    ).to(device)

    # Energy tracking
    snn_tracker = EnergyTracker(hardware=HardwareModel.LOIHI)
    ann_tracker = EnergyTracker(hardware=HardwareModel.GPU_FP32)

    for i in range(len(layer_sizes) - 1):
        snn_tracker.register_layer(layer_sizes[i], layer_sizes[i + 1])
        ann_tracker.register_layer(layer_sizes[i], layer_sizes[i + 1])

    # Run comparison
    snn_energies = []
    ann_energies = []
    snn_accuracies = []
    ann_accuracies = []

    for i in range(num_iterations):
        x = torch.randn(batch_size, 64, device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)

        # SNN forward
        snn.reset()
        snn_tracker.reset()
        snn_out = snn(x, num_steps=num_timesteps, return_all_layers=True)

        # Count SNN spikes
        for layer_idx, spikes in enumerate(snn_out):
            snn_tracker.count_spikes(spikes, layer_idx)

        snn_energy = snn_tracker.estimate_energy(num_timesteps)
        snn_pred = snn_out[-1].mean(dim=1).argmax(dim=-1)
        snn_acc = (snn_pred == targets).float().mean().item()

        # ANN forward
        ann_out = ann(x)
        ann_pred = ann_out.argmax(dim=-1)
        ann_acc = (ann_pred == targets).float().mean().item()

        # ANN energy (simplified: count MACs)
        total_params = sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1))
        ann_energy_pj = total_params * batch_size * 100  # 100 pJ per MAC

        snn_energies.append(snn_energy.total_energy_pj)
        ann_energies.append(ann_energy_pj)
        snn_accuracies.append(snn_acc)
        ann_accuracies.append(ann_acc)

    # Summary
    avg_snn_energy = sum(snn_energies) / len(snn_energies)
    avg_ann_energy = sum(ann_energies) / len(ann_energies)
    avg_snn_acc = sum(snn_accuracies) / len(snn_accuracies)
    avg_ann_acc = sum(ann_accuracies) / len(ann_accuracies)

    efficiency_gain = avg_ann_energy / avg_snn_energy if avg_snn_energy > 0 else float('inf')

    results = {
        "snn": {
            "avg_energy_pj": avg_snn_energy,
            "avg_accuracy": avg_snn_acc,
        },
        "ann": {
            "avg_energy_pj": avg_ann_energy,
            "avg_accuracy": avg_ann_acc,
        },
        "comparison": {
            "efficiency_gain": efficiency_gain,
            "snn_sparsity": 1.0 - (sum(snn_energies) / sum(ann_energies)),
        },
    }

    print(f"\nResults over {num_iterations} iterations:")
    print(f"  SNN Energy: {avg_snn_energy:.2f} pJ, Accuracy: {avg_snn_acc:.2%}")
    print(f"  ANN Energy: {avg_ann_energy:.2f} pJ, Accuracy: {avg_ann_acc:.2%}")
    print(f"  Efficiency Gain: {efficiency_gain:.2f}x")

    return results


def demo_liquid_state_machine(
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Demonstrate Liquid State Machine for temporal processing.
    """
    print("\n=== Liquid State Machine Demo ===")

    # Create LSM
    lsm = LiquidStateMachine(
        input_size=16,
        reservoir_size=128,
        output_size=4,
        spectral_radius=0.9,
        sparsity=0.1,
    ).to(device)

    # Generate temporal sequence
    batch_size = 8
    seq_length = 50

    # Create patterns with temporal structure
    patterns = []
    labels = []

    for i in range(100):
        pattern_type = i % 4
        seq = torch.zeros(batch_size, seq_length, 16, device=device)

        if pattern_type == 0:
            # Rising pattern
            for t in range(seq_length):
                seq[:, t, :4] = t / seq_length
        elif pattern_type == 1:
            # Falling pattern
            for t in range(seq_length):
                seq[:, t, 4:8] = 1 - t / seq_length
        elif pattern_type == 2:
            # Oscillating
            for t in range(seq_length):
                seq[:, t, 8:12] = 0.5 + 0.5 * torch.sin(torch.tensor(t * 0.5))
        else:
            # Random walk
            seq[:, 0, 12:16] = 0.5
            for t in range(1, seq_length):
                seq[:, t, 12:16] = seq[:, t-1, 12:16] + 0.1 * torch.randn(batch_size, 4, device=device)

        patterns.append(seq)
        labels.append(torch.full((batch_size,), pattern_type, device=device))

    # Train readout
    optimizer = torch.optim.Adam(lsm.readout.parameters(), lr=1e-3)

    for epoch in range(50):
        total_loss = 0
        correct = 0
        total = 0

        for seq, label in zip(patterns, labels):
            optimizer.zero_grad()

            output, reservoir_states = lsm(seq)
            loss = F.cross_entropy(output, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += batch_size

            lsm.reset()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/50: Loss={total_loss/len(patterns):.4f}, Acc={correct/total:.2%}")

    return {
        "final_accuracy": correct / total,
        "final_loss": total_loss / len(patterns),
    }


def main():
    parser = argparse.ArgumentParser(description="Train neuromorphic swarm")
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["snn", "stdp", "reward_stdp", "swarm", "compare", "lsm", "all"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    device = args.device

    if args.experiment in ["snn", "all"]:
        train_spiking_network(num_epochs=args.epochs, device=device)

    if args.experiment in ["stdp", "all"]:
        train_stdp_unsupervised(num_iterations=500, device=device)

    if args.experiment in ["reward_stdp", "all"]:
        train_reward_modulated_stdp(num_episodes=200, device=device)

    if args.experiment in ["swarm", "all"]:
        train_spiking_swarm(num_iterations=300, device=device)

    if args.experiment in ["compare", "all"]:
        compare_snn_vs_ann(num_iterations=100, device=device)

    if args.experiment in ["lsm", "all"]:
        demo_liquid_state_machine(device=device)

    print("\n=== Neuromorphic Training Complete ===")


if __name__ == "__main__":
    main()
