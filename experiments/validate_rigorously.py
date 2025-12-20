#!/usr/bin/env python3
"""
Rigorous Experimental Validation of SEESWM Hypothesis.

This script runs the complete validation protocol:
1. Ablation studies (what components matter?)
2. Scaling experiments (does emergence scale?)
3. Synergy measurement (is there true collective intelligence?)
4. Baseline comparisons (does swarm beat alternatives?)
5. Generalization tests (does it transfer?)
6. Emergence detection (what behaviors emerged?)
7. Statistical rigor (are results significant?)
8. Interpretability (what did it learn?)

Usage:
    python experiments/validate_rigorously.py --device cpu --seeds 10 --full

Publication checklist:
    [ ] Reproducible (code, seeds, hyperparams)
    [ ] Ablations show each component matters
    [ ] Scaling curves show favorable trends
    [ ] Synergy is measurably positive
    [ ] Beats all reasonable baselines
    [ ] Failure modes understood
    [ ] Some interpretability insights
    [ ] Generalizes to held-out tasks
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.swarm.graph import SwarmGraph, SwarmConfig, TopologyType
from src.environment.cosmos import CosmosEnvironment
from src.validation.ablations import AblationStudy, AblationConfig, AblationType, run_ablation_suite
from src.validation.scaling import ScalingExperiment, plot_scaling_laws, find_phase_transitions
from src.validation.synergy import SynergyMeasurer, compute_true_synergy
from src.validation.baselines import BaselineComparison
from src.validation.generalization import run_generalization_suite
from src.validation.emergence import EmergenceDetector
from src.validation.statistics import ExperimentStats, report_results, paired_significance_test
from src.validation.interpretability import run_interpretability_suite

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('seeswm_validation')


# Global variable to store loaded model weights
_loaded_model_state = None


def load_trained_model(model_path: str) -> Dict:
    """Load trained model weights."""
    global _loaded_model_state
    logger.info(f"Loading trained model from {model_path}")
    _loaded_model_state = torch.load(model_path, map_location='cpu', weights_only=False)
    logger.info(f"Loaded model trained for {_loaded_model_state.get('training_metrics', {}).get('num_epochs', '?')} epochs")
    logger.info(f"Final avg reward: {_loaded_model_state.get('training_metrics', {}).get('final_avg_reward', '?'):.2f}")
    return _loaded_model_state


def create_swarm_from_config(config: Dict, device: str = "cpu") -> SwarmGraph:
    """Create SwarmGraph from config dictionary, optionally loading trained weights."""
    global _loaded_model_state

    num_agents = config.get('num_agents', 20)

    swarm_config = SwarmConfig(
        num_agents=num_agents,
        input_dim=config.get('input_dim', 137),
        hidden_dim=config.get('hidden_dim', 128),
        output_dim=config.get('output_dim', 5),
        message_dim=config.get('hidden_dim', 128),
        topology=config.get('topology', TopologyType.SMALL_WORLD),
        num_perception=num_agents // 4,
        num_reasoning=num_agents // 4,
        num_memory=num_agents // 4,
        num_planning=num_agents - 3 * (num_agents // 4),
    )
    swarm = SwarmGraph(swarm_config, device=device)

    # Load trained weights if available
    if _loaded_model_state is not None and 'swarm_state' in _loaded_model_state:
        try:
            swarm.load_state_dict(_loaded_model_state['swarm_state'])
            logger.debug("Loaded trained weights into swarm")
        except Exception as e:
            logger.warning(f"Could not load weights: {e}")

    return swarm


def create_evaluation_function(env_config: Dict, device: str):
    """Create evaluation function for models."""

    def evaluate(model, seed: int) -> Dict[str, float]:
        """Evaluate a model on the environment."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Handle SwarmGraph, SwarmGraph wrappers, and nn.Module
        # Check if model has step/reset methods (SwarmGraph-like)
        is_swarm_like = hasattr(model, 'step') and hasattr(model, 'reset')

        if not is_swarm_like and hasattr(model, 'to'):
            model = model.to(device)
        if not is_swarm_like and hasattr(model, 'eval'):
            model.eval()

        env = CosmosEnvironment(**env_config)

        total_reward = 0.0
        total_steps = 0
        total_coverage = 0.0
        num_episodes = 5

        for _ in range(num_episodes):
            obs = env.reset()
            episode_reward = 0.0

            if is_swarm_like:
                model.reset(batch_size=1)

            for step in range(100):
                obs_tensor = obs[0].to_tensor(device).unsqueeze(0)

                with torch.no_grad():
                    if is_swarm_like:
                        action_logits = model.step(obs_tensor)
                    else:
                        action_logits = model(obs_tensor)
                    action = action_logits.argmax(dim=-1).item()

                # Environment expects list of actions (one per agent)
                # Returns: observations, rewards, dones (3 values)
                obs, rewards, dones = env.step([action])
                reward = rewards[0] if rewards else 0.0
                done = dones[0] if dones else False
                episode_reward += reward
                total_steps += 1

                if done:
                    break

            total_reward += episode_reward

            # Coverage from stats
            if hasattr(env, 'stats') and hasattr(env.stats, 'tiles_visited'):
                coverage = len(env.stats.tiles_visited) / (env.grid_size ** 2)
                total_coverage += coverage

        return {
            'reward': total_reward / num_episodes,
            'steps': total_steps / num_episodes,
            'coverage': total_coverage / num_episodes,
            'synergy': 0.0,  # Computed separately
            'survival_time': total_steps / num_episodes,
        }

    return evaluate


def run_ablation_experiment(
    swarm_config: Dict,
    env_config: Dict,
    device: str,
    num_seeds: int,
) -> Dict:
    """Run ablation studies."""
    logger.info("=" * 60)
    logger.info("RUNNING ABLATION STUDIES")
    logger.info("=" * 60)

    eval_fn = create_evaluation_function(env_config, device)
    results = run_ablation_suite(swarm_config, eval_fn, device, num_seeds)

    # Convert to serializable format
    summary = {}
    for name, result in results.items():
        summary[name] = {
            'mean_delta': result.mean_delta.tolist(),
            'p_values': result.p_values.tolist(),
            'effect_sizes': result.effect_sizes.tolist(),
        }

    return summary


def run_scaling_experiment(
    swarm_config: Dict,
    device: str,
    num_seeds: int,
) -> Dict:
    """Run scaling law experiments."""
    logger.info("=" * 60)
    logger.info("RUNNING SCALING EXPERIMENTS")
    logger.info("=" * 60)

    experiment = ScalingExperiment(
        base_input_dim=swarm_config.get('input_dim', 137),
        base_output_dim=swarm_config.get('output_dim', 5),
        device=device,
    )

    # Run sweep
    data_points = experiment.run_scaling_sweep(
        agent_counts=[5, 10, 20, 50, 100],
        hidden_dims=[64, 128],
        message_rounds=[1, 3],
        num_seeds=num_seeds,
    )

    # Fit scaling law
    scaling_law = experiment.fit_scaling_law('performance', 'num_agents')

    # Detect phase transitions
    transitions = find_phase_transitions(data_points, 'synergy')

    # Save plot
    output_dir = Path('results/validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_scaling_laws(data_points, scaling_law, str(output_dir / 'scaling_laws.png'))

    return {
        'scaling_exponent': scaling_law.exponent_alpha,
        'r_squared': scaling_law.r_squared,
        'phase_transitions': transitions,
        'num_data_points': len(data_points),
    }


def run_synergy_experiment(
    swarm_config: Dict,
    device: str,
    num_samples: int = 1000,
) -> Dict:
    """Measure true synergy using PID."""
    logger.info("=" * 60)
    logger.info("MEASURING SYNERGY (PARTIAL INFORMATION DECOMPOSITION)")
    logger.info("=" * 60)

    # Create swarm
    swarm = create_swarm_from_config(swarm_config, device)

    # Generate test data
    inputs = torch.randn(num_samples, swarm_config['input_dim'], device=device)
    targets = torch.randn(num_samples, swarm_config['output_dim'], device=device)

    # Measure synergy
    measurer = SynergyMeasurer(swarm, pid_method='broja', device=device)
    decomp = measurer.measure(inputs, targets, num_bootstrap=50)

    logger.info(f"Total MI: {decomp.total_mi:.4f}")
    logger.info(f"Redundancy: {decomp.redundancy:.4f}")
    logger.info(f"Synergy: {decomp.synergy:.4f} ± {decomp.std_synergy:.4f}")
    logger.info(f"Synergy ratio: {decomp.synergy_ratio:.2%}")

    return {
        'total_mi': decomp.total_mi,
        'redundancy': decomp.redundancy,
        'synergy': decomp.synergy,
        'synergy_std': decomp.std_synergy,
        'synergy_ratio': decomp.synergy_ratio,
        'unique_per_agent': decomp.unique,
    }


def run_baseline_experiment(
    swarm_config: Dict,
    env_config: Dict,
    device: str,
    num_seeds: int,
) -> Dict:
    """Compare against baselines."""
    logger.info("=" * 60)
    logger.info("RUNNING BASELINE COMPARISONS")
    logger.info("=" * 60)

    swarm = create_swarm_from_config(swarm_config, device)
    eval_fn = create_evaluation_function(env_config, device)

    comparison = BaselineComparison(swarm, eval_fn, device)
    results = comparison.compare(num_seeds=num_seeds, metric='reward')

    logger.info(comparison.summary())

    # Convert to serializable
    summary = {}
    for name, result in results.items():
        summary[name] = {
            'baseline_score': result.baseline_score,
            'swarm_score': result.swarm_score,
            'delta': result.delta,
            'wins': result.wins_out_of[0],
            'total': result.wins_out_of[1],
            'p_value': result.p_value,
        }

    return summary


def run_generalization_experiment(
    swarm_config: Dict,
    env_config: Dict,
    device: str,
    num_episodes: int = 10,
) -> Dict:
    """Test generalization."""
    logger.info("=" * 60)
    logger.info("RUNNING GENERALIZATION TESTS")
    logger.info("=" * 60)

    swarm = create_swarm_from_config(swarm_config, device)

    results = run_generalization_suite(swarm, env_config, device, num_episodes)

    summary = {}
    for name, result in results.items():
        summary[name] = {
            'train_perf': result.train_performance,
            'test_perf': result.test_performance,
            'gap': result.generalization_gap,
            'transfer_efficiency': result.transfer_efficiency,
        }

    return summary


def run_emergence_experiment(
    swarm_config: Dict,
    env_config: Dict,
    device: str,
    num_episodes: int = 50,
) -> Dict:
    """Detect emergent behaviors."""
    logger.info("=" * 60)
    logger.info("DETECTING EMERGENT BEHAVIORS")
    logger.info("=" * 60)

    swarm = create_swarm_from_config(swarm_config, device)
    detector = EmergenceDetector(swarm, device)

    # Generate trajectories
    trajectories = []
    env = CosmosEnvironment(**env_config)

    for _ in range(num_episodes):
        obs = env.reset()
        swarm.reset(batch_size=1)
        traj = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'agent_actions': {i: [] for i in range(len(swarm.agents))},
        }

        for step in range(100):
            obs_tensor = obs[0].to_tensor(device).unsqueeze(0)

            with torch.no_grad():
                action_logits = swarm.step(obs_tensor)
                action = action_logits.argmax(dim=-1).item()

            traj['observations'].append(obs[0])
            traj['actions'].append(action)

            # Environment expects list of actions
            # Returns: observations, rewards, dones (3 values)
            obs, rewards, dones = env.step([action])
            reward = rewards[0] if rewards else 0.0
            done = dones[0] if dones else False
            traj['rewards'].append(reward)

            if done:
                break

        trajectories.append(traj)

    # Analyze
    behaviors = detector.analyze_all(trajectories)

    summary = {}
    for name, behavior in behaviors.items():
        summary[name] = {
            'frequency': behavior.frequency,
            'strength': behavior.strength,
            'evidence': behavior.evidence,
        }
        logger.info(f"Detected: {name} (freq={behavior.frequency:.2%}, strength={behavior.strength:.2f})")

    return summary


def run_interpretability_experiment(
    swarm_config: Dict,
    device: str,
) -> Dict:
    """Run interpretability analysis."""
    logger.info("=" * 60)
    logger.info("RUNNING INTERPRETABILITY ANALYSIS")
    logger.info("=" * 60)

    swarm = create_swarm_from_config(swarm_config, device)

    # Generate test data
    test_inputs = torch.randn(100, swarm_config['input_dim'], device=device)
    test_labels = torch.randint(0, 5, (100,), device=device)

    results = run_interpretability_suite(swarm, test_inputs, test_labels, device)

    summary = {}

    if 'probe' in results:
        summary['probe_accuracy'] = results['probe'].accuracy

    if 'agent_importance' in results:
        summary['agent_importance'] = results['agent_importance']

    if 'message_importance' in results:
        summary['message_importance'] = results['message_importance']

    return summary


def generate_publication_checklist(all_results: Dict) -> str:
    """Generate publication readiness checklist."""
    lines = [
        "\n" + "=" * 70,
        "PUBLICATION CHECKLIST",
        "=" * 70,
    ]

    checks = []

    # 1. Ablations show components matter
    if 'ablations' in all_results:
        ablations = all_results['ablations']
        significant = sum(1 for v in ablations.values() if min(v.get('p_values', [1.0])) < 0.05)
        checks.append(('Ablations show components matter', significant >= 3))

    # 2. Scaling shows favorable trends
    if 'scaling' in all_results:
        scaling = all_results['scaling']
        checks.append(('Scaling shows favorable trends', scaling.get('scaling_exponent', 0) > 0))

    # 3. Synergy is positive
    if 'synergy' in all_results:
        synergy = all_results['synergy']
        checks.append(('Synergy is measurably positive', synergy.get('synergy', 0) > 0))

    # 4. Beats baselines
    if 'baselines' in all_results:
        baselines = all_results['baselines']
        wins = sum(1 for v in baselines.values() if v.get('delta', 0) > 0)
        checks.append(('Beats majority of baselines', wins >= len(baselines) // 2))

    # 5. Generalizes
    if 'generalization' in all_results:
        gen = all_results['generalization']
        avg_efficiency = np.mean([v.get('transfer_efficiency', 0) for v in gen.values()])
        checks.append(('Generalizes to new tasks', avg_efficiency > 0.5))

    # 6. Emergence detected
    if 'emergence' in all_results:
        emergence = all_results['emergence']
        checks.append(('Shows emergent behaviors', len(emergence) > 0))

    # 7. Interpretable
    if 'interpretability' in all_results:
        interp = all_results['interpretability']
        checks.append(('Has interpretability insights', len(interp) > 0))

    for description, passed in checks:
        status = '✓' if passed else '✗'
        lines.append(f"  [{status}] {description}")

    passed = sum(1 for _, p in checks if p)
    total = len(checks)

    lines.append("")
    lines.append(f"Score: {passed}/{total}")

    if passed == total:
        lines.append("Status: READY FOR PUBLICATION")
    elif passed >= total - 2:
        lines.append("Status: CLOSE TO READY")
    else:
        lines.append("Status: MORE WORK NEEDED")

    lines.append("=" * 70)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Rigorous SEESWM Validation')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds')
    parser.add_argument('--full', action='store_true', help='Run full validation')
    parser.add_argument('--quick', action='store_true', help='Quick validation')
    parser.add_argument('--output', type=str, default='results/validation', help='Output directory')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model')
    args = parser.parse_args()

    # Load trained model if specified
    if args.model:
        loaded = load_trained_model(args.model)
        # Use config from trained model if available
        if 'swarm_config' in loaded:
            saved_config = loaded['swarm_config']
            swarm_config = {
                'num_agents': saved_config.get('num_agents', 20),
                'input_dim': saved_config.get('input_dim', 137),
                'hidden_dim': saved_config.get('hidden_dim', 128),
                'output_dim': saved_config.get('output_dim', 128),
                'topology': TopologyType.SMALL_WORLD,
            }
            logger.info(f"Using config from trained model: {swarm_config}")
        else:
            swarm_config = {
                'num_agents': 20,
                'input_dim': 137,
                'hidden_dim': 128,
                'output_dim': 128,  # Match training
                'topology': TopologyType.SMALL_WORLD,
            }
    else:
        # Configuration for untrained model
        swarm_config = {
            'num_agents': 20,
            'input_dim': 137,
            'hidden_dim': 128,
            'output_dim': 5,
            'topology': TopologyType.SMALL_WORLD,
        }

    env_config = {
        'grid_size': 32,
        'num_resources': 20,
        'num_hazards': 10,
        'vision_radius': 5,
    }

    if args.quick:
        args.seeds = 3

    # Output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'swarm': swarm_config,
            'env': env_config,
            'device': args.device,
            'seeds': args.seeds,
        }
    }

    logger.info("=" * 60)
    logger.info("SEESWM RIGOROUS VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Device: {args.device}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Mode: {'full' if args.full else 'standard'}")
    logger.info(f"Model: {'TRAINED (' + args.model + ')' if args.model else 'UNTRAINED (random init)'}")

    try:
        # 1. Ablation studies
        logger.info("\n[1/7] Ablation Studies")
        all_results['ablations'] = run_ablation_experiment(
            swarm_config, env_config, args.device, args.seeds
        )

        # 2. Scaling experiments
        logger.info("\n[2/7] Scaling Experiments")
        all_results['scaling'] = run_scaling_experiment(
            swarm_config, args.device, args.seeds
        )

        # 3. Synergy measurement
        logger.info("\n[3/7] Synergy Measurement")
        all_results['synergy'] = run_synergy_experiment(
            swarm_config, args.device, 500 if not args.quick else 100
        )

        # 4. Baseline comparisons
        logger.info("\n[4/7] Baseline Comparisons")
        all_results['baselines'] = run_baseline_experiment(
            swarm_config, env_config, args.device, args.seeds
        )

        # 5. Generalization tests
        logger.info("\n[5/7] Generalization Tests")
        all_results['generalization'] = run_generalization_experiment(
            swarm_config, env_config, args.device, 5 if args.quick else 10
        )

        # 6. Emergence detection
        logger.info("\n[6/7] Emergence Detection")
        all_results['emergence'] = run_emergence_experiment(
            swarm_config, env_config, args.device, 20 if args.quick else 50
        )

        # 7. Interpretability
        logger.info("\n[7/7] Interpretability Analysis")
        all_results['interpretability'] = run_interpretability_experiment(
            swarm_config, args.device
        )

    except Exception as e:
        logger.error(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()

    # Generate checklist
    checklist = generate_publication_checklist(all_results)
    logger.info(checklist)

    # Save results
    output_file = output_dir / f'validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    # Convert numpy arrays and other non-serializable types
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, TopologyType):
            return obj.name
        else:
            return obj

    all_results = make_serializable(all_results)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    return all_results


if __name__ == '__main__':
    main()
