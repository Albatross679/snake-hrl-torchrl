#!/usr/bin/env python
"""Train hierarchical RL policy for full snake predation task."""

import argparse
import sys
import torch
from pathlib import Path

from snake_hrl.trainers import HRLTrainer
from snake_hrl.configs.training import HRLConfig, PPOConfig
from snake_hrl.configs.network import HRLNetworkConfig, NetworkConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train hierarchical RL policy")
    parser.add_argument(
        "--strategy",
        type=str,
        default="sequential",
        choices=["sequential", "joint", "pretrain_skills"],
        help="Training strategy",
    )
    parser.add_argument(
        "--approach-frames",
        type=int,
        default=500_000,
        help="Frames for approach skill training",
    )
    parser.add_argument(
        "--coil-frames",
        type=int,
        default=500_000,
        help="Frames for coil skill training",
    )
    parser.add_argument(
        "--manager-frames",
        type=int,
        default=500_000,
        help="Frames for manager training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Log directory",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="snake_hrl",
        help="Experiment name",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate resume checkpoint path if provided
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"Error: Resume checkpoint not found: {args.resume}")
            sys.exit(1)
        if not resume_path.is_file():
            print(f"Error: Resume path is not a file: {args.resume}")
            sys.exit(1)

    print("=" * 60)
    print("Training Hierarchical RL Policy for Snake Predation")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Device: {args.device}")
    print(f"Approach frames: {args.approach_frames:,}")
    print(f"Coil frames: {args.coil_frames:,}")
    print(f"Manager frames: {args.manager_frames:,}")
    print("=" * 60)

    # Set random seeds
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Create skill training configs
    approach_config = PPOConfig(
        total_frames=args.approach_frames,
        frames_per_batch=4096,
        learning_rate=3e-4,
        experiment_name=f"{args.experiment_name}_approach",
        log_dir=args.log_dir,
        save_dir=args.save_dir,
    )

    coil_config = PPOConfig(
        total_frames=args.coil_frames,
        frames_per_batch=4096,
        learning_rate=3e-4,
        experiment_name=f"{args.experiment_name}_coil",
        log_dir=args.log_dir,
        save_dir=args.save_dir,
    )

    manager_config = PPOConfig(
        total_frames=args.manager_frames,
        frames_per_batch=2048,
        learning_rate=1e-4,
        entropy_coef=0.05,  # Higher entropy for discrete skill selection
        experiment_name=f"{args.experiment_name}_manager",
        log_dir=args.log_dir,
        save_dir=args.save_dir,
    )

    # Create HRL config
    hrl_config = HRLConfig(
        approach_config=approach_config,
        coil_config=coil_config,
        manager_config=manager_config,
        training_strategy=args.strategy,
        approach_frames=args.approach_frames,
        coil_frames=args.coil_frames,
        manager_frames=args.manager_frames,
        use_curriculum=True,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
    )

    # Create network configs
    worker_config = NetworkConfig()
    worker_config.actor.hidden_dims = [256, 256, 128]
    worker_config.critic.hidden_dims = [256, 256, 128]

    manager_network_config = NetworkConfig()
    manager_network_config.actor.hidden_dims = [128, 128]
    manager_network_config.critic.hidden_dims = [128, 128]

    network_config = HRLNetworkConfig(
        manager=manager_network_config,
        worker_approach=worker_config,
        worker_coil=worker_config,
    )

    # Create trainer
    trainer = HRLTrainer(
        config=hrl_config,
        network_config=network_config,
        device=args.device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    results = trainer.train()

    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)

    if "approach" in results:
        print(f"Approach skill - Best reward: {results['approach']['best_reward']:.2f}")
    if "coil" in results:
        print(f"Coil skill - Best reward: {results['coil']['best_reward']:.2f}")

    # Final evaluation
    print("\nRunning final evaluation...")
    eval_results = trainer.evaluate(num_episodes=20)
    print(f"Evaluation - Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Evaluation - Success rate: {eval_results['success_rate'] * 100:.1f}%")

    return results


if __name__ == "__main__":
    main()
