#!/usr/bin/env python
"""Train the approach skill using PPO."""

import argparse
import torch
from pathlib import Path

from snake_hrl.envs import ApproachEnv
from snake_hrl.trainers import PPOTrainer
from snake_hrl.configs.env import ApproachEnvConfig
from snake_hrl.configs.training import PPOConfig
from snake_hrl.configs.network import NetworkConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train approach skill")
    parser.add_argument(
        "--total-frames",
        type=int,
        default=500_000,
        help="Total training frames",
    )
    parser.add_argument(
        "--frames-per-batch",
        type=int,
        default=4096,
        help="Frames per training batch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
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
        default="approach_skill",
        help="Experiment name",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Training Approach Skill")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Total frames: {args.total_frames:,}")
    print(f"Frames per batch: {args.frames_per_batch}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    # Set random seeds
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Create environment
    env_config = ApproachEnvConfig(
        max_episode_steps=500,
        use_reward_shaping=True,
    )
    env = ApproachEnv(config=env_config, device=args.device)

    # Create training config
    train_config = PPOConfig(
        total_frames=args.total_frames,
        frames_per_batch=args.frames_per_batch,
        learning_rate=args.lr,
        num_epochs=10,
        mini_batch_size=256,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        experiment_name=args.experiment_name,
        seed=args.seed,
    )

    # Create network config
    network_config = NetworkConfig()
    network_config.actor.hidden_dims = [256, 256, 128]
    network_config.critic.hidden_dims = [256, 256, 128]

    # Create trainer
    trainer = PPOTrainer(
        env=env,
        config=train_config,
        network_config=network_config,
        device=args.device,
    )

    # Train
    results = trainer.train()

    # Print final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total frames: {results['total_frames']:,}")
    print(f"Total episodes: {results['total_episodes']}")
    print(f"Best reward: {results['best_reward']:.2f}")

    # Final evaluation
    print("\nRunning evaluation...")
    eval_results = trainer.evaluate(num_episodes=20)
    print(f"Evaluation - Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"Evaluation - Mean length: {eval_results['mean_length']:.0f}")

    return results


if __name__ == "__main__":
    main()
