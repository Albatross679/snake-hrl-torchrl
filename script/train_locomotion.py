#!/usr/bin/env python
"""Training script for planar snake locomotion (Bing et al., IJCAI 2019).

Usage:
    python script/train_locomotion.py --task power_velocity --seed 1
    python script/train_locomotion.py --task target_tracking --track wave --seed 1
    python script/train_locomotion.py --task power_velocity --total-frames 50000 --seed 1
"""

import argparse
import sys

from locomotion.configs import (
    LocomotionEnvConfig,
    LocomotionNetworkConfig,
    LocomotionPPOConfig,
    LocomotionPhysicsConfig,
)
from locomotion import PlanarSnakeEnv
from trainers.ppo import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train planar snake locomotion")
    parser.add_argument(
        "--task", type=str, default="power_velocity",
        choices=["power_velocity", "target_tracking"],
        help="Task type (default: power_velocity)",
    )
    parser.add_argument(
        "--track", type=str, default="line",
        choices=["line", "wave", "zigzag", "circle", "random"],
        help="Track type for target_tracking task (default: line)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-frames", type=int, default=None, help="Override total frames")
    parser.add_argument("--target-v", type=float, default=None, help="Override target velocity")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Checkpoint dir")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Log directory")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build configs
    physics_config = LocomotionPhysicsConfig()
    env_config = LocomotionEnvConfig(
        physics=physics_config,
        task=args.task,
        track_type=args.track,
        device=args.device,
    )
    if args.target_v is not None:
        env_config.target_v = args.target_v

    ppo_config = LocomotionPPOConfig(
        seed=args.seed,
        device=args.device,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        experiment_name=f"locomotion_{args.task}",
    )
    if args.total_frames is not None:
        ppo_config.total_frames = args.total_frames

    network_config = LocomotionNetworkConfig(device=args.device)

    # Create environment
    env = PlanarSnakeEnv(config=env_config, device=args.device)

    # Create trainer
    trainer = PPOTrainer(
        env=env,
        config=ppo_config,
        network_config=network_config,
        device=args.device,
    )

    # Train
    print(f"Training locomotion: task={args.task}, seed={args.seed}")
    print(f"  Total frames: {ppo_config.total_frames}")
    print(f"  Network: 2x64 MLP (tanh)")
    print(f"  PPO: clip={ppo_config.clip_epsilon}, entropy={ppo_config.entropy_coef}")

    stats = trainer.train()

    print(f"\nTraining complete!")
    print(f"  Total frames: {stats.get('total_frames', 'N/A')}")
    print(f"  Best reward: {stats.get('best_reward', 'N/A'):.4f}")

    # Save final checkpoint
    trainer.save_checkpoint("final")
    print(f"  Saved checkpoint to {args.save_dir}/final.pt")


if __name__ == "__main__":
    main()
