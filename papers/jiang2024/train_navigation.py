#!/usr/bin/env python3
"""Train COBRA snake robot navigation (Jiang et al., 2024).

Usage:
    python -m jiang2024.train_navigation --task waypoint
    python -m jiang2024.train_navigation --task maze --total-frames 1000000
"""

import argparse

from jiang2024.configs_jiang2024 import (
    CobraNavigationConfig,
    CobraEnvConfig,
    CobraMazeEnvConfig,
)
from jiang2024.env_jiang2024 import CobraNavigationEnv, CobraMazeEnv
from src.configs import setup_run_dir, ConsoleLogger
from src.configs.base import resolve_device
from src.trainers.ddpg import DDPGTrainer


def main():
    parser = argparse.ArgumentParser(description="Train COBRA navigation")
    parser.add_argument(
        "--task", type=str, default="waypoint", choices=["waypoint", "maze"],
        help="Navigation task (default: waypoint)",
    )
    parser.add_argument("--total-frames", type=int, default=None, help="Override total frames")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    args = parser.parse_args()
    device = resolve_device(args.device)

    # Build config
    config = CobraNavigationConfig(
        seed=args.seed,
        device=device,
        name=f"cobra_{args.task}",
    )

    if args.total_frames is not None:
        config.total_frames = args.total_frames

    # Build environment
    if args.task == "waypoint":
        env_config = CobraEnvConfig(device=device)
        env = CobraNavigationEnv(config=env_config, device=device)
    else:
        env_config = CobraMazeEnvConfig(device=device)
        env = CobraMazeEnv(config=env_config, device=device)

    config.env = env_config

    # Setup run directory
    run_dir = setup_run_dir(config)

    with ConsoleLogger(run_dir, config.console):
        print(f"Training COBRA navigation ({args.task} task)")
        print(f"  Run directory: {run_dir}")
        print(f"  Total frames: {config.total_frames:,}")
        print(f"  Device: {config.device}")

        # Train
        trainer = DDPGTrainer(
            env=env,
            config=config,
            network_config=config.network,
            device=device,
            run_dir=run_dir,
        )

        results = trainer.train()

        print(f"\nTraining complete!")
        print(f"  Total frames: {results['total_frames']:,}")
        print(f"  Total episodes: {results['total_episodes']}")
        print(f"  Best reward: {results['best_reward']:.2f}")

    env.close()


if __name__ == "__main__":
    main()
