#!/usr/bin/env python3
"""Evaluate trained COBRA navigation policy (Jiang et al., 2024).

Usage:
    python -m jiang2024.evaluate_navigation --checkpoint model/cobra_navigation/best.pt
    python -m jiang2024.evaluate_navigation --checkpoint model/cobra_navigation/best.pt --task maze
"""

import argparse
import numpy as np

from src.configs.base import resolve_device
from jiang2024.configs_jiang2024 import CobraEnvConfig, CobraMazeEnvConfig
from jiang2024.env_jiang2024 import CobraNavigationEnv, CobraMazeEnv
from src.trainers.ddpg import DDPGTrainer


def main():
    parser = argparse.ArgumentParser(description="Evaluate COBRA navigation policy")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument(
        "--task", type=str, default="waypoint", choices=["waypoint", "maze"],
        help="Navigation task",
    )
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of eval episodes")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    args = parser.parse_args()
    device = resolve_device(args.device)

    # Build environment
    if args.task == "waypoint":
        env_config = CobraEnvConfig(device=device)
        env = CobraNavigationEnv(config=env_config, device=device)
    else:
        env_config = CobraMazeEnvConfig(device=device)
        env = CobraMazeEnv(config=env_config, device=device)

    # Load trainer and checkpoint
    from jiang2024.configs_jiang2024 import CobraNavigationConfig, CobraNetworkConfig

    config = CobraNavigationConfig(device=device)
    trainer = DDPGTrainer(
        env=env,
        config=config,
        network_config=config.network,
        device=device,
    )
    trainer.load_checkpoint(args.checkpoint)

    # Evaluate
    print(f"Evaluating {args.task} task ({args.num_episodes} episodes)...")
    metrics = trainer.evaluate(num_episodes=args.num_episodes)

    print(f"\nResults:")
    print(f"  Mean reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    print(f"  Mean length: {metrics['mean_length']:.1f}")
    print(f"  Min reward:  {metrics['min_reward']:.2f}")
    print(f"  Max reward:  {metrics['max_reward']:.2f}")

    env.close()


if __name__ == "__main__":
    main()
