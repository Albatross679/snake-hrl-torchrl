"""Evaluation script for snake locomotion.

Usage:
    python -m locomotion.evaluate --gait forward --checkpoint model/best.pt
"""

import argparse

from locomotion.config import GaitType, LocomotionConfig
from locomotion.env import LocomotionEnv
from src.trainers.ppo import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate snake locomotion")
    parser.add_argument(
        "--gait",
        type=str,
        default="forward",
        choices=[g.value for g in GaitType],
        help="Locomotion gait type",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--num-episodes", type=int, default=10, help="Eval episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    config = LocomotionConfig(seed=args.seed, device=args.device)
    config.env.gait = GaitType(args.gait)
    config.env.device = args.device

    env = LocomotionEnv(config.env, device=args.device)

    trainer = PPOTrainer(
        env=env,
        config=config,
        network_config=config.network,
        device=args.device,
    )
    trainer.load_checkpoint(args.checkpoint)

    results = trainer.evaluate(num_episodes=args.num_episodes, deterministic=True)
    print(f"Evaluation ({args.num_episodes} episodes, gait={args.gait}):")
    print(f"  Mean reward: {results['mean_reward']:.3f} +/- {results['std_reward']:.3f}")
    print(f"  Min/Max: {results['min_reward']:.3f} / {results['max_reward']:.3f}")
    print(f"  Mean length: {results['mean_length']:.1f}")


if __name__ == "__main__":
    main()
