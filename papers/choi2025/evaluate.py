"""Evaluation script for soft manipulator (Choi & Tong, 2025).

Supports both SAC and PPO algorithm checkpoints.

Usage:
    python -m choi2025.evaluate --task follow_target --checkpoint model/best.pt --algo sac
    python -m choi2025.evaluate --task follow_target --checkpoint model/best.pt --algo ppo
"""

import argparse

from choi2025.config import Choi2025Config, Choi2025PPOConfig, TaskType
from choi2025.env import SoftManipulatorEnv
from src.configs.base import resolve_device
from src.trainers.sac import SACTrainer
from src.trainers.ppo import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate soft manipulator")
    parser.add_argument(
        "--task",
        type=str,
        default="follow_target",
        choices=[t.value for t in TaskType],
        help="Task type",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "ppo"],
        help="Algorithm used to train the checkpoint",
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Eval episodes")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # Select config and trainer based on algorithm
    if args.algo == "sac":
        config = Choi2025Config(seed=args.seed, device=device)
        config.env.task = TaskType(args.task)
        config.env.device = device
        env = SoftManipulatorEnv(config.env, device=device)
        trainer = SACTrainer(
            env=env,
            config=config,
            network_config=config.network,
            device=device,
        )
    else:
        config = Choi2025PPOConfig(seed=args.seed, device=device)
        config.env.task = TaskType(args.task)
        config.env.device = device
        env = SoftManipulatorEnv(config.env, device=device)
        trainer = PPOTrainer(
            env=env,
            config=config,
            network_config=config.network,
            device=device,
        )

    trainer.load_checkpoint(args.checkpoint)

    results = trainer.evaluate(num_episodes=args.num_episodes, deterministic=True)
    print(f"Evaluation ({args.algo.upper()}, {args.num_episodes} episodes):")
    print(f"  Mean reward: {results['mean_reward']:.3f} +/- {results['std_reward']:.3f}")
    print(f"  Min/Max: {results['min_reward']:.3f} / {results['max_reward']:.3f}")
    print(f"  Mean length: {results['mean_length']:.1f}")


if __name__ == "__main__":
    main()
