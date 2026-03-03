#!/usr/bin/env python
"""Training script for CPG-regulated locomotion (Liu et al. 2023).

Phase 1: PPO + CPG with fixed K_f = 1.0 and curriculum training.

Usage:
    python -m liu2023.train_liu2023 --seed 1
    python -m liu2023.train_liu2023 --total-frames 10000 --seed 1
    python -m liu2023.train_liu2023 --no-curriculum --seed 1
"""

import argparse

from liu2023.configs_liu2023 import (
    Liu2023Config,
    Liu2023EnvConfig,
    Liu2023NetworkConfig,
    Liu2023PhysicsConfig,
)
from liu2023 import SoftSnakeEnv
from configs import setup_run_dir, ConsoleLogger
from configs.base import resolve_device
from trainers.ppo import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CPG-regulated snake locomotion (Liu et al. 2023)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--total-frames", type=int, default=None, help="Override total frames")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum")
    parser.add_argument("--no-domain-rand", action="store_true", help="Disable domain randomization")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # Build configs
    physics_config = Liu2023PhysicsConfig()
    if args.no_domain_rand:
        physics_config.randomize_friction = False
        physics_config.randomize_mass = False
        physics_config.randomize_max_pressure = False

    env_config = Liu2023EnvConfig(
        physics=physics_config,
        device=device,
    )
    if args.no_curriculum:
        env_config.curriculum.enabled = False

    ppo_config = Liu2023Config(
        seed=args.seed,
        device=device,
        name="liu2023_cpg",
    )
    if args.total_frames is not None:
        ppo_config.total_frames = args.total_frames

    network_config = Liu2023NetworkConfig(device=device)

    # Setup run directory
    run_dir = setup_run_dir(ppo_config)

    # Create environment
    env = SoftSnakeEnv(config=env_config, device=device)

    with ConsoleLogger(run_dir, ppo_config.console):
        # Create trainer
        trainer = PPOTrainer(
            env=env,
            config=ppo_config,
            network_config=network_config,
            device=device,
            run_dir=run_dir,
        )

        # Train
        curriculum_str = "enabled" if env_config.curriculum.enabled else "disabled"
        print(f"Training Liu 2023 CPG locomotion (Phase 1: PPO+CPG)")
        print(f"  Run directory: {run_dir}")
        print(f"  Seed: {args.seed}")
        print(f"  Total frames: {ppo_config.total_frames}")
        print(f"  Network: 4x128 MLP (tanh)")
        print(f"  Curriculum: {curriculum_str}")
        print(f"  Domain randomization: {not args.no_domain_rand}")

        stats = trainer.train()

        print(f"\nTraining complete!")
        print(f"  Total frames: {stats.get('total_frames', 'N/A')}")
        print(f"  Best reward: {stats.get('best_reward', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
