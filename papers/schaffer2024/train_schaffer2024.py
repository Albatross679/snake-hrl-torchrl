"""Training script for biohybrid worm PPO (Schaffer et al., 2024).

Usage:
    python -m schaffer2024.train_schaffer2024 --total-frames 500000
    python -m schaffer2024.train_schaffer2024 --no-adaptation --seed 0
"""

import argparse

from schaffer2024.configs_schaffer2024 import Schaffer2024Config, Schaffer2024EnvConfig
from schaffer2024.env_schaffer2024 import LatticeWormEnv
from src.configs import setup_run_dir, ConsoleLogger
from src.configs.base import resolve_device
from src.trainers.ppo import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train biohybrid worm PPO")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument(
        "--total-frames", type=int, default=None, help="Total training frames"
    )
    parser.add_argument(
        "--no-adaptation",
        action="store_true",
        help="Disable muscle adaptation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # Build config
    config = Schaffer2024Config(seed=args.seed, device=device)
    config.env.device = device

    if args.total_frames is not None:
        config.total_frames = args.total_frames
    if args.no_adaptation:
        config.env.enable_adaptation = False

    # Setup consolidated run directory
    run_dir = setup_run_dir(config)

    # Create environment
    env = LatticeWormEnv(config.env, device=device)

    with ConsoleLogger(run_dir, config.console):
        # Create trainer
        trainer = PPOTrainer(
            env=env,
            config=config,
            network_config=config.network,
            device=device,
            run_dir=run_dir,
        )

        # Train
        adaptation_str = "enabled" if config.env.enable_adaptation else "disabled"
        print(f"Training biohybrid worm (adaptation {adaptation_str})")
        print(f"  Run directory: {run_dir}")
        print(f"  Frames: {config.total_frames}, Muscles: {config.env.muscles.num_muscles}")
        results = trainer.train()
        print(f"Done: {results['total_episodes']} episodes, best={results['best_reward']:.2f}")


if __name__ == "__main__":
    main()
