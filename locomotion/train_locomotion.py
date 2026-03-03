"""Training script for free-body snake locomotion via PPO.

Usage:
    python -m locomotion.train_locomotion --gait forward --total-frames 2000000
    python -m locomotion.train_locomotion --gait turn_left --seed 0
    python -m locomotion.train_locomotion --gait u_turn --max-wall-time 30m
"""

import argparse
import re

from configs import setup_run_dir, ConsoleLogger
from configs.base import resolve_device
from locomotion.configs_locomotion import GaitType, LocomotionConfig, LocomotionEnvConfig
from locomotion.env_locomotion import LocomotionEnv
from trainers.ppo import PPOTrainer


def parse_wall_time(s: str) -> float:
    """Parse a wall-time string into seconds.

    Accepts: '30m', '2h', '1h30m', '90s', '3600' (bare number = seconds).
    """
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", s)
    if not m or not any(m.groups()):
        raise argparse.ArgumentTypeError(
            f"Invalid wall-time format: '{s}'. Use e.g. '30m', '2h', '1h30m', or '3600'."
        )
    hours = int(m.group(1) or 0)
    minutes = int(m.group(2) or 0)
    seconds = int(m.group(3) or 0)
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train snake locomotion via PPO")
    parser.add_argument(
        "--gait",
        type=str,
        default="forward",
        choices=[g.value for g in GaitType],
        help="Locomotion gait type",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument(
        "--total-frames", type=int, default=None, help="Total training frames"
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel envs"
    )
    parser.add_argument(
        "--max-wall-time",
        type=str,
        default=None,
        help="Wall-clock time limit, e.g. '30m', '2h', '1h30m', or '3600' (seconds)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = resolve_device(args.device)

    # Build config
    env_config = LocomotionEnvConfig(gait=GaitType(args.gait), device=device)
    config = LocomotionConfig(seed=args.seed, device=device, env=env_config)

    if args.total_frames is not None:
        config.total_frames = args.total_frames
    if args.max_wall_time is not None:
        config.max_wall_time = parse_wall_time(args.max_wall_time)
    if args.num_envs > 1:
        config.num_envs = args.num_envs

    # Setup consolidated run directory
    run_dir = setup_run_dir(config)

    # Create environment
    if config.num_envs > 1:
        from torchrl.envs import SerialEnv

        env = SerialEnv(
            config.num_envs,
            lambda: LocomotionEnv(config.env, device=device),
        )
    else:
        env = LocomotionEnv(config.env, device=device)

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
        wall_msg = ""
        if config.max_wall_time is not None:
            mins = config.max_wall_time / 60
            wall_msg = f", max wall time {mins:.0f}min"
        print(f"Training {args.gait} locomotion with {config.total_frames} frames{wall_msg}")
        print(f"  Device: {device}")
        print(f"  Run directory: {run_dir}")
        results = trainer.train()
        print(f"Done: {results['total_episodes']} episodes, best={results['best_reward']:.2f}")


if __name__ == "__main__":
    main()
