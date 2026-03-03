"""Training script for soft manipulator SAC (Choi & Tong, 2025).

Usage:
    python -m choi2025.train_choi2025 --task follow_target --total-frames 1000000
    python -m choi2025.train_choi2025 --task inverse_kinematics --seed 0
    python -m choi2025.train_choi2025 --task tight_obstacles --max-wall-time 30m
"""

import argparse
import re

from choi2025.configs_choi2025 import Choi2025Config, Choi2025EnvConfig, TaskType
from choi2025.env_choi2025 import SoftManipulatorEnv
from configs import setup_run_dir, ConsoleLogger
from configs.base import resolve_device
from trainers.sac import SACTrainer


def parse_wall_time(s: str) -> float:
    """Parse a wall-time string into seconds.

    Accepts: '30m', '2h', '1h30m', '90s', '3600' (bare number = seconds).
    """
    s = s.strip()
    # Bare number → seconds
    try:
        return float(s)
    except ValueError:
        pass
    # Match patterns like '1h30m', '30m', '2h', '90s'
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
    parser = argparse.ArgumentParser(description="Train soft manipulator SAC")
    parser.add_argument(
        "--task",
        type=str,
        default="follow_target",
        choices=[t.value for t in TaskType],
        help="Task type",
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

    # Resolve device: "auto" → "cuda" if available, else "cpu"
    device = resolve_device(args.device)

    # Build config (construct env first so __post_init__ sees the task)
    env_config = Choi2025EnvConfig(task=TaskType(args.task), device=device)
    config = Choi2025Config(seed=args.seed, device=device, env=env_config)

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
            lambda: SoftManipulatorEnv(config.env, device=device),
        )
    else:
        env = SoftManipulatorEnv(config.env, device=device)

    with ConsoleLogger(run_dir, config.console):
        # Create trainer
        trainer = SACTrainer(
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
        print(f"Training {args.task} with {config.total_frames} frames{wall_msg}")
        print(f"  Device: {device}")
        print(f"  Run directory: {run_dir}")
        results = trainer.train()
        print(f"Done: {results['total_episodes']} episodes, best={results['best_reward']:.2f}")


if __name__ == "__main__":
    main()
