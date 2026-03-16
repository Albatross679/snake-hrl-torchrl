"""RL training with the surrogate environment.

Mirrors locomotion_elastica/train.py but uses SurrogateLocomotionEnv
instead of LocomotionElasticaEnv. No ParallelEnv needed — the surrogate
env is internally GPU-batched.

Usage:
    python -m aprx_model_elastica.train_rl --surrogate-checkpoint output/surrogate
    python -m aprx_model_elastica.train_rl --surrogate-checkpoint output/surrogate --total-frames 2000000
"""

import argparse
import re

from src.configs import setup_run_dir, ConsoleLogger
from src.configs.base import resolve_device
from aprx_model_elastica.train_config import SurrogateEnvConfig, SurrogateRLConfig
from aprx_model_elastica.env import SurrogateLocomotionEnv
from src.trainers.ppo import PPOTrainer


def parse_wall_time(s: str) -> float:
    """Parse a wall-time string into seconds."""
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
    parser = argparse.ArgumentParser(
        description="Train snake locomotion via PPO (surrogate env)"
    )
    parser.add_argument(
        "--surrogate-checkpoint", type=str, required=True,
        help="Path to surrogate model directory (containing model.pt, normalizer.pt, config.json)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument(
        "--total-frames", type=int, default=None, help="Total training frames"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Number of parallel environments in the surrogate (GPU batch size)",
    )
    parser.add_argument(
        "--frames-per-batch", type=int, default=None,
        help="Frames per batch (default: config value, 8192)",
    )
    parser.add_argument(
        "--max-wall-time", type=str, default=None,
        help="Wall-clock time limit, e.g. '30m', '2h', '1h30m'",
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False, help="Enable W&B logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="snake-hrl-surrogate", help="W&B project"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default="", help="W&B entity"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # Build config
    env_config = SurrogateEnvConfig(
        surrogate_checkpoint=args.surrogate_checkpoint,
        device=device,
        batch_size=args.batch_size,
    )
    config = SurrogateRLConfig(seed=args.seed, device=device, env=env_config)

    if args.total_frames is not None:
        config.total_frames = args.total_frames
    if args.max_wall_time is not None:
        config.max_wall_time = parse_wall_time(args.max_wall_time)
    if args.frames_per_batch is not None:
        config.frames_per_batch = args.frames_per_batch
    if args.wandb:
        config.wandb.enabled = True
        config.wandb.project = args.wandb_project
        if args.wandb_entity:
            config.wandb.entity = args.wandb_entity

    # Setup run directory
    run_dir = setup_run_dir(config)

    # Create surrogate environment with transforms
    from torchrl.envs import TransformedEnv
    from torchrl.envs.transforms import StepCounter, RewardSum

    base_env = SurrogateLocomotionEnv(config.env, device=device)
    env = TransformedEnv(base_env)
    env.append_transform(StepCounter())
    env.append_transform(RewardSum())

    with ConsoleLogger(run_dir, config.console):
        trainer = PPOTrainer(
            env=env,
            config=config,
            network_config=config.network,
            device=device,
            run_dir=run_dir,
        )

        wall_msg = ""
        if config.max_wall_time is not None:
            mins = config.max_wall_time / 60
            wall_msg = f", max wall time {mins:.0f}min"
        print(
            f"Training locomotion (surrogate) with {config.total_frames} frames{wall_msg}"
        )
        print(f"  Device: {device}")
        print(f"  Surrogate batch size: {args.batch_size}")
        print(f"  Surrogate checkpoint: {args.surrogate_checkpoint}")
        print(f"  Run directory: {run_dir}")
        results = trainer.train()
        print(f"Done: {results['total_episodes']} episodes, best={results['best_reward']:.2f}")


if __name__ == "__main__":
    main()
