"""Training script for free-body snake locomotion via PPO (PyElastica backend).

Usage:
    # Single config (baseline)
    python -m locomotion_elastica.train --gait forward --total-frames 5000000 --wandb

    # Select a specific config variant
    python -m locomotion_elastica.train --config high_lr --gait forward --wandb

    # Sequential auto-batch: run multiple configs in one process
    python -m locomotion_elastica.train --batch baseline,high_lr,large_net --gait forward --wandb
"""

import os
# Limit thread spawning per process for parallel envs (must be set before numpy import)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import re

from src.configs import setup_run_dir, ConsoleLogger
from src.configs.base import resolve_device
from src.utils.cleanup import cleanup_vram
from locomotion_elastica.config import (
    GaitType,
    LocomotionElasticaConfig,
    LocomotionElasticaEnvConfig,
    CONFIG_REGISTRY,
)
from locomotion_elastica.env import LocomotionElasticaEnv
from src.trainers.ppo import PPOTrainer


class _EnvFactory:
    """Picklable env factory for ParallelEnv.

    ParallelEnv spawns child processes via multiprocessing, which requires
    pickling the env constructor. Lambdas and closures are not picklable,
    but a plain class with dataclass attributes is.
    """

    def __init__(self, config: LocomotionElasticaEnvConfig, device: str):
        self.config = config
        self.device = device

    def __call__(self):
        return LocomotionElasticaEnv(self.config, device=self.device)


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
    parser = argparse.ArgumentParser(
        description="Train snake locomotion via PPO (PyElastica)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="baseline",
        choices=list(CONFIG_REGISTRY.keys()),
        help=f"Config variant to use (choices: {', '.join(CONFIG_REGISTRY.keys())})",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=None,
        help="Comma-separated config names for sequential auto-batch (e.g. baseline,high_lr,large_net)",
    )
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
        "--num-envs", type=int, default=None,
        help="Number of parallel envs (default: CPU cores - 1)",
    )
    parser.add_argument(
        "--frames-per-batch", type=int, default=None,
        help="Frames per batch (default: config value, 8192)",
    )
    parser.add_argument(
        "--max-wall-time",
        type=str,
        default=None,
        help="Wall-clock time limit, e.g. '30m', '2h', '1h30m', or '3600' (seconds)",
    )
    parser.add_argument(
        "--wandb", action="store_true", default=False, help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project", type=str, default="snake-hrl", help="W&B project name"
    )
    parser.add_argument(
        "--wandb-entity", type=str, default="", help="W&B entity (team/user)"
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace, config_name: str) -> LocomotionElasticaConfig:
    """Build a config from a registry name + CLI overrides."""
    ConfigClass = CONFIG_REGISTRY[config_name]
    device = resolve_device(args.device)

    env_config = LocomotionElasticaEnvConfig(gait=GaitType(args.gait), device=device)
    config = ConfigClass(seed=args.seed, device=device, env=env_config)

    if args.total_frames is not None:
        config.total_frames = args.total_frames
    if args.max_wall_time is not None:
        config.max_wall_time = parse_wall_time(args.max_wall_time)
    if args.num_envs is not None:
        config.num_envs = args.num_envs
    if args.frames_per_batch is not None:
        config.frames_per_batch = args.frames_per_batch
    if args.wandb:
        config.wandb.enabled = True
        config.wandb.project = args.wandb_project
        if args.wandb_entity:
            config.wandb.entity = args.wandb_entity

    return config


def create_env(config: LocomotionElasticaConfig, device: str):
    """Create the TorchRL environment (single or parallel)."""
    from torchrl.envs import TransformedEnv
    from torchrl.envs.transforms import StepCounter, RewardSum

    if config.num_envs > 1:
        from torchrl.envs import ParallelEnv

        base_env = ParallelEnv(
            config.num_envs,
            _EnvFactory(config.env, "cpu"),
        )
    else:
        base_env = LocomotionElasticaEnv(config.env, device=device)

    env = TransformedEnv(base_env)
    env.append_transform(StepCounter())
    env.append_transform(RewardSum())
    return env


def run_single_config(args: argparse.Namespace, config_name: str) -> dict:
    """Train a single config variant. Returns results dict."""
    config = build_config(args, config_name)
    device = resolve_device(args.device)

    # Setup consolidated run directory
    run_dir = setup_run_dir(config)

    # Create environment
    env = create_env(config, device)

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
            f"Training [{config_name}] {config.env.gait.value} locomotion (PyElastica) "
            f"with {config.total_frames} frames{wall_msg}"
        )
        print(f"  Device: {device}")
        print(f"  Parallel envs: {config.num_envs}")
        print(f"  Run directory: {run_dir}")
        results = trainer.train()
        print(f"Done [{config_name}]: {results['total_episodes']} episodes, "
              f"best={results['best_reward']:.2f}, stop_reason={results['stop_reason']}")

    return results


def main():
    args = parse_args()

    # Determine which configs to run
    if args.batch:
        config_names = [name.strip() for name in args.batch.split(",")]
        for name in config_names:
            if name not in CONFIG_REGISTRY:
                raise ValueError(
                    f"Unknown config '{name}'. Available: {', '.join(CONFIG_REGISTRY.keys())}"
                )
    else:
        config_names = [args.config]

    # Sequential auto-batch: run configs one by one with VRAM cleanup
    for i, config_name in enumerate(config_names):
        if i > 0:
            print(f"\n{'='*60}")
            print(f"Cleaning VRAM before config {i+1}/{len(config_names)}: {config_name}")
            print(f"{'='*60}\n")
            cleanup_vram()

        run_single_config(args, config_name)

    if len(config_names) > 1:
        print(f"\nAll {len(config_names)} configs completed.")


if __name__ == "__main__":
    from src.utils.gpu_lock import GpuLock
    with GpuLock():
        main()
