"""Training script for soft manipulator PPO (Choi & Tong, 2025 setup).

Usage:
    python -m choi2025.train_ppo --task follow_target --total-frames 1000000
    python -m choi2025.train_ppo --task inverse_kinematics --seed 0
    python -m choi2025.train_ppo --task tight_obstacles --max-wall-time 30m
"""

import argparse
import os

# Limit thread spawning for parallel envs to avoid pthread exhaustion
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from torchrl.envs import RewardSum
from torchrl.envs.transforms import ObservationNorm, RewardScaling

from choi2025.config import Choi2025PPOConfig, Choi2025EnvConfig, TaskType
from choi2025.env import SoftManipulatorEnv
from choi2025.train import parse_wall_time
from src.configs import setup_run_dir, ConsoleLogger
from src.configs.base import resolve_device
from src.trainers.ppo import PPOTrainer


def _make_env(env_config, device):
    """Top-level env factory for picklability with ParallelEnv."""
    return SoftManipulatorEnv(env_config, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train soft manipulator PPO")
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
        "--num-envs", type=int, default=32, help="Number of parallel envs"
    )
    parser.add_argument(
        "--max-wall-time",
        type=str,
        default=None,
        help="Wall-clock time limit, e.g. '30m', '2h', '1h30m', or '3600' (seconds)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g. output/.../checkpoints/final.pt)",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable curriculum: ramp target speed from 0 to full over warmup episodes",
    )
    parser.add_argument(
        "--warmup-episodes",
        type=int,
        default=200,
        help="Per-worker episodes before reaching full target speed (default: 200)",
    )
    parser.add_argument(
        "--dist-weight",
        type=float,
        default=1.0,
        help="Base distance reward weight (0.0=disabled, 1.0=default). Set to 0 for PBRS-only.",
    )
    parser.add_argument(
        "--heading-weight",
        type=float,
        default=0.0,
        help="Heading reward weight: bonus for pointing tip toward target (default: 0.0, try 0.3)",
    )
    parser.add_argument(
        "--pbrs-gamma",
        type=float,
        default=0.0,
        help="PBRS discount factor (0.0=disabled, 0.99=typical). Adds policy-invariant shaping F=prev_dist-gamma*dist.",
    )
    parser.add_argument(
        "--smooth-weight",
        type=float,
        default=0.0,
        help="Action smoothness penalty weight (normalized to [-1,0]). Typical: 0.01-0.05. 0.0=disabled.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve device: "auto" -> "cuda" if available, else "cpu"
    device = resolve_device(args.device)

    # Build config (construct env first so __post_init__ sees the task)
    env_config = Choi2025EnvConfig(task=TaskType(args.task), device=device)

    # Set base distance reward weight
    env_config.dist_weight = args.dist_weight

    # Enable heading reward if requested
    if args.heading_weight > 0:
        env_config.heading_weight = args.heading_weight

    # Enable PBRS if requested
    if args.pbrs_gamma > 0:
        env_config.pbrs_gamma = args.pbrs_gamma
    if args.smooth_weight > 0:
        env_config.smooth_weight = args.smooth_weight

    # Enable curriculum learning if requested
    if args.curriculum:
        from choi2025.config import CurriculumConfig
        env_config.target.curriculum = CurriculumConfig(
            enabled=True, warmup_episodes=args.warmup_episodes,
        )

    config = Choi2025PPOConfig(seed=args.seed, device=device, env=env_config)

    if args.total_frames is not None:
        config.total_frames = args.total_frames
    if args.max_wall_time is not None:
        config.max_wall_time = parse_wall_time(args.max_wall_time)
    if args.num_envs > 1:
        config.num_envs = args.num_envs

    # Re-run __post_init__ to update name with correct num_envs
    config.__post_init__()

    # Setup consolidated run directory
    run_dir = setup_run_dir(config)

    # Create environment
    # ParallelEnv workers run on CPU (physics is CPU-bound numpy);
    # the training loop handles GPU transfer for policy/critic inference.
    if config.num_envs > 1:
        from torchrl.envs import ParallelEnv

        env = ParallelEnv(
            num_workers=config.num_envs,
            create_env_fn=[
                lambda cfg=env_config: _make_env(cfg, "cpu")
            ] * config.num_envs,
        )
    else:
        env = SoftManipulatorEnv(config.env, device=device)

    # Normalize observations (running mean/std) — critical for PPO with
    # heterogeneous-scale observations (positions, velocities, curvatures).
    obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
    env = env.append_transform(obs_norm)

    # Initialize running stats from a few random rollouts
    if config.num_envs > 1:
        obs_norm.init_stats(num_iter=200, reduce_dim=[0, 1], cat_dim=0)
    else:
        obs_norm.init_stats(num_iter=200, reduce_dim=0)

    # Accumulate per-step rewards into episode_reward for monitoring
    env = env.append_transform(RewardSum())

    try:
        with ConsoleLogger(run_dir, config.console):
            # Create trainer
            trainer = PPOTrainer(
                env=env,
                config=config,
                network_config=config.network,
                device=device,
                run_dir=run_dir,
            )

            # Resume from checkpoint if specified
            if args.resume:
                print(f"Resuming from checkpoint: {args.resume}")
                trainer.load_checkpoint(args.resume)
                print(f"  Resumed at frame {trainer.total_frames}, best_reward={trainer.best_reward:.2f}")

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
    finally:
        env.close()


if __name__ == "__main__":
    from src.utils.gpu_lock import GpuLock

    with GpuLock():
        main()
