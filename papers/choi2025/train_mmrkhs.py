"""Training script for soft manipulator MM-RKHS (Choi & Tong, 2025 setup).

Usage:
    python -m choi2025.train_mmrkhs --task follow_target --total-frames 1000000
    python -m choi2025.train_mmrkhs --task inverse_kinematics --seed 0
    python -m choi2025.train_mmrkhs --task tight_obstacles --max-wall-time 30m
"""

import argparse
import os

# Limit thread spawning for parallel envs to avoid pthread exhaustion
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from torchrl.envs import RewardSum

from choi2025.config import Choi2025MMRKHSConfig, Choi2025EnvConfig, TaskType
from choi2025.env import SoftManipulatorEnv
from choi2025.train import parse_wall_time
from src.configs import setup_run_dir, ConsoleLogger
from src.configs.base import resolve_device
from src.trainers.mmrkhs import MMRKHSTrainer


def _make_env(env_config, device):
    """Top-level env factory for picklability with ParallelEnv."""
    return SoftManipulatorEnv(env_config, device=device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train soft manipulator MM-RKHS (Gupta & Mahajan)")
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
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve device: "auto" -> "cuda" if available, else "cpu"
    device = resolve_device(args.device)

    # Build config (construct env first so __post_init__ sees the task)
    env_config = Choi2025EnvConfig(task=TaskType(args.task), device=device)
    config = Choi2025MMRKHSConfig(seed=args.seed, device=device, env=env_config)

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

    # Accumulate per-step rewards into episode_reward for monitoring
    env = env.append_transform(RewardSum())

    try:
        with ConsoleLogger(run_dir, config.console):
            # Create trainer
            trainer = MMRKHSTrainer(
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
            print(f"Training {args.task} with MM-RKHS, {config.total_frames} frames{wall_msg}")
            print(f"  Device: {device}")
            print(f"  Run directory: {run_dir}")
            results = trainer.train()
            print(f"Done: {results['total_episodes']} episodes, best={results['best_reward']:.2f}")
    finally:
        try:
            env.close()
        except RuntimeError:
            pass  # Already closed by collector


if __name__ == "__main__":
    # Skip GpuLock when running on a dedicated GPU (e.g. CUDA_VISIBLE_DEVICES=1)
    # to allow concurrent training on separate GPUs
    if os.environ.get("SKIP_GPU_LOCK"):
        main()
    else:
        from src.utils.gpu_lock import GpuLock

        with GpuLock():
            main()
