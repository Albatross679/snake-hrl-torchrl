"""Training script for hierarchical path following (Liu, Guo & Fang, 2022).

PPO training of the RL policy that outputs gait offset φ_o.
The gait execution layer converts φ_o into lateral undulatory joint angles.

Converges in ~1M timesteps (proposed hierarchical method)
vs ~2M timesteps for end-to-end baseline.

Usage:
    python -m liu2022.train_liu2022 --total-frames 1000000
    python -m liu2022.train_liu2022 --path-type sinusoidal --seed 42
"""

import argparse

from liu2022.configs_liu2022 import Liu2022Config, PathType
from liu2022.env_liu2022 import PathFollowingSnakeEnv
from src.configs import setup_run_dir, ConsoleLogger
from src.configs.base import resolve_device
from src.trainers.ppo import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train hierarchical path following (Liu et al., 2022)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument(
        "--total-frames", type=int, default=None, help="Total training frames"
    )
    parser.add_argument(
        "--path-type",
        type=str,
        default="straight_line",
        choices=["straight_line", "sinusoidal", "circle"],
        help="Desired path type",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    config = Liu2022Config(seed=args.seed)

    if args.total_frames is not None:
        config.total_frames = args.total_frames

    config.env.path.path_type = PathType(args.path_type)
    config.env.device = device

    # Setup consolidated run directory
    run_dir = setup_run_dir(config)

    # Create environment
    env = PathFollowingSnakeEnv(config.env, device=device)

    with ConsoleLogger(run_dir, config.console):
        # Train with PPO
        trainer = PPOTrainer(
            env=env,
            config=config,
            network_config=config.network,
            device=device,
            run_dir=run_dir,
        )

        print("Hierarchical Path Following (Liu, Guo & Fang, 2022)")
        print(f"  Run directory: {run_dir}")
        print(f"  Path: {args.path_type}")
        print(f"  Obs: ({env._obs_dim},), Action: (1,) gait offset")
        print(f"  Gait: α={config.env.gait.amplitude}, ω={config.env.gait.angular_freq}, δ={config.env.gait.phase_diff}")
        print(f"  Frames: {config.total_frames}")
        results = trainer.train()
        print(f"Done: {results['total_episodes']} episodes, best={results['best_reward']:.2f}")

    env.close()


if __name__ == "__main__":
    main()
