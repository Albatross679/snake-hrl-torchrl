"""Training script for Elastica-RL benchmark (Naughton et al., 2021).

Supports all five algorithms benchmarked in the paper (SAC, TD3, DDPG, PPO, TRPO),
defaulting to SAC as the best performer.

Usage:
    python -m naughton2021.train_naughton2021 --case case1_tracking --algo sac
    python -m naughton2021.train_naughton2021 --case case3_structured --total-frames 5000000
"""

import argparse

from naughton2021.configs_naughton2021 import (
    BenchmarkCase,
    Naughton2021Config,
    Naughton2021EnvConfig,
)
from naughton2021.env_naughton2021 import ElasticaControlEnv
from src.configs import setup_run_dir, ConsoleLogger
from src.configs.base import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Elastica-RL benchmark")
    parser.add_argument(
        "--case",
        type=str,
        default="case1_tracking",
        choices=[c.value for c in BenchmarkCase],
        help="Benchmark case",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["sac", "td3", "ddpg", "ppo", "trpo"],
        help="RL algorithm (SAC and TD3 are best)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument(
        "--total-frames", type=int, default=None, help="Total training frames"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # Build config (construct env first so __post_init__ sees the case)
    env_config = Naughton2021EnvConfig(case=BenchmarkCase(args.case), device=device)
    config = Naughton2021Config(seed=args.seed, device=device, env=env_config)

    if args.total_frames is not None:
        config.total_frames = args.total_frames

    # Adjust frames for on-policy vs off-policy
    if args.algo in ("ppo", "trpo") and args.total_frames is None:
        config.total_frames = 10_000_000  # On-policy needs more samples

    # Setup consolidated run directory
    run_dir = setup_run_dir(config)

    # Create environment
    env = ElasticaControlEnv(config.env, device=device)

    with ConsoleLogger(run_dir, config.console):
        # Select trainer
        if args.algo == "sac":
            from src.trainers.sac import SACTrainer

            trainer = SACTrainer(
                env=env,
                config=config,
                network_config=config.network,
                device=device,
                run_dir=run_dir,
            )
        elif args.algo == "ppo":
            from src.configs.training import PPOConfig
            from src.trainers.ppo import PPOTrainer

            ppo_config = PPOConfig(
                name="naughton2021",
                seed=args.seed,
                device=device,
                total_frames=config.total_frames,
                experiment_name=f"naughton2021_{args.case}_ppo",
            )
            trainer = PPOTrainer(
                env=env,
                config=ppo_config,
                network_config=config.network,
                device=device,
                run_dir=run_dir,
            )
        elif args.algo == "ddpg":
            from src.trainers.ddpg import DDPGTrainer
            from src.configs.training import DDPGConfig

            ddpg_config = DDPGConfig(
                name="naughton2021",
                seed=args.seed,
                device=device,
                total_frames=config.total_frames,
                experiment_name=f"naughton2021_{args.case}_ddpg",
            )
            trainer = DDPGTrainer(
                env=env,
                config=ddpg_config,
                network_config=config.network,
                device=device,
                run_dir=run_dir,
            )
        else:
            print(f"Algorithm '{args.algo}' not yet implemented in trainers/")
            return

        # Train
        case_name = args.case.replace("_", " ").title()
        print(f"Elastica-RL Benchmark: {case_name}")
        print(f"  Run directory: {run_dir}")
        print(f"  Algorithm: {args.algo.upper()}, Frames: {config.total_frames}")
        print(f"  Action dim: {config.env.control.action_dim}")
        results = trainer.train()
        print(f"Done: {results['total_episodes']} episodes, best={results['best_reward']:.2f}")


if __name__ == "__main__":
    main()
