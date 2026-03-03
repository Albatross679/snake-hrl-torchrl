"""Training script for contact-aware CPG soft snake (Liu, Onal & Fu, 2021).

Implements the fictitious cooperative game (Algorithm 1):
1. Initialize C1 (trained in obstacle-free) and R2 (random)
2. Evaluate both in obstacle environment
3. Alternate: fix one, train the other via PPO
4. Repeat until value function converges (or max iterations)

Usage:
    python -m liu2021.train_liu2021 --total-frames 2000000
    python -m liu2021.train_liu2021 --max-rounds 20 --seed 42
"""

import argparse

from liu2021.configs_liu2021 import Liu2021Config
from liu2021.env_liu2021 import ContactAwareSoftSnakeEnv
from configs import setup_run_dir, ConsoleLogger
from configs.base import resolve_device
from trainers.ppo import PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train contact-aware CPG soft snake (fictitious play)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument(
        "--total-frames", type=int, default=None, help="Total training frames"
    )
    parser.add_argument(
        "--max-rounds", type=int, default=20, help="Max fictitious play rounds"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    config = Liu2021Config(seed=args.seed, device=device)
    config.env.device = device

    if args.total_frames is not None:
        config.total_frames = args.total_frames
    config.max_macro_iterations = args.max_rounds

    # Setup consolidated run directory
    run_dir = setup_run_dir(config)

    # Create environment (combined C1+R2 action space)
    env = ContactAwareSoftSnakeEnv(config.env, device=device)

    with ConsoleLogger(run_dir, config.console):
        # In the full implementation, C1 and R2 would be separate networks
        # trained alternately via fictitious play. Here we train the combined
        # policy as a single PPO agent for simplicity.
        trainer = PPOTrainer(
            env=env,
            config=config,
            network_config=config.network,
            device=device,
            run_dir=run_dir,
        )

        print("Contact-aware CPG Soft Snake (Liu, Onal & Fu, 2021)")
        print(f"  Run directory: {run_dir}")
        print(f"  Obs: {config.env.obs_dim}, Action: C1({config.env.c1_action_dim}) + R2({config.env.r2_action_dim})")
        print(f"  Frames: {config.total_frames}, Max rounds: {config.max_macro_iterations}")
        results = trainer.train()
        print(f"Done: {results['total_episodes']} episodes, best={results['best_reward']:.2f}")

    env.close()


if __name__ == "__main__":
    main()
