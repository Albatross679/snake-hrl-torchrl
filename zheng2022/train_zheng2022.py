"""PPO training with curriculum learning for Zheng, Li & Hayashibe (2022).

Two-phase curriculum:
    Phase 1 (epochs 0-2000): Maximize forward velocity with power penalty.
    Phase 2 (epochs 2000+): Match decreasing target velocities with low power.

Uses PPOTrainer with a callback for curriculum reward rewriting.
"""

import argparse

import torch
import numpy as np

from zheng2022.configs_zheng2022 import (
    Zheng2022Config,
    Zheng2022EnvConfig,
    Zheng2022PhysicsConfig,
    WATER,
)
from zheng2022 import UnderwaterSnakeEnv
from zheng2022.rewards_zheng2022 import CurriculumReward
from configs import setup_run_dir, ConsoleLogger
from configs.base import resolve_device
from trainers.ppo import PPOTrainer


def rewrite_rewards(batch, curriculum: CurriculumReward) -> None:
    """Replace environment rewards with curriculum rewards in-place.

    The env stores raw head_velocity_x and power. This function applies the
    curriculum reward function to compute the actual training reward.
    """
    head_vx = batch["head_velocity_x"].squeeze(-1).cpu().numpy()
    power = batch["power"].squeeze(-1).cpu().numpy()

    rewards = np.zeros_like(head_vx)
    for i in range(len(head_vx)):
        rewards[i] = curriculum.compute_reward(float(head_vx[i]), float(power[i]))

    batch["reward"] = torch.tensor(
        rewards, dtype=torch.float32, device=batch.device,
    ).unsqueeze(-1)


def train(config: Zheng2022Config):
    """Run PPO training with curriculum learning."""
    run_dir = setup_run_dir(config)

    env = UnderwaterSnakeEnv(config=config.env, device=config.device)
    curriculum = CurriculumReward(config.env.curriculum)

    batch_count = 0

    def curriculum_callback(metrics):
        nonlocal batch_count
        batch_count += 1
        curriculum.set_epoch(batch_count)

    with ConsoleLogger(run_dir, config.console):
        trainer = PPOTrainer(
            env=env,
            config=config,
            network_config=config.network,
            device=config.device,
            run_dir=run_dir,
        )

        # Monkey-patch the collector to rewrite rewards after each batch
        original_train = trainer.train

        def train_with_curriculum(callback=None):
            def combined_callback(metrics):
                curriculum_callback(metrics)
                if callback:
                    callback(metrics)

            # Override the collector's iterator to rewrite rewards
            original_iter = trainer.collector.__iter__

            def patched_iter():
                for batch in original_iter():
                    rewrite_rewards(batch, curriculum)
                    yield batch

            trainer.collector.__iter__ = patched_iter
            return original_train(callback=combined_callback)

        trainer.train = train_with_curriculum
        trainer.train()


def main():
    parser = argparse.ArgumentParser(description="Train underwater snake (Zheng et al. 2022)")
    parser.add_argument("--epochs", type=int, default=None, help="Override total epochs")
    parser.add_argument("--stiffness", type=float, default=0.0, help="Joint stiffness (Nm/rad)")
    parser.add_argument("--fluid-density", type=float, default=1000.0, help="Fluid density (kg/m^3)")
    parser.add_argument("--fluid-viscosity", type=float, default=0.0009, help="Fluid viscosity (Pa-s)")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--name", type=str, default=None, help="Experiment name override")
    args = parser.parse_args()

    device = resolve_device(args.device)

    config = Zheng2022Config()
    config.device = device
    config.env.physics.joint_stiffness = args.stiffness
    config.env.physics.fluid_density = args.fluid_density
    config.env.physics.fluid_viscosity = args.fluid_viscosity

    if args.name is not None:
        config.name = args.name
        config.experiment_name = args.name

    if args.epochs is not None:
        config.total_frames = args.epochs * config.frames_per_batch

    train(config)


if __name__ == "__main__":
    main()
