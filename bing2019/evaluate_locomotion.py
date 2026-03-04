#!/usr/bin/env python
"""Evaluation script for planar snake locomotion.

Two evaluation modes:
1. Power-velocity sweep: Sweep target_v from 0.025 to 0.255, measure velocity and power.
2. Target tracking: Run on various track types, measure trajectory following.

Usage:
    python -m bing2019.evaluate_locomotion --task power_velocity --checkpoint checkpoints/final.pt
    python -m bing2019.evaluate_locomotion --task target_tracking --checkpoint checkpoints/final.pt
"""

import argparse
import csv
import os

import numpy as np
import torch

from bing2019.configs_bing2019 import (
    LocomotionEnvConfig,
    LocomotionNetworkConfig,
    LocomotionPhysicsConfig,
)
from bing2019 import PlanarSnakeEnv
from bing2019.rewards_bing2019 import compute_energy_normalized
from src.configs.base import resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate planar snake locomotion")
    parser.add_argument(
        "--task", type=str, default="power_velocity",
        choices=["power_velocity", "target_tracking"],
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--num-steps", type=int, default=1000, help="Steps per evaluation")
    parser.add_argument("--warmup-steps", type=int, default=200, help="Warmup steps to discard")
    return parser.parse_args()


def load_actor(checkpoint_path, device):
    """Load actor from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def evaluate_power_velocity(args):
    """Sweep target velocities and measure achieved velocity + power."""
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    actor_state = checkpoint["actor_state_dict"]

    results = []
    target_vs = np.arange(0.025, 0.260, 0.005)

    for target_v in target_vs:
        env_config = LocomotionEnvConfig(
            task="power_velocity",
            target_v=float(target_v),
            device=args.device,
        )
        env = PlanarSnakeEnv(config=env_config, device=args.device)

        # Manually set target_v (bypass cycling)
        env._target_v = float(target_v)

        td = env.reset()
        velocities = []
        powers = []

        for step in range(args.num_steps):
            # For evaluation without a trained actor, use zero actions
            action = torch.zeros(8, dtype=torch.float32, device=args.device)
            td["action"] = action
            td = env.step(td)

            if step >= args.warmup_steps:
                sensor_frcs = env._get_sensor_actuatorfrcs()
                joint_vels = env._get_joint_velocities()
                gear = float(env.model.actuator_gear.max())
                fmax = float(env.model.actuator_forcerange.max())
                pn, _ = compute_energy_normalized(sensor_frcs, joint_vels, gear, fmax)

                head_vel = env._get_sensor_head_velocity()
                velocities.append(abs(head_vel))
                powers.append(pn)

            td = td["next"] if "next" in td.keys() else env.reset()

        mean_vel = np.mean(velocities) if velocities else 0.0
        mean_power = np.mean(powers) if powers else 0.0
        results.append({
            "target_v": float(target_v),
            "mean_velocity": mean_vel,
            "mean_power_normalized": mean_power,
        })
        print(f"  target_v={target_v:.3f}: vel={mean_vel:.4f}, power={mean_power:.4f}")
        env.close()

    # Save CSV
    csv_path = os.path.join(args.output_dir, "power_velocity_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target_v", "mean_velocity", "mean_power_normalized"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")


def evaluate_target_tracking(args):
    """Evaluate target tracking on various track types."""
    os.makedirs(args.output_dir, exist_ok=True)

    track_types = ["line", "wave", "zigzag", "random"]
    num_steps = 3000

    results = []
    for track_type in track_types:
        env_config = LocomotionEnvConfig(
            task="target_tracking",
            track_type=track_type,
            device=args.device,
        )
        env_config.max_episode_steps = num_steps + 100
        env = PlanarSnakeEnv(config=env_config, device=args.device)

        td = env.reset()
        distances = []
        head_xs, head_ys = [], []
        target_xs, target_ys = [], []

        for step in range(num_steps):
            action = torch.zeros(8, dtype=torch.float32, device=args.device)
            td["action"] = action
            td = env.step(td)

            dist = env._calc_distance()
            head_x, head_y = env._get_head_pos()
            target_x, target_y = env._get_target_pos()

            distances.append(dist)
            head_xs.append(head_x)
            head_ys.append(head_y)
            target_xs.append(target_x)
            target_ys.append(target_y)

            td = td["next"] if "next" in td.keys() else env.reset()

        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        results.append({
            "track_type": track_type,
            "mean_distance": mean_dist,
            "std_distance": std_dist,
            "target_distance": env.config.target_distance,
        })
        print(f"  {track_type}: mean_dist={mean_dist:.3f} +/- {std_dist:.3f}")
        env.close()

    csv_path = os.path.join(args.output_dir, "target_tracking_eval.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track_type", "mean_distance", "std_distance", "target_distance"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")


def main():
    args = parse_args()
    device = resolve_device(args.device)
    args.device = device

    print(f"Evaluating locomotion: task={args.task}")
    print(f"  Checkpoint: {args.checkpoint}")

    if args.task == "power_velocity":
        evaluate_power_velocity(args)
    else:
        evaluate_target_tracking(args)


if __name__ == "__main__":
    main()
