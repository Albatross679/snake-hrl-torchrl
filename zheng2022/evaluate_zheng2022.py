"""Evaluation script for trained underwater snake policies.

Measures velocity, power, efficiency, and plots gait patterns.
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from networks.actor import create_actor
from configs.network import ActorConfig

from zheng2022.configs_zheng2022 import Zheng2022EnvConfig, Zheng2022PhysicsConfig
from zheng2022 import UnderwaterSnakeEnv


def load_policy(checkpoint_path: str, config: Zheng2022EnvConfig):
    """Load trained actor from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    actor_config = ActorConfig(
        hidden_dims=[256, 256],
        activation="relu",
    )

    env = UnderwaterSnakeEnv(config=config, device=config.device)
    actor = create_actor(
        obs_dim=config.physics.obs_dim,
        action_spec=env.action_spec,
        config=actor_config,
        device=config.device,
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor, env


def evaluate_policy(
    actor, env: UnderwaterSnakeEnv, num_episodes: int = 10, deterministic: bool = True,
) -> dict:
    """Run evaluation episodes and collect metrics."""
    all_velocities = []
    all_powers = []
    all_joint_angles = []  # Per-step joint angles for gait analysis
    all_joint_velocities = []
    episode_rewards = []
    episode_lengths = []

    for ep in range(num_episodes):
        td = env.reset()
        ep_reward = 0.0
        ep_len = 0
        ep_vx = []
        ep_power = []
        ep_angles = []
        ep_angvels = []

        done = False
        while not done:
            with torch.no_grad():
                if deterministic:
                    actor(td)
                    td["action"] = td["loc"]  # Use mean action
                else:
                    td = actor(td)

            td = env.step(td)

            # TorchRL puts step results under "next" key
            next_td = td["next"]
            ep_vx.append(next_td["head_velocity_x"].item())
            ep_power.append(next_td["power"].item())
            ep_reward += next_td["reward"].item()
            ep_len += 1

            # Record joint angles and velocities
            obs = next_td["observation"].cpu().numpy()
            ep_angles.append(obs[8:14].copy())  # Joint angles from obs
            ep_angvels.append(obs[1:7].copy())  # Joint angular velocities

            done = next_td["done"].item()

            # Prepare for next step
            td = next_td

        all_velocities.append(ep_vx)
        all_powers.append(ep_power)
        all_joint_angles.append(np.array(ep_angles))
        all_joint_velocities.append(np.array(ep_angvels))
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

    # Aggregate statistics
    mean_vx_per_ep = [np.mean(v) for v in all_velocities]
    mean_power_per_ep = [np.mean(p) for p in all_powers]
    efficiency = [v / max(p, 1e-8) for v, p in zip(mean_vx_per_ep, mean_power_per_ep)]

    results = {
        "mean_velocity": np.mean(mean_vx_per_ep),
        "std_velocity": np.std(mean_vx_per_ep),
        "mean_power": np.mean(mean_power_per_ep),
        "std_power": np.std(mean_power_per_ep),
        "mean_efficiency": np.mean(efficiency),
        "mean_reward": np.mean(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "velocities": all_velocities,
        "powers": all_powers,
        "joint_angles": all_joint_angles,
        "joint_velocities": all_joint_velocities,
    }
    return results


def plot_results(results: dict, output_dir: Path):
    """Generate evaluation plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Velocity and power over time (first episode)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = np.arange(len(results["velocities"][0])) * 0.04  # 25 Hz

    axes[0].plot(t, results["velocities"][0], "b-", linewidth=0.8)
    axes[0].set_ylabel("Forward velocity (m/s)")
    axes[0].set_title("Evaluation episode")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, results["powers"][0], "r-", linewidth=0.8)
    axes[1].set_ylabel("Power (W)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "velocity_power_timeseries.png", dpi=150)
    plt.close()

    # Joint angle trajectories (gait pattern)
    fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
    angles = results["joint_angles"][0]
    t = np.arange(len(angles)) * 0.04

    for j in range(6):
        ax = axes[j // 2, j % 2]
        ax.plot(t, np.degrees(angles[:, j]), linewidth=0.8)
        ax.set_ylabel(f"Joint {j+1} (deg)")
        ax.grid(True, alpha=0.3)
        if j >= 4:
            ax.set_xlabel("Time (s)")

    fig.suptitle("Joint angle trajectories (gait pattern)")
    plt.tight_layout()
    plt.savefig(output_dir / "gait_pattern.png", dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained underwater snake policy")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--stiffness", type=float, default=0.0, help="Joint stiffness (Nm/rad)")
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    config = Zheng2022EnvConfig()
    config.physics.joint_stiffness = args.stiffness
    actor, env = load_policy(args.checkpoint, config)

    print(f"Evaluating {args.checkpoint} for {args.episodes} episodes...")
    results = evaluate_policy(actor, env, num_episodes=args.episodes, deterministic=args.deterministic)

    print(f"\nResults:")
    print(f"  Mean velocity: {results['mean_velocity']:.4f} +/- {results['std_velocity']:.4f} m/s")
    print(f"  Mean power:    {results['mean_power']:.4f} +/- {results['std_power']:.4f} W")
    print(f"  Mean efficiency (v/P): {results['mean_efficiency']:.4f}")
    print(f"  Mean reward:   {results['mean_reward']:.4f}")
    print(f"  Mean length:   {results['mean_length']:.0f} steps")

    output_dir = Path(args.output_dir) / "zheng2022"
    plot_results(results, output_dir)

    # Save raw results
    torch.save(results, output_dir / "eval_results.pt")
    print(f"Results saved to {output_dir / 'eval_results.pt'}")


if __name__ == "__main__":
    main()
