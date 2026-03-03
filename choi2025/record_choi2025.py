#!/usr/bin/env python
"""Record video of soft manipulator rollout (Choi & Tong, 2025).

Runs a trained SAC policy (or zero-action baseline) through the DisMech
environment, collects node positions at each step, and renders to MP4 or
GIF via matplotlib animation.

Usage:
    # Passive dynamics (zero action)
    python -m choi2025.record_choi2025 --steps 200 \
        --output media/manipulator_passive.mp4

    # With trained policy
    python -m choi2025.record_choi2025 \
        --checkpoint model/choi2025/best.pt \
        --task follow_target \
        --output media/manipulator_trained.mp4

    # GIF output, custom view angle
    python -m choi2025.record_choi2025 \
        --checkpoint model/choi2025/best.pt \
        --output media/manipulator.gif \
        --elevation 30 --azimuth 45

    # Multiple episodes stitched together
    python -m choi2025.record_choi2025 \
        --checkpoint model/choi2025/best.pt \
        --num-episodes 3 --output media/manipulator_3ep.mp4
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from choi2025.configs_choi2025 import Choi2025Config, Choi2025NetworkConfig, TaskType
from choi2025.env_choi2025 import SoftManipulatorEnv
from configs.base import resolve_device
from networks.actor import create_actor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record DisMech soft manipulator rollout"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="follow_target",
        choices=[t.value for t in TaskType],
        help="Task type",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to trained .pt checkpoint"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Number of episodes to record",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Max steps per episode (default: env max_episode_steps)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="media/manipulator.mp4",
        help="Output path (.mp4 or .gif)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    parser.add_argument(
        "--elevation", type=float, default=25.0, help="Camera elevation angle"
    )
    parser.add_argument(
        "--azimuth", type=float, default=45.0, help="Camera azimuth angle"
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_actor(checkpoint_path: str, env: SoftManipulatorEnv, device: str):
    """Reconstruct actor network and load trained weights.

    Args:
        checkpoint_path: Path to .pt checkpoint.
        env: Environment (needed for specs).
        device: Torch device.

    Returns:
        TorchRL ProbabilisticActor with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    network_config = Choi2025NetworkConfig()
    obs_dim = env.observation_spec["observation"].shape[-1]

    actor = create_actor(
        obs_dim=obs_dim,
        action_spec=env.action_spec,
        config=network_config.actor,
        device=device,
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    frames_trained = checkpoint.get("total_frames", "N/A")
    best_reward = checkpoint.get("best_reward", "N/A")
    print(f"  Loaded checkpoint: {checkpoint_path}")
    print(f"  Frames trained: {frames_trained}")
    print(f"  Best reward:    {best_reward}")

    return actor


def collect_rollout(env, actor, max_steps, device):
    """Run one episode and collect trajectory data.

    Args:
        env: SoftManipulatorEnv instance.
        actor: Trained actor (or None for zero actions).
        max_steps: Maximum steps per episode.
        device: Torch device.

    Returns:
        Dict with keys:
            positions: (T, num_nodes, 3) node positions per step
            targets: (T, 3) target positions per step
            rewards: (T,) reward per step
            tip_positions: (T, 3) tip position per step
    """
    td = env.reset()

    positions_list = []
    targets_list = []
    rewards_list = []
    tip_positions_list = []

    for step in range(max_steps):
        # Record state before action
        node_pos = env._get_positions()  # (num_nodes, 3)
        target_pos = env._target.position.copy()
        tip_pos = node_pos[-1].copy()

        positions_list.append(node_pos)
        targets_list.append(target_pos)
        tip_positions_list.append(tip_pos)

        # Select action
        if actor is not None:
            with torch.no_grad():
                actor(td)
                td["action"] = td["loc"]  # deterministic (mean)
        else:
            td["action"] = torch.zeros(
                env.action_spec.shape, dtype=torch.float32, device=device
            )

        # Step
        td = env.step(td)
        step_result = td["next"] if "next" in td.keys() else td
        rewards_list.append(step_result["reward"].item())

        if step_result["done"].item():
            break

        td = step_result

    return {
        "positions": np.array(positions_list),
        "targets": np.array(targets_list),
        "rewards": np.array(rewards_list),
        "tip_positions": np.array(tip_positions_list),
    }


def build_animation(
    rollouts,
    task,
    elevation,
    azimuth,
    fps,
    obstacles=None,
):
    """Build a matplotlib FuncAnimation from collected rollout data.

    Args:
        rollouts: List of rollout dicts from collect_rollout().
        task: TaskType enum.
        elevation: Camera elevation angle.
        azimuth: Camera azimuth angle.
        fps: Frames per second.
        obstacles: Optional ObstacleManager for rendering obstacles.

    Returns:
        (fig, ani) — matplotlib figure and FuncAnimation.
    """
    # Concatenate all episodes
    all_positions = np.concatenate([r["positions"] for r in rollouts], axis=0)
    all_targets = np.concatenate([r["targets"] for r in rollouts], axis=0)
    all_rewards = np.concatenate([r["rewards"] for r in rollouts], axis=0)
    all_tips = np.concatenate([r["tip_positions"] for r in rollouts], axis=0)

    n_frames = len(all_positions)
    num_nodes = all_positions.shape[1]

    # Episode boundaries for display
    ep_boundaries = []
    offset = 0
    for r in rollouts:
        ep_len = len(r["positions"])
        ep_boundaries.append((offset, offset + ep_len))
        offset += ep_len

    # Compute axis limits from all data
    all_pts = np.concatenate([all_positions.reshape(-1, 3), all_targets], axis=0)
    margin = 0.15
    x_min, x_max = all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin
    y_min, y_max = all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin
    z_min, z_max = all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin

    # Create figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elevation, azim=azimuth)

    # Initial plot elements
    (rod_line,) = ax.plot([], [], [], "o-", color="#1f78b4", lw=2.5, ms=4, label="Rod")
    (tip_marker,) = ax.plot([], [], [], "s", color="#e31a1c", ms=8, label="Tip")
    (target_marker,) = ax.plot(
        [], [], [], "*", color="#33a02c", ms=14, label="Target"
    )
    # Tip trail (fading line showing recent tip trajectory)
    trail_len = 30
    (tip_trail,) = ax.plot([], [], [], "-", color="#e31a1c", alpha=0.3, lw=1)

    # Clamp marker (base node)
    (clamp_marker,) = ax.plot([], [], [], "D", color="#333333", ms=8, label="Clamp")

    # Obstacle spheres (drawn once if present)
    if obstacles is not None and len(obstacles.positions) > 0:
        for obs_pos, obs_r in zip(obstacles.positions, obstacles.radii):
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 15)
            xs = obs_r * np.outer(np.cos(u), np.sin(v)) + obs_pos[0]
            ys = obs_r * np.outer(np.sin(u), np.sin(v)) + obs_pos[1]
            zs = obs_r * np.outer(np.ones_like(u), np.cos(v)) + obs_pos[2]
            ax.plot_surface(xs, ys, zs, alpha=0.25, color="gray")

    # Text overlay
    info_text = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=9)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Soft Manipulator — {task.value}")
    ax.legend(loc="upper right", fontsize=8)

    def _find_episode(frame_idx):
        for ep_num, (start, end) in enumerate(ep_boundaries):
            if start <= frame_idx < end:
                return ep_num, frame_idx - start
        return len(ep_boundaries) - 1, 0

    def update(frame):
        pos = all_positions[frame]  # (num_nodes, 3)
        target = all_targets[frame]
        reward = all_rewards[frame]
        tip = all_tips[frame]

        # Rod
        rod_line.set_data(pos[:, 0], pos[:, 1])
        rod_line.set_3d_properties(pos[:, 2])

        # Tip
        tip_marker.set_data([tip[0]], [tip[1]])
        tip_marker.set_3d_properties([tip[2]])

        # Target
        target_marker.set_data([target[0]], [target[1]])
        target_marker.set_3d_properties([target[2]])

        # Clamp (first node)
        clamp_marker.set_data([pos[0, 0]], [pos[0, 1]])
        clamp_marker.set_3d_properties([pos[0, 2]])

        # Tip trail
        trail_start = max(0, frame - trail_len)
        trail = all_tips[trail_start : frame + 1]
        tip_trail.set_data(trail[:, 0], trail[:, 1])
        tip_trail.set_3d_properties(trail[:, 2])

        # Info text
        ep_num, ep_step = _find_episode(frame)
        dist = np.linalg.norm(tip - target)
        info_text.set_text(
            f"Episode {ep_num + 1}/{len(ep_boundaries)}  "
            f"Step {ep_step + 1}  "
            f"Reward {reward:+.3f}  "
            f"Tip-Target {dist:.3f}m"
        )

        return rod_line, tip_marker, target_marker, clamp_marker, tip_trail, info_text

    interval_ms = 1000.0 / fps
    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)
    return fig, ani


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # Build config and environment
    config = Choi2025Config(seed=args.seed, device=device)
    config.env.task = TaskType(args.task)
    config.env.device = device

    env = SoftManipulatorEnv(config.env, device=device)
    env.set_seed(args.seed)

    max_steps = args.steps or config.env.max_episode_steps

    # Load actor
    actor = None
    if args.checkpoint is not None:
        actor = load_actor(args.checkpoint, env, device)
    else:
        print("No checkpoint provided — recording zero-action passive dynamics.")

    # Collect rollouts
    print(
        f"Recording {args.num_episodes} episode(s), "
        f"up to {max_steps} steps each (task={args.task})..."
    )
    rollouts = []
    total_reward = 0.0
    total_steps = 0

    for ep in range(args.num_episodes):
        rollout = collect_rollout(env, actor, max_steps, device)
        ep_reward = rollout["rewards"].sum()
        ep_len = len(rollout["rewards"])
        total_reward += ep_reward
        total_steps += ep_len
        rollouts.append(rollout)
        print(f"  Episode {ep + 1}: reward={ep_reward:.3f}, length={ep_len}")

    mean_reward = total_reward / args.num_episodes
    print(f"Mean reward: {mean_reward:.3f} over {args.num_episodes} episode(s)")

    # Get obstacle state from the last reset (for rendering)
    obstacles = env._obstacles if hasattr(env, "_obstacles") else None

    env.close()

    # Build animation
    print("Rendering animation...")
    fig, ani = build_animation(
        rollouts,
        task=TaskType(args.task),
        elevation=args.elevation,
        azimuth=args.azimuth,
        fps=args.fps,
        obstacles=obstacles,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".gif":
        ani.save(str(output_path), writer="pillow", fps=args.fps, dpi=args.dpi)
    else:
        ani.save(str(output_path), writer="ffmpeg", fps=args.fps, dpi=args.dpi)

    plt.close(fig)
    print(f"Saved {total_steps} frames to {output_path}")


if __name__ == "__main__":
    main()
