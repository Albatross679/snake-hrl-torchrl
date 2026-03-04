#!/usr/bin/env python
"""Record video of snake locomotion rollout.

Top-down view (elevation=90) with ground grid, CoM trail, and heading arrow.
Supports MP4 and GIF output.

Usage:
    # Passive dynamics (zero action)
    python -m locomotion.record --steps 200 \
        --output media/locomotion_passive.mp4

    # With trained policy
    python -m locomotion.record \
        --checkpoint model/locomotion/best.pt \
        --gait forward \
        --output media/locomotion_forward.mp4

    # GIF output
    python -m locomotion.record \
        --checkpoint model/locomotion/best.pt \
        --gait turn_left \
        --output media/locomotion_turn.gif
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation

from locomotion.config import (
    GaitType,
    LocomotionConfig,
    LocomotionNetworkConfig,
)
from locomotion.env import LocomotionEnv
from src.networks.actor import create_actor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record snake locomotion rollout")
    parser.add_argument(
        "--gait",
        type=str,
        default="forward",
        choices=[g.value for g in GaitType],
        help="Locomotion gait type",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to trained .pt checkpoint"
    )
    parser.add_argument("--num-episodes", type=int, default=1, help="Episodes to record")
    parser.add_argument(
        "--steps", type=int, default=None, help="Max steps per episode (default: env max)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="media/locomotion.mp4",
        help="Output path (.mp4 or .gif)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    parser.add_argument(
        "--elevation", type=float, default=90.0, help="Camera elevation (90=top-down)"
    )
    parser.add_argument("--azimuth", type=float, default=0.0, help="Camera azimuth")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_actor(checkpoint_path: str, env: LocomotionEnv, device: str):
    """Reconstruct actor network and load trained weights."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    network_config = LocomotionNetworkConfig()
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

    Returns:
        Dict with keys:
            positions: (T, num_nodes, 3)
            com_trail: (T, 2)
            headings: (T, 2)
            rewards: (T,)
    """
    td = env.reset()

    positions_list = []
    com_trail = []
    headings_list = []
    rewards_list = []

    for step in range(max_steps):
        node_pos = env._get_positions()
        com_xy = node_pos[:, :2].mean(axis=0)
        heading = env._get_heading(node_pos)

        positions_list.append(node_pos)
        com_trail.append(com_xy)
        headings_list.append(heading)

        # Select action
        if actor is not None:
            with torch.no_grad():
                actor(td)
                td["action"] = td["loc"]  # deterministic
        else:
            td["action"] = torch.zeros(
                env.action_spec.shape, dtype=torch.float32, device=device
            )

        td = env.step(td)
        rewards_list.append(td["reward"].item())

        if td["done"].item():
            break

        td = td["next"]

    return {
        "positions": np.array(positions_list),
        "com_trail": np.array(com_trail),
        "headings": np.array(headings_list),
        "rewards": np.array(rewards_list),
    }


def build_animation(rollouts, gait, elevation, azimuth, fps, snake_radius):
    """Build matplotlib FuncAnimation with top-down ground view."""
    # Concatenate episodes
    all_positions = np.concatenate([r["positions"] for r in rollouts], axis=0)
    all_com = np.concatenate([r["com_trail"] for r in rollouts], axis=0)
    all_headings = np.concatenate([r["headings"] for r in rollouts], axis=0)
    all_rewards = np.concatenate([r["rewards"] for r in rollouts], axis=0)

    n_frames = len(all_positions)

    # Episode boundaries
    ep_boundaries = []
    offset = 0
    for r in rollouts:
        ep_len = len(r["positions"])
        ep_boundaries.append((offset, offset + ep_len))
        offset += ep_len

    # Axis limits from data
    all_xy = all_positions[:, :, :2].reshape(-1, 2)
    margin = 0.15
    x_min, x_max = all_xy[:, 0].min() - margin, all_xy[:, 0].max() + margin
    y_min, y_max = all_xy[:, 1].min() - margin, all_xy[:, 1].max() + margin

    # Enforce square aspect ratio
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half_span = max(x_max - x_min, y_max - y_min) / 2
    x_min, x_max = cx - half_span, cx + half_span
    y_min, y_max = cy - half_span, cy + half_span

    fig, ax = plt.subplots(figsize=(8, 8))

    # Ground grid
    grid_spacing = 0.1
    for x in np.arange(np.floor(x_min / grid_spacing) * grid_spacing,
                       x_max + grid_spacing, grid_spacing):
        ax.axvline(x, color="#e0e0e0", linewidth=0.5, zorder=0)
    for y in np.arange(np.floor(y_min / grid_spacing) * grid_spacing,
                       y_max + grid_spacing, grid_spacing):
        ax.axhline(y, color="#e0e0e0", linewidth=0.5, zorder=0)

    # Plot elements
    (rod_line,) = ax.plot([], [], "o-", color="#1f78b4", lw=3, ms=3, label="Snake", zorder=3)
    (head_marker,) = ax.plot([], [], "o", color="#e31a1c", ms=8, zorder=4, label="Head")
    (com_trail_line,) = ax.plot([], [], "-", color="#33a02c", alpha=0.5, lw=1.5,
                                 label="CoM trail", zorder=2)
    heading_arrow = ax.annotate(
        "", xy=(0, 0), xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color="#e31a1c", lw=2),
        zorder=5,
    )

    info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, fontsize=9,
                        verticalalignment="top", family="monospace",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Snake Locomotion — {gait.value}")
    ax.legend(loc="upper right", fontsize=8)

    trail_len = 100  # CoM trail history

    def _find_episode(frame_idx):
        for ep_num, (start, end) in enumerate(ep_boundaries):
            if start <= frame_idx < end:
                return ep_num, frame_idx - start
        return len(ep_boundaries) - 1, 0

    def update(frame):
        pos = all_positions[frame]  # (num_nodes, 3)
        com = all_com[frame]
        heading = all_headings[frame]
        reward = all_rewards[frame]

        # Snake body (XY projection)
        rod_line.set_data(pos[:, 0], pos[:, 1])

        # Head marker (last node)
        head_marker.set_data([pos[-1, 0]], [pos[-1, 1]])

        # CoM trail
        trail_start = max(0, frame - trail_len)
        trail = all_com[trail_start:frame + 1]
        com_trail_line.set_data(trail[:, 0], trail[:, 1])

        # Heading arrow from CoM
        arrow_len = 0.05
        heading_arrow.xy = (com[0] + arrow_len * heading[0],
                            com[1] + arrow_len * heading[1])
        heading_arrow.set_position((com[0], com[1]))

        # Info text
        ep_num, ep_step = _find_episode(frame)
        info_text.set_text(
            f"Episode {ep_num + 1}/{len(ep_boundaries)}  "
            f"Step {ep_step + 1}\n"
            f"Reward {reward:+.3f}  "
            f"CoM ({com[0]:.3f}, {com[1]:.3f})"
        )

        return rod_line, head_marker, com_trail_line, heading_arrow, info_text

    interval_ms = 1000.0 / fps
    ani = FuncAnimation(fig, update, frames=n_frames, interval=interval_ms, blit=False)
    return fig, ani


def main():
    args = parse_args()

    config = LocomotionConfig(seed=args.seed, device=args.device)
    config.env.gait = GaitType(args.gait)
    config.env.device = args.device

    env = LocomotionEnv(config.env, device=args.device)
    env.set_seed(args.seed)

    max_steps = args.steps or config.env.max_episode_steps

    # Load actor
    actor = None
    if args.checkpoint is not None:
        actor = load_actor(args.checkpoint, env, args.device)
    else:
        print("No checkpoint provided — recording zero-action passive dynamics.")

    # Collect rollouts
    print(
        f"Recording {args.num_episodes} episode(s), "
        f"up to {max_steps} steps each (gait={args.gait})..."
    )
    rollouts = []
    total_reward = 0.0
    total_steps = 0

    for ep in range(args.num_episodes):
        rollout = collect_rollout(env, actor, max_steps, args.device)
        ep_reward = rollout["rewards"].sum()
        ep_len = len(rollout["rewards"])
        total_reward += ep_reward
        total_steps += ep_len
        rollouts.append(rollout)
        print(f"  Episode {ep + 1}: reward={ep_reward:.3f}, length={ep_len}")

    mean_reward = total_reward / args.num_episodes
    print(f"Mean reward: {mean_reward:.3f} over {args.num_episodes} episode(s)")

    env.close()

    # Build animation
    print("Rendering animation...")
    fig, ani = build_animation(
        rollouts,
        gait=GaitType(args.gait),
        elevation=args.elevation,
        azimuth=args.azimuth,
        fps=args.fps,
        snake_radius=config.env.physics.snake_radius,
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
