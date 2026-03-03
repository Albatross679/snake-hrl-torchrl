#!/usr/bin/env python
"""Record video of planar snake locomotion (Bing et al., IJCAI 2019).

Renders the MuJoCo environment via offscreen EGL and saves to MP4 (ffmpeg)
or GIF (Pillow).

Usage:
    # Zero actions (passive dynamics)
    python record_locomotion.py --task power_velocity --steps 500 \
        --output media/locomotion_passive.mp4

    # With trained policy
    python record_locomotion.py \
        --checkpoint checkpoints/locomotion_power_velocity/final.pt \
        --output media/locomotion_trained.mp4

    # GIF output
    python record_locomotion.py --steps 300 --output media/locomotion.gif
"""

import argparse
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from bing2019.configs_bing2019 import LocomotionEnvConfig, LocomotionNetworkConfig
from bing2019 import PlanarSnakeEnv
from configs.base import resolve_device
from networks.actor import create_actor


def parse_args():
    parser = argparse.ArgumentParser(description="Record locomotion video")
    parser.add_argument(
        "--task", type=str, default="power_velocity",
        choices=["power_velocity", "target_tracking"],
    )
    parser.add_argument(
        "--track", type=str, default="line",
        choices=["line", "wave", "zigzag", "circle", "random"],
    )
    parser.add_argument("--steps", type=int, default=500, help="Number of steps to record")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    parser.add_argument("--output", type=str, default="media/locomotion.mp4", help="Output path")
    parser.add_argument("--width", type=int, default=640, help="Render width")
    parser.add_argument("--height", type=int, default=480, help="Render height")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_actor(checkpoint_path, env, device):
    """Reconstruct actor network and load trained weights from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        env: PlanarSnakeEnv instance (needed for action_spec)
        device: Torch device

    Returns:
        TorchRL ProbabilisticActor with loaded weights
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct actor with same architecture used during training
    network_config = LocomotionNetworkConfig(device=device)
    obs_dim = env.observation_spec["observation"].shape[-1]

    actor = create_actor(
        obs_dim=obs_dim,
        action_spec=env.action_spec,
        config=network_config.actor,
        device=device,
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    print(f"  Frames trained: {checkpoint.get('total_frames', 'N/A')}")
    print(f"  Best reward:    {checkpoint.get('best_reward', 'N/A')}")

    return actor


def render_frame(renderer, data, camera):
    """Render a single RGB frame.

    Args:
        renderer: mujoco.Renderer
        data: mujoco.MjData
        camera: Camera object (mujoco.MjvCamera) or camera ID (int)
    """
    renderer.update_scene(data, camera=camera)
    return renderer.render().copy()  # (H, W, 3) uint8


def make_tracking_camera(env):
    """Create a free camera that looks down at the snake from above."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Position: look down from above
    cam.distance = 3.0
    cam.elevation = -60.0
    cam.azimuth = 90.0
    # Initial lookat = snake head position
    head_pos = env.data.geom_xpos[env._geom_head_id]
    cam.lookat[:] = head_pos
    return cam


def update_tracking_camera(cam, env):
    """Update camera lookat to follow the snake head smoothly."""
    head_pos = env.data.geom_xpos[env._geom_head_id]
    # Smooth tracking with exponential moving average
    alpha = 0.1
    for i in range(3):
        cam.lookat[i] = cam.lookat[i] * (1 - alpha) + head_pos[i] * alpha


def save_mp4(frames, output_path, fps):
    """Pipe frames to ffmpeg to create MP4."""
    h, w, _ = frames[0].shape
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        raise RuntimeError(f"ffmpeg failed (code {proc.returncode}): {stderr}")


def save_gif(frames, output_path, fps):
    """Save frames as GIF using Pillow."""
    from PIL import Image

    images = [Image.fromarray(f) for f in frames]
    duration_ms = int(1000 / fps)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # Create environment
    env_config = LocomotionEnvConfig(
        task=args.task,
        track_type=args.track,
        device=device,
    )
    env = PlanarSnakeEnv(config=env_config, device=device)
    env.set_seed(args.seed)

    # Create offscreen renderer
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)

    # Create tracking camera that follows the snake
    camera = make_tracking_camera(env)

    # Load actor if checkpoint provided
    actor = None
    if args.checkpoint is not None:
        actor = load_actor(args.checkpoint, env, device)
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Collect frames
    print(f"Recording {args.steps} steps (task={args.task})...")
    frames = []
    td = env.reset()

    for step in range(args.steps):
        # Get action
        if actor is not None:
            with torch.no_grad():
                td = actor(td)
                # Use mean action (deterministic) for evaluation
                td["action"] = td["loc"]
        else:
            td["action"] = torch.zeros(8, dtype=torch.float32, device=device)

        td = env.step(td)

        # Update camera to follow snake
        update_tracking_camera(camera, env)

        # Render
        frame = render_frame(renderer, env.data, camera)
        frames.append(frame)

        # Handle episode end
        td = td["next"]
        if td["done"].item():
            td = env.reset()

    renderer.close()
    env.close()

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.output.endswith(".gif"):
        save_gif(frames, args.output, args.fps)
    else:
        save_mp4(frames, args.output, args.fps)

    print(f"Saved {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
