#!/usr/bin/env python
"""Record video of planar snake locomotion (Bing et al., IJCAI 2019).

Renders the MuJoCo environment via offscreen EGL and saves to MP4 (ffmpeg)
or GIF (Pillow).

Usage:
    # Zero actions (passive dynamics)
    python -m bing2019.record_locomotion --task power_velocity --steps 500 \
        --output media/locomotion_power_velocity.mp4

    # With trained policy
    python -m bing2019.record_locomotion --task power_velocity \
        --checkpoint checkpoints/final.pt --output media/locomotion_trained.mp4

    # GIF output
    python -m bing2019.record_locomotion --task power_velocity --steps 300 \
        --output media/locomotion.gif
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from bing2019.configs_bing2019 import LocomotionEnvConfig
from bing2019 import PlanarSnakeEnv
from src.configs.base import resolve_device


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


def load_actor(checkpoint_path, device):
    """Load actor network from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def render_frame(renderer, data, camera_id=-1):
    """Render a single RGB frame from the top-down camera."""
    renderer.update_scene(data, camera=camera_id)
    return renderer.render().copy()  # (H, W, 3) uint8


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

    # Find top-down camera (or use default free camera)
    try:
        cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "top")
    except Exception:
        cam_id = -1  # Free camera

    # Load actor if checkpoint provided
    actor = None
    if args.checkpoint is not None:
        actor = load_actor(args.checkpoint, device)
        print(f"Loaded checkpoint: {args.checkpoint}")

    # Collect frames
    print(f"Recording {args.steps} steps (task={args.task})...")
    frames = []
    td = env.reset()

    for step in range(args.steps):
        # Get action
        if actor is not None:
            with torch.no_grad():
                # Assumes checkpoint has actor that takes observation
                obs = td["observation"]
                action = actor(obs)
        else:
            action = torch.zeros(8, dtype=torch.float32, device=device)

        td["action"] = action
        td = env.step(td)

        # Render
        frame = render_frame(renderer, env.data, camera_id=cam_id)
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
