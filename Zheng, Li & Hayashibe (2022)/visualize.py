"""MuJoCo viewer for trained underwater snake policies.

Renders the snake swimming in real-time or records video.
"""

import sys
import argparse
import time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from snake_hrl.networks.actor import create_actor
from snake_hrl.configs.network import ActorConfig

from configs import ZhengConfig
from env import UnderwaterSnakeEnv


def load_policy(checkpoint_path: str, config: ZhengConfig):
    """Load trained actor from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=config.device)

    actor_config = ActorConfig(
        hidden_dims=config.hidden_dims,
        activation=config.activation,
    )

    env = UnderwaterSnakeEnv(config=config, device=config.device)
    actor = create_actor(
        obs_dim=config.obs_dim,
        action_spec=env.action_spec,
        config=actor_config,
        device=config.device,
    )
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()
    return actor, env


def visualize_live(actor, env: UnderwaterSnakeEnv, num_episodes: int = 3):
    """Visualize policy with MuJoCo interactive viewer."""
    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(num_episodes):
            td = env.reset()
            step = 0
            done = False

            while not done and viewer.is_running():
                with torch.no_grad():
                    actor(td)
                    td["action"] = td["loc"]  # Deterministic (mean action)

                td = env.step(td)
                next_td = td["next"]
                step += 1
                done = next_td["done"].item()
                td = next_td

                viewer.sync()
                time.sleep(env.config.control_dt)  # Real-time playback

            if not viewer.is_running():
                break

            vx = td["head_velocity_x"].item()
            print(f"Episode {ep + 1}: {step} steps, final vx={vx:.4f} m/s")


def visualize_offscreen(
    actor, env: UnderwaterSnakeEnv, output_path: str = "swimming.mp4",
    num_steps: int = 500, width: int = 1280, height: int = 720,
):
    """Render policy to video file."""
    import mujoco
    import mediapy

    renderer = mujoco.Renderer(env.model, height=height, width=width)
    frames = []

    td = env.reset()
    for _ in range(num_steps):
        with torch.no_grad():
            actor(td)
            td["action"] = td["loc"]

        td = env.step(td)
        td = td["next"]

        renderer.update_scene(env.data, camera="tracking")
        frames.append(renderer.render())

    renderer.close()

    fps = int(1.0 / env.config.control_dt)  # 25 FPS
    mediapy.write_video(output_path, frames, fps=fps)
    print(f"Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize trained underwater snake")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--mode", choices=["live", "video"], default="live")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes (live mode)")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps (video mode)")
    parser.add_argument("--output", type=str, default="swimming.mp4", help="Output video path")
    parser.add_argument("--stiffness", type=float, default=0.0, help="Joint stiffness (Nm/rad)")
    args = parser.parse_args()

    config = ZhengConfig(joint_stiffness=args.stiffness)
    actor, env = load_policy(args.checkpoint, config)

    if args.mode == "live":
        visualize_live(actor, env, num_episodes=args.episodes)
    else:
        visualize_offscreen(actor, env, output_path=args.output, num_steps=args.steps)


if __name__ == "__main__":
    main()
