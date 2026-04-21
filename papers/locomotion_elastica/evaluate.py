"""Evaluate a trained locomotion policy: load checkpoint, run episodes, report metrics.

Usage:
    python -m locomotion_elastica.evaluate --checkpoint output/.../checkpoints/best.pt
    python -m locomotion_elastica.evaluate --checkpoint output/.../checkpoints/best.pt --num-episodes 50
"""
import argparse
import sys
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs.utils import set_exploration_type, ExplorationType

from locomotion_elastica.config import (
    LocomotionElasticaConfig,
    LocomotionElasticaEnvConfig,
    GaitType,
)
from locomotion_elastica.env import LocomotionElasticaEnv
from src.networks.actor import create_actor


def evaluate(checkpoint_path: str, num_episodes: int = 20, deterministic: bool = True):
    """Load a trained actor and run evaluation episodes."""
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"Checkpoint: step={ckpt.get('total_frames')}, best_reward={ckpt.get('best_reward', 'N/A')}")

    # Create environment
    env_config = LocomotionElasticaEnvConfig(gait=GaitType.FORWARD, device="cpu")
    env = LocomotionElasticaEnv(env_config, device="cpu")

    # Create actor and load weights (use checkpoint's network config for correct architecture)
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_spec = env.action_spec
    actor_config = ckpt["config"].network.actor if "config" in ckpt else None
    actor = create_actor(obs_dim=obs_dim, action_spec=action_spec, config=actor_config, device="cpu")
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    # Run episodes
    all_rewards = []
    all_lengths = []
    all_final_dists = []
    all_displacements = []
    goal_reached = 0
    starvation = 0
    truncated = 0

    for ep in range(num_episodes):
        td = env.reset()
        episode_reward = 0.0
        step_count = 0
        initial_com = env._get_com(env._get_positions()).copy()

        done = False
        while not done:
            with torch.no_grad():
                td_in = TensorDict({"observation": td["observation"].unsqueeze(0)}, batch_size=[1])
                if deterministic:
                    with set_exploration_type(ExplorationType.DETERMINISTIC):
                        td_out = actor(td_in)
                else:
                    td_out = actor(td_in)
                td_action = td.clone()
                td_action["action"] = td_out["action"].squeeze(0)

            td_step = env.step(td_action)
            td = td_step["next"]
            episode_reward += td["reward"].item()
            step_count += 1
            done = td["done"].item()

        final_com = env._get_com(env._get_positions())
        final_dist = env._get_dist_to_goal(final_com)
        displacement = np.linalg.norm(final_com - initial_com)

        # Determine termination reason
        if final_dist <= env.config.goal.goal_radius:
            goal_reached += 1
            term_reason = "GOAL"
        elif env._no_progress_count >= env.config.goal.starvation_timeout:
            starvation += 1
            term_reason = "STARVATION"
        else:
            truncated += 1
            term_reason = "TRUNCATED"

        all_rewards.append(episode_reward)
        all_lengths.append(step_count)
        all_final_dists.append(final_dist)
        all_displacements.append(displacement)

        direction = final_com - initial_com
        print(f"  Ep {ep+1:2d}: reward={episode_reward:7.2f}, steps={step_count:3d}, "
              f"disp={displacement:.3f}m, final_dist={final_dist:.3f}m, "
              f"dir=({direction[0]:.3f},{direction[1]:.3f}), {term_reason}")

    # Summary
    goal_dist = env.config.goal.goal_distance
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY ({num_episodes} episodes, {'deterministic' if deterministic else 'stochastic'})")
    print(f"{'='*60}")
    print(f"  Mean reward:         {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
    print(f"  Mean episode length: {np.mean(all_lengths):.1f} +/- {np.std(all_lengths):.1f}")
    print(f"  Mean displacement:   {np.mean(all_displacements):.4f} m")
    print(f"  Mean final dist:     {np.mean(all_final_dists):.4f} m  (goal at {goal_dist}m)")
    print(f"  Goal reached:        {goal_reached}/{num_episodes} ({100*goal_reached/num_episodes:.1f}%)")
    print(f"  Starvation:          {starvation}/{num_episodes} ({100*starvation/num_episodes:.1f}%)")
    print(f"  Truncated:           {truncated}/{num_episodes} ({100*truncated/num_episodes:.1f}%)")

    # Movement analysis
    print(f"\n{'='*60}")
    print("MOVEMENT ANALYSIS")
    print(f"{'='*60}")
    mean_disp = np.mean(all_displacements)
    if mean_disp < 0.01:
        print(f"  *** Snake barely moves ({mean_disp:.4f}m mean displacement) ***")
    elif mean_disp < 0.1:
        print(f"  Snake moves slightly ({mean_disp:.4f}m) — marginal locomotion.")
    else:
        print(f"  Snake displaces {mean_disp:.3f}m on average — locomotion confirmed.")

    if goal_reached > 0:
        print(f"  Reaches {goal_dist}m goal in {goal_reached}/{num_episodes} episodes.")
    else:
        dist_reduction = goal_dist - np.mean(all_final_dists)
        pct = 100 * dist_reduction / goal_dist
        print(f"  Distance to goal reduced by {dist_reduction:.3f}m ({pct:.1f}% of {goal_dist}m).")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained snake locomotion policy")
    parser.add_argument("--checkpoint", type=str,
                        default="output/locomotion_elastica_forward_20260307_021959/checkpoints/best.pt",
                        help="Path to checkpoint")
    parser.add_argument("--num-episodes", type=int, default=20, help="Number of evaluation episodes")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic (sampled) actions")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.checkpoint, num_episodes=args.num_episodes, deterministic=not args.stochastic)
