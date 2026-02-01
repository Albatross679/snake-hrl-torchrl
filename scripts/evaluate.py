#!/usr/bin/env python
"""Evaluate trained snake HRL policies."""

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

from snake_hrl.envs import ApproachEnv, CoilEnv, HRLEnv
from snake_hrl.trainers import PPOTrainer, HRLTrainer
from snake_hrl.configs.env import ApproachEnvConfig, CoilEnvConfig, HRLEnvConfig
from snake_hrl.configs.training import PPOConfig, HRLConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained policies")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default="hrl",
        choices=["approach", "coil", "hrl"],
        help="Type of policy to evaluate",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes (if supported)",
    )
    parser.add_argument(
        "--save-videos",
        action="store_true",
        help="Save episode videos",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="./videos",
        help="Directory to save videos",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Evaluation device",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )
    return parser.parse_args()


def evaluate_approach(
    checkpoint_path: str,
    num_episodes: int,
    deterministic: bool,
    device: str,
) -> dict:
    """Evaluate approach skill policy."""
    print("Evaluating Approach Skill...")

    # Create environment and trainer
    env = ApproachEnv(device=device)
    trainer = PPOTrainer(env=env, device=device)
    trainer.load_checkpoint(checkpoint_path)

    # Run evaluation
    results = {
        "rewards": [],
        "lengths": [],
        "final_distances": [],
        "successes": [],
    }

    for episode in range(num_episodes):
        td = env.reset()
        episode_reward = 0.0
        episode_length = 0

        done = False
        while not done:
            with torch.no_grad():
                if deterministic:
                    trainer.actor(td)
                    td["action"] = td["loc"]
                else:
                    td = trainer.actor(td)

            td = env.step(td)
            episode_reward += td["reward"].item()
            episode_length += 1
            done = td["done"].item()

        results["rewards"].append(episode_reward)
        results["lengths"].append(episode_length)
        results["final_distances"].append(env.state["prey_distance"])
        results["successes"].append(float(env.is_success()))

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}")

    return results


def evaluate_coil(
    checkpoint_path: str,
    num_episodes: int,
    deterministic: bool,
    device: str,
) -> dict:
    """Evaluate coil skill policy."""
    print("Evaluating Coil Skill...")

    # Create environment and trainer
    env = CoilEnv(device=device)
    trainer = PPOTrainer(env=env, device=device)
    trainer.load_checkpoint(checkpoint_path)

    # Run evaluation
    results = {
        "rewards": [],
        "lengths": [],
        "contact_fractions": [],
        "wrap_counts": [],
        "successes": [],
    }

    for episode in range(num_episodes):
        td = env.reset()
        episode_reward = 0.0
        episode_length = 0

        done = False
        while not done:
            with torch.no_grad():
                if deterministic:
                    trainer.actor(td)
                    td["action"] = td["loc"]
                else:
                    td = trainer.actor(td)

            td = env.step(td)
            episode_reward += td["reward"].item()
            episode_length += 1
            done = td["done"].item()

        results["rewards"].append(episode_reward)
        results["lengths"].append(episode_length)
        results["contact_fractions"].append(env.state["contact_fraction"])
        results["wrap_counts"].append(abs(env.state["wrap_count"]))
        results["successes"].append(float(env.is_success()))

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}")

    return results


def evaluate_hrl(
    checkpoint_path: str,
    num_episodes: int,
    deterministic: bool,
    device: str,
) -> dict:
    """Evaluate hierarchical RL policy."""
    print("Evaluating Hierarchical RL Policy...")

    # Create trainer and load checkpoint
    trainer = HRLTrainer(device=device)
    trainer.load_checkpoint(checkpoint_path)

    # Run evaluation
    results = {
        "rewards": [],
        "lengths": [],
        "approach_successes": [],
        "coil_successes": [],
        "full_successes": [],
        "skill_switches": [],
    }

    for episode in range(num_episodes):
        td = trainer.hrl_env.reset()
        episode_reward = 0.0
        episode_length = 0
        num_switches = 0
        prev_skill = 0

        done = False
        while not done:
            with torch.no_grad():
                # Manager selects skill
                obs = td["observation"]
                logits = trainer.manager_actor(obs.unsqueeze(0))

                if deterministic:
                    skill_idx = logits.argmax().item()
                else:
                    probs = torch.softmax(logits, dim=-1)
                    skill_idx = torch.multinomial(probs, 1).item()

                if skill_idx != prev_skill:
                    num_switches += 1
                    prev_skill = skill_idx

                # Execute skill action
                if skill_idx == 0:
                    td = trainer.approach_trainer.actor(td)
                else:
                    td = trainer.coil_trainer.actor(td)

            td = trainer.hrl_env.step(td)
            episode_reward += td["reward"].item()
            episode_length += 1
            done = td["done"].item()

        results["rewards"].append(episode_reward)
        results["lengths"].append(episode_length)
        results["approach_successes"].append(float(trainer.hrl_env.approach_complete))
        results["coil_successes"].append(float(trainer.hrl_env.coil_complete))
        results["full_successes"].append(float(trainer.hrl_env.is_success()))
        results["skill_switches"].append(num_switches)

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes}")

    return results


def print_results(results: dict, policy_type: str) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print(f"Evaluation Results ({policy_type})")
    print("=" * 60)

    print(f"Episodes: {len(results['rewards'])}")
    print(f"Mean Reward: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
    print(f"Mean Length: {np.mean(results['lengths']):.1f} ± {np.std(results['lengths']):.1f}")
    print(f"Success Rate: {np.mean(results['successes']) * 100:.1f}%")

    if policy_type == "approach":
        print(f"Mean Final Distance: {np.mean(results['final_distances']):.3f}")
    elif policy_type == "coil":
        print(f"Mean Contact Fraction: {np.mean(results['contact_fractions']):.3f}")
        print(f"Mean Wrap Count: {np.mean(results['wrap_counts']):.2f}")
    elif policy_type == "hrl":
        print(f"Approach Success Rate: {np.mean(results['approach_successes']) * 100:.1f}%")
        print(f"Coil Success Rate: {np.mean(results['coil_successes']) * 100:.1f}%")
        print(f"Full Task Success Rate: {np.mean(results['full_successes']) * 100:.1f}%")
        print(f"Mean Skill Switches: {np.mean(results['skill_switches']):.1f}")


def save_results(results: dict, output_path: str) -> None:
    """Save results to JSON file."""
    import json

    # Convert numpy arrays to lists
    serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, list):
            serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in value]
        else:
            serializable[key] = value

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Snake HRL Policy Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Policy type: {args.policy_type}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print("=" * 60)

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return

    # Run evaluation based on policy type
    if args.policy_type == "approach":
        results = evaluate_approach(
            args.checkpoint,
            args.num_episodes,
            args.deterministic,
            args.device,
        )
    elif args.policy_type == "coil":
        results = evaluate_coil(
            args.checkpoint,
            args.num_episodes,
            args.deterministic,
            args.device,
        )
    else:  # hrl
        results = evaluate_hrl(
            args.checkpoint,
            args.num_episodes,
            args.deterministic,
            args.device,
        )

    # Print results
    if args.policy_type == "hrl":
        results["successes"] = results["full_successes"]
    print_results(results, args.policy_type)

    # Save results if output specified
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
