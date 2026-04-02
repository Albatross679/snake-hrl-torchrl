"""Diagnostic experiments for PPO follow_target failure analysis.

Systematically tests 6 hypotheses for why PPO cannot learn the follow_target task:
  H1: Reward sparsity (exp(-5*d) near zero at typical distances)
  H2: Integrative action space (delta curvature accumulates)
  H3: Mock physics limitations (damped dynamics)
  H4: 3D reachability (upper hemisphere unreachable)
  H5: Observation overload (148 dims for 8192-frame batches)
  H6: Network overparameterization (4x512 with small batches)

Usage:
    python -m choi2025.diagnose_ppo probe           # Run fast probes (EXP1-3)
    python -m choi2025.diagnose_ppo train --exp all  # Run all training experiments
    python -m choi2025.diagnose_ppo train --exp static_target
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import RewardSum
from torchrl.envs.transforms import ObservationNorm, Transform
from torchrl.data import Composite, Unbounded

from choi2025.config import (
    Choi2025EnvConfig,
    Choi2025PPOConfig,
    Choi2025PPONetworkConfig,
    TaskType,
)
from choi2025.env import SoftManipulatorEnv
from choi2025.rewards import compute_follow_target_reward
from choi2025.tasks import TargetGenerator
from src.configs import setup_run_dir, ConsoleLogger
from src.configs.base import resolve_device
from src.configs.network import ActorConfig, CriticConfig, NetworkConfig
from src.trainers.ppo import PPOTrainer


OUTPUT_DIR = Path("output/diagnostics")


def _ensure_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_json(data: dict, filename: str) -> Path:
    _ensure_output_dir()
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    return path


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ============================================================================
# EXP1: Reward Landscape Probe
# ============================================================================


def exp1_reward_landscape(rng: np.random.Generator) -> dict:
    """Probe reward landscape with random actions over 50 episodes."""
    print("\n" + "=" * 70)
    print("EXP1: Reward Landscape Probe")
    print("=" * 70)

    env_config = Choi2025EnvConfig(task=TaskType.FOLLOW_TARGET)
    env = SoftManipulatorEnv(env_config, device="cpu")

    all_rewards = []
    all_dists = []
    all_tip_pos = []
    all_target_pos = []

    n_episodes = 50
    for ep in range(n_episodes):
        td = env.reset()
        done = False
        while not done:
            action = torch.tensor(
                rng.uniform(-1, 1, size=env.action_spec.shape).astype(np.float32)
            )
            td["action"] = action
            td = env.step(td)["next"]

            reward = td["reward"].item()
            obs = td["observation"].numpy()
            # Tip position: obs indices 60:63 (positions[-1] for 21 nodes * 3)
            tip_pos = obs[60:63]
            # Target position: obs indices 145:148
            target_pos = obs[145:148]
            dist = np.linalg.norm(tip_pos - target_pos)

            all_rewards.append(reward)
            all_dists.append(dist)
            all_tip_pos.append(tip_pos.tolist())
            all_target_pos.append(target_pos.tolist())

            done = td["done"].item()

    rewards = np.array(all_rewards)
    dists = np.array(all_dists)

    # Reward histogram bins
    bins = [0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    hist_counts, _ = np.histogram(rewards, bins=bins)
    hist_fracs = hist_counts / len(rewards)

    # Alternative reward scales
    alt_exp2 = np.exp(-2.0 * dists)
    alt_exp1 = np.exp(-1.0 * dists)

    results = {
        "experiment": "exp1_reward_landscape",
        "n_episodes": n_episodes,
        "n_steps": len(rewards),
        "reward_stats": {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "median": float(np.median(rewards)),
        },
        "distance_stats": {
            "mean": float(np.mean(dists)),
            "std": float(np.std(dists)),
            "min": float(np.min(dists)),
            "max": float(np.max(dists)),
            "median": float(np.median(dists)),
        },
        "reward_histogram": {
            "bins": bins,
            "counts": hist_counts.tolist(),
            "fractions": hist_fracs.tolist(),
        },
        "reward_thresholds": {
            "frac_gt_0.01": float(np.mean(rewards > 0.01)),
            "frac_gt_0.05": float(np.mean(rewards > 0.05)),
            "frac_gt_0.1": float(np.mean(rewards > 0.1)),
        },
        "alternative_rewards": {
            "exp_neg2d": {
                "mean": float(np.mean(alt_exp2)),
                "std": float(np.std(alt_exp2)),
                "median": float(np.median(alt_exp2)),
            },
            "exp_neg1d": {
                "mean": float(np.mean(alt_exp1)),
                "std": float(np.std(alt_exp1)),
                "median": float(np.median(alt_exp1)),
            },
        },
    }

    env.close()

    print(f"  Steps collected: {len(rewards)}")
    print(f"  Reward: mean={results['reward_stats']['mean']:.6f}, "
          f"std={results['reward_stats']['std']:.6f}, "
          f"median={results['reward_stats']['median']:.6f}")
    print(f"  Distance: mean={results['distance_stats']['mean']:.4f}, "
          f"std={results['distance_stats']['std']:.4f}")
    print(f"  Fraction reward > 0.01: {results['reward_thresholds']['frac_gt_0.01']:.4f}")
    print(f"  Fraction reward > 0.05: {results['reward_thresholds']['frac_gt_0.05']:.4f}")
    print(f"  Fraction reward > 0.1:  {results['reward_thresholds']['frac_gt_0.1']:.4f}")
    print(f"  Alt exp(-2*d): mean={results['alternative_rewards']['exp_neg2d']['mean']:.6f}")
    print(f"  Alt exp(-1*d): mean={results['alternative_rewards']['exp_neg1d']['mean']:.6f}")

    return results


# ============================================================================
# EXP2: Action Impact Probe
# ============================================================================


def exp2_action_impact(rng: np.random.Generator) -> dict:
    """Compare zero-action vs random-action displacement and state change."""
    print("\n" + "=" * 70)
    print("EXP2: Action Impact Probe")
    print("=" * 70)

    env_config = Choi2025EnvConfig(task=TaskType.FOLLOW_TARGET)

    def _run_episodes(env, n_eps, action_fn):
        tip_displacements = []
        state_changes = []
        for _ in range(n_eps):
            td = env.reset()
            prev_obs = td["observation"].numpy()
            prev_tip = prev_obs[60:63].copy()
            done = False
            while not done:
                td["action"] = action_fn()
                td = env.step(td)["next"]
                obs = td["observation"].numpy()
                tip = obs[60:63]
                tip_displacements.append(float(np.linalg.norm(tip - prev_tip)))
                state_changes.append(float(np.linalg.norm(obs - prev_obs)))
                prev_obs = obs
                prev_tip = tip.copy()
                done = td["done"].item()
        return np.array(tip_displacements), np.array(state_changes)

    # Zero actions
    env_zero = SoftManipulatorEnv(env_config, device="cpu")
    action_dim = env_zero.action_spec.shape[0]
    zero_tip, zero_state = _run_episodes(
        env_zero, 10, lambda: torch.zeros(action_dim)
    )
    env_zero.close()

    # Random actions
    env_rand = SoftManipulatorEnv(env_config, device="cpu")
    rand_tip, rand_state = _run_episodes(
        env_rand, 10,
        lambda: torch.tensor(rng.uniform(-1, 1, size=(action_dim,)).astype(np.float32))
    )
    env_rand.close()

    # Avoid division by zero
    zero_tip_mean = float(np.mean(zero_tip))
    rand_tip_mean = float(np.mean(rand_tip))
    ratio = rand_tip_mean / zero_tip_mean if zero_tip_mean > 1e-10 else float("inf")

    results = {
        "experiment": "exp2_action_impact",
        "zero_action": {
            "n_episodes": 10,
            "n_steps": len(zero_tip),
            "tip_displacement": {
                "mean": zero_tip_mean,
                "std": float(np.std(zero_tip)),
            },
            "state_change": {
                "mean": float(np.mean(zero_state)),
                "std": float(np.std(zero_state)),
            },
        },
        "random_action": {
            "n_episodes": 10,
            "n_steps": len(rand_tip),
            "tip_displacement": {
                "mean": rand_tip_mean,
                "std": float(np.std(rand_tip)),
            },
            "state_change": {
                "mean": float(np.mean(rand_state)),
                "std": float(np.std(rand_state)),
            },
        },
        "displacement_ratio": ratio,
        "interpretation": (
            "actions_have_effect" if ratio > 2.0
            else "actions_have_minimal_effect" if ratio > 1.2
            else "actions_have_no_effect"
        ),
    }

    print(f"  Zero-action tip displacement: mean={zero_tip_mean:.6f}, "
          f"std={results['zero_action']['tip_displacement']['std']:.6f}")
    print(f"  Random-action tip displacement: mean={rand_tip_mean:.6f}, "
          f"std={results['random_action']['tip_displacement']['std']:.6f}")
    print(f"  Displacement ratio (random/zero): {ratio:.4f}")
    print(f"  Zero-action state change: mean={results['zero_action']['state_change']['mean']:.6f}")
    print(f"  Random-action state change: mean={results['random_action']['state_change']['mean']:.6f}")
    print(f"  Interpretation: {results['interpretation']}")

    return results


# ============================================================================
# EXP3: Reachability Analysis
# ============================================================================


def exp3_reachability(rng: np.random.Generator) -> dict:
    """Analyze target reachability by hemisphere with random-action rollouts."""
    print("\n" + "=" * 70)
    print("EXP3: Reachability Analysis")
    print("=" * 70)

    from choi2025.config import TargetConfig
    target_config = TargetConfig()
    target_gen = TargetGenerator(target_config, rng)

    env_config = Choi2025EnvConfig(task=TaskType.FOLLOW_TARGET)
    env = SoftManipulatorEnv(env_config, device="cpu")
    action_dim = env.action_spec.shape[0]

    # Sample 20 targets, 10 rollouts each, 100 steps
    n_targets = 20
    n_rollouts = 10
    n_steps = 100

    target_results = []
    for t_idx in range(n_targets):
        target_gen.sample(TaskType.FOLLOW_TARGET)
        target_pos = target_gen.position.copy()

        min_dist = float("inf")
        for _ in range(n_rollouts):
            td = env.reset()
            # Override target position in the env
            env._target.position = target_pos.copy()
            env._target._velocity = np.zeros(3)  # Keep it static for reachability

            for step in range(n_steps):
                action = torch.tensor(
                    rng.uniform(-1, 1, size=(action_dim,)).astype(np.float32)
                )
                td["action"] = action
                td = env.step(td)["next"]
                obs = td["observation"].numpy()
                tip = obs[60:63]
                d = float(np.linalg.norm(tip - target_pos))
                min_dist = min(min_dist, d)

                if td["done"].item():
                    break

        target_results.append({
            "target_pos": target_pos.tolist(),
            "min_dist": min_dist,
            "z": float(target_pos[2]),
        })

    env.close()

    # Analyze by hemisphere
    min_dists = np.array([r["min_dist"] for r in target_results])
    z_vals = np.array([r["z"] for r in target_results])

    def _stats(mask, name):
        if np.sum(mask) == 0:
            return {"n": 0, "frac_lt_0.3": 0.0, "frac_lt_0.1": 0.0, "frac_lt_0.05": 0.0}
        d = min_dists[mask]
        return {
            "n": int(np.sum(mask)),
            "mean_min_dist": float(np.mean(d)),
            "frac_lt_0.3": float(np.mean(d < 0.3)),
            "frac_lt_0.1": float(np.mean(d < 0.1)),
            "frac_lt_0.05": float(np.mean(d < 0.05)),
        }

    upper = z_vals > 0.15
    lower = z_vals < -0.15
    equatorial = np.abs(z_vals) <= 0.15

    results = {
        "experiment": "exp3_reachability",
        "n_targets": n_targets,
        "n_rollouts_per_target": n_rollouts,
        "n_steps_per_rollout": n_steps,
        "overall": {
            "frac_lt_0.3": float(np.mean(min_dists < 0.3)),
            "frac_lt_0.1": float(np.mean(min_dists < 0.1)),
            "frac_lt_0.05": float(np.mean(min_dists < 0.05)),
            "mean_min_dist": float(np.mean(min_dists)),
            "std_min_dist": float(np.std(min_dists)),
        },
        "by_hemisphere": {
            "upper": _stats(upper, "upper"),
            "lower": _stats(lower, "lower"),
            "equatorial": _stats(equatorial, "equatorial"),
        },
        "per_target": target_results,
    }

    print(f"  Targets sampled: {n_targets}")
    print(f"  Overall fraction within 0.3m: {results['overall']['frac_lt_0.3']:.4f}")
    print(f"  Overall fraction within 0.1m: {results['overall']['frac_lt_0.1']:.4f}")
    print(f"  Overall fraction within 0.05m: {results['overall']['frac_lt_0.05']:.4f}")
    print(f"  Mean min distance: {results['overall']['mean_min_dist']:.4f}")
    for hemi in ["upper", "lower", "equatorial"]:
        h = results["by_hemisphere"][hemi]
        if h["n"] > 0:
            print(f"  {hemi.capitalize()} (n={h['n']}): "
                  f"<0.3m={h['frac_lt_0.3']:.2f}, <0.1m={h['frac_lt_0.1']:.2f}")

    return results


# ============================================================================
# Training Experiments (EXP4-8)
# ============================================================================


def _make_training_env(env_config, device, obs_transform=None):
    """Create environment with ObservationNorm + RewardSum transforms."""
    env = SoftManipulatorEnv(env_config, device=device)

    # Normalize observations
    obs_norm = ObservationNorm(in_keys=["observation"], standard_normal=True)
    env = env.append_transform(obs_norm)
    obs_norm.init_stats(num_iter=200, reduce_dim=0)

    # Optional extra transform (e.g., ReducedObsTransform)
    if obs_transform is not None:
        env = env.append_transform(obs_transform)

    # Accumulate episode reward
    env = env.append_transform(RewardSum())

    return env


def _run_training_experiment(
    name: str,
    exp_num: int,
    env_config: Choi2025EnvConfig,
    network_config: NetworkConfig = None,
    obs_transform=None,
    reward_patch=None,
) -> dict:
    """Run a 200K-frame PPO training experiment."""
    print(f"\n{'=' * 70}")
    print(f"EXP{exp_num}: {name}")
    print("=" * 70)

    device = resolve_device("auto")

    out_dir = OUTPUT_DIR / f"exp{exp_num}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Choi2025PPOConfig(
        device=device,
        env=env_config,
        num_envs=1,
    )
    config.total_frames = 200_000
    config.max_wall_time = 1800  # 30 min safety
    config.frames_per_batch = 2048  # Smaller for single env
    config.mini_batch_size = 256

    if network_config is not None:
        config.network = network_config

    config.name = f"diag_exp{exp_num}_{name}"
    config.experiment_name = config.name

    # Setup run directory inside our diagnostics output
    run_dir = setup_run_dir(config, base_dir=str(out_dir))

    env = _make_training_env(env_config, device, obs_transform=obs_transform)

    try:
        # Optionally patch reward function
        ctx = (
            patch("choi2025.rewards.compute_follow_target_reward", reward_patch)
            if reward_patch is not None
            else _nullcontext()
        )
        with ctx:
            with ConsoleLogger(run_dir, config.console):
                trainer = PPOTrainer(
                    env=env,
                    config=config,
                    network_config=config.network,
                    device=device,
                    run_dir=run_dir,
                )
                print(f"  Training {config.total_frames} frames, device={device}")
                print(f"  Run dir: {run_dir}")
                results = trainer.train()
    finally:
        env.close()

    summary = {
        "experiment": f"exp{exp_num}_{name}",
        "best_reward": results.get("best_reward", 0.0),
        "final_mean_reward": results.get("final_mean_reward", 0.0),
        "total_episodes": results.get("total_episodes", 0),
        "run_dir": str(run_dir),
    }

    # Save results
    result_path = out_dir / "results.json"
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    print(f"  Done: episodes={summary['total_episodes']}, "
          f"best={summary['best_reward']:.4f}, "
          f"final_mean={summary['final_mean_reward']:.4f}")

    return summary


class _nullcontext:
    """Simple null context manager for Python 3.10 compatibility."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        return False


# --- Reduced observation transform for EXP7 ---

class ReducedObsTransform(Transform):
    """Replace 148-dim observation with 9 dims: tip_pos, tip_vel, target_pos.

    Indices in the original 148-dim observation (21 nodes, 3D):
      - positions: 0:63 (21*3), tip = positions[-1] = indices 60:63
      - velocities: 63:126 (21*3), tip_vel = velocities[-1] = indices 123:126
      - curvatures: 126:145 (19 bend springs)
      - target_pos: 145:148
    """

    def __init__(self):
        super().__init__(in_keys=["observation"], out_keys=["observation"])

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        # Handle batched or unbatched
        if obs.dim() == 1:
            tip_pos = obs[60:63]
            tip_vel = obs[123:126]
            target = obs[145:148]
            return torch.cat([tip_pos, tip_vel, target], dim=-1)
        else:
            tip_pos = obs[..., 60:63]
            tip_vel = obs[..., 123:126]
            target = obs[..., 145:148]
            return torch.cat([tip_pos, tip_vel, target], dim=-1)

    def _reset(self, tensordict, tensordict_reset):
        # Transform the observation in the reset output
        if "observation" in tensordict_reset.keys():
            tensordict_reset["observation"] = self._apply_transform(
                tensordict_reset["observation"]
            )
        return tensordict_reset

    def transform_observation_spec(self, observation_spec):
        """Override obs spec to reflect reduced dimension."""
        old_spec = observation_spec["observation"]
        observation_spec["observation"] = Unbounded(
            shape=(9,),
            dtype=old_spec.dtype,
            device=old_spec.device,
        )
        return observation_spec


def exp4_static_target() -> dict:
    """EXP4: Static target (target_speed=0)."""
    env_config = Choi2025EnvConfig(task=TaskType.FOLLOW_TARGET)
    env_config.target.target_speed = 0.0
    return _run_training_experiment("static_target", 4, env_config)


def exp5_dense_reward() -> dict:
    """EXP5: Dense reward with exp(-2*d) instead of exp(-5*d)."""
    env_config = Choi2025EnvConfig(task=TaskType.FOLLOW_TARGET)

    def dense_reward_wrapper(
        tip_pos, target_pos, prev_tip_pos, **kwargs,
    ):
        """Wrapper that uses exp(-2*d) instead of exp(-5*d)."""
        dist = np.linalg.norm(tip_pos - target_pos)
        dist_reward = float(np.exp(-2.0 * dist))

        # Keep other components the same
        return_components = kwargs.get("return_components", False)
        if return_components:
            return dist_reward, {
                "dist_to_goal": float(dist),
                "reward_dist": dist_reward,
                "reward_align": 0.0,
                "reward_pbrs": 0.0,
                "reward_improve": 0.0,
                "reward_smooth": 0.0,
            }
        return dist_reward

    return _run_training_experiment(
        "dense_reward", 5, env_config, reward_patch=dense_reward_wrapper
    )


def exp6_pbrs() -> dict:
    """EXP6: PBRS with gamma=0.99."""
    env_config = Choi2025EnvConfig(task=TaskType.FOLLOW_TARGET)
    env_config.pbrs_gamma = 0.99
    return _run_training_experiment("pbrs", 6, env_config)


def exp7_reduced_obs() -> dict:
    """EXP7: Reduced 9-dim observation with 2x64 network."""
    env_config = Choi2025EnvConfig(task=TaskType.FOLLOW_TARGET)

    # Small network for 9-dim input
    network_config = NetworkConfig(
        actor=ActorConfig(
            hidden_dims=[64, 64],
            activation="relu",
            ortho_init=True,
            init_gain=0.01,
            min_std=0.1,
            max_std=1.0,
            init_std=0.5,
        ),
        critic=CriticConfig(
            hidden_dims=[64, 64],
            activation="relu",
            ortho_init=True,
            init_gain=1.0,
        ),
    )

    return _run_training_experiment(
        "reduced_obs", 7, env_config,
        network_config=network_config,
        obs_transform=ReducedObsTransform(),
    )


def exp8_small_network() -> dict:
    """EXP8: Small 2x128 network with default obs and reward."""
    env_config = Choi2025EnvConfig(task=TaskType.FOLLOW_TARGET)

    network_config = NetworkConfig(
        actor=ActorConfig(
            hidden_dims=[128, 128],
            activation="relu",
            ortho_init=True,
            init_gain=0.01,
            min_std=0.1,
            max_std=1.0,
            init_std=0.5,
        ),
        critic=CriticConfig(
            hidden_dims=[128, 128],
            activation="relu",
            ortho_init=True,
            init_gain=1.0,
        ),
    )

    return _run_training_experiment(
        "small_network", 8, env_config, network_config=network_config
    )


# ============================================================================
# Subcommand Handlers
# ============================================================================


TRAIN_EXPERIMENTS = {
    "static_target": exp4_static_target,
    "dense_reward": exp5_dense_reward,
    "pbrs": exp6_pbrs,
    "reduced_obs": exp7_reduced_obs,
    "small_network": exp8_small_network,
}


def run_probes():
    """Run all 3 probe experiments (EXP1-3)."""
    rng = np.random.default_rng(42)

    print("=" * 70)
    print("PPO Follow-Target Diagnostic Probes")
    print("=" * 70)

    t0 = time.time()

    r1 = exp1_reward_landscape(rng)
    _save_json(r1, "exp1_reward_landscape.json")

    r2 = exp2_action_impact(rng)
    _save_json(r2, "exp2_action_impact.json")

    r3 = exp3_reachability(rng)
    _save_json(r3, "exp3_reachability.json")

    elapsed = time.time() - t0

    # Summary table
    print("\n" + "=" * 70)
    print("PROBE SUMMARY")
    print("=" * 70)
    print(f"  Time elapsed: {elapsed:.1f}s")
    print()
    print("  EXP1 Reward Landscape:")
    print(f"    Mean reward (exp(-5d)): {r1['reward_stats']['mean']:.6f}")
    print(f"    Mean distance: {r1['distance_stats']['mean']:.4f}m")
    print(f"    Frac reward > 0.01: {r1['reward_thresholds']['frac_gt_0.01']:.4f}")
    print(f"    Alt exp(-2d) mean: {r1['alternative_rewards']['exp_neg2d']['mean']:.6f}")
    print(f"    Alt exp(-1d) mean: {r1['alternative_rewards']['exp_neg1d']['mean']:.6f}")
    print()
    print("  EXP2 Action Impact:")
    print(f"    Zero-action displacement: {r2['zero_action']['tip_displacement']['mean']:.6f}")
    print(f"    Random-action displacement: {r2['random_action']['tip_displacement']['mean']:.6f}")
    print(f"    Ratio: {r2['displacement_ratio']:.4f}")
    print(f"    Verdict: {r2['interpretation']}")
    print()
    print("  EXP3 Reachability:")
    print(f"    Targets within 0.3m: {r3['overall']['frac_lt_0.3']:.2%}")
    print(f"    Targets within 0.1m: {r3['overall']['frac_lt_0.1']:.2%}")
    print(f"    Mean min distance: {r3['overall']['mean_min_dist']:.4f}m")
    print()
    print(f"  Results saved to {OUTPUT_DIR}/")


def run_train(exp_name: str):
    """Run training experiment(s)."""
    if exp_name == "all":
        results = {}
        for name, fn in TRAIN_EXPERIMENTS.items():
            results[name] = fn()

        # Comparison table
        print("\n" + "=" * 70)
        print("TRAINING EXPERIMENT COMPARISON")
        print("=" * 70)
        print(f"  {'Experiment':<20} {'Best Reward':>12} {'Final Mean':>12} {'Episodes':>10}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
        for name, r in results.items():
            print(f"  {name:<20} {r['best_reward']:>12.4f} {r['final_mean_reward']:>12.4f} "
                  f"{r['total_episodes']:>10}")
    else:
        fn = TRAIN_EXPERIMENTS[exp_name]
        fn()


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="PPO follow_target diagnostic experiments"
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # Probe subcommand
    probe_parser = subparsers.add_parser(
        "probe", help="Run fast probe experiments (EXP1-3, <5 min)"
    )

    # Train subcommand
    train_parser = subparsers.add_parser(
        "train", help="Run training experiments (EXP4-8)"
    )
    train_parser.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=list(TRAIN_EXPERIMENTS.keys()) + ["all"],
        help="Which training experiment to run",
    )

    args = parser.parse_args()

    if args.command == "probe":
        run_probes()
    elif args.command == "train":
        run_train(args.exp)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
