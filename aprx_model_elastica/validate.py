"""Validate surrogate accuracy against real PyElastica environment.

Compares multi-step surrogate rollouts against ground-truth trajectories.
Reports per-step RMSE, trajectory drift, and per-component errors.

Usage:
    python -m aprx_model_elastica.validate --surrogate-checkpoint output/surrogate
    python -m aprx_model_elastica.validate --surrogate-checkpoint output/surrogate --num-episodes 20
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

from locomotion_elastica.config import LocomotionElasticaEnvConfig
from locomotion_elastica.env import LocomotionElasticaEnv
from aprx_model_elastica.train_config import SurrogateModelConfig
from aprx_model_elastica.model import SurrogateModel
from aprx_model_elastica.state import (
    RodState2D,
    StateNormalizer,
    action_to_omega,
    encode_phase,
    POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate surrogate accuracy")
    parser.add_argument(
        "--surrogate-checkpoint", type=str, required=True,
        help="Path to surrogate model directory",
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Episodes to evaluate")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--save-plots", type=str, default="figures/surrogate_validation",
                        help="Directory to save validation plots")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    return parser.parse_args()


def collect_real_trajectory(env: LocomotionElasticaEnv, rng: np.random.Generator):
    """Run one episode with random actions, return states and actions."""
    td = env.reset()
    states = [RodState2D.pack_from_rod(env._rod)]
    actions = []
    serp_times = [env._serpenoid._time]

    done = False
    while not done:
        action = rng.uniform(-1.0, 1.0, size=(5,)).astype(np.float32)
        actions.append(action)

        td_action = td.clone()
        td_action["action"] = torch.tensor(action, dtype=torch.float32)
        td = env._step(td_action)

        states.append(RodState2D.pack_from_rod(env._rod))
        serp_times.append(env._serpenoid._time)
        done = td["done"].item()

    return {
        "states": np.stack(states),                          # (T+1, 124)
        "actions": np.stack(actions),                        # (T, 5)
        "serpenoid_times": np.array(serp_times[:-1], dtype=np.float32),  # (T,)
    }


def rollout_surrogate(
    model: SurrogateModel,
    normalizer: StateNormalizer,
    initial_state: np.ndarray,
    actions: np.ndarray,
    serpenoid_times: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Autoregressively unroll surrogate for the given action sequence.

    Returns predicted states (T+1, 124) including initial state.
    """
    model.eval()
    states = [initial_state.copy()]
    state = torch.tensor(initial_state, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        for t in range(len(actions)):
            action_np = actions[t]
            action = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)

            # Encode oscillation phase omega*t (not raw t)
            omega = action_to_omega(action_np)
            phase = omega * serpenoid_times[t]
            time_enc = torch.tensor(
                encode_phase(phase),
                dtype=torch.float32, device=device,
            ).unsqueeze(0)

            state = model.predict_next_state(state, action, time_enc, normalizer)
            states.append(state.squeeze(0).cpu().numpy())

    return np.stack(states)


def compute_errors(real_states: np.ndarray, pred_states: np.ndarray) -> dict:
    """Compute per-step and cumulative errors between real and predicted trajectories."""
    T = min(len(real_states), len(pred_states))
    real = real_states[:T]
    pred = pred_states[:T]

    diff = pred - real
    per_step_rmse = np.sqrt(np.mean(diff ** 2, axis=1))  # (T,)

    # Per-component RMSE
    component_rmse = {
        "pos_x": np.sqrt(np.mean(diff[:, POS_X] ** 2, axis=1)),
        "pos_y": np.sqrt(np.mean(diff[:, POS_Y] ** 2, axis=1)),
        "vel_x": np.sqrt(np.mean(diff[:, VEL_X] ** 2, axis=1)),
        "vel_y": np.sqrt(np.mean(diff[:, VEL_Y] ** 2, axis=1)),
        "yaw": np.sqrt(np.mean(diff[:, YAW] ** 2, axis=1)),
        "omega_z": np.sqrt(np.mean(diff[:, OMEGA_Z] ** 2, axis=1)),
    }

    # CoM drift
    real_com_x = real[:, POS_X].mean(axis=1)
    real_com_y = real[:, POS_Y].mean(axis=1)
    pred_com_x = pred[:, POS_X].mean(axis=1)
    pred_com_y = pred[:, POS_Y].mean(axis=1)
    com_drift = np.sqrt(
        (pred_com_x - real_com_x) ** 2 + (pred_com_y - real_com_y) ** 2
    )

    # Heading drift
    real_hx = real[:, 20] - real[:, 0]   # head_x - tail_x
    real_hy = real[:, 41] - real[:, 21]  # head_y - tail_y
    pred_hx = pred[:, 20] - pred[:, 0]
    pred_hy = pred[:, 41] - pred[:, 21]
    real_heading = np.arctan2(real_hy, real_hx)
    pred_heading = np.arctan2(pred_hy, pred_hx)
    heading_drift = np.abs(
        (pred_heading - real_heading + np.pi) % (2 * np.pi) - np.pi
    )

    return {
        "per_step_rmse": per_step_rmse,
        "component_rmse": component_rmse,
        "com_drift": com_drift,
        "heading_drift": heading_drift,
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Load surrogate
    ckpt_dir = Path(args.surrogate_checkpoint)
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            model_config = SurrogateModelConfig(**json.load(f))
    else:
        model_config = SurrogateModelConfig()

    model = SurrogateModel(model_config).to(args.device)
    model.load_state_dict(
        torch.load(ckpt_dir / "model.pt", map_location=args.device, weights_only=True)
    )
    normalizer = StateNormalizer.load(str(ckpt_dir / "normalizer.pt"), device=args.device)

    # Create real environment
    env_config = LocomotionElasticaEnvConfig()
    env = LocomotionElasticaEnv(env_config, device="cpu")

    # Collect trajectories and compare
    all_errors = []
    horizons = [10, 50, 100, 200, 500]

    print(f"Validating surrogate on {args.num_episodes} episodes...")
    for ep in range(args.num_episodes):
        env._set_seed(args.seed + ep)
        traj = collect_real_trajectory(env, rng)

        pred_states = rollout_surrogate(
            model, normalizer,
            traj["states"][0],
            traj["actions"],
            traj["serpenoid_times"],
            device=args.device,
        )

        errors = compute_errors(traj["states"], pred_states)
        all_errors.append(errors)

        T = len(traj["actions"])
        horizon_rmse = {
            h: errors["per_step_rmse"][min(h, T - 1)] for h in horizons if h < T
        }
        horizon_str = " | ".join(f"{h}:{v:.4f}" for h, v in horizon_rmse.items())
        print(f"  Episode {ep + 1}: T={T} | RMSE at steps: {horizon_str}")

    # Aggregate statistics
    print("\n--- Aggregate Results ---")

    # Mean RMSE at different horizons
    for h in horizons:
        rmses = []
        for e in all_errors:
            if h < len(e["per_step_rmse"]):
                rmses.append(e["per_step_rmse"][h])
        if rmses:
            print(f"  Step {h:3d} RMSE:  {np.mean(rmses):.6f} ± {np.std(rmses):.6f}")

    # Mean CoM drift at different horizons
    print("\n  CoM drift (meters):")
    for h in horizons:
        drifts = []
        for e in all_errors:
            if h < len(e["com_drift"]):
                drifts.append(e["com_drift"][h])
        if drifts:
            print(f"    Step {h:3d}:  {np.mean(drifts):.6f} ± {np.std(drifts):.6f}")

    # Mean heading drift
    print("\n  Heading drift (radians):")
    for h in horizons:
        drifts = []
        for e in all_errors:
            if h < len(e["heading_drift"]):
                drifts.append(e["heading_drift"][h])
        if drifts:
            print(f"    Step {h:3d}:  {np.mean(drifts):.4f} ± {np.std(drifts):.4f}")

    # Per-component mean RMSE at step 50
    h = 50
    print(f"\n  Per-component RMSE at step {h}:")
    for comp in ["pos_x", "pos_y", "vel_x", "vel_y", "yaw", "omega_z"]:
        vals = []
        for e in all_errors:
            if h < len(e["component_rmse"][comp]):
                vals.append(e["component_rmse"][comp][h])
        if vals:
            print(f"    {comp:8s}: {np.mean(vals):.6f} ± {np.std(vals):.6f}")

    # Save plots if matplotlib available
    try:
        _save_plots(all_errors, horizons, args.save_plots)
    except ImportError:
        print("\nmatplotlib not available — skipping plot generation")

    env.close()


def _save_plots(all_errors: list, horizons: list, save_dir: str):
    """Save validation plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Plot 1: RMSE over time
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, e in enumerate(all_errors):
        ax.plot(e["per_step_rmse"], alpha=0.3, color="blue")
    # Mean
    max_len = max(len(e["per_step_rmse"]) for e in all_errors)
    mean_rmse = np.zeros(max_len)
    counts = np.zeros(max_len)
    for e in all_errors:
        n = len(e["per_step_rmse"])
        mean_rmse[:n] += e["per_step_rmse"]
        counts[:n] += 1
    mean_rmse = mean_rmse / np.maximum(counts, 1)
    ax.plot(mean_rmse, color="red", linewidth=2, label="Mean")
    ax.set_xlabel("Step")
    ax.set_ylabel("RMSE")
    ax.set_title("Surrogate vs Real: Per-Step RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path / "rmse_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: CoM drift over time
    fig, ax = plt.subplots(figsize=(10, 5))
    for e in all_errors:
        ax.plot(e["com_drift"], alpha=0.3, color="green")
    mean_drift = np.zeros(max_len)
    counts = np.zeros(max_len)
    for e in all_errors:
        n = len(e["com_drift"])
        mean_drift[:n] += e["com_drift"]
        counts[:n] += 1
    mean_drift = mean_drift / np.maximum(counts, 1)
    ax.plot(mean_drift, color="darkgreen", linewidth=2, label="Mean")
    ax.set_xlabel("Step")
    ax.set_ylabel("CoM Drift (m)")
    ax.set_title("Surrogate vs Real: Center-of-Mass Drift")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_path / "com_drift.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPlots saved to {save_path}/")


if __name__ == "__main__":
    main()
