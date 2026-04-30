"""Validate surrogate accuracy against real DisMech environment.

Compares multi-step surrogate rollouts against ground-truth trajectories
from DisMech's implicit Euler solver.

Usage:
    python -m aprx_model_dismech validate --surrogate-checkpoint output/surrogate_dismech
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

from src.physics.snake_robot import SnakeRobot
from src.configs.physics import DismechConfig
from aprx_model_dismech.train_config import SurrogateModelConfig
from aprx_model_dismech.model import SurrogateModel
from aprx_model_dismech.state import (
    RodState2D,
    StateNormalizer,
    action_to_omega,
    encode_phase,
    encode_n_cycles,
    raw_to_relative,
    relative_to_raw,
    REL_STATE_DIM,
    POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z,
)
from aprx_model_dismech.collect_data import _generate_serpenoid_curvatures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate DisMech surrogate accuracy")
    parser.add_argument(
        "--surrogate-checkpoint", type=str, required=True,
        help="Path to surrogate model directory",
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Episodes to evaluate")
    parser.add_argument("--steps-per-episode", type=int, default=100, help="Steps per episode")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--save-plots", type=str, default="figures/surrogate_dismech_validation",
                        help="Directory to save validation plots")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    return parser.parse_args()


def collect_real_trajectory(
    config: DismechConfig,
    rng: np.random.Generator,
    num_steps: int = 100,
):
    """Run one episode with random actions on DisMech, return states and actions."""
    snake_robot = SnakeRobot(config)
    serpenoid_time = 0.0

    states = [RodState2D.pack_from_dismech(snake_robot)]
    actions = []
    serp_times = [serpenoid_time]

    for _ in range(num_steps):
        action = rng.uniform(-1.0, 1.0, size=(5,)).astype(np.float32)
        actions.append(action)

        curvatures = _generate_serpenoid_curvatures(action, serpenoid_time)
        snake_robot.set_curvature_control(curvatures)
        snake_robot.step()
        serpenoid_time += config.dt

        states.append(RodState2D.pack_from_dismech(snake_robot))
        serp_times.append(serpenoid_time)

    return {
        "states": np.stack(states),
        "actions": np.stack(actions),
        "serpenoid_times": np.array(serp_times[:-1], dtype=np.float32),
    }


def rollout_surrogate(
    model: SurrogateModel,
    normalizer: StateNormalizer,
    initial_state: np.ndarray,
    actions: np.ndarray,
    serpenoid_times: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Autoregressively unroll surrogate for the given action sequence."""
    model.eval()
    states_raw = [initial_state.copy()]

    state_raw_t = torch.tensor(initial_state, dtype=torch.float32, device=device).unsqueeze(0)
    state = raw_to_relative(state_raw_t)

    with torch.no_grad():
        for t in range(len(actions)):
            action_np = actions[t]
            action = torch.tensor(action_np, dtype=torch.float32, device=device).unsqueeze(0)

            omega = action_to_omega(action_np)
            phase = omega * serpenoid_times[t]
            phase_enc = encode_phase(phase)
            ncycles_enc = encode_n_cycles(action_np)
            time_enc = torch.tensor(
                np.concatenate([phase_enc, ncycles_enc]),
                dtype=torch.float32, device=device,
            ).unsqueeze(0)

            state = model.predict_next_state(state, action, time_enc, normalizer)
            state_raw = relative_to_raw(state)
            states_raw.append(state_raw.squeeze(0).cpu().numpy())

    return np.stack(states_raw)


def compute_errors(real_states: np.ndarray, pred_states: np.ndarray) -> dict:
    """Compute per-step and cumulative errors between real and predicted trajectories."""
    T = min(len(real_states), len(pred_states))
    real = real_states[:T]
    pred = pred_states[:T]

    diff = pred - real
    per_step_rmse = np.sqrt(np.mean(diff ** 2, axis=1))

    # CoM drift
    real_com_x = real[:, POS_X].mean(axis=1)
    real_com_y = real[:, POS_Y].mean(axis=1)
    pred_com_x = pred[:, POS_X].mean(axis=1)
    pred_com_y = pred[:, POS_Y].mean(axis=1)
    com_drift = np.sqrt(
        (pred_com_x - real_com_x) ** 2 + (pred_com_y - real_com_y) ** 2
    )

    return {
        "per_step_rmse": per_step_rmse,
        "com_drift": com_drift,
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

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

    physics_config = DismechConfig()
    all_errors = []
    horizons = [10, 20, 50, 100]

    print(f"Validating DisMech surrogate on {args.num_episodes} episodes...")
    for ep in range(args.num_episodes):
        traj = collect_real_trajectory(physics_config, rng, num_steps=args.steps_per_episode)
        pred_states = rollout_surrogate(
            model, normalizer, traj["states"][0], traj["actions"],
            traj["serpenoid_times"], device=args.device,
        )
        errors = compute_errors(traj["states"], pred_states)
        all_errors.append(errors)

        T = len(traj["actions"])
        horizon_rmse = {h: errors["per_step_rmse"][min(h, T - 1)] for h in horizons if h < T}
        horizon_str = " | ".join(f"{h}:{v:.4f}" for h, v in horizon_rmse.items())
        print(f"  Episode {ep + 1}: T={T} | RMSE at steps: {horizon_str}")

    print("\n--- Aggregate Results ---")
    for h in horizons:
        rmses = [e["per_step_rmse"][h] for e in all_errors if h < len(e["per_step_rmse"])]
        if rmses:
            print(f"  Step {h:3d} RMSE:  {np.mean(rmses):.6f} +/- {np.std(rmses):.6f}")


if __name__ == "__main__":
    main()
