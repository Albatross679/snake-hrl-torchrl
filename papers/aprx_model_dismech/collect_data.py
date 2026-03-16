"""Collect training data from the DisMech physics environment.

Runs SnakeRobot (DisMech implicit Euler) with random CPG actions,
capturing (state, action, serpenoid_time) -> next_state transitions.

Each worker process runs 1 SnakeRobot. Parallelism = num_workers.

Usage:
    python -m aprx_model_dismech collect --num-transitions 100000
    python -m aprx_model_dismech collect --num-workers 24
"""

import argparse
import multiprocessing as mp
import os
import signal
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import numpy as np
import torch

# Guarded wandb import -- collection must not fail if wandb is unavailable
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False

from src.physics.snake_robot import SnakeRobot
from src.configs.physics import DismechConfig
from aprx_model_dismech.state import RodState2D, ACTION_DIM
from aprx_model_dismech.collect_config import DataCollectionConfig
from aprx_model_dismech.health import log_event, validate_episode_finite


def _generate_serpenoid_curvatures(
    action: np.ndarray,
    t: float,
    num_joints: int = 19,
    freq_range: tuple = (0.5, 3.0),
) -> np.ndarray:
    """Generate per-joint curvatures from CPG action parameters.

    Args:
        action: (5,) array [amplitude, frequency, wave_number, phase_offset, turn_bias]
                normalized in [-1, 1].
        t: Current serpenoid time (seconds).
        num_joints: Number of internal joints (num_segments - 1).
        freq_range: (min_freq, max_freq) in Hz.

    Returns:
        (num_joints,) array of target curvatures.
    """
    # Denormalize action components
    A = (float(np.clip(action[0], -1.0, 1.0)) + 1.0) / 2.0 * 5.0  # [0, 5] rad/m
    freq_norm = (float(np.clip(action[1], -1.0, 1.0)) + 1.0) / 2.0
    frequency = freq_range[0] + freq_norm * (freq_range[1] - freq_range[0])
    omega = 2.0 * np.pi * frequency
    wave_norm = (float(np.clip(action[2], -1.0, 1.0)) + 1.0) / 2.0
    wave_number = 0.5 + wave_norm * 3.0
    k = 2.0 * np.pi * wave_number
    phi = float(np.clip(action[3], -1.0, 1.0)) * np.pi
    turn_bias = float(np.clip(action[4], -1.0, 1.0)) * 2.0  # [-2, 2] rad/m

    # Joint arc-length positions uniformly spaced in [0, 1]
    s_j = np.linspace(0.0, 1.0, num_joints, dtype=np.float64)

    curvatures = A * np.sin(k * s_j + omega * t + phi) + turn_bias
    return curvatures.astype(np.float64)


def _generate_sobol_action(engine, dim: int = 5) -> np.ndarray:
    """Generate one Sobol quasi-random action in [-1, 1]^dim."""
    point = engine.random()  # (1, dim) in [0, 1]
    return (point[0] * 2.0 - 1.0).astype(np.float32)


def _generate_random_action(rng: np.random.Generator, dim: int = 5) -> np.ndarray:
    """Generate one uniform random action in [-1, 1]^dim."""
    return rng.uniform(-1.0, 1.0, size=(dim,)).astype(np.float32)


def _perturb_rod_state(snake_robot: SnakeRobot, config: DataCollectionConfig, rng: np.random.Generator) -> None:
    """Apply random perturbation to the rod state for diversity."""
    q = snake_robot._dismech_robot.state.q
    u = snake_robot._dismech_robot.state.u
    num_nodes = config.physics.num_segments + 1

    # Perturb positions (x, y only)
    pos_noise = rng.normal(0, config.perturb_position_std, size=(num_nodes, 2))
    for i in range(num_nodes):
        q[3 * i + 0] += pos_noise[i, 0]
        q[3 * i + 1] += pos_noise[i, 1]

    # Perturb velocities
    vel_noise = rng.normal(0, config.perturb_velocity_std, size=(num_nodes, 2))
    for i in range(num_nodes):
        u[3 * i + 0] += vel_noise[i, 0]
        u[3 * i + 1] += vel_noise[i, 1]

    # Apply initial curvature
    curv_amp = rng.uniform(0, config.perturb_curvature_max)
    curv = curv_amp * np.sin(np.linspace(0, 2 * np.pi, config.physics.num_segments - 1))
    snake_robot.set_curvature_control(curv)


def _collection_loop(
    worker_id: int,
    config: DataCollectionConfig,
    transitions_per_worker: int,
    result_queue: mp.Queue,
    shutdown_event: mp.Event,
    event_log_path: Path,
) -> None:
    """Worker loop: collect transitions from DisMech."""
    rng = np.random.default_rng(config.seed + worker_id)

    # Sobol engine for quasi-random actions
    sobol = None
    if config.use_sobol_actions:
        from scipy.stats.qmc import Sobol
        sobol = Sobol(d=ACTION_DIM, scramble=True, seed=config.seed + worker_id)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Batch accumulators
    batch_states = []
    batch_next_states = []
    batch_actions = []
    batch_t_starts = []
    batch_episode_ids = []
    batch_step_ids = []

    total_collected = 0
    episode_id = worker_id * 100_000  # unique episode IDs per worker
    batch_counter = 0

    log_event(event_log_path, "worker_status", "info", worker_id,
              {"status": "started", "transitions_target": transitions_per_worker})

    while total_collected < transitions_per_worker and not shutdown_event.is_set():
        # Create a fresh SnakeRobot
        snake_robot = SnakeRobot(config.physics)
        serpenoid_time = 0.0

        # Optionally perturb initial state
        if rng.random() < config.perturbation_fraction:
            _perturb_rod_state(snake_robot, config, rng)

        # Generate action for this run
        if sobol is not None and rng.random() > config.random_action_fraction:
            action = _generate_sobol_action(sobol)
        else:
            action = _generate_random_action(rng)

        # Pack state before step
        state_before = RodState2D.pack_from_dismech(snake_robot)

        # Apply serpenoid curvature and step
        curvatures = _generate_serpenoid_curvatures(action, serpenoid_time)
        snake_robot.set_curvature_control(curvatures)
        snake_robot.step()

        # Pack state after step
        state_after = RodState2D.pack_from_dismech(snake_robot)

        # Check for NaN/Inf
        ep_data = {"states": state_before[np.newaxis], "next_states": state_after[np.newaxis],
                    "actions": action[np.newaxis]}
        if not validate_episode_finite(ep_data):
            log_event(event_log_path, "nan_discard", "warn", worker_id,
                      {"episode_id": episode_id})
            episode_id += 1
            continue

        # Accumulate
        batch_states.append(state_before)
        batch_next_states.append(state_after)
        batch_actions.append(action)
        batch_t_starts.append(serpenoid_time)
        batch_episode_ids.append(episode_id)
        batch_step_ids.append(0)

        total_collected += 1
        episode_id += 1

        # Save batch periodically
        if len(batch_states) >= config.episodes_per_save:
            batch_data = {
                "states": torch.tensor(np.stack(batch_states), dtype=torch.float32),
                "next_states": torch.tensor(np.stack(batch_next_states), dtype=torch.float32),
                "actions": torch.tensor(np.stack(batch_actions), dtype=torch.float32),
                "t_start": torch.tensor(np.array(batch_t_starts), dtype=torch.float32),
                "episode_ids": torch.tensor(np.array(batch_episode_ids), dtype=torch.int64),
                "step_ids": torch.tensor(np.array(batch_step_ids), dtype=torch.int64),
            }

            batch_name = f"batch_{worker_id:04d}_{batch_counter:06d}.pt"
            tmp_path = save_dir / f".tmp_{batch_name}"
            final_path = save_dir / batch_name
            torch.save(batch_data, tmp_path)
            os.replace(tmp_path, final_path)

            batch_counter += 1
            batch_states.clear()
            batch_next_states.clear()
            batch_actions.clear()
            batch_t_starts.clear()
            batch_episode_ids.clear()
            batch_step_ids.clear()

            log_event(event_log_path, "worker_status", "info", worker_id,
                      {"status": "alive", "transitions": total_collected})

    # Save remaining
    if batch_states:
        batch_data = {
            "states": torch.tensor(np.stack(batch_states), dtype=torch.float32),
            "next_states": torch.tensor(np.stack(batch_next_states), dtype=torch.float32),
            "actions": torch.tensor(np.stack(batch_actions), dtype=torch.float32),
            "t_start": torch.tensor(np.array(batch_t_starts), dtype=torch.float32),
            "episode_ids": torch.tensor(np.array(batch_episode_ids), dtype=torch.int64),
            "step_ids": torch.tensor(np.array(batch_step_ids), dtype=torch.int64),
        }
        batch_name = f"batch_{worker_id:04d}_{batch_counter:06d}.pt"
        torch.save(batch_data, save_dir / batch_name)

    result_queue.put((worker_id, total_collected))
    log_event(event_log_path, "worker_status", "info", worker_id,
              {"status": "done", "transitions": total_collected})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect DisMech surrogate training data")
    parser.add_argument("--num-transitions", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = DataCollectionConfig()

    if args.num_transitions is not None:
        config.num_transitions = args.num_transitions
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.seed is not None:
        config.seed = args.seed

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = save_dir / "events.jsonl"

    print(f"Collecting {config.num_transitions:,} transitions with {config.num_workers} workers")
    print(f"Save dir: {save_dir}")
    print(f"Physics: DisMech (dt={config.physics.dt}s, implicit Euler)")

    shutdown_event = mp.Event()
    result_queue = mp.Queue()

    transitions_per_worker = config.num_transitions // config.num_workers

    def _signal_handler(signum, frame):
        print(f"\nSignal {signum} received, shutting down workers...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Launch workers
    workers = []
    for wid in range(config.num_workers):
        p = mp.Process(
            target=_collection_loop,
            args=(wid, config, transitions_per_worker, result_queue, shutdown_event, event_log_path),
        )
        p.start()
        workers.append(p)

    # Wait for completion
    total = 0
    for _ in range(config.num_workers):
        wid, count = result_queue.get()
        total += count
        print(f"  Worker {wid} done: {count:,} transitions")

    for p in workers:
        p.join()

    print(f"\nCollection complete: {total:,} transitions saved to {save_dir}")
    log_event(event_log_path, "shutdown_complete", "info", details={"total_transitions": total})


if __name__ == "__main__":
    main()
