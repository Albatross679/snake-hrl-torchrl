"""Collect training data from the real PyElastica environment.

Runs LocomotionElasticaEnv with random and/or trained policy actions,
capturing (state, action, serpenoid_time) -> next_state transitions
and force snapshots.

Each worker process runs 1 env. Parallelism = num_workers.

Health monitoring features:
- Per-worker progress tracking
- NaN/Inf episode filtering
- Atomic batch saves (tmp + os.replace)
- Graceful shutdown on SIGINT/SIGTERM
- Worker crash/stall detection and auto-respawn
- W&B alerts for critical events
- Structured JSONL event log

Usage:
    python -m aprx_model_elastica.collect_data --num-transitions 100000
    python -m aprx_model_elastica.collect_data --num-workers 24
    python -m aprx_model_elastica.collect_data --policy-checkpoint output/best.pt
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

# Guarded wandb import — collection must not fail if wandb is unavailable
try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _wandb = None
    _WANDB_AVAILABLE = False

from locomotion_elastica.config import LocomotionElasticaEnvConfig
from locomotion_elastica.env import LocomotionElasticaEnv
from aprx_model_elastica.state import RodState2D, ACTION_DIM
from aprx_model_elastica.health import log_event, validate_episode_finite


def _find_next_batch_idx(save_dir: Path, prefix: str, save_format: str) -> int:
    """Find the next available batch index for a given prefix.

    Scans save_dir for existing batch files matching the prefix pattern
    and returns max_existing_index + 1 so new files don't overwrite old ones.

    Args:
        save_dir: Directory containing batch files.
        prefix: File name prefix (e.g. "batch" or "batch_w00").
        save_format: "pt" or "parquet".

    Returns:
        Next available batch index (0 if no existing files).
    """
    import re

    ext = "parquet" if save_format == "parquet" else "pt"
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.{ext}$")

    max_idx = -1
    if save_dir.exists():
        for f in save_dir.iterdir():
            m = pattern.match(f.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))

    return max_idx + 1


def _find_max_episode_id(save_dir: Path) -> int:
    """Find the maximum episode ID across all existing batch files.

    Used to offset new episode IDs so they don't collide with existing data.

    Args:
        save_dir: Directory containing batch files.

    Returns:
        max_episode_id + 1, or 0 if no existing files.
    """
    max_eid = -1
    if not save_dir.exists():
        return 0

    for f in sorted(save_dir.glob("batch_*.pt")):
        try:
            data = torch.load(f, map_location="cpu", weights_only=True)
            file_max = data["episode_ids"].max().item()
            max_eid = max(max_eid, file_max)
        except Exception:
            continue

    for f in sorted(save_dir.glob("batch_*.parquet")):
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(f, columns=["episode_ids"])
            eids = table["episode_ids"].to_numpy()
            file_max = int(eids.max())
            max_eid = max(max_eid, file_max)
        except Exception:
            continue

    return max_eid + 1 if max_eid >= 0 else 0


class SobolActionSampler:
    """Quasi-random action sampler using scrambled Sobol sequences.

    Gives better space-filling coverage of the 5D action hypercube
    than uniform random sampling.
    """

    def __init__(self, action_dim: int = ACTION_DIM, seed: int = 42):
        self.engine = torch.quasirandom.SobolEngine(
            dimension=action_dim, scramble=True, seed=seed
        )

    def sample(self) -> np.ndarray:
        """Draw next quasi-random action in [-1, 1]^d."""
        point = self.engine.draw(1).squeeze(0).numpy()  # [0, 1]^d
        return (2.0 * point - 1.0).astype(np.float32)   # [-1, 1]^d


def perturb_rod_state(
    env: LocomotionElasticaEnv,
    rng: np.random.Generator,
    position_std: float = 0.002,
    velocity_std: float = 0.01,
    omega_std: float = 0.05,
    curvature_max: float = 3.0,
) -> None:
    """Add perturbations to rod state after reset for exploration.

    Perturbs positions/velocities/omega with Gaussian noise and sets
    an initial sinusoidal rest curvature so the surrogate trains on
    transitions starting from bent configurations (not just straight).

    Args:
        env: LocomotionElasticaEnv with initialized rod.
        rng: Random number generator.
        position_std: Gaussian noise std for node positions (meters).
        velocity_std: Gaussian noise std for node velocities (m/s).
        omega_std: Gaussian noise std for angular velocities (rad/s).
        curvature_max: Max amplitude for initial sinusoidal rest_kappa (rad/m).
            Serpenoid range is 0-5 rad/m; 3.0 is moderate. Set to 0 to skip.
    """
    rod = env._rod
    n_nodes = rod.n_elems + 1
    n_elems = rod.n_elems

    if position_std > 0:
        rod.position_collection[:2, :] += rng.normal(
            0, position_std, size=(2, n_nodes)
        ).astype(np.float64)

    if velocity_std > 0:
        rod.velocity_collection[:2, :] += rng.normal(
            0, velocity_std, size=(2, n_nodes)
        ).astype(np.float64)

    if omega_std > 0:
        rod.omega_collection[2, :] += rng.normal(
            0, omega_std, size=(n_elems,)
        ).astype(np.float64)

    # Initial curvature: randomized sinusoidal rest_kappa
    # Mimics mid-locomotion configurations the serpenoid controller produces
    if curvature_max > 0:
        n_kappa = rod.rest_kappa.shape[1]  # n_elems - 1 = 19
        joint_positions = np.linspace(0, 1, n_kappa)

        # Random serpenoid-like parameters
        amplitude = rng.uniform(0.0, curvature_max)
        wave_number = rng.uniform(0.5, 3.5)  # matches action range
        phase = rng.uniform(0.0, 2 * np.pi)
        turn_bias = rng.uniform(-1.0, 1.0)   # mild steering bias

        curvatures = (
            amplitude * np.sin(2 * np.pi * wave_number * joint_positions + phase)
            + turn_bias
        )
        rod.rest_kappa[0, :] = curvatures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect surrogate training data")
    parser.add_argument(
        "--num-transitions", type=int, default=None,
        help="Total transitions to collect",
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help="Number of parallel worker processes (1 env each)",
    )
    parser.add_argument(
        "--policy-checkpoint", type=str, default=None,
        help="Path to trained policy checkpoint (empty = random only)",
    )
    parser.add_argument(
        "--random-fraction", type=float, default=None,
        help="Fraction of episodes using random actions (vs policy)",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Output directory for batch files",
    )
    parser.add_argument(
        "--episodes-per-save", type=int, default=None,
        help="Save to disk every N episodes",
    )
    parser.add_argument(
        "--collect-forces", action="store_true", default=None,
        help="Collect force/torque data",
    )
    parser.add_argument(
        "--no-collect-forces", action="store_true", default=None,
        help="Skip force/torque data collection",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--save-format", type=str, default=None, choices=["pt", "parquet"],
        help="Output format: 'pt' (torch tensors) or 'parquet'",
    )
    parser.add_argument(
        "--sobol", action="store_true", default=None,
        help="Use Sobol quasi-random actions (better 5D coverage)",
    )
    parser.add_argument(
        "--no-sobol", action="store_true", default=None,
        help="Use uniform random actions instead of Sobol",
    )
    parser.add_argument(
        "--perturbation-fraction", type=float, default=None,
        help="Fraction of episodes with perturbed initial state (0=off)",
    )
    parser.add_argument(
        "--skip-disk-check", action="store_true", default=False,
        help="Skip the disk space pre-check",
    )
    parser.add_argument(
        "--baseline-fps", type=float, default=None,
        help="Expected FPS from smoke test (for schedule tracking in W&B)",
    )
    parser.add_argument(
        "--poll-interval", type=float, default=None,
        help="Seconds between health checks (default: 30)",
    )
    parser.add_argument(
        "--stall-intervals", type=int, default=None,
        help="Consecutive zero-progress polls before stall detection (default: 2)",
    )
    parser.add_argument(
        "--steps-per-run", type=int, default=None,
        help="env.step() calls per collection run (default: 4 → 2000 Elastica substeps)",
    )
    parser.add_argument(
        "--perturb-omega-std", type=float, default=None,
        help="Angular velocity perturbation std (default: 1.5 rad/s)",
    )
    parser.add_argument(
        "--flat-output", action="store_true", default=False,
        help="Save flat (states/next_states) format instead of substep_states format (Phase 02.2)",
    )
    return parser.parse_args()


def load_policy(checkpoint_path: str, device: str = "cpu"):
    """Load a trained actor network from checkpoint.

    Returns None if path is empty or file doesn't exist.
    """
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Extract actor from PPOTrainer checkpoint format
    if "actor_state_dict" in checkpoint:
        from src.networks.actor import build_actor
        from locomotion_elastica.config import LocomotionElasticaNetworkConfig

        net_config = LocomotionElasticaNetworkConfig()
        actor = build_actor(
            obs_dim=14,
            action_dim=5,
            config=net_config.actor,
        )
        actor.load_state_dict(checkpoint["actor_state_dict"])
        actor.eval()
        return actor
    return None


def collect_episode(
    env: LocomotionElasticaEnv,
    policy=None,
    use_random: bool = True,
    rng: np.random.Generator = None,
    collect_forces: bool = True,
    sobol_sampler: SobolActionSampler = None,
    perturb: bool = False,
    perturb_position_std: float = 0.002,
    perturb_velocity_std: float = 0.01,
    perturb_omega_std: float = 0.05,
    perturb_curvature_max: float = 3.0,
) -> dict:
    """Collect one episode of transitions.

    Args:
        env: LocomotionElasticaEnv instance.
        policy: Optional trained actor network.
        use_random: If True, use random actions regardless of policy.
        rng: Random number generator.
        collect_forces: If True, capture force/torque snapshots each step.
        sobol_sampler: If provided, use Sobol quasi-random actions for random episodes.
        perturb: If True, perturb rod state after reset.
        perturb_position_std: Gaussian noise std for node positions (meters).
        perturb_velocity_std: Gaussian noise std for node velocities (m/s).
        perturb_omega_std: Gaussian noise std for angular velocities (rad/s).
        perturb_curvature_max: Max amplitude for initial sinusoidal curvature (rad/m).

    Returns:
        Dict with arrays: states, actions, serpenoid_times, next_states,
        forces (if collect_forces), step_indices.
    """
    td = env.reset()

    # Optionally perturb rod state after reset for exploration
    if perturb:
        perturb_rod_state(
            env, rng,
            position_std=perturb_position_std,
            velocity_std=perturb_velocity_std,
            omega_std=perturb_omega_std,
            curvature_max=perturb_curvature_max,
        )

    states = []
    actions = []
    serpenoid_times = []
    next_states = []
    ext_forces = []
    int_forces = []
    ext_torques = []
    int_torques = []
    step_indices = []

    step = 0
    done = False
    while not done:
        # Capture pre-step state
        state = RodState2D.pack_from_rod(env._rod)
        serp_time = env._serpenoid._time

        # Select action
        if use_random or policy is None:
            if sobol_sampler is not None:
                action = sobol_sampler.sample()
            else:
                action = rng.uniform(-1.0, 1.0, size=(5,)).astype(np.float32)
            action_tensor = torch.tensor(action, dtype=torch.float32)
        else:
            with torch.no_grad():
                obs = td["observation"].unsqueeze(0)
                action_tensor = policy(obs).squeeze(0)
                # Add small exploration noise
                action_tensor = action_tensor + 0.1 * torch.randn_like(action_tensor)
                action_tensor = action_tensor.clamp(-1.0, 1.0)
            action = action_tensor.numpy()

        # Step the environment
        td_action = td.clone()
        td_action["action"] = action_tensor
        td = env._step(td_action)

        # Capture post-step state
        next_state = RodState2D.pack_from_rod(env._rod)

        # Store transition
        states.append(state)
        actions.append(action)
        serpenoid_times.append(serp_time)
        next_states.append(next_state)
        step_indices.append(step)

        # Optionally capture forces
        if collect_forces:
            forces = RodState2D.pack_forces(env._rod)
            ext_forces.append(forces["external_forces"])
            int_forces.append(forces["internal_forces"])
            ext_torques.append(forces["external_torques"])
            int_torques.append(forces["internal_torques"])

        done = td["done"].item()
        step += 1

        if done:
            break

    result = {
        "states": np.stack(states),              # (T, 124)
        "actions": np.stack(actions),             # (T, 5)
        "serpenoid_times": np.array(serpenoid_times, dtype=np.float32),  # (T,)
        "next_states": np.stack(next_states),     # (T, 124)
        "step_indices": np.array(step_indices, dtype=np.int64),  # (T,)
    }
    if collect_forces:
        result["forces"] = {
            "external_forces": np.stack(ext_forces),    # (T, 3, 21)
            "internal_forces": np.stack(int_forces),    # (T, 3, 21)
            "external_torques": np.stack(ext_torques),  # (T, 3, 20)
            "internal_torques": np.stack(int_torques),  # (T, 3, 20)
        }
    return result


def validate_run_finite(run_data: dict) -> bool:
    """Return True if all states in a checkpoint run are finite (no NaN/Inf).

    Args:
        run_data: dict returned by collect_checkpoint_run(), must contain
                  key 'substep_states' of shape (K+1, 124).

    Returns:
        True if all elements of substep_states are finite.
    """
    return bool(np.all(np.isfinite(run_data["substep_states"])))


def collect_checkpoint_run(
    env: LocomotionElasticaEnv,
    action: np.ndarray,
    steps_per_run: int = 4,
    collect_forces: bool = False,
) -> dict:
    """Collect one checkpoint run: call env.step(action) steps_per_run times.

    Each call to env.step() runs substeps_per_action=500 Elastica substeps internally.
    Returns states at each macro-step boundary: initial state + post-step states.

    The env must already be reset (and optionally perturbed) by the caller before
    calling this function. This function does NOT call env.reset().

    Args:
        env: LocomotionElasticaEnv with rod already reset and perturbed (caller's
             responsibility to call env.reset() before this function).
        action: (5,) action array normalized in [-1, 1]. Same action for all steps.
        steps_per_run: Number of env.step() calls (default 4 → 5 states, 4 pairs).
        collect_forces: If True, capture force/torque snapshots after each env._step().
                        Returns forces dict with (K, 3, 21/20) arrays. Default False
                        (backward compatible with Phase 02.1).

    Returns:
        dict with:
            substep_states: (steps_per_run + 1, 124) float32 — state at each boundary
                            (may be shorter if done_early=True)
            action:         (5,) float32 — same action used for all steps
            t_start:        float — env._serpenoid._time at run start
            done_early:     bool — True if env signaled done before steps_per_run calls
            forces:         (optional, only if collect_forces=True) dict with:
                                external_forces:  (K, 3, 21) float32
                                internal_forces:  (K, 3, 21) float32
                                external_torques: (K, 3, 20) float32
                                internal_torques: (K, 3, 20) float32
    """
    from tensordict import TensorDict

    t_start = float(env._serpenoid._time)
    action_tensor = torch.tensor(action, dtype=torch.float32, device=env._device)

    # Capture initial state (env already reset by caller)
    states = [RodState2D.pack_from_rod(env._rod)]
    done_early = False

    # Force accumulation lists (only populated when collect_forces=True)
    ext_forces: list = []
    int_forces: list = []
    ext_torques: list = []
    int_torques: list = []

    # Build a minimal TensorDict for _step — env._step only reads tensordict["action"],
    # so we pass a stub observation to satisfy any shape checks.
    td = TensorDict(
        {
            "action": action_tensor,
            "observation": torch.zeros(env.OBS_DIM, dtype=torch.float32, device=env._device),
        },
        batch_size=[],
        device=env._device,
    )

    for _ in range(steps_per_run):
        td["action"] = action_tensor
        td_result = env._step(td)
        states.append(RodState2D.pack_from_rod(env._rod))
        if collect_forces:
            f = RodState2D.pack_forces(env._rod)
            ext_forces.append(f["external_forces"])
            int_forces.append(f["internal_forces"])
            ext_torques.append(f["external_torques"])
            int_torques.append(f["internal_torques"])
        if td_result["done"].item():
            done_early = True
            break

    result = {
        "substep_states": np.stack(states, axis=0).astype(np.float32),  # (K+1, 124)
        "action": action.copy().astype(np.float32),                      # (5,)
        "t_start": t_start,
        "done_early": done_early,
    }
    if collect_forces:
        result["forces"] = {
            "external_forces": np.stack(ext_forces),    # (K, 3, 21)
            "internal_forces": np.stack(int_forces),    # (K, 3, 21)
            "external_torques": np.stack(ext_torques),  # (K, 3, 20)
            "internal_torques": np.stack(int_torques),  # (K, 3, 20)
        }
    return result


# ---------------------------------------------------------------------------
# Collection loop (shared by single-process and worker modes)
# ---------------------------------------------------------------------------

def _collection_loop(
    config,
    rng: np.random.Generator,
    env: LocomotionElasticaEnv,
    policy,
    sobol_sampler,
    save_dir: Path,
    prefix: str = "batch",
    episode_id_offset: int = 0,
    shared_counter=None,
    worker_counter=None,
    shared_nan_counter=None,
    shutdown_event=None,
    worker_id: int = -1,
    wb_run=None,
):
    """Run the checkpoint-format collection loop with a single env.

    Phase 2.1+ format: each "run" calls env.step(action) steps_per_run times,
    capturing state at each macro-step boundary. Replaces the episode-based
    loop (collect_episode). episodes_per_save now means "runs per save batch".

    Args:
        config: DataCollectionConfig.
        rng: Numpy random generator for this worker.
        env: Single LocomotionElasticaEnv instance.
        policy: Loaded policy (or None for random-only).
        sobol_sampler: SobolActionSampler or None.
        save_dir: Directory to save batch files.
        prefix: Batch file name prefix (e.g. "batch" or "batch_w00").
        episode_id_offset: Starting run ID offset for this worker.
        shared_counter: mp.Value for total transitions (or None for single-process).
        worker_counter: mp.Value for per-worker transitions (or None).
        shared_nan_counter: mp.Value for per-worker NaN discard count (or None).
        shutdown_event: mp.Event for graceful shutdown (or None).
        worker_id: Worker ID for logging (-1 = single-process).
        wb_run: Optional W&B run object for per-batch progress logging.
                Logging is non-fatal — any exception is silently caught.
    """
    log_prefix = f"[W{worker_id}] " if worker_id >= 0 else ""
    steps_per_run = config.steps_per_run

    total_transitions = 0
    total_transitions_saved = 0  # tracks transitions written to disk for W&B
    total_runs = 0
    batch_idx = _find_next_batch_idx(save_dir, prefix, "pt")
    collection_start_time = time.time()
    if batch_idx > 0:
        print(f"{log_prefix}Resuming: found existing files, starting at batch_idx={batch_idx}")

    # Checkpoint-format batch accumulators
    batch_substep_states: list = []   # each (K+1, 124)
    batch_actions: list = []          # each (5,)
    batch_t_starts: list = []         # each float
    batch_episode_ids: list = []      # each int (run id)
    batch_run_ids: list = []          # each int (global step within run sequence)

    # Flat-format batch accumulators (Phase 02.2 — only used when config.flat_output=True)
    # These are separate from the checkpoint-format lists above.
    flat_states: list = []            # each (124,) — pre-step state
    flat_next_states: list = []       # each (124,) — post-step state
    flat_actions: list = []           # each (5,)
    flat_t_starts: list = []          # each float
    flat_episode_ids: list = []       # each int
    flat_run_ids: list = []           # each int
    flat_forces: list = []            # each forces dict or None
    runs_in_batch = 0

    t_start_wall = time.monotonic()

    def _target_reached():
        if shared_counter is not None:
            return shared_counter.value >= config.num_transitions
        return total_transitions >= config.num_transitions

    def _should_stop():
        if shutdown_event is not None and shutdown_event.is_set():
            return True
        return _target_reached()

    while not _should_stop():
        # Reset env and optionally perturb rod state
        env.reset()
        perturb = (
            config.perturbation_fraction > 0
            and rng.random() < config.perturbation_fraction
        )
        if perturb:
            perturb_rod_state(
                env, rng,
                position_std=config.perturb_position_std,
                velocity_std=config.perturb_velocity_std,
                omega_std=config.perturb_omega_std,
                curvature_max=config.perturb_curvature_max,
            )

        # Sample action (Sobol or random)
        use_random = rng.random() < config.random_action_fraction or policy is None
        if use_random and sobol_sampler is not None:
            action = sobol_sampler.sample()
        else:
            action = rng.uniform(-1.0, 1.0, size=(5,)).astype(np.float32)

        # Collect one checkpoint run
        run_data = collect_checkpoint_run(
            env, action,
            steps_per_run=steps_per_run,
            collect_forces=config.collect_forces,
        )

        # Skip runs with done_early or non-finite states
        if run_data["done_early"] or not validate_run_finite(run_data):
            if shared_nan_counter is not None:
                with shared_nan_counter.get_lock():
                    shared_nan_counter.value += 1
            continue

        # Number of valid (state, action, next_state) pairs in this run
        n_pairs = len(run_data["substep_states"]) - 1  # typically 4

        batch_substep_states.append(run_data["substep_states"])
        batch_actions.append(run_data["action"])
        batch_t_starts.append(run_data["t_start"])
        batch_episode_ids.append(episode_id_offset + total_runs)
        batch_run_ids.append(episode_id_offset + total_runs)

        # Flat-format accumulation (Phase 02.2)
        if config.flat_output:
            ss = run_data["substep_states"]  # (K+1, 124)
            run_forces = run_data.get("forces")  # dict of (K, 3, 21/20) or None
            run_action = run_data["action"]
            run_t_start = run_data["t_start"]
            # total_runs not yet incremented at this point
            run_episode_id = episode_id_offset + total_runs
            for k in range(ss.shape[0] - 1):
                flat_states.append(ss[k])
                flat_next_states.append(ss[k + 1])
                flat_actions.append(run_action)
                flat_t_starts.append(run_t_start)
                flat_episode_ids.append(run_episode_id)
                flat_run_ids.append(run_episode_id)
                if run_forces is not None:
                    flat_forces.append({key: val[k] for key, val in run_forces.items()})
                else:
                    flat_forces.append(None)

        total_transitions += n_pairs
        total_runs += 1
        runs_in_batch += 1

        # Update shared counter (increment by pairs)
        if shared_counter is not None:
            with shared_counter.get_lock():
                shared_counter.value += n_pairs

        # Update per-worker counter
        if worker_counter is not None:
            with worker_counter.get_lock():
                worker_counter.value += n_pairs

        # Save batch to disk (episodes_per_save now means runs per save)
        if runs_in_batch >= config.episodes_per_save:
            if config.flat_output:
                _save_batch_flat(
                    save_dir=save_dir,
                    batch_idx=batch_idx,
                    batch_states=flat_states,
                    batch_next_states=flat_next_states,
                    batch_actions=flat_actions,
                    batch_t_starts=flat_t_starts,
                    batch_episode_ids=flat_episode_ids,
                    batch_run_ids=flat_run_ids,
                    batch_forces=flat_forces,
                    prefix=prefix,
                )
            else:
                _save_batch(
                    save_dir=save_dir,
                    batch_idx=batch_idx,
                    batch_substep_states=batch_substep_states,
                    batch_actions=batch_actions,
                    batch_t_starts=batch_t_starts,
                    batch_episode_ids=batch_episode_ids,
                    batch_run_ids=batch_run_ids,
                    prefix=prefix,
                )
            total_transitions_saved += n_pairs * runs_in_batch

            # W&B per-batch progress logging (non-fatal)
            if wb_run is not None:
                try:
                    gb_on_disk = sum(
                        os.path.getsize(os.path.join(str(save_dir), f))
                        for f in os.listdir(str(save_dir))
                        if f.endswith(".pt")
                    ) / (1024 ** 3)
                    elapsed_wb = time.time() - collection_start_time
                    throughput = total_transitions_saved / max(elapsed_wb, 1.0)
                    wb_run.log({
                        "total_transitions": total_transitions_saved,
                        "throughput_transitions_per_sec": throughput,
                        "gb_collected": gb_on_disk,
                        "batch_count": batch_idx,
                        "worker_count": config.num_workers,
                    })
                except Exception as _wb_exc:
                    print(f"[wandb] log failed (continuing): {_wb_exc}")

            batch_idx += 1
            batch_substep_states = []
            batch_actions = []
            batch_t_starts = []
            batch_episode_ids = []
            batch_run_ids = []
            flat_states = []
            flat_next_states = []
            flat_actions = []
            flat_t_starts = []
            flat_episode_ids = []
            flat_run_ids = []
            flat_forces = []
            runs_in_batch = 0

        # Progress report
        elapsed = time.monotonic() - t_start_wall
        fps = total_transitions / elapsed if elapsed > 0 else 0
        if total_runs % 50 == 0:
            if shared_counter is not None:
                global_total = shared_counter.value
                print(
                    f"{log_prefix}Runs: {total_runs:,} | "
                    f"Global: {global_total:,}/{config.num_transitions:,} | "
                    f"FPS: {fps:.0f}"
                )
            else:
                print(
                    f"Runs: {total_runs:,} | "
                    f"Transitions: {total_transitions:,}/{config.num_transitions:,} | "
                    f"FPS: {fps:.0f} | "
                    f"Elapsed: {elapsed:.0f}s"
                )

    # Save remaining batch
    if runs_in_batch > 0:
        if config.flat_output:
            _save_batch_flat(
                save_dir=save_dir,
                batch_idx=batch_idx,
                batch_states=flat_states,
                batch_next_states=flat_next_states,
                batch_actions=flat_actions,
                batch_t_starts=flat_t_starts,
                batch_episode_ids=flat_episode_ids,
                batch_run_ids=flat_run_ids,
                batch_forces=flat_forces,
                prefix=prefix,
            )
        else:
            _save_batch(
                save_dir=save_dir,
                batch_idx=batch_idx,
                batch_substep_states=batch_substep_states,
                batch_actions=batch_actions,
                batch_t_starts=batch_t_starts,
                batch_episode_ids=batch_episode_ids,
                batch_run_ids=batch_run_ids,
                prefix=prefix,
            )

    elapsed = time.monotonic() - t_start_wall
    print(
        f"{log_prefix}Done: {total_transitions:,} transitions "
        f"from {total_runs:,} runs in {elapsed:.0f}s "
        f"({total_transitions / max(1, elapsed):.0f} FPS)"
    )

    env.close()


# ---------------------------------------------------------------------------
# Worker function for multiprocess collection
# ---------------------------------------------------------------------------

def _worker_fn(
    worker_id, config, shared_counter, worker_counter, shared_nan_counter,
    shutdown_event, existing_episode_offset=0, seed_offset=0,
):
    """Worker process: create 1 env and run collection loop."""
    # Workers ignore SIGINT — only main process handles it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Ensure thread env vars are set (safety for forkserver)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Worker-specific seed (with offset for respawned workers)
    worker_seed = config.seed + worker_id * 137 + seed_offset
    rng = np.random.default_rng(worker_seed)

    save_dir = Path(config.save_dir)

    # Create single env for this worker
    env = LocomotionElasticaEnv(config.env, device="cpu")
    env._set_seed(worker_seed)

    # Load policy
    policy = load_policy(config.policy_checkpoint)

    # Sobol sampler with worker-specific seed for diverse coverage
    sobol_sampler = None
    if config.use_sobol_actions:
        sobol_sampler = SobolActionSampler(seed=config.seed + worker_id * 1000 + seed_offset)

    print(f"[W{worker_id}] Started, seed={worker_seed}")

    _collection_loop(
        config=config,
        rng=rng,
        env=env,
        policy=policy,
        sobol_sampler=sobol_sampler,
        save_dir=save_dir,
        prefix=f"batch_w{worker_id:02d}",
        episode_id_offset=existing_episode_offset + worker_id * 10_000_000,
        shared_counter=shared_counter,
        worker_counter=worker_counter,
        shared_nan_counter=shared_nan_counter,
        shutdown_event=shutdown_event,
        worker_id=worker_id,
    )


# ---------------------------------------------------------------------------
# Worker respawn and alerting helpers
# ---------------------------------------------------------------------------

def _respawn_worker(
    worker_id, config, shared_counter, worker_counter, nan_counter,
    shutdown_event, existing_episode_offset, respawn_count, save_dir,
):
    """Respawn a dead/stalled worker with a fresh seed."""
    # Clean up any .tmp files from the dead worker
    prefix = f"batch_w{worker_id:02d}"
    for tmp in save_dir.glob(f"{prefix}_*.tmp"):
        tmp.unlink(missing_ok=True)

    # Reset the worker's per-worker counter
    with worker_counter.get_lock():
        worker_counter.value = 0

    # New seed offset to avoid collision
    seed_offset = respawn_count * 7919

    p = mp.Process(
        target=_worker_fn,
        args=(worker_id, config, shared_counter, worker_counter, nan_counter,
              shutdown_event, existing_episode_offset, seed_offset),
        name=f"collector-{worker_id}-r{respawn_count}",
    )
    p.start()
    return p


def _send_alert(wandb_run, title: str, text: str, level: str = "WARN", wait_duration: int = 300):
    """Send W&B alert if wandb is active. Silently no-op otherwise."""
    if wandb_run is None:
        return
    try:
        from wandb import AlertLevel
        import wandb
        level_map = {"INFO": AlertLevel.INFO, "WARN": AlertLevel.WARN, "ERROR": AlertLevel.ERROR}
        wandb.alert(
            title=title[:63],
            text=text,
            level=level_map.get(level, AlertLevel.WARN),
            wait_duration=wait_duration,
        )
    except Exception:
        pass  # Never crash collection on alert failure


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Defaults from dataclass, CLI overrides take precedence
    from aprx_model_elastica.collect_config import DataCollectionConfig
    config = DataCollectionConfig()

    # CLI overrides
    if args.num_transitions is not None:
        config.num_transitions = args.num_transitions
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    if args.policy_checkpoint is not None:
        config.policy_checkpoint = args.policy_checkpoint
    if args.random_fraction is not None:
        config.random_action_fraction = args.random_fraction
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    if args.episodes_per_save is not None:
        config.episodes_per_save = args.episodes_per_save
    if args.seed is not None:
        config.seed = args.seed
    if args.no_collect_forces:
        config.collect_forces = False
    elif args.collect_forces:
        config.collect_forces = True
    if args.save_format is not None:
        config.save_format = args.save_format
    if args.no_sobol:
        config.use_sobol_actions = False
    elif args.sobol:
        config.use_sobol_actions = True
    if args.perturbation_fraction is not None:
        config.perturbation_fraction = args.perturbation_fraction
    if args.poll_interval is not None:
        config.poll_interval = args.poll_interval
    if args.stall_intervals is not None:
        config.stall_intervals = args.stall_intervals
    if args.steps_per_run is not None:
        config.steps_per_run = args.steps_per_run
    if args.perturb_omega_std is not None:
        config.perturb_omega_std = args.perturb_omega_std
    if args.flat_output:
        config.flat_output = True
    # Auto-set flat_output when collecting 1-step runs with forces (Phase 02.2 mode)
    if config.steps_per_run == 1 and config.collect_forces:
        config.flat_output = True

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Disk space pre-check
    if not args.skip_disk_check:
        disk_usage = shutil.disk_usage(save_dir)
        free_bytes = disk_usage.free
        # Estimate: 1 KB per transition, with 50% safety margin
        required_bytes = int(config.num_transitions * 1024 * 1.5)
        if free_bytes < required_bytes:
            free_gb = free_bytes / (1024 ** 3)
            required_gb = required_bytes / (1024 ** 3)
            print(
                f"ERROR: Not enough disk space.\n"
                f"  Free:     {free_gb:.1f} GB\n"
                f"  Required: {required_gb:.1f} GB "
                f"({config.num_transitions:,} transitions × 1 KB × 1.5 safety margin)\n"
                f"  Use --skip-disk-check to bypass this check."
            )
            sys.exit(1)

    # Print config summary
    policy = load_policy(config.policy_checkpoint)
    if policy is not None:
        print(f"Loaded policy from {config.policy_checkpoint}")
    else:
        print("No policy loaded — using random actions only")
    print(f"Force/torque collection: {'ON' if config.collect_forces else 'OFF'}")
    print(f"Save format: {config.save_format}")
    print(f"Action sampling: {'Sobol quasi-random' if config.use_sobol_actions else 'uniform random'}")
    print(
        f"State perturbation: {config.perturbation_fraction*100:.0f}% of episodes"
        f" (curvature_max={config.perturb_curvature_max} rad/m)"
    )

    if config.num_workers > 1:
        _multiprocess_collect(config, baseline_fps=args.baseline_fps)
    else:
        _single_process_collect(config, policy)


def _single_process_collect(config, policy):
    """Single-process collection (1 worker, 1 env)."""
    rng = np.random.default_rng(config.seed)
    save_dir = Path(config.save_dir)

    # Detect existing data for append mode
    existing_episode_offset = _find_max_episode_id(save_dir)
    if existing_episode_offset > 0:
        print(f"Append mode: found existing data (max episode_id={existing_episode_offset - 1})")

    env = LocomotionElasticaEnv(config.env, device="cpu")
    env._set_seed(config.seed)

    sobol_sampler = None
    if config.use_sobol_actions:
        sobol_sampler = SobolActionSampler(seed=config.seed)

    print("Single-process mode: 1 worker, 1 env")

    # W&B run for per-batch progress logging (non-fatal)
    wb_run = None
    if _WANDB_AVAILABLE:
        try:
            wb_run = _wandb.init(
                project="snake-surrogate-data",
                name="phase02.2-rl-step-collection",
                config={
                    "steps_per_run": config.steps_per_run,
                    "num_workers": config.num_workers,
                    "num_transitions": config.num_transitions,
                    "perturb_omega_std": config.perturb_omega_std,
                    "perturbation_fraction": config.perturbation_fraction,
                    "collect_forces": config.collect_forces,
                    "flat_output": config.flat_output,
                    "seed": config.seed,
                },
                resume="allow",
            )
            print(f"[wandb] Run initialized: {wb_run.url}")
        except Exception as _wb_init_exc:
            print(f"[wandb] init failed (continuing without logging): {_wb_init_exc}")
            wb_run = None

    _collection_loop(
        config=config,
        rng=rng,
        env=env,
        policy=policy,
        sobol_sampler=sobol_sampler,
        save_dir=save_dir,
        episode_id_offset=existing_episode_offset,
        wb_run=wb_run,
    )

    # Finalize W&B run
    if wb_run is not None:
        try:
            wb_run.finish()
        except Exception:
            pass

    print(f"Saved to {save_dir}/")


def _get_dir_size_gb(path: Path) -> float:
    """Get total size of a directory in GB."""
    total = 0
    if path.exists():
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    return total / (1024 ** 3)


def _multiprocess_collect(config, baseline_fps: float | None = None):
    """Multiprocess collection with health monitoring, respawn, and alerting."""
    try:
        mp.set_start_method("forkserver")
    except RuntimeError:
        pass  # already set

    # Detect existing data for append mode
    save_dir = Path(config.save_dir)
    existing_episode_offset = _find_max_episode_id(save_dir)
    if existing_episode_offset > 0:
        print(f"Append mode: found existing data (max episode_id={existing_episode_offset - 1})")

    shared_counter = mp.Value("l", 0)
    num_workers = config.num_workers

    # Per-worker health tracking
    worker_counters = [mp.Value("l", 0) for _ in range(num_workers)]
    nan_counters = [mp.Value("l", 0) for _ in range(num_workers)]
    shutdown_event = mp.Event()
    event_log_path = save_dir / "events.jsonl"

    print(f"Multiprocess mode: {num_workers} workers (1 env each)")

    # --- Signal handling for graceful shutdown ---
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    def _shutdown_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n{sig_name} received, requesting graceful shutdown...")
        shutdown_event.set()
        log_event(event_log_path, "shutdown_requested", "info", details={"signal": sig_name})

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    # --- W&B init (optional) ---
    wandb_run = None
    if _WANDB_AVAILABLE:
        try:
            wandb_run = _wandb.init(
                project="snake-surrogate-data",
                name="phase02.2-rl-step-collection",
                config={
                    "num_workers": config.num_workers,
                    "num_transitions": config.num_transitions,
                    "baseline_fps": baseline_fps,
                    "save_dir": config.save_dir,
                    "use_sobol_actions": config.use_sobol_actions,
                    "perturbation_fraction": config.perturbation_fraction,
                    "perturb_omega_std": config.perturb_omega_std,
                    "collect_forces": config.collect_forces,
                    "flat_output": config.flat_output,
                    "seed": config.seed,
                },
                resume="allow",
            )
            print(f"W&B run initialized: {wandb_run.url}")
        except Exception as e:
            print(f"WARNING: wandb init failed ({e}) — skipping W&B logging")
            wandb_run = None
    else:
        print("WARNING: wandb not installed — skipping W&B logging")

    log_event(event_log_path, "collection_started", "info",
              details={"num_workers": num_workers, "target": config.num_transitions})

    t_start = time.monotonic()

    workers = []
    for w in range(num_workers):
        p = mp.Process(
            target=_worker_fn,
            args=(w, config, shared_counter, worker_counters[w], nan_counters[w],
                  shutdown_event, existing_episode_offset, 0),
            name=f"collector-{w}",
        )
        p.start()
        workers.append(p)

    # --- Monitoring loop with health checks ---
    prev_worker_counts = [0] * num_workers
    stall_counts = [0] * num_workers
    respawn_counts = [0] * num_workers
    prev_count = 0
    prev_time = t_start

    try:
        while True:
            # Check termination conditions
            if shutdown_event.is_set():
                break
            if shared_counter.value >= config.num_transitions:
                time.sleep(2)  # let workers finish current episode
                break

            # Wait for next poll (responsive to shutdown)
            shutdown_event.wait(timeout=config.poll_interval)
            if shutdown_event.is_set():
                break

            now = time.monotonic()
            elapsed = now - t_start
            current_count = shared_counter.value

            # --- Per-worker health checks ---
            for i in range(num_workers):
                p = workers[i]
                wc = worker_counters[i]
                current_wc = wc.value
                delta = current_wc - prev_worker_counts[i]

                if p.is_alive():
                    # Only count stalls after worker has produced at least 1 transition
                    # (PyElastica initialization takes >60s)
                    if delta == 0 and current_wc > 0:
                        stall_counts[i] += 1
                    elif delta > 0:
                        stall_counts[i] = 0

                    # Stall detection
                    if stall_counts[i] >= config.stall_intervals:
                        log_event(event_log_path, "worker_stalled", "warn", worker_id=i,
                                  details={"stall_intervals": stall_counts[i], "pid": p.pid})
                        _send_alert(wandb_run, "Worker Stalled",
                                    f"Worker {i} (PID {p.pid}) stalled for {stall_counts[i]} intervals. Restarting.",
                                    level="WARN", wait_duration=config.alert_wait_duration)
                        p.terminate()
                        p.join(timeout=10)
                        respawn_counts[i] += 1
                        workers[i] = _respawn_worker(
                            i, config, shared_counter, worker_counters[i], nan_counters[i],
                            shutdown_event, existing_episode_offset, respawn_counts[i], save_dir)
                        stall_counts[i] = 0
                        log_event(event_log_path, "worker_respawned", "info", worker_id=i,
                                  details={"respawn_count": respawn_counts[i], "reason": "stall"})
                    else:
                        status = "alive"
                        log_event(event_log_path, "worker_status", "info", worker_id=i,
                                  details={"status": status, "transitions": current_wc, "delta": delta})
                else:
                    # Worker died
                    exitcode = p.exitcode
                    log_event(event_log_path, "worker_died", "error", worker_id=i,
                              details={"exitcode": exitcode, "pid": p.pid})
                    _send_alert(wandb_run, "Worker Died",
                                f"Worker {i} (PID {p.pid}) exited with code {exitcode}. Respawning.",
                                level="ERROR", wait_duration=config.alert_wait_duration)
                    respawn_counts[i] += 1
                    workers[i] = _respawn_worker(
                        i, config, shared_counter, worker_counters[i], nan_counters[i],
                        shutdown_event, existing_episode_offset, respawn_counts[i], save_dir)
                    stall_counts[i] = 0
                    log_event(event_log_path, "worker_respawned", "info", worker_id=i,
                              details={"respawn_count": respawn_counts[i], "reason": "death"})

                prev_worker_counts[i] = current_wc

            # --- NaN rate alerting ---
            total_nan = sum(nc.value for nc in nan_counters)
            total_episodes_approx = max(1, current_count // 500)
            nan_rate = total_nan / max(1, total_nan + total_episodes_approx)
            if nan_rate > config.nan_rate_threshold:
                _send_alert(wandb_run, "High NaN Rate",
                            f"NaN discard rate {nan_rate:.1%} exceeds threshold {config.nan_rate_threshold:.1%}. "
                            f"Total discards: {total_nan}",
                            level="WARN", wait_duration=config.alert_wait_duration * 2)
                log_event(event_log_path, "high_nan_rate", "warn",
                          details={"nan_rate": round(nan_rate, 4), "total_discards": total_nan})

            # --- Aggregate metrics ---
            interval_count = current_count - prev_count
            interval_time = now - prev_time

            fps_current = interval_count / max(1e-6, interval_time)
            fps_rolling = current_count / max(1e-6, elapsed)
            pct_complete = current_count / max(1, config.num_transitions) * 100
            remaining = config.num_transitions - current_count
            eta_hours = remaining / max(1, fps_rolling) / 3600

            disk_used_gb = _get_dir_size_gb(save_dir)
            disk_free_gb = shutil.disk_usage(save_dir).free / (1024 ** 3)

            # Schedule tracking
            schedule_delta_pct = 0.0
            if baseline_fps is not None and baseline_fps > 0:
                expected = baseline_fps * elapsed
                schedule_delta_pct = (current_count - expected) / max(1, expected) * 100

            # Log to stdout
            fps_baseline_str = f"  base={baseline_fps:.0f}" if baseline_fps else ""
            total_respawns = sum(respawn_counts)
            print(
                f"[monitor] {current_count:,}/{config.num_transitions:,} "
                f"({pct_complete:.1f}%) | "
                f"FPS cur={fps_current:.0f} roll={fps_rolling:.0f}{fps_baseline_str} | "
                f"ETA {eta_hours:.1f}h | "
                f"NaN={total_nan} respawns={total_respawns} | "
                f"disk {disk_used_gb:.1f}GB used, {disk_free_gb:.1f}GB free"
            )

            # Log to W&B — includes the 5 required Phase 02.2 metrics plus health metrics
            if wandb_run is not None:
                try:
                    metrics: dict[str, float] = {
                        # Phase 02.2 required metrics
                        "total_transitions": current_count,
                        "throughput_transitions_per_sec": fps_rolling,
                        "gb_collected": disk_used_gb,
                        "batch_count": int(current_count / max(1, config.episodes_per_save)),
                        "worker_count": config.num_workers,
                        # Health metrics
                        "fps_current": fps_current,
                        "fps_rolling": fps_rolling,
                        "pct_complete": pct_complete,
                        "eta_hours": eta_hours,
                        "disk_free_gb": disk_free_gb,
                        "nan_discards_total": total_nan,
                        "respawns_total": total_respawns,
                    }
                    if baseline_fps is not None:
                        metrics["fps_baseline"] = baseline_fps
                        metrics["schedule_delta_pct"] = schedule_delta_pct
                    wandb_run.log(metrics)
                except Exception:
                    pass

            prev_count = current_count
            prev_time = now

    finally:
        # Restore original signal handlers
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

        if shutdown_event.is_set():
            print("Graceful shutdown: waiting for workers to drain...")
            for p in workers:
                p.join(timeout=60)
            for p in workers:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=5)
        else:
            # Normal completion — wait for workers
            for p in workers:
                if p.is_alive():
                    p.join(timeout=30)

        log_event(event_log_path, "shutdown_complete", "info",
                  details={"total_transitions": shared_counter.value})

        # Finish W&B run
        if wandb_run is not None:
            try:
                import wandb as _wb
                _wb.finish()
            except Exception:
                pass

    elapsed = time.monotonic() - t_start
    total = shared_counter.value
    print(
        f"\nAll workers done! {total:,} transitions in {elapsed:.0f}s "
        f"({total / max(1, elapsed):.0f} FPS)"
    )
    print(f"Saved to {config.save_dir}/")


# ---------------------------------------------------------------------------
# Batch saving — Phase 2.1+ checkpoint format (with atomic writes)
# ---------------------------------------------------------------------------

def _save_batch(
    save_dir: Path,
    batch_idx: int,
    batch_substep_states: list,
    batch_actions: list,
    batch_t_starts: list,
    batch_episode_ids: list,
    batch_run_ids: list,
    prefix: str = "batch",
) -> None:
    """Save a batch of checkpoint runs to disk as a .pt file.

    Phase 2.1+ format: always .pt. Parquet not supported.

    Each run contributes one row to the batch tensors:
        substep_states: (N_runs, K+1, 124)  — states at each macro-step boundary
        actions:        (N_runs, 5)          — same action applied for all K steps
        t_start:        (N_runs,)            — serpenoid time at run start
        episode_ids:    (N_runs,)            — run identifier
        step_ids:       (N_runs,)            — same as episode_ids (alias for compat)

    Uses atomic write (tmp file + os.replace) to avoid partial files on crash.
    """
    n_runs = len(batch_substep_states)
    # Stack: each element is (K+1, 124) → (N_runs, K+1, 124)
    stacked_states = np.stack(batch_substep_states, axis=0)  # (N_runs, K+1, 124)
    stacked_actions = np.array(batch_actions, dtype=np.float32)  # (N_runs, 5)

    data = {
        "substep_states": torch.from_numpy(stacked_states),              # (N_runs, K+1, 124)
        "actions": torch.from_numpy(stacked_actions),                    # (N_runs, 5)
        "t_start": torch.tensor(batch_t_starts, dtype=torch.float32),   # (N_runs,)
        "episode_ids": torch.tensor(batch_episode_ids, dtype=torch.int64),  # (N_runs,)
        "step_ids": torch.tensor(batch_run_ids, dtype=torch.int64),     # (N_runs,)
    }

    path = save_dir / f"{prefix}_{batch_idx:04d}.pt"
    tmp_path = save_dir / f"{prefix}_{batch_idx:04d}.pt.tmp"
    torch.save(data, str(tmp_path))
    os.replace(str(tmp_path), str(path))
    n_pairs = n_runs * (stacked_states.shape[1] - 1)
    print(f"  Saved {n_runs:,} runs ({n_pairs:,} pairs) to {path}")


def _save_batch_flat(
    save_dir: Path,
    batch_idx: int,
    batch_states: list,
    batch_next_states: list,
    batch_actions: list,
    batch_t_starts: list,
    batch_episode_ids: list,
    batch_run_ids: list,
    batch_forces: list,
    prefix: str = "batch",
) -> None:
    """Save flat (state, action, next_state, forces) batch — Phase 1 consistent format.

    Phase 02.2+ format: each transition is stored as a flat (state, next_state) pair
    rather than the checkpoint substep_states format used in Phase 02.1.
    Forces dict is included when available (collect_forces=True).

    Args:
        save_dir: Output directory.
        batch_idx: Batch file index (used in filename).
        batch_states: List of (124,) ndarrays — pre-step rod states.
        batch_next_states: List of (124,) ndarrays — post-step rod states.
        batch_actions: List of (5,) ndarrays.
        batch_t_starts: List of float — serpenoid time at step start.
        batch_episode_ids: List of int — episode/run identifiers.
        batch_run_ids: List of int — step identifiers within run sequence.
        batch_forces: List of forces dicts (or None entries). If any element is not
                      None, forces are stacked and saved under the 'forces' key.
        prefix: Filename prefix (e.g. "batch" or "batch_w00").
    """
    data = {
        "states":      torch.from_numpy(np.stack(batch_states).astype(np.float32)),       # (N, 124)
        "next_states": torch.from_numpy(np.stack(batch_next_states).astype(np.float32)),  # (N, 124)
        "actions":     torch.from_numpy(np.array(batch_actions, dtype=np.float32)),       # (N, 5)
        "t_start":     torch.tensor(batch_t_starts, dtype=torch.float32),                 # (N,)
        "episode_ids": torch.tensor(batch_episode_ids, dtype=torch.int64),                # (N,)
        "step_ids":    torch.tensor(batch_run_ids, dtype=torch.int64),                    # (N,)
    }
    # Include forces if at least one entry is non-None
    if batch_forces and batch_forces[0] is not None:
        force_keys = ("external_forces", "internal_forces", "external_torques", "internal_torques")
        data["forces"] = {
            k: torch.from_numpy(np.stack([f[k] for f in batch_forces]).astype(np.float32))
            for k in force_keys
        }

    path = save_dir / f"{prefix}_{batch_idx:04d}.pt"
    tmp_path = path.with_suffix(".pt.tmp")
    torch.save(data, str(tmp_path))
    os.replace(str(tmp_path), str(path))
    print(f"  Saved {len(batch_states):,} flat transitions to {path}")


if __name__ == "__main__":
    main()
