"""Collect training data from the real PyElastica environment.

Runs LocomotionElasticaEnv with random and/or trained policy actions,
capturing (state, action, serpenoid_time) -> next_state transitions
and force snapshots.

Each worker process runs 1 env. Parallelism = num_workers.

Usage:
    python -m aprx_model_elastica.collect_data --num-transitions 100000
    python -m aprx_model_elastica.collect_data --num-workers 24
    python -m aprx_model_elastica.collect_data --policy-checkpoint output/best.pt
"""

import argparse
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

from locomotion_elastica.config import LocomotionElasticaEnvConfig
from locomotion_elastica.env import LocomotionElasticaEnv
from aprx_model_elastica.state import RodState2D, ACTION_DIM


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
    worker_id: int = -1,
):
    """Run the episode collection loop with a single env.

    Args:
        config: DataCollectionConfig.
        rng: Numpy random generator for this worker.
        env: Single LocomotionElasticaEnv instance.
        policy: Loaded policy (or None for random-only).
        sobol_sampler: SobolActionSampler or None.
        save_dir: Directory to save batch files.
        prefix: Batch file name prefix (e.g. "batch" or "batch_w00").
        episode_id_offset: Starting episode ID for this worker.
        shared_counter: mp.Value for total transitions (or None for single-process).
        worker_id: Worker ID for logging (-1 = single-process).
    """
    collect_forces = config.collect_forces
    save_format = config.save_format
    log_prefix = f"[W{worker_id}] " if worker_id >= 0 else ""

    total_transitions = 0
    total_episodes = 0
    batch_idx = _find_next_batch_idx(save_dir, prefix, save_format)
    if batch_idx > 0:
        print(f"{log_prefix}Resuming: found existing files, starting at batch_idx={batch_idx}")
    batch_states = []
    batch_actions = []
    batch_serp_times = []
    batch_next_states = []
    batch_ext_forces = []
    batch_int_forces = []
    batch_ext_torques = []
    batch_int_torques = []
    batch_episode_ids = []
    batch_step_indices = []
    episodes_in_batch = 0

    t_start = time.monotonic()

    def _target_reached():
        if shared_counter is not None:
            return shared_counter.value >= config.num_transitions
        return total_transitions >= config.num_transitions

    while not _target_reached():
        use_random = rng.random() < config.random_action_fraction or policy is None
        perturb = (
            config.perturbation_fraction > 0
            and rng.random() < config.perturbation_fraction
        )

        ep_data = collect_episode(
            env, policy=policy, use_random=use_random, rng=rng,
            collect_forces=collect_forces,
            sobol_sampler=sobol_sampler if use_random else None,
            perturb=perturb,
            perturb_position_std=config.perturb_position_std,
            perturb_velocity_std=config.perturb_velocity_std,
            perturb_omega_std=config.perturb_omega_std,
            perturb_curvature_max=config.perturb_curvature_max,
        )
        n_steps = len(ep_data["states"])

        batch_states.append(ep_data["states"])
        batch_actions.append(ep_data["actions"])
        batch_serp_times.append(ep_data["serpenoid_times"])
        batch_next_states.append(ep_data["next_states"])
        if collect_forces:
            batch_ext_forces.append(ep_data["forces"]["external_forces"])
            batch_int_forces.append(ep_data["forces"]["internal_forces"])
            batch_ext_torques.append(ep_data["forces"]["external_torques"])
            batch_int_torques.append(ep_data["forces"]["internal_torques"])
        batch_episode_ids.append(
            np.full(n_steps, episode_id_offset + total_episodes, dtype=np.int64)
        )
        batch_step_indices.append(ep_data["step_indices"])

        total_transitions += n_steps
        total_episodes += 1
        episodes_in_batch += 1

        # Update shared counter
        if shared_counter is not None:
            with shared_counter.get_lock():
                shared_counter.value += n_steps

        # Save batch to disk
        if episodes_in_batch >= config.episodes_per_save:
            _save_batch(
                save_dir, batch_idx,
                batch_states, batch_actions, batch_serp_times,
                batch_next_states,
                batch_ext_forces, batch_int_forces,
                batch_ext_torques, batch_int_torques,
                batch_episode_ids, batch_step_indices,
                collect_forces=collect_forces,
                save_format=save_format,
                prefix=prefix,
            )
            batch_idx += 1
            batch_states, batch_actions, batch_serp_times = [], [], []
            batch_next_states = []
            batch_ext_forces, batch_int_forces = [], []
            batch_ext_torques, batch_int_torques = [], []
            batch_episode_ids, batch_step_indices = [], []
            episodes_in_batch = 0

        # Progress report
        elapsed = time.monotonic() - t_start
        fps = total_transitions / elapsed if elapsed > 0 else 0
        if total_episodes % 50 == 0:
            if shared_counter is not None:
                global_total = shared_counter.value
                print(
                    f"{log_prefix}Episodes: {total_episodes:,} | "
                    f"Global: {global_total:,}/{config.num_transitions:,} | "
                    f"FPS: {fps:.0f}"
                )
            else:
                print(
                    f"Episodes: {total_episodes:,} | "
                    f"Transitions: {total_transitions:,}/{config.num_transitions:,} | "
                    f"FPS: {fps:.0f} | "
                    f"Elapsed: {elapsed:.0f}s"
                )

    # Save remaining batch
    if episodes_in_batch > 0:
        _save_batch(
            save_dir, batch_idx,
            batch_states, batch_actions, batch_serp_times,
            batch_next_states,
            batch_ext_forces, batch_int_forces,
            batch_ext_torques, batch_int_torques,
            batch_episode_ids, batch_step_indices,
            collect_forces=collect_forces,
            save_format=save_format,
            prefix=prefix,
        )

    elapsed = time.monotonic() - t_start
    print(
        f"{log_prefix}Done: {total_transitions:,} transitions "
        f"from {total_episodes:,} episodes in {elapsed:.0f}s "
        f"({total_transitions / elapsed:.0f} FPS)"
    )

    env.close()


# ---------------------------------------------------------------------------
# Worker function for multiprocess collection
# ---------------------------------------------------------------------------

def _worker_fn(worker_id, config, shared_counter, existing_episode_offset=0):
    """Worker process: create 1 env and run collection loop."""
    # Ensure thread env vars are set (safety for forkserver)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Worker-specific seed
    worker_seed = config.seed + worker_id * 137
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
        sobol_sampler = SobolActionSampler(seed=config.seed + worker_id * 1000)

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
        worker_id=worker_id,
    )


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
        _multiprocess_collect(config)
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

    _collection_loop(
        config=config,
        rng=rng,
        env=env,
        policy=policy,
        sobol_sampler=sobol_sampler,
        save_dir=save_dir,
        episode_id_offset=existing_episode_offset,
    )

    print(f"Saved to {save_dir}/")


def _multiprocess_collect(config):
    """Multiprocess collection with shared transition counter."""
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

    print(f"Multiprocess mode: {num_workers} workers (1 env each)")

    t_start = time.monotonic()

    workers = []
    for w in range(num_workers):
        p = mp.Process(
            target=_worker_fn,
            args=(w, config, shared_counter, existing_episode_offset),
            name=f"collector-{w}",
        )
        p.start()
        workers.append(p)

    try:
        for p in workers:
            p.join()
    except KeyboardInterrupt:
        print("\nInterrupted — terminating workers...")
        for p in workers:
            p.terminate()
        for p in workers:
            p.join(timeout=5)

    elapsed = time.monotonic() - t_start
    total = shared_counter.value
    print(
        f"\nAll workers done! {total:,} transitions in {elapsed:.0f}s "
        f"({total / max(1, elapsed):.0f} FPS)"
    )
    print(f"Saved to {config.save_dir}/")


# ---------------------------------------------------------------------------
# Batch saving
# ---------------------------------------------------------------------------

def _save_batch(
    save_dir: Path,
    batch_idx: int,
    states, actions, serp_times, next_states,
    ext_forces, int_forces, ext_torques, int_torques,
    episode_ids, step_indices,
    collect_forces: bool = True,
    save_format: str = "pt",
    prefix: str = "batch",
):
    """Save a batch of transitions to disk."""
    cat_states = np.concatenate(states, axis=0)
    cat_actions = np.concatenate(actions, axis=0)
    cat_serp_times = np.concatenate(serp_times, axis=0)
    cat_next_states = np.concatenate(next_states, axis=0)
    cat_episode_ids = np.concatenate(episode_ids, axis=0)
    cat_step_indices = np.concatenate(step_indices, axis=0)
    n = cat_states.shape[0]

    if save_format == "parquet":
        _save_batch_parquet(
            save_dir, batch_idx, n,
            cat_states, cat_actions, cat_serp_times, cat_next_states,
            cat_episode_ids, cat_step_indices,
            ext_forces, int_forces, ext_torques, int_torques,
            collect_forces,
            prefix=prefix,
        )
    else:
        _save_batch_pt(
            save_dir, batch_idx, n,
            cat_states, cat_actions, cat_serp_times, cat_next_states,
            cat_episode_ids, cat_step_indices,
            ext_forces, int_forces, ext_torques, int_torques,
            collect_forces,
            prefix=prefix,
        )


def _save_batch_pt(
    save_dir, batch_idx, n,
    states, actions, serp_times, next_states,
    episode_ids, step_indices,
    ext_forces, int_forces, ext_torques, int_torques,
    collect_forces,
    prefix="batch",
):
    """Save batch as .pt (torch tensor) file."""
    data = {
        "states": torch.from_numpy(states),
        "actions": torch.from_numpy(actions),
        "serpenoid_times": torch.from_numpy(serp_times),
        "next_states": torch.from_numpy(next_states),
        "episode_ids": torch.from_numpy(episode_ids),
        "step_indices": torch.from_numpy(step_indices),
    }
    if collect_forces and ext_forces:
        data["forces"] = {
            "external_forces": torch.from_numpy(np.concatenate(ext_forces, axis=0)),
            "internal_forces": torch.from_numpy(np.concatenate(int_forces, axis=0)),
            "external_torques": torch.from_numpy(np.concatenate(ext_torques, axis=0)),
            "internal_torques": torch.from_numpy(np.concatenate(int_torques, axis=0)),
        }
    path = save_dir / f"{prefix}_{batch_idx:04d}.pt"
    torch.save(data, path)
    print(f"  Saved {n:,} transitions to {path}")


def _save_batch_parquet(
    save_dir, batch_idx, n,
    states, actions, serp_times, next_states,
    episode_ids, step_indices,
    ext_forces, int_forces, ext_torques, int_torques,
    collect_forces,
    prefix="batch",
):
    """Save batch as .parquet file.

    Multi-dimensional arrays (states, actions, forces) are stored as
    fixed-size-list columns for efficient columnar access.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    arrays = {
        "states": pa.FixedSizeListArray.from_arrays(
            pa.array(states.ravel(), type=pa.float32()), list_size=states.shape[1]
        ),
        "actions": pa.FixedSizeListArray.from_arrays(
            pa.array(actions.ravel(), type=pa.float32()), list_size=actions.shape[1]
        ),
        "serpenoid_times": pa.array(serp_times, type=pa.float32()),
        "next_states": pa.FixedSizeListArray.from_arrays(
            pa.array(next_states.ravel(), type=pa.float32()), list_size=next_states.shape[1]
        ),
        "episode_ids": pa.array(episode_ids, type=pa.int64()),
        "step_indices": pa.array(step_indices, type=pa.int64()),
    }
    if collect_forces and ext_forces:
        for name, data_list in [
            ("external_forces", ext_forces),
            ("internal_forces", int_forces),
            ("external_torques", ext_torques),
            ("internal_torques", int_torques),
        ]:
            arr = np.concatenate(data_list, axis=0)  # (T, 3, N)
            flat_size = arr.shape[1] * arr.shape[2]  # 3*21 or 3*20
            arrays[name] = pa.FixedSizeListArray.from_arrays(
                pa.array(arr.reshape(-1), type=pa.float32()), list_size=flat_size
            )

    table = pa.table(arrays)
    path = save_dir / f"{prefix}_{batch_idx:04d}.parquet"
    pq.write_table(table, path, compression="zstd")
    print(f"  Saved {n:,} transitions to {path}")


if __name__ == "__main__":
    main()
