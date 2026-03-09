#!/usr/bin/env python3
"""Post-collection data validation for surrogate training data.

Loads all batch files from a data directory and checks integrity, coverage,
and episode structure. Prints a formatted summary report to stdout.

Usage:
    python script/validate_surrogate_data.py [--data-dir data/surrogate/]
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure project root is on sys.path so aprx_model_elastica can be imported
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from aprx_model_elastica.state import (
    ACTION_DIM,
    OMEGA_Z,
    POS_X,
    POS_Y,
    STATE_DIM,
    VEL_X,
    VEL_Y,
    YAW,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_pt_batch(path: Path) -> Dict[str, torch.Tensor]:
    """Load a .pt batch file."""
    return torch.load(path, map_location="cpu", weights_only=True)


def load_all_batches(data_dir: Path) -> Tuple[Dict[str, torch.Tensor], int, int]:
    """Load and concatenate all batch files.

    Returns:
        (data_dict, num_pt_files, num_pq_files)
    """
    pt_files = sorted(data_dir.glob("batch_*.pt"))
    pq_files = sorted(data_dir.glob("batch_*.parquet"))

    if not pt_files and not pq_files:
        print(f"ERROR: No batch_*.pt or batch_*.parquet files in {data_dir}")
        sys.exit(1)

    all_states: List[torch.Tensor] = []
    all_actions: List[torch.Tensor] = []
    all_next_states: List[torch.Tensor] = []
    all_serp_times: List[torch.Tensor] = []
    all_episode_ids: List[torch.Tensor] = []
    all_step_indices: List[torch.Tensor] = []

    for f in pt_files:
        data = _load_pt_batch(f)
        all_states.append(data["states"])
        all_actions.append(data["actions"])
        all_next_states.append(data["next_states"])
        all_serp_times.append(data["serpenoid_times"])
        all_episode_ids.append(data["episode_ids"])
        all_step_indices.append(data["step_indices"])

    if pq_files:
        from aprx_model_elastica.dataset import _load_parquet_batch

        for f in pq_files:
            data = _load_parquet_batch(f)
            all_states.append(data["states"])
            all_actions.append(data["actions"])
            all_next_states.append(data["next_states"])
            all_serp_times.append(data["serpenoid_times"])
            all_episode_ids.append(data["episode_ids"])
            all_step_indices.append(data["step_indices"])

    combined = {
        "states": torch.cat(all_states, dim=0),
        "actions": torch.cat(all_actions, dim=0),
        "next_states": torch.cat(all_next_states, dim=0),
        "serpenoid_times": torch.cat(all_serp_times, dim=0),
        "episode_ids": torch.cat(all_episode_ids, dim=0),
        "step_indices": torch.cat(all_step_indices, dim=0),
    }
    return combined, len(pt_files), len(pq_files)


# ---------------------------------------------------------------------------
# Integrity checks
# ---------------------------------------------------------------------------


def check_nan_inf(data: Dict[str, torch.Tensor]) -> List[str]:
    """Check for NaN/Inf values in key tensors. Returns failure messages."""
    failures: List[str] = []
    fields = ["states", "actions", "next_states", "serpenoid_times"]
    print("  NaN/Inf check:")
    for name in fields:
        t = data[name]
        nan_count = int(torch.isnan(t).sum().item())
        inf_count = int(torch.isinf(t).sum().item())
        status = "OK" if (nan_count == 0 and inf_count == 0) else "FAIL"
        print(f"    {name:20s}  NaN={nan_count:,}  Inf={inf_count:,}  [{status}]")
        if nan_count > 0:
            failures.append(f"{name} has {nan_count:,} NaN values")
        if inf_count > 0:
            failures.append(f"{name} has {inf_count:,} Inf values")
    return failures


def check_dimensions(data: Dict[str, torch.Tensor]) -> List[str]:
    """Verify tensor dimensions. Returns failure messages."""
    failures: List[str] = []
    n = data["states"].shape[0]
    checks = [
        ("states", data["states"].shape, (n, STATE_DIM)),
        ("next_states", data["next_states"].shape, (n, STATE_DIM)),
        ("actions", data["actions"].shape, (n, ACTION_DIM)),
    ]
    print("  Dimension check:")
    for name, actual, expected in checks:
        status = "OK" if actual == expected else "FAIL"
        print(f"    {name:20s}  {str(actual):20s}  expected {str(expected):20s}  [{status}]")
        if actual != expected:
            failures.append(f"{name} shape {actual}, expected {expected}")
    return failures


def check_episode_uniqueness(episode_ids: torch.Tensor) -> List[str]:
    """Check episode ID uniqueness. Returns failure messages."""
    failures: List[str] = []
    unique_ids = episode_ids.unique()
    total = episode_ids.shape[0]
    n_unique = unique_ids.shape[0]

    # Check for duplicate global IDs
    if n_unique != torch.unique(episode_ids).shape[0]:
        failures.append("Episode ID collision detected")

    # Worker-level check (IDs offset by worker_id * 10_000_000)
    worker_bins = unique_ids // 10_000_000
    n_workers = worker_bins.unique().shape[0]
    print(f"  Episode uniqueness:")
    print(f"    Unique episodes:   {n_unique:,}")
    print(f"    Workers detected:  {n_workers}")

    for wb in worker_bins.unique():
        mask = worker_bins == wb
        ids_in_bin = (unique_ids[mask] % 10_000_000)
        if ids_in_bin.unique().shape[0] != ids_in_bin.shape[0]:
            failures.append(f"Episode ID collision within worker bin {wb.item()}")
            print(f"    Worker {wb.item():3d}: COLLISION")
        else:
            print(f"    Worker {wb.item():3d}: {ids_in_bin.shape[0]:,} episodes  [OK]")

    return failures


# ---------------------------------------------------------------------------
# Coverage statistics
# ---------------------------------------------------------------------------

STATE_GROUPS = [
    ("pos_x[0:21]", POS_X),
    ("pos_y[21:42]", POS_Y),
    ("vel_x[42:63]", VEL_X),
    ("vel_y[63:84]", VEL_Y),
    ("yaw[84:104]", YAW),
    ("omega_z[104:124]", OMEGA_Z),
]


def report_state_coverage(states: torch.Tensor) -> None:
    """Print per-state-group coverage statistics."""
    print("  Per-state-group coverage (across all elements in group):")
    print(f"    {'Group':25s}  {'Min':>12s}  {'Max':>12s}  {'Mean':>12s}  {'Std':>12s}")
    print("    " + "-" * 77)
    for name, sl in STATE_GROUPS:
        group = states[:, sl]
        print(
            f"    {name:25s}  {group.min().item():12.4f}  {group.max().item():12.4f}"
            f"  {group.mean().item():12.4f}  {group.std().item():12.4f}"
        )


def report_action_coverage(actions: torch.Tensor) -> None:
    """Print per-action-dimension statistics."""
    print(f"  Per-action-dimension stats:")
    print(f"    {'Dim':>5s}  {'Min':>12s}  {'Max':>12s}  {'Mean':>12s}")
    print("    " + "-" * 47)
    for i in range(actions.shape[1]):
        col = actions[:, i]
        print(f"    {i:5d}  {col.min().item():12.4f}  {col.max().item():12.4f}  {col.mean().item():12.4f}")


def report_episode_lengths(episode_ids: torch.Tensor, step_indices: torch.Tensor) -> None:
    """Print episode length distribution."""
    unique_eps = episode_ids.unique()
    lengths: List[int] = []
    for eid in unique_eps:
        mask = episode_ids == eid
        steps = step_indices[mask]
        ep_len = int(steps.max().item() - steps.min().item() + 1)
        lengths.append(ep_len)

    lengths_t = torch.tensor(lengths, dtype=torch.float32)
    median_val = float(lengths_t.median().item())
    print("  Episode length distribution:")
    print(f"    Episodes:  {len(lengths):,}")
    print(f"    Min:       {min(lengths):,}")
    print(f"    Max:       {max(lengths):,}")
    print(f"    Mean:      {lengths_t.mean().item():.1f}")
    print(f"    Median:    {median_val:.1f}")


# ---------------------------------------------------------------------------
# Disk size helper
# ---------------------------------------------------------------------------


def get_dir_size(path: Path) -> float:
    """Return total size of all files in directory in GB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 ** 3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate surrogate training data")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/surrogate/",
        help="Directory containing batch_*.pt / batch_*.parquet files",
    )
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"ERROR: Directory does not exist: {data_dir}")
        sys.exit(1)

    print("=" * 70)
    print("Surrogate Data Validation")
    print("=" * 70)
    print(f"  Data dir: {data_dir.resolve()}")
    print()

    # Load data
    print("[1/5] Loading data...")
    data, n_pt, n_pq = load_all_batches(data_dir)
    n = data["states"].shape[0]
    n_episodes = int(data["episode_ids"].unique().shape[0])
    disk_gb = get_dir_size(data_dir)
    print(f"  Total transitions: {n:,}")
    print(f"  Total episodes:    {n_episodes:,}")
    print(f"  Total files:       {n_pt + n_pq} ({n_pt} .pt, {n_pq} .parquet)")
    print(f"  Size on disk:      {disk_gb:.3f} GB")
    print()

    # Integrity checks
    failures: List[str] = []

    print("[2/5] Integrity checks...")
    failures.extend(check_dimensions(data))
    failures.extend(check_nan_inf(data))
    failures.extend(check_episode_uniqueness(data["episode_ids"]))
    print()

    # Coverage stats
    print("[3/5] State coverage...")
    report_state_coverage(data["states"])
    print()

    print("[4/5] Action coverage...")
    report_action_coverage(data["actions"])
    print()

    print("[5/5] Episode lengths...")
    report_episode_lengths(data["episode_ids"], data["step_indices"])
    print()

    # Final report
    print("=" * 70)
    if failures:
        print("VALIDATION FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL INTEGRITY CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
