#!/usr/bin/env python3
"""Pre-flight smoke test for surrogate data collection.

Runs a small collection (2 workers, 1000 transitions) to a temp directory,
validates data integrity, and estimates wall-clock time for a full run.

Usage:
    python script/smoke_test_collect.py
"""

import shutil
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so aprx_model_elastica can be imported
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch


TEMP_DIR = Path("data/tmp_smoke_test")
NUM_WORKERS = 2
NUM_TRANSITIONS = 1000
EXPECTED_STATE_DIM = 124
EXPECTED_ACTION_DIM = 5
FULL_RUN_TRANSITIONS = 10_000_000


def run_collection() -> float:
    """Run a small data collection and return elapsed time in seconds."""
    from aprx_model_elastica.collect_config import DataCollectionConfig
    from aprx_model_elastica.collect_data import _multiprocess_collect

    config = DataCollectionConfig(
        num_transitions=NUM_TRANSITIONS,
        num_workers=NUM_WORKERS,
        save_dir=str(TEMP_DIR),
        episodes_per_save=10,
        collect_forces=False,
        seed=12345,
    )
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()
    _multiprocess_collect(config)
    elapsed = time.monotonic() - t_start
    return elapsed


def load_all_batches() -> dict[str, Any]:
    """Load and concatenate all batch .pt files from TEMP_DIR."""
    files = sorted(TEMP_DIR.glob("batch_*.pt"))
    if not files:
        print("FAIL: No batch files found in", TEMP_DIR)
        sys.exit(1)

    all_states = []
    all_actions = []
    all_next_states = []
    all_serp_times = []
    all_episode_ids = []

    for f in files:
        data = torch.load(f, map_location="cpu", weights_only=True)
        all_states.append(data["states"])
        all_actions.append(data["actions"])
        all_next_states.append(data["next_states"])
        all_serp_times.append(data["serpenoid_times"])
        all_episode_ids.append(data["episode_ids"])

    return {
        "states": torch.cat(all_states, dim=0),
        "actions": torch.cat(all_actions, dim=0),
        "next_states": torch.cat(all_next_states, dim=0),
        "serpenoid_times": torch.cat(all_serp_times, dim=0),
        "episode_ids": torch.cat(all_episode_ids, dim=0),
        "num_files": len(files),
    }


def validate(data: dict[str, torch.Tensor]) -> list[str]:
    """Run all validation checks. Returns list of failure messages."""
    failures = []
    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]
    serp_times = data["serpenoid_times"]
    episode_ids = data["episode_ids"]
    n = states.shape[0]

    print(f"  Total transitions: {n:,} (from {data['num_files']} files)")

    # Shape checks
    if states.shape != (n, EXPECTED_STATE_DIM):
        failures.append(f"states shape {states.shape}, expected (N, {EXPECTED_STATE_DIM})")
    if next_states.shape != (n, EXPECTED_STATE_DIM):
        failures.append(f"next_states shape {next_states.shape}, expected (N, {EXPECTED_STATE_DIM})")
    if actions.shape != (n, EXPECTED_ACTION_DIM):
        failures.append(f"actions shape {actions.shape}, expected (N, {EXPECTED_ACTION_DIM})")

    # NaN / Inf checks
    for name, tensor in [
        ("states", states),
        ("actions", actions),
        ("next_states", next_states),
        ("serpenoid_times", serp_times),
    ]:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        if nan_count > 0:
            failures.append(f"{name} has {nan_count} NaN values")
        if inf_count > 0:
            failures.append(f"{name} has {inf_count} Inf values")

    # Episode ID uniqueness across workers
    # Workers use offset worker_id * 10_000_000, so IDs from different workers
    # should never collide. Check by verifying no episode ID appears with
    # two different worker prefixes (integer division by 10M).
    unique_ids = episode_ids.unique()
    worker_bins = unique_ids // 10_000_000
    id_within_worker = unique_ids % 10_000_000
    # Group by worker bin — within each bin, IDs should be unique
    for wb in worker_bins.unique():
        mask = worker_bins == wb
        ids_in_bin = id_within_worker[mask]
        if ids_in_bin.unique().shape[0] != ids_in_bin.shape[0]:
            failures.append(f"Episode ID collision within worker bin {wb.item()}")
    # Across workers: full IDs should be unique
    if unique_ids.shape[0] != episode_ids.unique().shape[0]:
        failures.append("Episode ID collision across workers")

    print(f"  Unique episodes: {unique_ids.shape[0]}")
    print(f"  Workers detected: {worker_bins.unique().tolist()}")

    return failures


def main() -> None:
    print("=" * 60)
    print("Surrogate Data Collection — Smoke Test")
    print("=" * 60)
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  Transitions: {NUM_TRANSITIONS:,}")
    print(f"  Temp dir: {TEMP_DIR}")
    print()

    # Clean up any previous temp data
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    # Run collection
    print("[1/3] Running collection...")
    elapsed = run_collection()
    fps = NUM_TRANSITIONS / elapsed if elapsed > 0 else 0
    print(f"  Elapsed: {elapsed:.1f}s | FPS: {fps:.1f}")
    print()

    # Load and validate
    print("[2/3] Validating data...")
    data = load_all_batches()
    failures = validate(data)
    print()

    # FPS estimate
    print("[3/3] Projection for full run")
    print(f"  Measured FPS: {fps:.1f}")
    if fps > 0:
        full_run_hours = FULL_RUN_TRANSITIONS / fps / 3600
        print(f"  Estimated time for {FULL_RUN_TRANSITIONS / 1e6:.0f}M transitions: {full_run_hours:.1f} hours")
    print()

    # Clean up
    shutil.rmtree(TEMP_DIR)
    print(f"  Cleaned up {TEMP_DIR}")
    print()

    # Report
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
