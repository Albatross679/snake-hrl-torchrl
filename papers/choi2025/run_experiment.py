"""Experiment matrix runner for Choi2025 soft manipulator replication.

Orchestrates the full 8-run experiment matrix (4 tasks x 2 algorithms).
Runs sequentially (one at a time for GPU safety) with GPU cleanup between runs.

Usage:
    python -m choi2025.run_experiment --total-frames 1000000 --num-envs 32
    python -m choi2025.run_experiment --quick  # 100K frames for validation
    python -m choi2025.run_experiment --tasks follow_target --algos sac
"""

import argparse
import gc
import subprocess
import sys
import time
from datetime import datetime

import torch

from choi2025.config import TaskType


# Full experiment matrix: 4 tasks x 2 algorithms = 8 runs
EXPERIMENT_MATRIX = [
    {"task": t.value, "algo": a}
    for t in TaskType
    for a in ["sac", "ppo"]
]

# Training scripts per algorithm
TRAIN_SCRIPTS = {
    "sac": "choi2025.train",
    "ppo": "choi2025.train_ppo",
}

# Watchdog timeout after max_wall_time expires (10 minutes)
WATCHDOG_TIMEOUT = 600


def parse_wall_time_seconds(s: str) -> float:
    """Parse wall-time string into seconds (simplified)."""
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        pass
    import re
    m = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", s)
    if not m or not any(m.groups()):
        raise ValueError(f"Invalid wall-time: '{s}'")
    h = int(m.group(1) or 0)
    mi = int(m.group(2) or 0)
    sec = int(m.group(3) or 0)
    return float(h * 3600 + mi * 60 + sec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Choi2025 experiment matrix (4 tasks x 2 algos)"
    )
    parser.add_argument(
        "--total-frames", type=int, default=1_000_000,
        help="Total training frames per run (default: 1M)",
    )
    parser.add_argument(
        "--num-envs", type=int, default=32,
        help="Number of parallel environments (default: 32)",
    )
    parser.add_argument(
        "--max-wall-time", type=str, default="4h",
        help="Max wall-clock time per run (default: 4h)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick validation mode: 100K frames, 30m wall time",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Filter to specific tasks (e.g. follow_target inverse_kinematics)",
    )
    parser.add_argument(
        "--algos", nargs="+", default=None,
        choices=["sac", "ppo"],
        help="Filter to specific algorithms",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Quick mode overrides
    if args.quick:
        total_frames = 100_000
        max_wall_time = "30m"
    else:
        total_frames = args.total_frames
        max_wall_time = args.max_wall_time

    wall_time_seconds = parse_wall_time_seconds(max_wall_time)

    # Filter experiment matrix
    matrix = EXPERIMENT_MATRIX
    if args.tasks:
        matrix = [e for e in matrix if e["task"] in args.tasks]
    if args.algos:
        matrix = [e for e in matrix if e["algo"] in args.algos]

    if not matrix:
        print("No experiments match the given filters.")
        return

    print(f"=== Choi2025 Experiment Matrix ===")
    print(f"Runs: {len(matrix)}")
    print(f"Frames/run: {total_frames:,}")
    print(f"Envs/run: {args.num_envs}")
    print(f"Wall time/run: {max_wall_time}")
    if args.quick:
        print("  (quick validation mode)")
    print()

    results = []

    for i, entry in enumerate(matrix):
        task = entry["task"]
        algo = entry["algo"]
        script = TRAIN_SCRIPTS[algo]

        print(f"--- Run {i+1}/{len(matrix)}: {task} / {algo} ---")
        start = datetime.now()
        print(f"  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")

        cmd = [
            sys.executable, "-m", script,
            "--task", task,
            "--num-envs", str(args.num_envs),
            "--total-frames", str(total_frames),
            "--max-wall-time", max_wall_time,
        ]

        # Launch with watchdog timeout
        proc = subprocess.Popen(cmd)
        try:
            proc.wait(timeout=wall_time_seconds + WATCHDOG_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(
                f"  WATCHDOG: {task}/{algo} hung after "
                f"{wall_time_seconds + WATCHDOG_TIMEOUT:.0f}s, killing"
            )
            proc.kill()
            proc.wait()

        end = datetime.now()
        duration = end - start
        return_code = proc.returncode

        # Classify exit status
        if return_code in (137, 143):
            status = "hung (killed)"
        elif return_code == 0:
            status = "success"
        else:
            status = f"error (code {return_code})"

        print(f"  Finished: {end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {duration}")
        print(f"  Status: {status}")
        print()

        results.append({
            "task": task,
            "algo": algo,
            "return_code": return_code,
            "duration_s": duration.total_seconds(),
            "status": status,
        })

        # GPU cleanup between runs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Summary table
    print("=== Experiment Summary ===")
    print(f"{'Task':<25} {'Algo':<6} {'Status':<20} {'Duration':<12}")
    print("-" * 65)
    for r in results:
        mins = r["duration_s"] / 60
        print(f"{r['task']:<25} {r['algo']:<6} {r['status']:<20} {mins:.1f}m")

    # Overall status
    n_success = sum(1 for r in results if r["return_code"] == 0)
    n_hung = sum(1 for r in results if r["return_code"] in (137, 143))
    n_error = sum(1 for r in results if r["return_code"] not in (0, 137, 143))
    print(f"\nTotal: {n_success} success, {n_hung} hung, {n_error} errors")


if __name__ == "__main__":
    main()
