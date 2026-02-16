#!/usr/bin/env python3
"""Benchmark all physics backends under identical conditions.

Runs a 500-step serpenoid simulation on each available backend and records:
- Wall-clock time per step
- Head/CoG trajectory, heading angle, forward displacement
- Energy components (kinetic, elastic, gravitational, total)
- Solver residual norm (DisMech only)

Outputs CSV files to output/benchmark/ for analysis.

Usage:
    python script/benchmark_physics.py
    python script/benchmark_physics.py --steps 200 --frameworks dismech mujoco
    python script/benchmark_physics.py --output-dir output/benchmark_quick --steps 100
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

from configs.env import PhysicsConfig, SolverFramework
from physics import create_snake_robot

# ── Framework availability checks ──────────────────────────────────────

FRAMEWORK_INFO = {
    "dismech": {
        "enum": SolverFramework.DISMECH,
        "label": "DisMech (Python)",
    },
    "elastica": {
        "enum": SolverFramework.ELASTICA,
        "label": "PyElastica",
    },
    "dismech_rods": {
        "enum": SolverFramework.DISMECH_RODS,
        "label": "dismech-rods (C++)",
    },
    "mujoco": {
        "enum": SolverFramework.MUJOCO,
        "label": "MuJoCo",
    },
}


def _check_available(name: str) -> bool:
    """Check if a backend can be imported."""
    try:
        if name == "dismech":
            import dismech  # noqa: F401
        elif name == "elastica":
            import elastica  # noqa: F401
        elif name == "dismech_rods":
            import py_dismech  # noqa: F401
        elif name == "mujoco":
            import mujoco  # noqa: F401
        return True
    except ImportError:
        return False


# ── Serpenoid curvature generation ─────────────────────────────────────

class FixedSerpenoid:
    """Generate serpenoid curvatures with fixed parameters.

    kappa(s, t) = A * sin(k * s - omega * t + phi)
    """

    def __init__(
        self,
        num_joints: int = 19,
        amplitude: float = 1.0,
        frequency: float = 1.0,
        wave_number: float = 1.5,
        phase: float = 0.0,
    ):
        self.num_joints = num_joints
        self.amplitude = amplitude
        self.omega = 2 * np.pi * frequency
        self.k = 2 * np.pi * wave_number
        self.phase = phase
        self._joint_positions = np.linspace(0, 1, num_joints)
        self._time = 0.0

    def reset(self):
        self._time = 0.0

    def step(self, dt: float) -> np.ndarray:
        """Advance time and return curvatures."""
        self._time += dt
        return self.amplitude * np.sin(
            self.k * self._joint_positions - self.omega * self._time + self.phase
        )


# ── Metric helpers ─────────────────────────────────────────────────────

def _heading_angle(positions: np.ndarray) -> float:
    """Compute heading angle (radians) from head segment direction."""
    head_vec = positions[1] - positions[0]
    return float(np.arctan2(head_vec[1], head_vec[0]))


def _forward_displacement(positions: np.ndarray, initial_head: np.ndarray) -> float:
    """Forward displacement of head from initial position."""
    delta = positions[0] - initial_head
    return float(np.linalg.norm(delta[:2]))  # XY plane


# ── Per-framework benchmark ───────────────────────────────────────────

def run_benchmark(
    framework_name: str,
    num_steps: int,
    config: PhysicsConfig,
) -> dict:
    """Run benchmark for a single framework.

    Returns:
        dict with keys 'timeseries' (list of row dicts) and 'name'.
    """
    config.solver_framework = FRAMEWORK_INFO[framework_name]["enum"]
    robot = create_snake_robot(config)
    robot.reset()

    serpenoid = FixedSerpenoid(num_joints=config.num_segments - 1)
    dt = config.dt

    # Record initial head position for displacement
    initial_state = robot.get_state()
    initial_head = initial_state["positions"][0].copy()

    is_dismech = framework_name == "dismech"
    rows = []

    for step_idx in range(num_steps):
        curvatures = serpenoid.step(dt)
        robot.set_curvature_control(curvatures)

        t0 = time.perf_counter()
        state = robot.step()
        wall_time_ms = (time.perf_counter() - t0) * 1000.0

        energy = robot.get_energy()
        positions = state["positions"]
        head = positions[0]
        cog = positions.mean(axis=0)
        heading = _heading_angle(positions)
        fwd_disp = _forward_displacement(positions, initial_head)

        # DisMech residual norm
        residual = None
        if is_dismech and hasattr(robot, "_last_residual_norm"):
            residual = robot._last_residual_norm

        rows.append({
            "step": step_idx,
            "sim_time": round((step_idx + 1) * dt, 6),
            "wall_time_ms": round(wall_time_ms, 4),
            "head_x": head[0],
            "head_y": head[1],
            "head_z": head[2],
            "cog_x": cog[0],
            "cog_y": cog[1],
            "cog_z": cog[2],
            "heading_angle": heading,
            "forward_disp": fwd_disp,
            "kinetic_energy": energy["kinetic"],
            "elastic_energy": energy["elastic"],
            "gravitational_energy": energy["gravitational"],
            "total_energy": energy["total"],
            "residual_norm": residual,
        })

    return {"timeseries": rows, "name": framework_name}


# ── Summary statistics ─────────────────────────────────────────────────

def compute_summary(result: dict) -> dict:
    """Compute summary statistics from a framework's timeseries."""
    rows = result["timeseries"]
    wall_times = [r["wall_time_ms"] for r in rows]
    energies = [r["total_energy"] for r in rows]

    initial_energy = energies[0] if energies[0] != 0 else 1e-12
    energy_drift = (energies[-1] - energies[0]) / abs(initial_energy)

    heading_start = rows[0]["heading_angle"]
    heading_end = rows[-1]["heading_angle"]
    heading_drift_deg = np.degrees(heading_end - heading_start)

    return {
        "framework": result["name"],
        "mean_step_ms": round(np.mean(wall_times), 4),
        "std_step_ms": round(np.std(wall_times), 4),
        "min_step_ms": round(np.min(wall_times), 4),
        "max_step_ms": round(np.max(wall_times), 4),
        "median_step_ms": round(np.median(wall_times), 4),
        "total_forward_disp": round(rows[-1]["forward_disp"], 6),
        "heading_drift_deg": round(heading_drift_deg, 4),
        "energy_mean": round(np.mean(energies), 6),
        "energy_std": round(np.std(energies), 6),
        "energy_drift": round(energy_drift, 6),
    }


# ── Cross-framework comparison ────────────────────────────────────────

def compute_cross_framework(results: list[dict]) -> list[dict]:
    """Compute pairwise cross-framework metrics."""
    comparisons = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]
            rows_a = a["timeseries"]
            rows_b = b["timeseries"]
            n = min(len(rows_a), len(rows_b))

            head_diffs_sq = []
            cog_diffs_sq = []
            max_head_div = 0.0

            for k in range(n):
                ha = np.array([rows_a[k]["head_x"], rows_a[k]["head_y"], rows_a[k]["head_z"]])
                hb = np.array([rows_b[k]["head_x"], rows_b[k]["head_y"], rows_b[k]["head_z"]])
                head_diff = np.sum((ha - hb) ** 2)
                head_diffs_sq.append(head_diff)
                max_head_div = max(max_head_div, np.sqrt(head_diff))

                ca = np.array([rows_a[k]["cog_x"], rows_a[k]["cog_y"], rows_a[k]["cog_z"]])
                cb = np.array([rows_b[k]["cog_x"], rows_b[k]["cog_y"], rows_b[k]["cog_z"]])
                cog_diffs_sq.append(np.sum((ca - cb) ** 2))

            comparisons.append({
                "framework_a": a["name"],
                "framework_b": b["name"],
                "head_trajectory_mse": round(np.mean(head_diffs_sq), 8),
                "cog_trajectory_mse": round(np.mean(cog_diffs_sq), 8),
                "max_head_divergence": round(max_head_div, 6),
            })

    return comparisons


# ── CSV output ─────────────────────────────────────────────────────────

TIMESERIES_FIELDS = [
    "step", "sim_time", "wall_time_ms",
    "head_x", "head_y", "head_z",
    "cog_x", "cog_y", "cog_z",
    "heading_angle", "forward_disp",
    "kinetic_energy", "elastic_energy", "gravitational_energy", "total_energy",
    "residual_norm",
]

SUMMARY_FIELDS = [
    "framework", "mean_step_ms", "std_step_ms", "min_step_ms", "max_step_ms",
    "median_step_ms", "total_forward_disp", "heading_drift_deg",
    "energy_mean", "energy_std", "energy_drift",
]

CROSS_FIELDS = [
    "framework_a", "framework_b",
    "head_trajectory_mse", "cog_trajectory_mse", "max_head_divergence",
]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    """Write rows to CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── Console output ─────────────────────────────────────────────────────

def print_summary_table(summaries: list[dict], skipped: list[str]):
    """Print a readable summary table to stdout."""
    if skipped:
        print(f"\nSkipped (unavailable): {', '.join(skipped)}")

    if not summaries:
        print("No frameworks were benchmarked.")
        return

    print(f"\n{'Framework':<22} {'Mean ms':>9} {'Std ms':>9} {'Median ms':>10} "
          f"{'Fwd disp':>10} {'Head drift':>11} {'E drift':>10}")
    print("-" * 85)
    for s in summaries:
        print(f"{s['framework']:<22} {s['mean_step_ms']:>9.3f} {s['std_step_ms']:>9.3f} "
              f"{s['median_step_ms']:>10.3f} {s['total_forward_disp']:>10.4f} "
              f"{s['heading_drift_deg']:>10.2f}° {s['energy_drift']:>10.4f}")
    print()


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark physics backends under identical serpenoid control."
    )
    parser.add_argument(
        "--output-dir", default="output/benchmark",
        help="Directory for CSV output files (default: output/benchmark)",
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Number of simulation steps (default: 500)",
    )
    parser.add_argument(
        "--frameworks", nargs="*", default=None,
        help="Frameworks to benchmark (default: all available). "
             "Choices: dismech, elastica, dismech_rods, mujoco",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which frameworks to run
    if args.frameworks:
        requested = args.frameworks
    else:
        requested = list(FRAMEWORK_INFO.keys())

    available = []
    skipped = []
    for name in requested:
        if name not in FRAMEWORK_INFO:
            print(f"Unknown framework: {name}")
            skipped.append(name)
            continue
        if _check_available(name):
            available.append(name)
        else:
            skipped.append(name)
            print(f"  {FRAMEWORK_INFO[name]['label']}: not installed, skipping")

    if not available:
        print("No frameworks available to benchmark.")
        sys.exit(1)

    # Shared physics config (default PhysicsConfig with higher max_iter
    # to prevent dismech-rods C++ from calling exit() on convergence failure)
    config = PhysicsConfig(max_iter=100)

    print(f"Benchmarking {len(available)} frameworks, {args.steps} steps each")
    print(f"Snake: {config.num_segments} segments, L={config.snake_length}m, "
          f"r={config.snake_radius}m, dt={config.dt}s, max_iter={config.max_iter}")
    print(f"Serpenoid: A=1.0, f=1.0, wn=1.5, phi=0")

    # Run benchmarks
    results = []
    for name in available:
        label = FRAMEWORK_INFO[name]["label"]
        print(f"\n  Running {label}...", end=" ", flush=True)
        t0 = time.perf_counter()
        result = run_benchmark(name, args.steps, PhysicsConfig(max_iter=config.max_iter))
        total_s = time.perf_counter() - t0
        print(f"done ({total_s:.1f}s)")
        results.append(result)

    # Write per-framework timeseries CSVs
    for result in results:
        path = output_dir / f"{result['name']}_timeseries.csv"
        write_csv(path, TIMESERIES_FIELDS, result["timeseries"])
        print(f"  Wrote {path}")

    # Compute and write summary
    summaries = [compute_summary(r) for r in results]
    summary_path = output_dir / "summary.csv"
    write_csv(summary_path, SUMMARY_FIELDS, summaries)
    print(f"  Wrote {summary_path}")

    # Compute and write cross-framework comparison
    if len(results) >= 2:
        cross = compute_cross_framework(results)
        cross_path = output_dir / "cross_framework.csv"
        write_csv(cross_path, CROSS_FIELDS, cross)
        print(f"  Wrote {cross_path}")

    # Print summary table
    print_summary_table(summaries, skipped)


if __name__ == "__main__":
    main()
