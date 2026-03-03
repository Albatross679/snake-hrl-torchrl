"""Stiffness and fluid property sweep experiments.

Reproduces the paper's experiments varying:
- Joint stiffness: 0, 0.5, 1.0, 2.0, 4.0 Nm/rad
- Fluid: water, propylene glycol, ethylene glycol
"""

import sys
import argparse
import subprocess
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from configs.base import resolve_device
from zheng2022.configs_zheng2022 import WATER, PROPYLENE_GLYCOL, ETHYLENE_GLYCOL


STIFFNESS_VALUES = [0.0, 0.5, 1.0, 2.0, 4.0]  # Nm/rad

FLUID_CONFIGS = {
    "water": WATER,
    "propylene_glycol": PROPYLENE_GLYCOL,
    "ethylene_glycol": ETHYLENE_GLYCOL,
}


def run_stiffness_sweep(
    stiffness_values: list = None,
    epochs: int = 5000,
    device: str = "cpu",
):
    """Launch training for each stiffness value."""
    stiffness_values = stiffness_values or STIFFNESS_VALUES
    script_dir = Path(__file__).resolve().parent

    for k in stiffness_values:
        name = f"stiffness_{k:.1f}"
        print(f"\n{'='*60}")
        print(f"Training with joint stiffness = {k} Nm/rad")
        print(f"{'='*60}")

        cmd = [
            sys.executable, str(script_dir / "train_zheng2022.py"),
            "--stiffness", str(k),
            "--epochs", str(epochs),
            "--device", device,
            "--name", name,
        ]
        subprocess.run(cmd, check=True)


def run_fluid_sweep(
    fluids: dict = None,
    epochs: int = 5000,
    device: str = "cpu",
):
    """Launch training for each fluid type."""
    fluids = fluids or FLUID_CONFIGS
    script_dir = Path(__file__).resolve().parent

    for fluid_name, fluid_params in fluids.items():
        name = f"fluid_{fluid_name}"
        print(f"\n{'='*60}")
        print(f"Training in {fluid_name} (rho={fluid_params['fluid_density']}, mu={fluid_params['fluid_viscosity']})")
        print(f"{'='*60}")

        cmd = [
            sys.executable, str(script_dir / "train_zheng2022.py"),
            "--fluid-density", str(fluid_params["fluid_density"]),
            "--fluid-viscosity", str(fluid_params["fluid_viscosity"]),
            "--epochs", str(epochs),
            "--device", device,
            "--name", name,
        ]
        subprocess.run(cmd, check=True)


def plot_stiffness_results(results_dir: Path, output_dir: Path):
    """Plot velocity vs power for different stiffness values."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    stiffnesses = []
    velocities = []
    powers = []
    efficiencies = []

    for k in STIFFNESS_VALUES:
        metrics_path = results_dir / f"stiffness_{k:.1f}" / "metrics.pt"
        if not metrics_path.exists():
            print(f"Skipping stiffness={k} (no results found at {metrics_path})")
            continue

        metrics = torch.load(metrics_path)
        # Use last 100 epochs for stable estimates
        last_n = min(100, len(metrics))
        mean_vx = np.mean([m["mean_head_vx"] for m in metrics[-last_n:]])
        mean_pow = np.mean([m["mean_power"] for m in metrics[-last_n:]])

        stiffnesses.append(k)
        velocities.append(mean_vx)
        powers.append(mean_pow)
        efficiencies.append(mean_vx / max(mean_pow, 1e-8))

    if not stiffnesses:
        print("No results found. Run sweep first.")
        return

    axes[0].bar(range(len(stiffnesses)), velocities, tick_label=[f"{k}" for k in stiffnesses])
    axes[0].set_xlabel("Joint stiffness (Nm/rad)")
    axes[0].set_ylabel("Mean forward velocity (m/s)")
    axes[0].set_title("Velocity vs stiffness")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(range(len(stiffnesses)), efficiencies, tick_label=[f"{k}" for k in stiffnesses],
                color="green")
    axes[1].set_xlabel("Joint stiffness (Nm/rad)")
    axes[1].set_ylabel("Efficiency (v/P)")
    axes[1].set_title("Efficiency vs stiffness")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "stiffness_sweep.png", dpi=150)
    plt.close()
    print(f"Plot saved to {output_dir / 'stiffness_sweep.png'}")


def main():
    parser = argparse.ArgumentParser(description="Run stiffness/fluid sweep experiments")
    parser.add_argument("--sweep", choices=["stiffness", "fluid", "both", "plot"], default="stiffness")
    parser.add_argument("--epochs", type=int, default=5000, help="Training epochs per run")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--results-dir", type=str, default="logs", help="Directory with training logs")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for plots")
    args = parser.parse_args()

    device = resolve_device(args.device)

    if args.sweep == "stiffness":
        run_stiffness_sweep(epochs=args.epochs, device=device)
    elif args.sweep == "fluid":
        run_fluid_sweep(epochs=args.epochs, device=device)
    elif args.sweep == "both":
        run_stiffness_sweep(epochs=args.epochs, device=device)
        run_fluid_sweep(epochs=args.epochs, device=device)
    elif args.sweep == "plot":
        plot_stiffness_results(Path(args.results_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
