"""Hyperparameter sweep runner for the surrogate model.

Launches 5 training configurations covering:
  - 3 learning rates: 1e-4, 3e-4, 1e-3
  - 3 model sizes: 256x3, 512x3, 512x4
  - lr=3e-4 as pivot connecting model sizes

Usage:
    python -m aprx_model_elastica.sweep
    python -m aprx_model_elastica.sweep --data-dir data/surrogate --device cuda
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


# 5-run sweep design:
#   All 3 LRs (1e-4, 3e-4, 1e-3), all 3 hidden_dims (256x3, 512x3, 512x4)
#   lr=3e-4 as pivot connecting model sizes
SWEEP_CONFIGS = [
    {
        "lr": 1e-4,
        "hidden_dims": "256,256,256",
        "name": "sweep_lr1e4_h256x3",
    },
    {
        "lr": 3e-4,
        "hidden_dims": "256,256,256",
        "name": "sweep_lr3e4_h256x3",
    },
    {
        "lr": 3e-4,
        "hidden_dims": "512,512,512",
        "name": "sweep_lr3e4_h512x3",
    },
    {
        "lr": 3e-4,
        "hidden_dims": "512,512,512,512",
        "name": "sweep_lr3e4_h512x4",
    },
    {
        "lr": 1e-3,
        "hidden_dims": "512,512,512",
        "name": "sweep_lr1e3_h512x3",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hyperparameter sweep for surrogate model training"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/surrogate",
        help="Data directory (passed through to train_surrogate)",
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (passed through to train_surrogate)",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Max epochs per run (default: 200)",
    )
    parser.add_argument(
        "--output-base", type=str, default="output/surrogate",
        help="Base output directory for sweep runs",
    )
    return parser.parse_args()


def run_sweep(args: argparse.Namespace) -> None:
    """Execute all sweep configurations sequentially."""
    total_start = time.monotonic()
    results = []

    print("=" * 70)
    print("SURROGATE MODEL HYPERPARAMETER SWEEP")
    print("=" * 70)
    print(f"  Configurations: {len(SWEEP_CONFIGS)}")
    print(f"  Data dir:       {args.data_dir}")
    print(f"  Device:         {args.device}")
    print(f"  Max epochs:     {args.epochs}")
    print(f"  Output base:    {args.output_base}")
    print()

    for i, cfg in enumerate(SWEEP_CONFIGS, 1):
        save_dir = str(Path(args.output_base) / cfg["name"])
        print("-" * 70)
        print(f"[{i}/{len(SWEEP_CONFIGS)}] {cfg['name']}")
        print(f"  LR: {cfg['lr']}, Hidden dims: {cfg['hidden_dims']}")
        print(f"  Save dir: {save_dir}")
        print("-" * 70)

        cmd = [
            sys.executable, "-m", "aprx_model_elastica.train_surrogate",
            "--lr", str(cfg["lr"]),
            "--hidden-dims", cfg["hidden_dims"],
            "--save-dir", save_dir,
            "--wandb",
            "--run-name", cfg["name"],
            "--epochs", str(args.epochs),
            "--save-best-val-loss",
            "--data-dir", args.data_dir,
            "--device", args.device,
        ]

        run_start = time.monotonic()
        result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))
        run_elapsed = time.monotonic() - run_start

        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        results.append({
            "name": cfg["name"],
            "lr": cfg["lr"],
            "hidden_dims": cfg["hidden_dims"],
            "status": status,
            "elapsed_s": run_elapsed,
            "returncode": result.returncode,
        })

        minutes = run_elapsed / 60
        print(f"\n  [{cfg['name']}] {status} in {minutes:.1f} min\n")

    total_elapsed = time.monotonic() - total_start

    # Print summary table
    print()
    print("=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Run':<30s} {'LR':<10s} {'Hidden':<18s} {'Val Loss':<14s} {'Time':<10s} {'Status'}")
    print("-" * 92)

    best_loss = float("inf")
    best_name = None

    for r in results:
        metrics_path = Path(args.output_base) / r["name"] / "metrics.json"
        val_loss_str = "N/A"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            val_loss = metrics.get("best_val_loss", float("inf"))
            val_loss_str = f"{val_loss:.6f}"
            if val_loss < best_loss:
                best_loss = val_loss
                best_name = r["name"]
        elapsed_str = f"{r['elapsed_s'] / 60:.1f} min"
        print(
            f"{r['name']:<30s} {r['lr']:<10.0e} {r['hidden_dims']:<18s} "
            f"{val_loss_str:<14s} {elapsed_str:<10s} {r['status']}"
        )

    print("-" * 92)
    total_min = total_elapsed / 60
    print(f"Total time: {total_min:.1f} min ({total_elapsed / 3600:.1f} hrs)")

    if best_name:
        print(f"\nBest run: {best_name} (val_loss={best_loss:.6f})")
        print(f"  Checkpoint: {args.output_base}/{best_name}/model.pt")
    else:
        print("\nNo runs completed successfully.")

    # Save sweep summary
    summary_path = Path(args.output_base) / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "configs": SWEEP_CONFIGS,
                "results": results,
                "best_run": best_name,
                "best_val_loss": best_loss if best_name else None,
                "total_elapsed_s": total_elapsed,
            },
            f,
            indent=2,
        )
    print(f"Sweep summary saved to {summary_path}")


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)
