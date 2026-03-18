"""15-config architecture sweep for the surrogate model.

All configs use Phase 02.2 FlatStepDataset data, rollout_weight=0.0,
auto-batch for GPU VRAM, and W&B logging.

Configurations:
  MLP (5 configs):
    M1: lr=1e-4, 256x3
    M2: lr=3e-4, 256x3
    M3: lr=1e-4, 512x3
    M4: lr=3e-4, 512x3
    M5: lr=1e-3, 512x3

  Residual MLP (3 configs):
    R1: lr=3e-4, 512x3
    R2: lr=1e-3, 512x3
    R3: lr=3e-4, 1024x3

  Wide/Deep MLP (3 configs):
    W1: lr=3e-4, 512x4
    W2: lr=3e-4, 1024x3
    W3: lr=1e-3, 1024x3

  FT-Transformer (4 configs):
    T1: lr=3e-4, 4 layers, 4 heads, d_model=128
    T2: lr=3e-4, 6 layers, 8 heads, d_model=256
    T3: lr=1e-4, 6 layers, 8 heads, d_model=256
    T4: lr=1e-4, 8 layers, 8 heads, d_model=512

Usage:
    python -m aprx_model_elastica.sweep --dry-run
    python -m aprx_model_elastica.sweep --data-dir data/surrogate_rl_step --device cuda
    python -m aprx_model_elastica.sweep --data-dir data/surrogate_rl_step --device cuda \\
        --epochs 200 --output-base output/surrogate
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


SWEEP_CONFIGS = [
    # --- MLP (5 configs) ---
    {
        "name": "M1",
        "lr": 1e-4,
        "hidden_dims": "256,256,256",
        "arch": "mlp",
        "rollout_weight": 0.0,
    },
    {
        "name": "M2",
        "lr": 3e-4,
        "hidden_dims": "256,256,256",
        "arch": "mlp",
        "rollout_weight": 0.0,
    },
    {
        "name": "M3",
        "lr": 1e-4,
        "hidden_dims": "512,512,512",
        "arch": "mlp",
        "rollout_weight": 0.0,
    },
    {
        "name": "M4",
        "lr": 3e-4,
        "hidden_dims": "512,512,512",
        "arch": "mlp",
        "rollout_weight": 0.0,
    },
    {
        "name": "M5",
        "lr": 1e-3,
        "hidden_dims": "512,512,512",
        "arch": "mlp",
        "rollout_weight": 0.0,
    },
    # --- Residual MLP (3 configs) ---
    {
        "name": "R1",
        "lr": 3e-4,
        "hidden_dims": "512,512,512",
        "arch": "residual",
        "rollout_weight": 0.0,
    },
    {
        "name": "R2",
        "lr": 1e-3,
        "hidden_dims": "512,512,512",
        "arch": "residual",
        "rollout_weight": 0.0,
    },
    {
        "name": "R3",
        "lr": 3e-4,
        "hidden_dims": "1024,1024,1024",
        "arch": "residual",
        "rollout_weight": 0.0,
    },
    # --- Wide/Deep MLP (3 configs) ---
    {
        "name": "W1",
        "lr": 3e-4,
        "hidden_dims": "512,512,512,512",
        "arch": "mlp",
        "rollout_weight": 0.0,
    },
    {
        "name": "W2",
        "lr": 3e-4,
        "hidden_dims": "1024,1024,1024",
        "arch": "mlp",
        "rollout_weight": 0.0,
    },
    {
        "name": "W3",
        "lr": 1e-3,
        "hidden_dims": "1024,1024,1024",
        "arch": "mlp",
        "rollout_weight": 0.0,
    },
    # --- FT-Transformer (4 configs) ---
    {
        "name": "T1",
        "lr": 3e-4,
        "hidden_dims": "256,256,256",  # not used for transformer, but kept for config uniformity
        "arch": "transformer",
        "rollout_weight": 0.0,
        "n_layers": 4,
        "n_heads": 4,
        "d_model": 128,
    },
    {
        "name": "T2",
        "lr": 3e-4,
        "hidden_dims": "256,256,256",
        "arch": "transformer",
        "rollout_weight": 0.0,
        "n_layers": 6,
        "n_heads": 8,
        "d_model": 256,
    },
    {
        "name": "T3",
        "lr": 1e-4,
        "hidden_dims": "256,256,256",
        "arch": "transformer",
        "rollout_weight": 0.0,
        "n_layers": 6,
        "n_heads": 8,
        "d_model": 256,
    },
    {
        "name": "T4",
        "lr": 1e-4,
        "hidden_dims": "256,256,256",
        "arch": "transformer",
        "rollout_weight": 0.0,
        "n_layers": 8,
        "n_heads": 8,
        "d_model": 512,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run 15-config architecture sweep for the surrogate model"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/surrogate_rl_step",
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
        help="Base output directory for sweep runs (default: output/surrogate)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print config table and exit without launching training",
    )
    return parser.parse_args()


def _arch_label(cfg: dict) -> str:
    """Human-readable architecture label."""
    arch = cfg.get("arch", "mlp")
    if arch == "transformer":
        return f"transformer({cfg.get('n_layers', '?')}L/{cfg.get('n_heads', '?')}H/{cfg.get('d_model', '?')}d)"
    elif arch == "residual":
        return "residual"
    else:
        return "mlp"


def _print_config_table(configs: list, output_base: str) -> None:
    print()
    print(f"{'#':<4s} {'Run':<8s} {'LR':<8s} {'Hidden':<14s} {'Arch':<28s} {'RW':<6s} {'Val Loss':<12s} {'Status'}")
    print("-" * 94)
    for i, cfg in enumerate(configs, 1):
        run_dir = Path(output_base) / cfg["name"]
        metrics_path = run_dir / "metrics.json"
        val_loss_str = "pending"
        status_str = "pending"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                val_loss_str = f"{metrics.get('best_val_loss', float('inf')):.6f}"
            except (json.JSONDecodeError, IOError):
                val_loss_str = "error"
            status_str = "done"
        elif (run_dir / "checkpoints" / "model.pt").exists():
            status_str = "in progress"
        hidden_str = cfg["hidden_dims"].replace(",", "x")
        print(
            f"{i:<4d} {cfg['name']:<8s} {cfg['lr']:<8.0e} {hidden_str:<14s} "
            f"{_arch_label(cfg):<28s} {cfg['rollout_weight']:<6.1f} "
            f"{val_loss_str:<12s} {status_str}"
        )
    print("-" * 94)


def run_sweep(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent
    output_base = Path(args.output_base)

    print("=" * 70)
    print("SURROGATE MODEL 15-CONFIG ARCHITECTURE SWEEP")
    print("=" * 70)
    print(f"  Configurations: {len(SWEEP_CONFIGS)}")
    print(f"  Data dir:       {args.data_dir}")
    print(f"  Device:         {args.device}")
    print(f"  Max epochs:     {args.epochs}")
    print(f"  Output base:    {args.output_base}")
    print()

    if args.dry_run:
        print("[DRY RUN] Config table (no training will be launched):")
        _print_config_table(SWEEP_CONFIGS, args.output_base)
        print()
        print(f"[DRY RUN] {len(SWEEP_CONFIGS)} configs would be launched. "
              f"Pass without --dry-run to start training.")
        return

    output_base.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    total_start = time.monotonic()

    for cfg_idx, cfg in enumerate(SWEEP_CONFIGS):
        run_name = cfg["name"]
        log_path = output_base / f"{run_name}.log"

        run_save_dir = str(output_base / run_name)
        cmd = [
            sys.executable, "-m", "aprx_model_elastica.train_surrogate",
            "--lr", str(cfg["lr"]),
            "--hidden-dims", cfg["hidden_dims"],
            "--run-name", run_name,
            "--epochs", str(args.epochs),
            "--data-dir", args.data_dir,
            "--device", args.device,
            "--rollout-weight", str(cfg["rollout_weight"]),
            "--arch", cfg["arch"],
            "--save-dir", run_save_dir,
        ]
        # Transformer-specific args
        if cfg["arch"] == "transformer":
            cmd.extend(["--n-layers", str(cfg["n_layers"])])
            cmd.extend(["--n-heads", str(cfg["n_heads"])])
            cmd.extend(["--d-model", str(cfg["d_model"])])

        print(f"\n[{cfg_idx + 1}/{len(SWEEP_CONFIGS)}] Launching {run_name} (arch={cfg['arch']})...")
        with open(log_path, "w") as log_f:
            proc = subprocess.run(
                cmd, cwd=str(repo_root), env=env,
                stdout=log_f, stderr=log_f,
            )
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  [{run_name}] {status}  (log: {log_path})")

    total_elapsed = time.monotonic() - total_start

    print()
    print("=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    _print_config_table(SWEEP_CONFIGS, args.output_base)

    # Find best run
    best_name = None
    best_val_loss = float("inf")
    results = []
    for cfg in SWEEP_CONFIGS:
        # Look for metrics.json in timestamped run dirs
        val_loss = float("inf")
        metrics_path = output_base / cfg["name"] / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    val_loss = json.load(f).get("best_val_loss", float("inf"))
            except (json.JSONDecodeError, IOError):
                pass
        results.append({**cfg, "val_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_name = cfg["name"]

    total_min = total_elapsed / 60
    print(f"\nTotal time: {total_min:.1f} min ({total_elapsed / 3600:.1f} hrs)")
    if best_name:
        print(f"Best run:   {best_name}  val_loss={best_val_loss:.6f}")
        print(f"Checkpoint: {args.output_base}/{best_name}/")
    else:
        print("No runs completed successfully.")

    summary_path = output_base / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "configs": SWEEP_CONFIGS,
                "results": results,
                "best_run": best_name,
                "best_val_loss": best_val_loss if best_name else None,
                "total_elapsed_s": total_elapsed,
            },
            f,
            indent=2,
        )
    print(f"Sweep summary saved to {summary_path}")


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)
