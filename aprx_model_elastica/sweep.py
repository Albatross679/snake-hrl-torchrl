"""15-config architecture sweep for the surrogate model.

Covers 4 architecture families (all rollout_weight=0.0, single-step MSE):

  MLP (5 configs): M1-M5 — varying LR and hidden dims
  Residual MLP (3 configs): R1-R3 — skip connections, varying LR/width
  Wide/Deep MLP (3 configs): W1-W3 — larger hidden dims
  FT-Transformer (4 configs): T1-T4 — varying layers/heads/d_model

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
    # --- Wide/Deep MLP (2 configs: W1-W2) ---
    {"name": "W1", "arch": "mlp", "lr": 3e-4, "hidden_dims": "512,512,512,512", "rollout_weight": 0.0},
    {"name": "W2", "arch": "mlp", "lr": 3e-4, "hidden_dims": "1024,1024,1024", "rollout_weight": 0.0},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run combined hyperparameter + architecture sweep for the surrogate model"
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
    return cfg.get("arch", "mlp")


def _print_config_table(configs: list[dict], output_base: str) -> None:
    print()
    print(f"{'#':<4s} {'Run':<8s} {'Arch':<14s} {'LR':<8s} {'Hidden':<18s} {'Val Loss':<12s} {'Status'}")
    print("-" * 80)
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
        elif (run_dir / "model.pt").exists():
            status_str = "in progress"
        hidden_str = cfg["hidden_dims"].replace(",", "×")
        arch_str = _arch_label(cfg)
        if cfg.get("n_layers"):
            arch_str += f" L{cfg['n_layers']}H{cfg['n_heads']}d{cfg['d_model']}"
        print(
            f"{i:<4d} {cfg['name']:<8s} {arch_str:<14s} {cfg['lr']:<8.0e} "
            f"{hidden_str:<18s} {val_loss_str:<12s} {status_str}"
        )
    print("-" * 80)


def run_sweep(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_base = Path(args.output_base)

    print("=" * 70)
    print("SURROGATE MODEL 15-CONFIG SWEEP (sequential)")
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
        print("[DRY RUN] Exiting — pass without --dry-run to launch training.")
        return

    output_base.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"

    total_start = time.monotonic()
    results_list = []

    for i, cfg in enumerate(SWEEP_CONFIGS, 1):
        save_dir = str(output_base / cfg["name"])
        log_path = output_base / f"{cfg['name']}.log"

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
            "--arch", cfg.get("arch", "mlp"),
            "--auto-batch",
        ]
        # Transformer-specific args
        if cfg.get("n_layers"):
            cmd.extend(["--n-layers", str(cfg["n_layers"])])
        if cfg.get("n_heads"):
            cmd.extend(["--n-heads", str(cfg["n_heads"])])
        if cfg.get("d_model"):
            cmd.extend(["--d-model", str(cfg["d_model"])])

        print(f"\n[{i}/{len(SWEEP_CONFIGS)}] Running {cfg['name']} ({cfg.get('arch', 'mlp')})...")
        run_start = time.monotonic()
        with open(log_path, "w") as log_f:
            result = subprocess.run(cmd, cwd=str(repo_root), env=env, stdout=log_f, stderr=log_f)
        run_elapsed = time.monotonic() - run_start

        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        results_list.append({"name": cfg["name"], "status": status, "elapsed_s": run_elapsed, "returncode": result.returncode})
        print(f"  [{cfg['name']}] {status} in {run_elapsed/60:.1f} min")

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
    for cfg, run_info in zip(SWEEP_CONFIGS, results_list):
        metrics_path = output_base / cfg["name"] / "metrics.json"
        val_loss = float("inf")
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    val_loss = json.load(f).get("best_val_loss", float("inf"))
            except (json.JSONDecodeError, IOError):
                pass
        results.append({**cfg, "val_loss": val_loss, **run_info})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_name = cfg["name"]

    total_min = total_elapsed / 60
    print(f"\nTotal time: {total_min:.1f} min ({total_elapsed / 3600:.1f} hrs)")
    if best_name:
        print(f"Best run:   {best_name}  val_loss={best_val_loss:.6f}")
        print(f"Checkpoint: {args.output_base}/{best_name}/model.pt")
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
