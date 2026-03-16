"""Combined hyperparameter + architecture sweep for the surrogate model.

All 10 runs launched in parallel, covering two orthogonal dimensions:

  LR / model-size sweep (rollout_weight=0.1, rollout_steps=8, base MLP):
    sweep_lr1e4_h256x3  : lr=1e-4, 256×3
    sweep_lr3e4_h256x3  : lr=3e-4, 256×3
    sweep_lr3e4_h512x3  : lr=3e-4, 512×3
    sweep_lr3e4_h512x4  : lr=3e-4, 512×4
    sweep_lr1e3_h512x3  : lr=1e-3, 512×3  ← shared baseline

  Rollout loss ablation (lr=1e-3, 512×3, base MLP):
    arch_A1_rw0.0       : rollout_weight=0.0, rollout_steps=8  (single-step only)
    arch_A3_rw0.3       : rollout_weight=0.3, rollout_steps=8
    arch_A4_rw0.5       : rollout_weight=0.5, rollout_steps=8
    arch_A5_rw0.3_s16   : rollout_weight=0.3, rollout_steps=16

  Architecture (lr=1e-3, 512×3, rollout_weight=0.1, rollout_steps=8):
    arch_B1_residual    : ResidualSurrogateModel

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
    # --- LR / model-size sweep ---
    {
        "name": "sweep_lr1e4_h256x3",
        "lr": 1e-4,
        "hidden_dims": "256,256,256",
        "rollout_weight": 0.1,
        "rollout_steps": 8,
        "use_residual": False,
    },
    {
        "name": "sweep_lr3e4_h256x3",
        "lr": 3e-4,
        "hidden_dims": "256,256,256",
        "rollout_weight": 0.1,
        "rollout_steps": 8,
        "use_residual": False,
    },
    {
        "name": "sweep_lr3e4_h512x3",
        "lr": 3e-4,
        "hidden_dims": "512,512,512",
        "rollout_weight": 0.1,
        "rollout_steps": 8,
        "use_residual": False,
    },
    {
        "name": "sweep_lr3e4_h512x4",
        "lr": 3e-4,
        "hidden_dims": "512,512,512,512",
        "rollout_weight": 0.1,
        "rollout_steps": 8,
        "use_residual": False,
    },
    {
        "name": "sweep_lr1e3_h512x3",
        "lr": 1e-3,
        "hidden_dims": "512,512,512",
        "rollout_weight": 0.1,
        "rollout_steps": 8,
        "use_residual": False,
    },
    # --- Rollout loss ablation (lr=1e-3, 512×3, base MLP) ---
    {
        "name": "arch_A1_rw0.0",
        "lr": 1e-3,
        "hidden_dims": "512,512,512",
        "rollout_weight": 0.0,
        "rollout_steps": 8,
        "use_residual": False,
    },
    {
        "name": "arch_A3_rw0.3",
        "lr": 1e-3,
        "hidden_dims": "512,512,512",
        "rollout_weight": 0.3,
        "rollout_steps": 8,
        "use_residual": False,
    },
    {
        "name": "arch_A4_rw0.5",
        "lr": 1e-3,
        "hidden_dims": "512,512,512",
        "rollout_weight": 0.5,
        "rollout_steps": 8,
        "use_residual": False,
    },
    {
        "name": "arch_A5_rw0.3_s16",
        "lr": 1e-3,
        "hidden_dims": "512,512,512",
        "rollout_weight": 0.3,
        "rollout_steps": 16,
        "use_residual": False,
    },
    # --- Architecture variant (lr=1e-3, 512×3, rollout_weight=0.1) ---
    {
        "name": "arch_B1_residual",
        "lr": 1e-3,
        "hidden_dims": "512,512,512",
        "rollout_weight": 0.1,
        "rollout_steps": 8,
        "use_residual": True,
    },
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
    return "residual" if cfg.get("use_residual") else "base MLP"


def _print_config_table(configs: list[dict], output_base: str) -> None:
    print()
    print(f"{'Run':<28s} {'LR':<8s} {'Hidden':<14s} {'RW':<6s} {'Steps':<6s} {'Arch':<10s} {'Val Loss':<12s} {'Status'}")
    print("-" * 98)
    for cfg in configs:
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
        print(
            f"{cfg['name']:<28s} {cfg['lr']:<8.0e} {hidden_str:<14s} "
            f"{cfg['rollout_weight']:<6.2f} {cfg['rollout_steps']:<6d} "
            f"{_arch_label(cfg):<10s} {val_loss_str:<12s} {status_str}"
        )
    print("-" * 98)


def run_sweep(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_base = Path(args.output_base)

    print("=" * 70)
    print("SURROGATE MODEL COMBINED SWEEP (parallel)")
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
    procs = []
    log_files = []

    for cfg in SWEEP_CONFIGS:
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
            "--rollout-weight", str(cfg["rollout_weight"]),
            "--rollout-steps", str(cfg["rollout_steps"]),
        ]
        if cfg.get("use_residual"):
            cmd.append("--use-residual")

        log_f = open(log_path, "w")
        proc = subprocess.Popen(cmd, cwd=str(repo_root), env=env, stdout=log_f, stderr=log_f)
        procs.append(proc)
        log_files.append((log_path, log_f))
        print(f"  Launched {cfg['name']} (pid={proc.pid}) → {log_path.name}")

    print(f"\nAll {len(procs)} runs launched. Waiting for completion...")
    print(f"Monitor: tail -f {args.output_base}/*.log\n")

    for proc, (log_path, log_f), cfg in zip(procs, log_files, SWEEP_CONFIGS):
        proc.wait()
        log_f.close()
        status = "OK" if proc.returncode == 0 else f"FAILED (rc={proc.returncode})"
        print(f"  [{cfg['name']}] {status}")

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
        metrics_path = output_base / cfg["name"] / "metrics.json"
        val_loss = float("inf")
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
