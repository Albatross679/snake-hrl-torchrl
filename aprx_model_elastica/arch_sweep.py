"""Architecture experiment sweep runner for surrogate model.

Experiments A and B:
  A1: rollout_weight=0.0, rollout_steps=8  (ablation — single-step only)
  A2: rollout_weight=0.1, rollout_steps=8  (BASELINE — Phase 3 result, injected)
  A3: rollout_weight=0.3, rollout_steps=8  (stronger rollout signal)
  A4: rollout_weight=0.5, rollout_steps=8  (aggressive rollout)
  A5: rollout_weight=0.3, rollout_steps=16 (longer horizon)
  B1: use_residual=True, rollout_weight=0.1, rollout_steps=8 (residual MLP)

Total new runs: 5 (A1, A3, A4, A5, B1). A2 is injected from Phase 3 results.

Usage:
    python -m aprx_model_elastica.arch_sweep --dry-run
    python -m aprx_model_elastica.arch_sweep --epochs 200 --device cuda
    python -m aprx_model_elastica.arch_sweep --epochs 200 --device cuda \\
        --output-base output/surrogate/arch_sweep
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# Phase 3 baseline checkpoint (A2 — default rollout config from sweep_lr1e3_h512x3)
BASELINE_CHECKPOINT = "output/surrogate/sweep_lr1e3_h512x3"
BASELINE_FALLBACK_VAL_LOSS = 0.2161

# 5 new configs to run (A2 is injected as known baseline, not re-trained)
ARCH_SWEEP_CONFIGS = [
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
    {
        "name": "arch_B1_residual",
        "lr": 1e-3,
        "hidden_dims": "512,512,512",
        "rollout_weight": 0.1,
        "rollout_steps": 8,
        "use_residual": True,
    },
]


def _load_baseline_val_loss(repo_root: Path) -> float:
    """Load Phase 3 baseline val_loss from metrics.json, fallback to hardcoded."""
    metrics_path = repo_root / BASELINE_CHECKPOINT / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
            return metrics.get("best_val_loss", BASELINE_FALLBACK_VAL_LOSS)
        except (json.JSONDecodeError, IOError):
            pass
    return BASELINE_FALLBACK_VAL_LOSS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run architecture experiments for surrogate model (Experiments A and B)"
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
        "--output-base", type=str, default="output/surrogate/arch_sweep",
        help="Base output directory for arch sweep runs (default: output/surrogate/arch_sweep)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Print config table and exit without launching any training subprocesses",
    )
    return parser.parse_args()


def _arch_label(cfg: dict) -> str:
    """Return display label for architecture column."""
    return "residual" if cfg.get("use_residual") else "base MLP"


def _print_config_table(configs: list[dict], baseline: dict, output_base: str) -> None:
    """Print sweep config table (used in both dry-run and live execution)."""
    print()
    print(f"{'Run':<30s} {'LR':<8s} {'RW':<6s} {'Steps':<6s} {'Arch':<10s} {'Val Loss':<12s} {'Time':<10s} {'Status'}")
    print("-" * 92)

    # Baseline row first
    b_loss = baseline.get("val_loss", BASELINE_FALLBACK_VAL_LOSS)
    b_loss_str = f"{b_loss:.6f}"
    print(
        f"{baseline['name']:<30s} {'1e-03':<8s} {'0.100':<6s} {'8':<6s} "
        f"{'base MLP':<10s} {b_loss_str:<12s} {'N/A':<10s} SKIPPED (Phase 3 result)"
    )

    for cfg in configs:
        rw_str = f"{cfg['rollout_weight']:.3f}"
        steps_str = str(cfg["rollout_steps"])
        arch_str = _arch_label(cfg)
        run_dir = Path(output_base) / cfg["name"]
        metrics_path = run_dir / "metrics.json"

        val_loss_str = "pending"
        time_str = "pending"
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

        print(
            f"{cfg['name']:<30s} {'1e-03':<8s} {rw_str:<6s} {steps_str:<6s} "
            f"{arch_str:<10s} {val_loss_str:<12s} {time_str:<10s} {status_str}"
        )

    print("-" * 92)


def run_arch_sweep(args: argparse.Namespace) -> None:
    """Execute all arch sweep configurations sequentially."""
    repo_root = Path(__file__).resolve().parent.parent
    total_start = time.monotonic()

    # Load baseline from Phase 3 results
    baseline_val_loss = _load_baseline_val_loss(repo_root)
    baseline = {
        "name": "arch_A2_rw0.1_BASELINE",
        "lr": 1e-3,
        "hidden_dims": "512,512,512",
        "rollout_weight": 0.1,
        "rollout_steps": 8,
        "use_residual": False,
        "status": "SKIPPED (Phase 3 result)",
        "elapsed_s": 0,
        "returncode": 0,
        "val_loss": baseline_val_loss,
    }

    print("=" * 70)
    print("ARCH SWEEP — Surrogate Architecture Experiments")
    print("=" * 70)
    print(f"  New configurations: {len(ARCH_SWEEP_CONFIGS)}")
    print(f"  Baseline (injected): {baseline['name']} val_loss={baseline_val_loss:.6f}")
    print(f"  Data dir:            {args.data_dir}")
    print(f"  Device:              {args.device}")
    print(f"  Max epochs:          {args.epochs}")
    print(f"  Output base:         {args.output_base}")
    print()

    # Dry-run: print config table and exit without training
    if args.dry_run:
        print("[DRY RUN] Config table (no training will be launched):")
        _print_config_table(ARCH_SWEEP_CONFIGS, baseline, args.output_base)
        print()
        print("[DRY RUN] Exiting — pass without --dry-run to launch training.")
        return

    # Ensure output directory exists
    output_base = Path(args.output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    results = [baseline]  # baseline is first entry in results

    for i, cfg in enumerate(ARCH_SWEEP_CONFIGS, 1):
        save_dir = str(output_base / cfg["name"])
        print("-" * 70)
        print(f"[{i}/{len(ARCH_SWEEP_CONFIGS)}] {cfg['name']}")
        print(f"  LR: {cfg['lr']}, Hidden dims: {cfg['hidden_dims']}")
        print(f"  Rollout weight: {cfg['rollout_weight']}, Steps: {cfg['rollout_steps']}")
        print(f"  Architecture: {_arch_label(cfg)}")
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
            "--rollout-weight", str(cfg["rollout_weight"]),
            "--rollout-steps", str(cfg["rollout_steps"]),
        ]
        if cfg.get("use_residual"):
            cmd.append("--use-residual")

        # Prevent OpenBLAS thread exhaustion in DataLoader workers
        env = os.environ.copy()
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"

        run_start = time.monotonic()
        result = subprocess.run(
            cmd, cwd=str(repo_root), env=env
        )
        run_elapsed = time.monotonic() - run_start

        status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
        run_result = {
            "name": cfg["name"],
            "lr": cfg["lr"],
            "hidden_dims": cfg["hidden_dims"],
            "rollout_weight": cfg["rollout_weight"],
            "rollout_steps": cfg["rollout_steps"],
            "use_residual": cfg.get("use_residual", False),
            "status": status,
            "elapsed_s": run_elapsed,
            "returncode": result.returncode,
        }

        # Read metrics.json
        metrics_path = output_base / cfg["name"] / "metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                run_result["val_loss"] = metrics.get("best_val_loss", float("inf"))
                run_result["final_epoch"] = metrics.get("final_epoch")
            except (json.JSONDecodeError, IOError):
                run_result["val_loss"] = float("inf")
        else:
            run_result["val_loss"] = float("inf")

        results.append(run_result)

        minutes = run_elapsed / 60
        print(f"\n  [{cfg['name']}] {status} in {minutes:.1f} min | val_loss={run_result['val_loss']:.6f}\n")

        # Save intermediate summary after each run
        _save_summary(output_base, baseline, ARCH_SWEEP_CONFIGS, results)

    total_elapsed = time.monotonic() - total_start

    # Print final summary table
    print()
    print("=" * 70)
    print("ARCH SWEEP SUMMARY")
    print("=" * 70)
    _print_config_table(ARCH_SWEEP_CONFIGS, baseline, args.output_base)

    # Find best run (excluding baseline)
    trained_results = [r for r in results if r.get("returncode") == 0 and r["name"] != baseline["name"]]
    best_run = None
    best_val_loss = baseline_val_loss  # start comparison at baseline

    for r in trained_results:
        vl = r.get("val_loss", float("inf"))
        if vl < best_val_loss:
            best_val_loss = vl
            best_run = r["name"]

    total_min = total_elapsed / 60
    print(f"\nTotal time: {total_min:.1f} min ({total_elapsed / 3600:.1f} hrs)")
    print(f"Baseline:   {baseline['name']} val_loss={baseline_val_loss:.6f}")

    if best_run:
        improvement = baseline_val_loss - best_val_loss
        print(f"Best run:   {best_run} val_loss={best_val_loss:.6f}")
        print(f"Improvement vs baseline: {improvement:+.6f}")
        print(f"  Checkpoint: {args.output_base}/{best_run}/model.pt")
    else:
        print("\nNo run beat the baseline.")
        best_run = baseline["name"]
        best_val_loss = baseline_val_loss

    # Save final summary JSON
    summary = _save_summary(output_base, baseline, ARCH_SWEEP_CONFIGS, results,
                            best_run=best_run, best_val_loss=best_val_loss,
                            total_elapsed_s=total_elapsed)
    summary_path = output_base / "arch_sweep_summary.json"
    print(f"\nArch sweep summary saved to {summary_path}")
    print(f"Monitor: tmux attach -t gsd-train  OR  tail -f output/surrogate/arch_sweep.log")


def _save_summary(
    output_base: Path,
    baseline: dict,
    configs: list[dict],
    results: list[dict],
    best_run: str | None = None,
    best_val_loss: float | None = None,
    total_elapsed_s: float = 0.0,
) -> dict:
    """Save arch_sweep_summary.json to output_base. Returns the summary dict."""
    # Determine best if not provided
    if best_run is None:
        trained = [r for r in results if r.get("name") != baseline["name"] and r.get("returncode") == 0]
        if trained:
            best_r = min(trained, key=lambda r: r.get("val_loss", float("inf")))
            best_run = best_r["name"]
            best_val_loss = best_r.get("val_loss", float("inf"))
        else:
            best_run = baseline["name"]
            best_val_loss = baseline.get("val_loss", BASELINE_FALLBACK_VAL_LOSS)

    summary = {
        "baseline": {
            "name": baseline["name"],
            "val_loss": baseline.get("val_loss", BASELINE_FALLBACK_VAL_LOSS),
        },
        "configs": configs,
        "results": results,
        "best_run": best_run,
        "best_val_loss": best_val_loss,
        "total_elapsed_s": total_elapsed_s,
    }

    output_base.mkdir(parents=True, exist_ok=True)
    summary_path = output_base / "arch_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    args = parse_args()
    run_arch_sweep(args)
