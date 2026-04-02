#!/usr/bin/env python3
"""Analyze reward design ablation results.

Usage:
    python3 script/analyze_reward_ablation.py output/ablation_YYYYMMDD_HHMMSS/

Reads *_metrics.jsonl files from the ablation results directory and produces
a comparison table + per-question analysis.
"""

import json
import os
import statistics
import sys
from pathlib import Path


def load_metrics(path: str) -> list[dict]:
    metrics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def summarize(metrics: list[dict]) -> dict:
    """Compute summary stats from a metrics file."""
    n = len(metrics)
    if n == 0:
        return {}

    last = metrics[-1]
    q = max(1, n // 4)

    # Quarter averages for trend detection
    q1 = metrics[:q]
    q4 = metrics[-q:]

    def qmean(data, key):
        vals = [m[key] for m in data if key in m]
        return statistics.mean(vals) if vals else float("nan")

    # Per-step SNR
    def snr(m):
        rm = abs(m.get("diagnostics/reward_mean", 0))
        rs = m.get("diagnostics/reward_std", 1e-8)
        return rm / (rs + 1e-8)

    return {
        "total_frames": last["total_frames"],
        "total_episodes": last["total_episodes"],
        # Primary metrics
        "dist_q1": qmean(q1, "mean_dist_to_goal"),
        "dist_q4": qmean(q4, "mean_dist_to_goal"),
        "dist_delta": qmean(q4, "mean_dist_to_goal") - qmean(q1, "mean_dist_to_goal"),
        "reward_q1": qmean(q1, "mean_episode_reward"),
        "reward_q4": qmean(q4, "mean_episode_reward"),
        "best_reward": last.get("best_reward", 0),
        # Diagnostics
        "ev_q1": qmean(q1, "explained_variance"),
        "ev_q4": qmean(q4, "explained_variance"),
        "std_min_q4": qmean(q4, "diagnostics/action_std_min"),
        "grad_norm_q4": qmean(q4, "grad_norm"),
        "snr_q1": statistics.mean([snr(m) for m in q1]),
        "snr_q4": statistics.mean([snr(m) for m in q4]),
        # Components (Q4 average, raw/unweighted)
        "comp_dist": qmean(q4, "mean_reward_dist"),
        "comp_pbrs": qmean(q4, "mean_reward_pbrs"),
        "comp_smooth": qmean(q4, "mean_reward_smooth"),
        "comp_align": qmean(q4, "mean_reward_align"),
    }


def print_table(experiments: dict[str, dict]):
    """Print comparison table."""
    # Header
    print(f"{'ID':4s} {'Dist(Q1)':>8s} {'Dist(Q4)':>8s} {'Delta':>7s} {'Rew(Q4)':>8s} "
          f"{'Best':>6s} {'EV(Q4)':>7s} {'SNR(Q4)':>8s} {'GradN':>7s} {'StdMin':>7s} "
          f"{'Frames':>10s}")
    print("-" * 100)

    for exp_id, s in sorted(experiments.items()):
        print(f"{exp_id:4s} "
              f"{s['dist_q1']:8.3f} {s['dist_q4']:8.3f} {s['dist_delta']:+7.3f} "
              f"{s['reward_q4']:8.2f} {s['best_reward']:6.1f} "
              f"{s['ev_q4']:7.3f} {s['snr_q4']:8.3f} "
              f"{s['grad_norm_q4']:7.1f} {s['std_min_q4']:7.4f} "
              f"{s['total_frames']:>10,}")


def analyze_comparisons(experiments: dict[str, dict]):
    """Run the comparison plan."""
    comparisons = [
        ("Does trained beat random?", "A1", None, "dist_q4", "lower",
         "If A1 dist ≈ A0, task too hard for ANY reward config without curriculum."),
        ("Does PBRS help dense?", "A1", "B2", "dist_q4", "lower",
         "B2 < A1 means PBRS provides useful shaping signal."),
        ("Does reduced dense + PBRS work?", "B2", "B3", "dist_q4", "lower",
         "B3 ≤ B2 means reducing dense weight doesn't hurt when PBRS compensates."),
        ("Does smoothness help?", "B3", "B4", "dist_q4", "lower",
         "B4 < B3 means smooth penalty aids task performance (not just action quality)."),
        ("Does heading help?", "B3", "B5", "dist_q4", "lower",
         "B5 < B3 means directional heading bonus aids approach."),
        ("Heading vs smoothness?", "B4", "B5", "dist_q4", "lower",
         "Compare B4 vs B5 to see which auxiliary signal helps more."),
        ("Does curriculum help vanilla?", "A1", "C1", "dist_q4", "lower",
         "C1 << A1 means curriculum breaks the plateau."),
        ("Does curriculum help dense+PBRS?", "B2", "C2", "dist_q4", "lower",
         "C2 << B2 means curriculum + PBRS is the winning combination."),
        ("Does curriculum help current config?", "B4", "C3", "dist_q4", "lower",
         "C3 << B4 means curriculum is the missing ingredient."),
        ("Does longer warmup help?", "C3", "C5", "dist_q4", "lower",
         "C5 < C3 means 200 warmup episodes isn't enough."),
    ]

    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS")
    print("=" * 80)

    for question, id_a, id_b, metric, direction, interpretation in comparisons:
        print(f"\n--- {question} ---")
        if id_a not in experiments:
            print(f"  SKIP: {id_a} not found")
            continue
        if id_b is not None and id_b not in experiments:
            print(f"  SKIP: {id_b} not found")
            continue

        val_a = experiments[id_a][metric]
        if id_b is not None:
            val_b = experiments[id_b][metric]
            diff = val_b - val_a
            better = id_b if (direction == "lower" and diff < 0) else id_a
            pct = abs(diff) / (abs(val_a) + 1e-8) * 100
            print(f"  {id_a}: {val_a:.4f}   {id_b}: {val_b:.4f}   delta: {diff:+.4f} ({pct:.1f}%)")
            print(f"  Winner: {better}")
        else:
            print(f"  {id_a}: {val_a:.4f}")
        print(f"  → {interpretation}")


def find_best(experiments: dict[str, dict]):
    """Identify best config."""
    print("\n" + "=" * 80)
    print("RANKINGS (by dist_to_goal Q4, lower = better)")
    print("=" * 80)

    ranked = sorted(experiments.items(), key=lambda x: x[1]["dist_q4"])
    for i, (exp_id, s) in enumerate(ranked, 1):
        health = "HEALTHY" if s["ev_q4"] > 0.3 and s["std_min_q4"] > 0.1 and s["snr_q4"] > 0.1 else "WARN"
        print(f"  {i}. {exp_id:4s} dist={s['dist_q4']:.3f} reward={s['reward_q4']:.2f} "
              f"ev={s['ev_q4']:.3f} snr={s['snr_q4']:.3f} [{health}]")

    print(f"\n  Best: {ranked[0][0]} (dist={ranked[0][1]['dist_q4']:.3f})")

    # Best with healthy diagnostics
    healthy = [(eid, s) for eid, s in ranked
               if s["ev_q4"] > 0.3 and s["std_min_q4"] > 0.1 and s["snr_q4"] > 0.1]
    if healthy:
        print(f"  Best (healthy): {healthy[0][0]} (dist={healthy[0][1]['dist_q4']:.3f})")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"Directory not found: {results_dir}")
        sys.exit(1)

    # Load all experiment metrics
    experiments = {}
    for f in sorted(results_dir.glob("*_metrics.jsonl")):
        exp_id = f.stem.replace("_metrics", "")
        metrics = load_metrics(str(f))
        if metrics:
            experiments[exp_id] = summarize(metrics)

    if not experiments:
        print("No metrics files found!")
        sys.exit(1)

    print(f"Loaded {len(experiments)} experiments from {results_dir}\n")

    # Component table (what was enabled in each run)
    print("=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print_table(experiments)

    # Pairwise comparisons
    analyze_comparisons(experiments)

    # Rankings
    find_best(experiments)

    # Reward component breakdown
    print("\n" + "=" * 80)
    print("REWARD COMPONENTS (Q4 avg, raw/unweighted)")
    print("=" * 80)
    print(f"{'ID':4s} {'dist':>8s} {'pbrs':>8s} {'smooth':>8s} {'align':>8s}")
    print("-" * 40)
    for exp_id, s in sorted(experiments.items()):
        print(f"{exp_id:4s} {s['comp_dist']:+8.4f} {s['comp_pbrs']:+8.4f} "
              f"{s['comp_smooth']:+8.4f} {s['comp_align']:+8.4f}")


if __name__ == "__main__":
    main()
