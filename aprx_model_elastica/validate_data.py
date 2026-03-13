"""Validate surrogate dataset: distribution, quality, temporal, and coverage analysis.

Loads all collected batch files (no train/val split), computes DVAL-01 through
DVAL-04 metrics, generates diagnostic figures, and writes a structured markdown
report with pass/fail rubric and actionable recommendations.

Usage:
    python -m aprx_model_elastica.validate_data
    python -m aprx_model_elastica.validate_data --data-dir data/surrogate --fig-dir figures/data_validation
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import numpy as np
from scipy import stats as scipy_stats

from aprx_model_elastica.state import (
    STATE_DIM,
    ACTION_DIM,
    NUM_NODES,
    NUM_ELEMENTS,
    POS_X,
    POS_Y,
    VEL_X,
    VEL_Y,
    YAW,
    OMEGA_Z,
)


# ---------------------------------------------------------------------------
# Action dimension names (for reporting)
# ---------------------------------------------------------------------------
ACTION_NAMES = ["amplitude", "frequency", "wave_number", "phase_offset", "direction_bias"]

# State group definitions for grouped reporting
STATE_GROUPS = {
    "pos_x": POS_X,
    "pos_y": POS_Y,
    "vel_x": VEL_X,
    "vel_y": VEL_Y,
    "yaw": YAW,
    "omega_z": OMEGA_Z,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_all_batches(data_dir: str) -> dict:
    """Load and concatenate ALL batch files without train/val split.

    Globs ``data_dir/batch_*.pt``, concatenates all tensors, and offsets
    episode_ids per file to make them globally unique.

    Args:
        data_dir: Directory containing ``batch_*.pt`` files.

    Returns:
        Dictionary with keys: states (N, 124), actions (N, 5),
        next_states (N, 124), episode_ids (N,), step_indices (N,),
        serpenoid_times (N,), n_batches (int).
    """
    data_path = Path(data_dir)
    batch_files = sorted(data_path.glob("batch_*.pt"))
    if not batch_files:
        raise FileNotFoundError(f"No batch_*.pt files found in {data_dir}")

    all_states: List[torch.Tensor] = []
    all_actions: List[torch.Tensor] = []
    all_next_states: List[torch.Tensor] = []
    all_episode_ids: List[torch.Tensor] = []
    all_step_indices: List[torch.Tensor] = []
    all_serp_times: List[torch.Tensor] = []

    episode_offset = 0
    for bf in batch_files:
        data = torch.load(bf, map_location="cpu", weights_only=True)
        all_states.append(data["states"])
        all_actions.append(data["actions"])
        all_next_states.append(data["next_states"])
        all_episode_ids.append(data["episode_ids"] + episode_offset)
        # Phase 2.2 flat format uses "step_ids"; Phase 1 uses "step_indices"
        step_key = "step_ids" if "step_ids" in data else "step_indices"
        all_step_indices.append(data[step_key])
        # Phase 2.2 flat format uses "t_start"; Phase 1 uses "serpenoid_times"
        time_key = "t_start" if "t_start" in data else "serpenoid_times"
        all_serp_times.append(data[time_key])
        episode_offset = all_episode_ids[-1].max().item() + 1

    return {
        "states": torch.cat(all_states, dim=0),
        "actions": torch.cat(all_actions, dim=0),
        "next_states": torch.cat(all_next_states, dim=0),
        "episode_ids": torch.cat(all_episode_ids, dim=0),
        "step_indices": torch.cat(all_step_indices, dim=0),
        "serpenoid_times": torch.cat(all_serp_times, dim=0),
        "n_batches": len(batch_files),
    }


# ---------------------------------------------------------------------------
# Summary feature extraction
# ---------------------------------------------------------------------------


def extract_summary_features(states: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Project 124-D states to 4 summary features (matches compute_density_weights).

    Args:
        states: (N, 124) state tensor.

    Returns:
        Dict mapping feature name to (N,) tensor:
        CoM_x, CoM_y, vel_mag, mean_omega.
    """
    com_x = states[:, POS_X].mean(dim=1)
    com_y = states[:, POS_Y].mean(dim=1)
    vel_mag = (states[:, VEL_X] ** 2 + states[:, VEL_Y] ** 2).sqrt().mean(dim=1)
    mean_omega = states[:, OMEGA_Z].abs().mean(dim=1)
    return {
        "CoM_x": com_x,
        "CoM_y": com_y,
        "vel_mag": vel_mag,
        "mean_omega": mean_omega,
    }


# ---------------------------------------------------------------------------
# DVAL-01: Distribution analysis
# ---------------------------------------------------------------------------


def analyze_distributions(data: dict) -> dict:
    """Compute distribution statistics for summary features and action dims.

    Args:
        data: Output of ``load_all_batches``.

    Returns:
        Dict with ``summary_features`` and ``action_dims`` sub-dicts, each
        containing per-feature stats (min, max, mean, std, skewness, kurtosis)
        and histogram data (bin_edges, counts).
    """
    n_bins = 50

    summary_feats = extract_summary_features(data["states"])
    actions = data["actions"]

    result: Dict[str, Any] = {"summary_features": {}, "action_dims": {}}

    # Summary features
    for name, tensor in summary_feats.items():
        arr = tensor.numpy()
        counts, bin_edges = np.histogram(arr, bins=n_bins)
        result["summary_features"][name] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "skewness": float(scipy_stats.skew(arr)),
            "kurtosis": float(scipy_stats.kurtosis(arr)),
            "bin_edges": bin_edges,
            "counts": counts,
        }

    # Action dimensions
    for dim_idx, dim_name in enumerate(ACTION_NAMES):
        arr = actions[:, dim_idx].numpy()
        counts, bin_edges = np.histogram(arr, bins=n_bins)
        result["action_dims"][dim_name] = {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "skewness": float(scipy_stats.skew(arr)),
            "kurtosis": float(scipy_stats.kurtosis(arr)),
            "bin_edges": bin_edges,
            "counts": counts,
        }

    return result


# ---------------------------------------------------------------------------
# DVAL-02: Data quality checks
# ---------------------------------------------------------------------------


def check_data_quality(data: dict) -> dict:
    """Run data quality checks: NaN/Inf, duplicates, constant features, outliers.

    Args:
        data: Output of ``load_all_batches``.

    Returns:
        Dict with quality metrics.
    """
    states = data["states"]
    next_states = data["next_states"]
    actions = data["actions"]
    n_total = states.shape[0]

    # --- NaN / Inf ---
    nan_inf_states = (~torch.isfinite(states)).sum().item()
    nan_inf_next = (~torch.isfinite(next_states)).sum().item()
    nan_inf_actions = (~torch.isfinite(actions)).sum().item()
    nan_inf_total = nan_inf_states + nan_inf_next + nan_inf_actions
    total_elements = states.numel() + next_states.numel() + actions.numel()
    nan_inf_rate = nan_inf_total / total_elements if total_elements > 0 else 0.0

    # --- Duplicate transitions ---
    # Hash approach: random projection to detect collisions
    torch.manual_seed(42)
    state_proj = torch.randn(STATE_DIM, 1)
    action_proj = torch.randn(ACTION_DIM, 1)
    state_hash = (states @ state_proj).squeeze(-1)
    action_hash = (actions @ action_proj).squeeze(-1)
    combined_hash = state_hash + action_hash * 1e6
    # Quantize to 6 decimal places to catch near-duplicates
    quantized = (combined_hash * 1e6).round()
    unique_count = len(torch.unique(quantized))
    duplicate_count = n_total - unique_count
    duplicate_rate = duplicate_count / n_total if n_total > 0 else 0.0

    # --- Constant / near-constant features ---
    all_features = torch.cat([states, actions], dim=1)  # (N, 129)
    feature_std = all_features.std(dim=0)
    constant_threshold = 1e-6
    constant_mask = feature_std < constant_threshold
    constant_indices = torch.where(constant_mask)[0].tolist()
    # Classify: state dims 0-123, action dims 124-128
    constant_state_dims = [i for i in constant_indices if i < STATE_DIM]
    constant_action_dims = [i - STATE_DIM for i in constant_indices if i >= STATE_DIM]

    # --- Outlier detection (>5 sigma per state dim) ---
    state_mean = states.mean(dim=0)
    state_std = states.std(dim=0)
    # Avoid division by zero for constant dims
    safe_std = state_std.clone()
    safe_std[safe_std < 1e-10] = 1.0
    z_scores = ((states - state_mean) / safe_std).abs()
    outlier_mask = z_scores > 5.0
    outlier_per_dim = outlier_mask.sum(dim=0).tolist()  # (124,)
    outlier_total = outlier_mask.sum().item()
    outlier_rate = outlier_total / states.numel() if states.numel() > 0 else 0.0

    # Group outliers by state component
    outlier_by_group = {}
    for group_name, slc in STATE_GROUPS.items():
        group_outliers = outlier_mask[:, slc].sum().item()
        outlier_by_group[group_name] = int(group_outliers)

    return {
        "nan_inf": {
            "states": nan_inf_states,
            "next_states": nan_inf_next,
            "actions": nan_inf_actions,
            "total": nan_inf_total,
            "rate": nan_inf_rate,
        },
        "duplicates": {
            "count": duplicate_count,
            "rate": duplicate_rate,
            "unique": unique_count,
        },
        "constant_features": {
            "count": len(constant_indices),
            "state_dims": constant_state_dims,
            "action_dims": constant_action_dims,
            "feature_stds_min": float(feature_std.min()),
        },
        "outliers": {
            "total": outlier_total,
            "rate": outlier_rate,
            "per_dim": outlier_per_dim,
            "by_group": outlier_by_group,
        },
    }


# ---------------------------------------------------------------------------
# DVAL-03: Temporal analysis
# ---------------------------------------------------------------------------


def analyze_temporal(data: dict) -> dict:
    """Analyze episode length distribution and step index bias.

    Args:
        data: Output of ``load_all_batches``.

    Returns:
        Dict with episode_lengths, step_histogram, bias_ratio.
    """
    episode_ids = data["episode_ids"]
    step_indices = data["step_indices"]

    # --- Episode lengths ---
    unique_eps, ep_counts = torch.unique(episode_ids, return_counts=True)
    ep_lengths = ep_counts.numpy().astype(float)

    ep_stats = {
        "min": float(ep_lengths.min()),
        "max": float(ep_lengths.max()),
        "mean": float(ep_lengths.mean()),
        "median": float(np.median(ep_lengths)),
        "std": float(ep_lengths.std()),
        "n_episodes": len(unique_eps),
    }

    # --- Step index distribution ---
    step_arr = step_indices.numpy()
    step_min, step_max = int(step_arr.min()), int(step_arr.max())
    n_step_bins = min(50, step_max - step_min + 1)
    step_counts, step_bin_edges = np.histogram(step_arr, bins=n_step_bins)

    # Early-vs-late bias: ratio of transitions in first quartile vs last quartile
    step_range = step_max - step_min
    q1_boundary = step_min + step_range * 0.25
    q4_boundary = step_min + step_range * 0.75
    first_q_count = int((step_arr <= q1_boundary).sum())
    last_q_count = int((step_arr >= q4_boundary).sum())
    bias_ratio = first_q_count / max(last_q_count, 1)

    return {
        "episode_lengths": ep_lengths,
        "episode_stats": ep_stats,
        "step_histogram": {
            "counts": step_counts,
            "bin_edges": step_bin_edges,
        },
        "step_range": {"min": step_min, "max": step_max},
        "bias_ratio": bias_ratio,
        "first_quartile_count": first_q_count,
        "last_quartile_count": last_q_count,
    }


# ---------------------------------------------------------------------------
# DVAL-04: Action coverage
# ---------------------------------------------------------------------------


def analyze_action_coverage(data: dict, n_bins: int = 20) -> dict:
    """Analyze action space coverage: per-dim fill and 5D joint fill fraction.

    Args:
        data: Output of ``load_all_batches``.
        n_bins: Number of bins per dimension.

    Returns:
        Dict with per_dim_histograms, per_dim_fill, joint_fill_fraction,
        under_sampled_regions.
    """
    actions = data["actions"]  # (N, 5)
    n_samples = actions.shape[0]

    per_dim_histograms = {}
    per_dim_fill = {}
    under_sampled_regions: Dict[str, List[str]] = {}

    for dim_idx, dim_name in enumerate(ACTION_NAMES):
        col = actions[:, dim_idx]
        # Bin in [-1, 1]
        bin_edges = torch.linspace(-1.0, 1.0, n_bins + 1)
        bin_idx = ((col - (-1.0)) / (2.0 / n_bins)).clamp(0, n_bins - 1).long()
        bin_counts = torch.zeros(n_bins, dtype=torch.long)
        for b in range(n_bins):
            bin_counts[b] = (bin_idx == b).sum()

        filled = (bin_counts > 0).sum().item()
        fill_frac = filled / n_bins

        # Under-sampled bins: count < mean_count / 3
        mean_count = bin_counts.float().mean().item()
        threshold = mean_count / 3.0
        under_sampled = []
        for b in range(n_bins):
            if bin_counts[b].item() < threshold:
                lo = bin_edges[b].item()
                hi = bin_edges[b + 1].item()
                under_sampled.append(f"[{lo:.2f}, {hi:.2f}]")

        per_dim_histograms[dim_name] = {
            "bin_edges": bin_edges.numpy(),
            "counts": bin_counts.numpy(),
        }
        per_dim_fill[dim_name] = fill_frac
        under_sampled_regions[dim_name] = under_sampled

    # --- 5D joint coverage ---
    # Compute flat bin index for each sample
    flat_bin_idx = torch.zeros(n_samples, dtype=torch.long)
    for dim_idx in range(ACTION_DIM):
        col = actions[:, dim_idx]
        bins = ((col - (-1.0)) / (2.0 / n_bins)).clamp(0, n_bins - 1).long()
        flat_bin_idx = flat_bin_idx * n_bins + bins

    unique_bins = torch.unique(flat_bin_idx)
    total_possible = n_bins ** ACTION_DIM
    joint_fill_fraction = len(unique_bins) / total_possible

    return {
        "per_dim_histograms": per_dim_histograms,
        "per_dim_fill": per_dim_fill,
        "joint_fill_fraction": joint_fill_fraction,
        "joint_occupied_bins": len(unique_bins),
        "joint_total_bins": total_possible,
        "under_sampled_regions": under_sampled_regions,
        "n_bins": n_bins,
    }


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_figures(
    dist_results: dict,
    quality_results: dict,
    temporal_results: dict,
    coverage_results: dict,
    save_dir: str,
) -> List[str]:
    """Generate and save all validation diagnostic figures.

    Follows established pattern: Agg backend, dpi=150, bbox_inches="tight".

    Args:
        dist_results: Output of ``analyze_distributions``.
        quality_results: Output of ``check_data_quality``.
        temporal_results: Output of ``analyze_temporal``.
        coverage_results: Output of ``analyze_action_coverage``.
        save_dir: Directory to save figures (created if needed).

    Returns:
        List of saved file paths.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    saved_files: List[str] = []

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    # ---- 1. Summary feature histograms (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Summary Feature Distributions", fontsize=14, fontweight="bold")
    for idx, (name, stats) in enumerate(dist_results["summary_features"].items()):
        ax = axes[idx // 2, idx % 2]
        bin_centers = (stats["bin_edges"][:-1] + stats["bin_edges"][1:]) / 2
        ax.bar(bin_centers, stats["counts"], width=np.diff(stats["bin_edges"]),
               color=colors[idx], alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_title(name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        info = f"mean={stats['mean']:.4f}\nstd={stats['std']:.4f}\nskew={stats['skewness']:.2f}"
        ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fpath = save_path / "summary_feature_histograms.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(str(fpath))

    # ---- 2. Action histograms (2x3 grid, 5 dims) ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Action Dimension Distributions", fontsize=14, fontweight="bold")
    action_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
    axes[1, 2].set_visible(False)  # hide unused subplot
    for idx, (name, stats) in enumerate(dist_results["action_dims"].items()):
        ax = action_axes[idx]
        bin_centers = (stats["bin_edges"][:-1] + stats["bin_edges"][1:]) / 2
        ax.bar(bin_centers, stats["counts"], width=np.diff(stats["bin_edges"]),
               color=colors[idx + 4], alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_title(name)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        info = f"mean={stats['mean']:.4f}\nstd={stats['std']:.4f}\nskew={stats['skewness']:.2f}"
        ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fpath = save_path / "action_histograms.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(str(fpath))

    # ---- 3. Outlier counts by state group ----
    fig, ax = plt.subplots(figsize=(10, 5))
    groups = list(quality_results["outliers"]["by_group"].keys())
    counts = [quality_results["outliers"]["by_group"][g] for g in groups]
    bar_colors = [colors[i] for i in range(len(groups))]
    ax.bar(groups, counts, color=bar_colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_title("Outlier Counts by State Group (>5 sigma)", fontsize=13, fontweight="bold")
    ax.set_xlabel("State Group")
    ax.set_ylabel("Outlier Count")
    for i, c in enumerate(counts):
        ax.text(i, c + max(counts) * 0.01, str(c), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fpath = save_path / "outlier_counts.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(str(fpath))

    # ---- 4. Episode length distribution ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ep_lengths = temporal_results["episode_lengths"]
    ax.hist(ep_lengths, bins=min(50, len(np.unique(ep_lengths))),
            color=colors[2], alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_title("Episode Length Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Episode Length (transitions)")
    ax.set_ylabel("Count")
    stats = temporal_results["episode_stats"]
    info = (f"N={stats['n_episodes']}\n"
            f"mean={stats['mean']:.1f}\n"
            f"median={stats['median']:.1f}\n"
            f"std={stats['std']:.1f}")
    ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    fig.tight_layout()
    fpath = save_path / "episode_length_distribution.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(str(fpath))

    # ---- 5. Step index distribution ----
    fig, ax = plt.subplots(figsize=(10, 5))
    step_hist = temporal_results["step_histogram"]
    bin_centers = (step_hist["bin_edges"][:-1] + step_hist["bin_edges"][1:]) / 2
    ax.bar(bin_centers, step_hist["counts"],
           width=np.diff(step_hist["bin_edges"]),
           color=colors[3], alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.set_title("Step Index Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Step Index")
    ax.set_ylabel("Count")
    # Draw quartile boundary lines
    sr = temporal_results["step_range"]
    step_range = sr["max"] - sr["min"]
    q1 = sr["min"] + step_range * 0.25
    q3 = sr["min"] + step_range * 0.75
    ax.axvline(q1, color="red", linestyle="--", alpha=0.7, label=f"Q1 ({q1:.0f})")
    ax.axvline(q3, color="red", linestyle="--", alpha=0.7, label=f"Q3 ({q3:.0f})")
    ax.legend()
    info = f"Bias ratio (Q1/Q4): {temporal_results['bias_ratio']:.3f}"
    ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    fig.tight_layout()
    fpath = save_path / "step_index_distribution.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(str(fpath))

    # ---- 6. Action coverage per dimension ----
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("Action Coverage Per Dimension", fontsize=14, fontweight="bold")
    for idx, dim_name in enumerate(ACTION_NAMES):
        ax = axes[idx]
        hist_data = coverage_results["per_dim_histograms"][dim_name]
        bin_edges = hist_data["bin_edges"]
        counts_arr = hist_data["counts"]
        mean_count = counts_arr.mean()
        threshold = mean_count / 3.0

        bar_colors_dim = []
        for c in counts_arr:
            if c < threshold:
                bar_colors_dim.append("red")
            else:
                bar_colors_dim.append(colors[idx])

        bin_centers_dim = (bin_edges[:-1] + bin_edges[1:]) / 2
        widths = np.diff(bin_edges)
        ax.bar(bin_centers_dim, counts_arr, width=widths,
               color=bar_colors_dim, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_title(f"{dim_name}\nfill={coverage_results['per_dim_fill'][dim_name]:.0%}")
        ax.set_xlabel("Value")
        if idx == 0:
            ax.set_ylabel("Count")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fpath = save_path / "action_coverage_per_dim.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(str(fpath))

    # ---- 7. Action coverage heatmap (5D fill fraction bar) ----
    fig, ax = plt.subplots(figsize=(6, 3))
    fill_pct = coverage_results["joint_fill_fraction"] * 100
    ax.barh(["5D Coverage"], [fill_pct], color=colors[1], alpha=0.8,
            edgecolor="black", linewidth=0.5, height=0.4)
    ax.set_xlim(0, max(fill_pct * 1.5, 10))
    ax.set_xlabel("Fill Fraction (%)")
    ax.set_title("5D Joint Action Coverage", fontsize=13, fontweight="bold")
    ax.text(fill_pct + 0.3, 0, f"{fill_pct:.2f}%", va="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fpath = save_path / "action_coverage_heatmap.png"
    fig.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_files.append(str(fpath))

    return saved_files


# ---------------------------------------------------------------------------
# Pass/Fail Rubric
# ---------------------------------------------------------------------------


def _evaluate_rubric(
    quality_results: dict,
    temporal_results: dict,
    coverage_results: dict,
) -> List[Dict[str, str]]:
    """Evaluate pass/fail rubric for all metrics.

    Returns list of dicts with keys: metric, value, threshold, status.
    """
    rubric: List[Dict[str, str]] = []

    # 1. NaN/Inf rate
    nan_rate = quality_results["nan_inf"]["rate"]
    nan_pct = nan_rate * 100
    if nan_rate == 0:
        status = "PASS"
    elif nan_pct < 0.1:
        status = "WARN"
    else:
        status = "FAIL"
    rubric.append({
        "metric": "NaN/Inf rate",
        "value": f"{nan_pct:.4f}%",
        "threshold": "PASS: 0% | WARN: <0.1% | FAIL: >=0.1%",
        "status": status,
    })

    # 2. Duplicate rate
    dup_rate = quality_results["duplicates"]["rate"]
    dup_pct = dup_rate * 100
    if dup_pct < 0.1:
        status = "PASS"
    elif dup_pct < 1.0:
        status = "WARN"
    else:
        status = "FAIL"
    rubric.append({
        "metric": "Duplicate rate",
        "value": f"{dup_pct:.4f}%",
        "threshold": "PASS: <0.1% | WARN: <1% | FAIL: >=1%",
        "status": status,
    })

    # 3. Constant features
    n_const = quality_results["constant_features"]["count"]
    if n_const == 0:
        status = "PASS"
    elif n_const <= 3:
        status = "WARN"
    else:
        status = "FAIL"
    rubric.append({
        "metric": "Constant features",
        "value": str(n_const),
        "threshold": "PASS: 0 | WARN: 1-3 | FAIL: >3",
        "status": status,
    })

    # 4. Outlier rate (>5 sigma)
    outlier_pct = quality_results["outliers"]["rate"] * 100
    if outlier_pct < 0.5:
        status = "PASS"
    elif outlier_pct < 2.0:
        status = "WARN"
    else:
        status = "FAIL"
    rubric.append({
        "metric": "Outlier rate (>5 sigma)",
        "value": f"{outlier_pct:.4f}%",
        "threshold": "PASS: <0.5% | WARN: <2% | FAIL: >=2%",
        "status": status,
    })

    # 5. Episode length CV
    ep_stats = temporal_results["episode_stats"]
    cv = ep_stats["std"] / ep_stats["mean"] if ep_stats["mean"] > 0 else 0.0
    if cv < 0.5:
        status = "PASS"
    elif cv < 1.0:
        status = "WARN"
    else:
        status = "FAIL"
    rubric.append({
        "metric": "Episode length CV",
        "value": f"{cv:.4f}",
        "threshold": "PASS: <0.5 | WARN: <1.0 | FAIL: >=1.0",
        "status": status,
    })

    # 6. Step index bias (Q1/Q4 ratio)
    bias = temporal_results["bias_ratio"]
    if 0.7 <= bias <= 1.3:
        status = "PASS"
    elif 0.5 <= bias <= 1.5:
        status = "WARN"
    else:
        status = "FAIL"
    rubric.append({
        "metric": "Step index bias (Q1/Q4)",
        "value": f"{bias:.4f}",
        "threshold": "PASS: 0.7-1.3 | WARN: 0.5-1.5 | FAIL: outside",
        "status": status,
    })

    # 7. Action 5D fill fraction
    fill_pct = coverage_results["joint_fill_fraction"] * 100
    if fill_pct > 5.0:
        status = "PASS"
    elif fill_pct > 1.0:
        status = "WARN"
    else:
        status = "FAIL"
    rubric.append({
        "metric": "Action 5D fill fraction",
        "value": f"{fill_pct:.2f}%",
        "threshold": "PASS: >5% | WARN: >1% | FAIL: <=1%",
        "status": status,
    })

    # 8. Per-dim action fill
    fills = coverage_results["per_dim_fill"]
    min_fill = min(fills.values())
    min_fill_pct = min_fill * 100
    if min_fill_pct > 90:
        status = "PASS"
    elif min_fill_pct > 70:
        status = "WARN"
    else:
        status = "FAIL"
    dim_detail = ", ".join(f"{n}={v:.0%}" for n, v in fills.items())
    rubric.append({
        "metric": "Per-dim action fill",
        "value": f"min={min_fill_pct:.0f}% ({dim_detail})",
        "threshold": "PASS: all >90% | WARN: all >70% | FAIL: any <=70%",
        "status": status,
    })

    return rubric


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------


def write_report(
    all_results: dict,
    report_path: str,
) -> None:
    """Write a structured markdown validation report.

    Args:
        all_results: Dict with keys: dist, quality, temporal, coverage, data_summary.
        report_path: Output path for the markdown report.
    """
    report_dir = Path(report_path).parent
    report_dir.mkdir(parents=True, exist_ok=True)

    dist = all_results["dist"]
    quality = all_results["quality"]
    temporal = all_results["temporal"]
    coverage = all_results["coverage"]
    summary = all_results["data_summary"]

    rubric = _evaluate_rubric(quality, temporal, coverage)

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pass_count = sum(1 for r in rubric if r["status"] == "PASS")
    warn_count = sum(1 for r in rubric if r["status"] == "WARN")
    fail_count = sum(1 for r in rubric if r["status"] == "FAIL")

    lines: List[str] = []

    # ---- Header ----
    lines.append("# Surrogate Data Validation Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Dataset:** {summary['n_transitions']:,} transitions, "
                 f"{summary['n_batches']} batch files, "
                 f"{summary['n_episodes']} episodes")
    lines.append("")

    # ---- Pass/Fail Rubric ----
    lines.append("## Pass/Fail Rubric")
    lines.append("")
    lines.append(f"**Summary:** {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")
    lines.append("")
    lines.append("| Metric | Value | Threshold | Status |")
    lines.append("|--------|-------|-----------|--------|")
    for r in rubric:
        status_icon = {"PASS": "PASS", "WARN": "WARN", "FAIL": "FAIL"}[r["status"]]
        lines.append(f"| {r['metric']} | {r['value']} | {r['threshold']} | **{status_icon}** |")
    lines.append("")

    # ---- Distribution Analysis ----
    lines.append("## Distribution Analysis (DVAL-01)")
    lines.append("")
    lines.append("### Summary Features")
    lines.append("")
    lines.append("| Feature | Min | Max | Mean | Std | Skewness | Kurtosis |")
    lines.append("|---------|-----|-----|------|-----|----------|----------|")
    for name, stats in dist["summary_features"].items():
        lines.append(
            f"| {name} | {stats['min']:.4f} | {stats['max']:.4f} | "
            f"{stats['mean']:.4f} | {stats['std']:.4f} | "
            f"{stats['skewness']:.4f} | {stats['kurtosis']:.4f} |"
        )
    lines.append("")
    lines.append("### Action Dimensions")
    lines.append("")
    lines.append("| Dimension | Min | Max | Mean | Std | Skewness | Kurtosis |")
    lines.append("|-----------|-----|-----|------|-----|----------|----------|")
    for name, stats in dist["action_dims"].items():
        lines.append(
            f"| {name} | {stats['min']:.4f} | {stats['max']:.4f} | "
            f"{stats['mean']:.4f} | {stats['std']:.4f} | "
            f"{stats['skewness']:.4f} | {stats['kurtosis']:.4f} |"
        )
    lines.append("")
    lines.append("See: `figures/data_validation/summary_feature_histograms.png`, "
                 "`figures/data_validation/action_histograms.png`")
    lines.append("")

    # ---- Data Quality ----
    lines.append("## Data Quality (DVAL-02)")
    lines.append("")

    nan = quality["nan_inf"]
    lines.append("### NaN/Inf Values")
    lines.append(f"- States: {nan['states']}")
    lines.append(f"- Next states: {nan['next_states']}")
    lines.append(f"- Actions: {nan['actions']}")
    lines.append(f"- **Total: {nan['total']} ({nan['rate']*100:.4f}%)**")
    lines.append("")

    dup = quality["duplicates"]
    lines.append("### Duplicate Transitions")
    lines.append(f"- Duplicates detected: {dup['count']:,} ({dup['rate']*100:.4f}%)")
    lines.append(f"- Unique transitions: {dup['unique']:,}")
    lines.append("")

    const = quality["constant_features"]
    lines.append("### Constant/Near-Constant Features (std < 1e-6)")
    lines.append(f"- Count: {const['count']}")
    if const["state_dims"]:
        lines.append(f"- State dims: {const['state_dims']}")
    if const["action_dims"]:
        lines.append(f"- Action dims: {const['action_dims']}")
    if const["count"] == 0:
        lines.append("- None detected")
    lines.append("")

    outliers = quality["outliers"]
    lines.append("### Outlier Detection (>5 sigma)")
    lines.append(f"- Total outliers: {outliers['total']:,} ({outliers['rate']*100:.4f}%)")
    lines.append("- By state group:")
    for group, count in outliers["by_group"].items():
        lines.append(f"  - {group}: {count:,}")
    lines.append("")
    lines.append("See: `figures/data_validation/outlier_counts.png`")
    lines.append("")

    # ---- Temporal Analysis ----
    lines.append("## Temporal Analysis (DVAL-03)")
    lines.append("")
    ep = temporal["episode_stats"]
    lines.append("### Episode Lengths")
    lines.append(f"- Episodes: {ep['n_episodes']}")
    lines.append(f"- Min: {ep['min']:.0f}, Max: {ep['max']:.0f}")
    lines.append(f"- Mean: {ep['mean']:.1f}, Median: {ep['median']:.1f}")
    lines.append(f"- Std: {ep['std']:.1f}")
    cv = ep["std"] / ep["mean"] if ep["mean"] > 0 else 0
    lines.append(f"- CV (std/mean): {cv:.4f}")
    lines.append("")
    lines.append("### Step Index Bias")
    lines.append(f"- Step range: {temporal['step_range']['min']} to {temporal['step_range']['max']}")
    lines.append(f"- First quartile transitions: {temporal['first_quartile_count']:,}")
    lines.append(f"- Last quartile transitions: {temporal['last_quartile_count']:,}")
    lines.append(f"- **Bias ratio (Q1/Q4): {temporal['bias_ratio']:.4f}**")
    lines.append("")
    lines.append("See: `figures/data_validation/episode_length_distribution.png`, "
                 "`figures/data_validation/step_index_distribution.png`")
    lines.append("")

    # ---- Action Coverage ----
    lines.append("## Action Coverage (DVAL-04)")
    lines.append("")
    lines.append("### Per-Dimension Fill")
    lines.append(f"- Bins per dimension: {coverage['n_bins']}")
    for dim_name, fill in coverage["per_dim_fill"].items():
        lines.append(f"- {dim_name}: {fill:.0%}")
    lines.append("")
    lines.append("### 5D Joint Coverage")
    lines.append(f"- Occupied bins: {coverage['joint_occupied_bins']:,} / "
                 f"{coverage['joint_total_bins']:,}")
    lines.append(f"- **Fill fraction: {coverage['joint_fill_fraction']*100:.2f}%**")
    lines.append("")
    lines.append("### Under-Sampled Regions")
    any_under = False
    for dim_name, regions in coverage["under_sampled_regions"].items():
        if regions:
            any_under = True
            lines.append(f"- **{dim_name}:** {', '.join(regions)}")
    if not any_under:
        lines.append("- None detected (all bins above threshold)")
    lines.append("")
    lines.append("See: `figures/data_validation/action_coverage_per_dim.png`, "
                 "`figures/data_validation/action_coverage_heatmap.png`")
    lines.append("")

    # ---- Recommendations ----
    lines.append("## Recommendations")
    lines.append("")
    recommendations = _generate_recommendations(quality, temporal, coverage, rubric)
    for rec in recommendations:
        lines.append(f"- {rec}")
    lines.append("")

    # ---- Overall Assessment ----
    lines.append("## Overall Assessment")
    lines.append("")
    if fail_count == 0 and warn_count <= 2:
        assessment = "**GO** -- Dataset is ready for surrogate model training."
    elif fail_count == 0:
        assessment = ("**CONDITIONAL GO** -- Dataset is usable but has warnings. "
                      "Consider addressing warnings before training.")
    else:
        assessment = ("**NO-GO** -- Dataset has failing metrics that should be "
                      "addressed before surrogate training.")
    lines.append(assessment)
    lines.append("")

    # Write
    with open(report_path, "w") as f:
        f.write("\n".join(lines))


def _generate_recommendations(
    quality: dict,
    temporal: dict,
    coverage: dict,
    rubric: List[Dict[str, str]],
) -> List[str]:
    """Generate prose recommendations based on analysis results."""
    recs: List[str] = []

    # NaN/Inf
    if quality["nan_inf"]["total"] > 0:
        recs.append(
            f"NaN/Inf values detected ({quality['nan_inf']['total']} total). "
            "Investigate physics stability -- check for divergence in PyElastica simulation."
        )

    # Duplicates
    if quality["duplicates"]["rate"] > 0.01:
        recs.append(
            f"Duplicate rate is {quality['duplicates']['rate']*100:.2f}%. "
            "Consider deduplicating before training or verifying collection pipeline."
        )

    # Outliers
    outlier_pct = quality["outliers"]["rate"] * 100
    if outlier_pct > 0.5:
        # Find the worst group
        worst_group = max(quality["outliers"]["by_group"],
                          key=lambda k: quality["outliers"]["by_group"][k])
        recs.append(
            f"Outlier rate is {outlier_pct:.2f}%, concentrated in {worst_group}. "
            "Consider checking physics stability or applying outlier clipping during training."
        )

    # Temporal bias
    bias = temporal["bias_ratio"]
    if bias < 0.7:
        recs.append(
            f"Step index biased toward late timesteps (bias ratio {bias:.2f}). "
            "Consider running more short episodes to increase early-step coverage."
        )
    elif bias > 1.3:
        recs.append(
            f"Step index biased toward early timesteps (bias ratio {bias:.2f}). "
            "Consider running longer episodes to increase late-step coverage."
        )

    # Action coverage
    for dim_name, regions in coverage["under_sampled_regions"].items():
        if len(regions) > 3:
            recs.append(
                f"Action dimension '{dim_name}' has {len(regions)} under-sampled bins "
                f"(out of {coverage['n_bins']}): {', '.join(regions[:5])}. "
                "Consider targeted recollection for this range."
            )

    fill_pct = coverage["joint_fill_fraction"] * 100
    if fill_pct < 5.0:
        recs.append(
            f"5D joint action coverage is {fill_pct:.2f}% "
            f"({coverage['joint_occupied_bins']:,}/{coverage['joint_total_bins']:,} bins). "
            "This is inherently sparse for 5D space; density-weighted sampling during "
            "training can compensate."
        )

    # Episode length
    ep = temporal["episode_stats"]
    cv = ep["std"] / ep["mean"] if ep["mean"] > 0 else 0
    if cv >= 1.0:
        recs.append(
            f"Episode length CV is {cv:.2f} (high variability). "
            "This may cause uneven step-index coverage. Consider standardizing episode lengths."
        )

    if not recs:
        recs.append("No significant issues found. Dataset appears well-suited for surrogate training.")

    return recs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def run_validation(
    data_dir: str = "data/surrogate",
    fig_dir: str = "figures/data_validation",
    report_path: str = "data/surrogate/validation_report.md",
) -> dict:
    """Run full validation pipeline: load, analyze, save figures, write report.

    Args:
        data_dir: Directory containing batch_*.pt files.
        fig_dir: Directory to save diagnostic figures.
        report_path: Path for the markdown report.

    Returns:
        Dict with all analysis results.
    """
    print(f"Loading data from {data_dir}...")
    data = load_all_batches(data_dir)

    n_transitions = data["states"].shape[0]
    n_episodes = int(torch.unique(data["episode_ids"]).shape[0])
    print(f"  Loaded {n_transitions:,} transitions from {data['n_batches']} batch files "
          f"({n_episodes} episodes)")

    print("Running distribution analysis (DVAL-01)...")
    dist_results = analyze_distributions(data)

    print("Running data quality checks (DVAL-02)...")
    quality_results = check_data_quality(data)

    print("Running temporal analysis (DVAL-03)...")
    temporal_results = analyze_temporal(data)

    print("Running action coverage analysis (DVAL-04)...")
    coverage_results = analyze_action_coverage(data)

    print(f"Saving figures to {fig_dir}...")
    saved_figs = save_figures(dist_results, quality_results, temporal_results,
                              coverage_results, fig_dir)
    print(f"  Saved {len(saved_figs)} figures")

    all_results = {
        "dist": dist_results,
        "quality": quality_results,
        "temporal": temporal_results,
        "coverage": coverage_results,
        "data_summary": {
            "n_transitions": n_transitions,
            "n_batches": data["n_batches"],
            "n_episodes": n_episodes,
        },
    }

    print(f"Writing report to {report_path}...")
    write_report(all_results, report_path)

    # Print summary to stdout
    rubric = _evaluate_rubric(quality_results, temporal_results, coverage_results)
    pass_count = sum(1 for r in rubric if r["status"] == "PASS")
    warn_count = sum(1 for r in rubric if r["status"] == "WARN")
    fail_count = sum(1 for r in rubric if r["status"] == "FAIL")

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Transitions: {n_transitions:,}")
    print(f"  Batches: {data['n_batches']}")
    print(f"  Episodes: {n_episodes}")
    print(f"  Results: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL")

    if fail_count == 0 and warn_count <= 2:
        print("  Assessment: GO -- ready for surrogate training")
    elif fail_count == 0:
        print("  Assessment: CONDITIONAL GO -- warnings present")
    else:
        print("  Assessment: NO-GO -- failing metrics need attention")
    print("=" * 60)

    return all_results


def main() -> None:
    """CLI entry point for data validation."""
    parser = argparse.ArgumentParser(
        description="Validate surrogate dataset quality and coverage"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/surrogate",
        help="Directory containing batch_*.pt files (default: data/surrogate)",
    )
    parser.add_argument(
        "--fig-dir",
        type=str,
        default="figures/data_validation",
        help="Directory to save figures (default: figures/data_validation)",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="data/surrogate/validation_report.md",
        help="Path for markdown report (default: data/surrogate/validation_report.md)",
    )
    args = parser.parse_args()

    run_validation(
        data_dir=args.data_dir,
        fig_dir=args.fig_dir,
        report_path=args.report_path,
    )


if __name__ == "__main__":
    main()
