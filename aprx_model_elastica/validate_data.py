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
        all_step_indices.append(data["step_indices"])
        all_serp_times.append(data["serpenoid_times"])
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
