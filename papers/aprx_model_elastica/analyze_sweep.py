"""Analyze 15-config architecture sweep results for surrogate model selection.

Produces:
    1. sweep_comparison.png:   Horizontal bar chart of val_loss for all configs
    2. per_component_rmse.png: Grouped bar chart of per-component RMSE (physical units)
    3. error_histograms.png:   Error distribution histograms for best model
    4. predicted_vs_actual.png: Scatter plots of predicted vs actual deltas

Usage:
    python3 -m aprx_model_elastica.analyze_sweep \
        --output-base output/surrogate \
        --figures-dir figures/surrogate_training
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from aprx_model_elastica.dataset import FlatStepDataset
from aprx_model_elastica.model import (
    ResidualSurrogateModel,
    SurrogateModel,
    TransformerSurrogateModel,
)
from aprx_model_elastica.state import (
    StateNormalizer,
    raw_to_relative,
    encode_phase_batch,
    encode_n_cycles_batch,
    action_to_omega_batch,
    REL_COM_X,
    REL_COM_Y,
    REL_COM_VEL_X,
    REL_COM_VEL_Y,
    REL_POS_X,
    REL_POS_Y,
    REL_VEL_X,
    REL_VEL_Y,
    REL_YAW,
    REL_OMEGA_Z,
    REL_STATE_DIM,
)
from aprx_model_elastica.train_config import SurrogateModelConfig, SurrogateTrainConfig


# ---------------------------------------------------------------------------
# Component definitions for per-component analysis
# ---------------------------------------------------------------------------

# Mapping from component name -> (slice into 130-dim relative state, physical unit, scale factor)
# Scale factors convert from the raw state units to physical display units.
# Positions are in meters -> multiply by 1000 for mm.
# Velocities are in m/s -> multiply by 1000 for mm/s.
# Yaw and omega_z are already in rad and rad/s.
COMPONENTS = {
    "pos_x": (REL_POS_X, "mm", 1000.0),
    "pos_y": (REL_POS_Y, "mm", 1000.0),
    "vel_x": (REL_VEL_X, "mm/s", 1000.0),
    "vel_y": (REL_VEL_Y, "mm/s", 1000.0),
    "yaw": (REL_YAW, "rad", 1.0),
    "omega_z": (REL_OMEGA_Z, "rad/s", 1.0),
}

# Architecture family -> color for sweep_comparison plot
ARCH_COLORS = {
    "mlp": "#4A90D9",         # Blue
    "residual": "#50C878",    # Green
    "wide_deep": "#E8A838",   # Orange (W1, W2, W3)
    "transformer": "#D9534F", # Red
}

# Wide/Deep MLP configs (identified by name prefix)
WIDE_DEEP_NAMES = {"W1", "W2", "W3"}


def _arch_family(cfg: dict) -> str:
    """Determine architecture family for coloring."""
    if cfg.get("name", "") in WIDE_DEEP_NAMES:
        return "wide_deep"
    return cfg.get("arch", "mlp")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_sweep_summary(output_base: Path) -> dict:
    """Load sweep_summary.json."""
    path = output_base / "sweep_summary.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run the sweep first.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def load_model_and_normalizer(
    config_dir: Path, device: str = "cpu"
) -> Tuple[Optional[nn.Module], Optional[StateNormalizer], Optional[dict]]:
    """Load model, normalizer, and config from a sweep config directory.

    Returns (model, normalizer, config_dict) or (None, None, None) if missing.
    """
    config_path = config_dir / "config.json"
    model_path = config_dir / "checkpoints" / "model.pt"
    normalizer_path = config_dir / "checkpoints" / "normalizer.pt"

    if not model_path.exists():
        return None, None, None

    # Load config
    with open(config_path) as f:
        config_dict = json.load(f)

    # Build model config
    model_cfg_dict = config_dict.get("model", {})
    model_cfg = SurrogateModelConfig(**{
        k: v for k, v in model_cfg_dict.items()
        if k in SurrogateModelConfig.__dataclass_fields__
    })

    # Build model based on architecture
    arch = model_cfg.arch
    if arch == "transformer":
        model = TransformerSurrogateModel(model_cfg)
    elif arch == "residual":
        model = ResidualSurrogateModel(model_cfg)
    else:
        model = SurrogateModel(model_cfg)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    normalizer = StateNormalizer.load(str(normalizer_path), device=device)

    return model, normalizer, config_dict


# ---------------------------------------------------------------------------
# Inference and metrics
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_inference(
    model: nn.Module,
    normalizer: StateNormalizer,
    val_states: torch.Tensor,
    val_actions: torch.Tensor,
    val_t_starts: torch.Tensor,
    val_next_states: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 8192,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run model inference on validation data.

    Returns:
        (predicted_deltas, actual_deltas) in physical (unnormalized, 130-dim) space.
    """
    n = val_states.shape[0]
    all_pred_deltas = []
    all_actual_deltas = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        states = val_states[start:end].to(device)
        actions = val_actions[start:end].to(device)
        t_starts = val_t_starts[start:end].to(device)
        next_states = val_next_states[start:end].to(device)

        # Build time encoding: [sin(phase), cos(phase), sin(n_cycles), cos(n_cycles)]
        omega = action_to_omega_batch(actions)
        phase = omega * t_starts
        phase_enc = encode_phase_batch(phase)
        n_cycles_enc = encode_n_cycles_batch(actions)
        time_enc = torch.cat([phase_enc, n_cycles_enc], dim=-1)

        # Normalize state
        state_norm = normalizer.normalize_state(states)

        # Predict delta (normalized)
        delta_norm = model(state_norm, actions, time_enc)

        # Denormalize delta
        pred_delta = normalizer.denormalize_delta(delta_norm)

        # Actual delta in physical space
        actual_delta = next_states - states

        all_pred_deltas.append(pred_delta.cpu())
        all_actual_deltas.append(actual_delta.cpu())

    return torch.cat(all_pred_deltas, dim=0), torch.cat(all_actual_deltas, dim=0)


def compute_component_metrics(
    pred_deltas: torch.Tensor,
    actual_deltas: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    """Compute per-component RMSE, MAE, and R-squared in physical units.

    Returns:
        Dict mapping component name -> {"rmse": float, "mae": float, "r2": float, "unit": str}
    """
    metrics = {}
    for name, (slc, unit, scale) in COMPONENTS.items():
        pred = pred_deltas[:, slc].numpy() * scale
        actual = actual_deltas[:, slc].numpy() * scale

        # Flatten: (N, n_features) -> (N * n_features,)
        pred_flat = pred.flatten()
        actual_flat = actual.flatten()

        errors = pred_flat - actual_flat
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        mae = float(np.mean(np.abs(errors)))

        # R-squared
        ss_res = np.sum(errors ** 2)
        ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
        r2 = float(1.0 - ss_res / (ss_tot + 1e-12))

        metrics[name] = {"rmse": rmse, "mae": mae, "r2": r2, "unit": unit}

    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_sweep_comparison(
    results: List[dict],
    figures_dir: Path,
) -> None:
    """Plot 1: Horizontal bar chart of val_loss for all configs."""
    # Sort by val_loss ascending (best at top)
    valid = [r for r in results if r["val_loss"] < float("inf")]
    valid.sort(key=lambda r: r["val_loss"])

    if not valid:
        print("WARNING: No valid results to plot for sweep_comparison.png")
        return

    names = [r["name"] for r in valid]
    losses = [r["val_loss"] for r in valid]
    colors = [ARCH_COLORS.get(_arch_family(r), "#888888") for r in valid]

    fig, ax = plt.subplots(figsize=(10, max(6, len(valid) * 0.5)))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, losses, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()  # Best at top
    ax.set_xlabel("Validation Loss (MSE)", fontsize=12)
    ax.set_title("Surrogate Model Sweep: Validation Loss by Configuration", fontsize=13)

    # Add value labels
    for bar, loss in zip(bars, losses):
        ax.text(
            bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{loss:.6f}", va="center", fontsize=8
        )

    # Legend for architecture families
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ARCH_COLORS["mlp"], edgecolor="black", label="MLP"),
        Patch(facecolor=ARCH_COLORS["residual"], edgecolor="black", label="Residual"),
        Patch(facecolor=ARCH_COLORS["wide_deep"], edgecolor="black", label="Wide/Deep"),
        Patch(facecolor=ARCH_COLORS["transformer"], edgecolor="black", label="Transformer"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = figures_dir / "sweep_comparison.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_per_component_rmse(
    top_metrics: Dict[str, Dict[str, Dict[str, float]]],
    figures_dir: Path,
) -> None:
    """Plot 2: Grouped bar chart of per-component RMSE for top models."""
    comp_names = list(COMPONENTS.keys())
    comp_labels = [
        f"{n} ({COMPONENTS[n][1]})" for n in comp_names
    ]
    model_names = list(top_metrics.keys())
    n_models = len(model_names)
    n_comps = len(comp_names)

    x = np.arange(n_comps)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.cm.Set2

    for i, model_name in enumerate(model_names):
        rmse_vals = [top_metrics[model_name][c]["rmse"] for c in comp_names]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, rmse_vals, width, label=model_name,
                      color=cmap(i / max(1, n_models - 1)), edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("RMSE (physical units)", fontsize=12)
    ax.set_title("Per-Component RMSE: Top 5 Models", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = figures_dir / "per_component_rmse.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_error_histograms(
    pred_deltas: torch.Tensor,
    actual_deltas: torch.Tensor,
    best_name: str,
    figures_dir: Path,
) -> None:
    """Plot 3: 6-panel error distribution histograms for best model."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    comp_names = list(COMPONENTS.keys())
    for idx, name in enumerate(comp_names):
        slc, unit, scale = COMPONENTS[name]
        errors = (pred_deltas[:, slc].numpy() - actual_deltas[:, slc].numpy()) * scale
        errors_flat = errors.flatten()
        rmse = float(np.sqrt(np.mean(errors_flat ** 2)))

        ax = axes[idx]
        ax.hist(errors_flat, bins=100, density=True, alpha=0.75,
                color="#4A90D9", edgecolor="black", linewidth=0.3)
        ax.set_title(f"{name} ({unit})  RMSE={rmse:.4f}", fontsize=11)
        ax.set_xlabel(f"Error ({unit})", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.7)

    fig.suptitle(f"Error Distributions: Best Model ({best_name})", fontsize=14, y=1.01)
    plt.tight_layout()
    path = figures_dir / "error_histograms.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_predicted_vs_actual(
    pred_deltas: torch.Tensor,
    actual_deltas: torch.Tensor,
    best_name: str,
    figures_dir: Path,
    max_points: int = 20000,
) -> None:
    """Plot 4: 6-panel scatter of predicted vs actual deltas for best model."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    comp_names = list(COMPONENTS.keys())
    for idx, name in enumerate(comp_names):
        slc, unit, scale = COMPONENTS[name]
        pred = pred_deltas[:, slc].numpy() * scale
        actual = actual_deltas[:, slc].numpy() * scale

        pred_flat = pred.flatten()
        actual_flat = actual.flatten()

        # Subsample for plotting
        if len(pred_flat) > max_points:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(pred_flat), max_points, replace=False)
            pred_flat = pred_flat[indices]
            actual_flat = actual_flat[indices]

        # R-squared
        ss_res = np.sum((pred_flat - actual_flat) ** 2)
        ss_tot = np.sum((actual_flat - np.mean(actual_flat)) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)

        ax = axes[idx]
        ax.scatter(actual_flat, pred_flat, s=1, alpha=0.15, color="#4A90D9", rasterized=True)

        # Diagonal reference line
        lims = [
            min(actual_flat.min(), pred_flat.min()),
            max(actual_flat.max(), pred_flat.max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7)

        ax.set_title(f"{name} ({unit})  R2={r2:.4f}", fontsize=11)
        ax.set_xlabel(f"Actual ({unit})", fontsize=9)
        ax.set_ylabel(f"Predicted ({unit})", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")

    fig.suptitle(f"Predicted vs Actual: Best Model ({best_name})", fontsize=14, y=1.01)
    plt.tight_layout()
    path = figures_dir / "predicted_vs_actual.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def print_ranked_table(
    ranked: List[Tuple[str, str, float, Dict[str, Dict[str, float]]]],
) -> None:
    """Print ranked summary table to stdout."""
    comp_names = list(COMPONENTS.keys())
    header = f"{'Rank':<5} {'Config':<8} {'Arch':<14} {'Val Loss':<12}"
    for c in comp_names:
        unit = COMPONENTS[c][1]
        header += f" {c}({unit})"
        header += " " * max(0, 10 - len(f"{c}({unit})"))
    print()
    print("=" * len(header))
    print("RANKED MODEL COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for rank, (name, arch, val_loss, metrics) in enumerate(ranked, 1):
        line = f"{rank:<5} {name:<8} {arch:<14} {val_loss:<12.6f}"
        for c in comp_names:
            rmse_str = f"{metrics[c]['rmse']:.4f}" if metrics else "N/A"
            line += f" {rmse_str}"
            line += " " * max(0, 10 - len(rmse_str))
        print(line)
    print("-" * len(header))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze surrogate model sweep results"
    )
    parser.add_argument(
        "--output-base", type=str, default="output/surrogate",
        help="Base output directory containing sweep results",
    )
    parser.add_argument(
        "--figures-dir", type=str, default="figures/surrogate_training",
        help="Directory for output figures",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/surrogate_rl_step",
        help="Data directory for validation inference",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for model inference (default: cpu)",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of top models to analyze in detail (default: 5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_base = Path(args.output_base)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load sweep summary
    print("Loading sweep summary...")
    summary = load_sweep_summary(output_base)
    results = summary.get("results", [])
    configs = summary.get("configs", [])

    if not results:
        print("ERROR: No results in sweep_summary.json", file=sys.stderr)
        sys.exit(1)

    # Filter to configs that actually completed (val_loss < inf)
    valid_results = [r for r in results if r.get("val_loss", float("inf")) < float("inf")]
    if not valid_results:
        print("ERROR: No configs completed successfully (all val_loss = inf).",
              file=sys.stderr)
        sys.exit(1)

    # Sort by val_loss ascending
    valid_results.sort(key=lambda r: r["val_loss"])
    top_k = min(args.top_k, len(valid_results))
    top_configs = valid_results[:top_k]

    print(f"  Total configs: {len(results)}")
    print(f"  Completed: {len(valid_results)}")
    print(f"  Best: {valid_results[0]['name']} (val_loss={valid_results[0]['val_loss']:.6f})")

    # 2. Load validation data
    print(f"\nLoading validation data from {args.data_dir}...")
    val_dataset = FlatStepDataset(args.data_dir, split="val")
    print(f"  Validation transitions: {len(val_dataset):,}")

    # Convert to 130-dim relative representation if needed
    if val_dataset.states.shape[-1] == 124:
        print("  Converting 124-dim -> 130-dim relative representation...")
        val_dataset.states = raw_to_relative(val_dataset.states)
        val_dataset.next_states = raw_to_relative(val_dataset.next_states)

    val_states = val_dataset.states
    val_actions = val_dataset.actions
    val_t_starts = val_dataset.t_starts
    val_next_states = val_dataset.next_states

    # 3. Run inference on top-K models and compute per-component metrics
    print(f"\nRunning inference on top {top_k} models...")
    top_metrics = {}  # config_name -> component metrics
    best_pred_deltas = None
    best_actual_deltas = None
    best_name = valid_results[0]["name"]

    ranked: List[Tuple[str, str, float, Dict[str, Dict[str, float]]]] = []

    for r in top_configs:
        config_name = r["name"]
        config_dir = output_base / config_name
        print(f"\n  Loading {config_name}...")

        model, normalizer, cfg_dict = load_model_and_normalizer(config_dir, device)
        if model is None:
            print(f"    SKIP: model.pt not found in {config_dir}")
            continue

        print(f"    Running inference...")
        pred_deltas, actual_deltas = run_inference(
            model, normalizer,
            val_states, val_actions, val_t_starts, val_next_states,
            device=device,
        )

        metrics = compute_component_metrics(pred_deltas, actual_deltas)
        top_metrics[config_name] = metrics

        arch_label = r.get("arch", "mlp")
        if config_name in WIDE_DEEP_NAMES:
            arch_label = "wide/deep"
        ranked.append((config_name, arch_label, r["val_loss"], metrics))

        # Store best model predictions for detailed plots
        if config_name == best_name:
            best_pred_deltas = pred_deltas
            best_actual_deltas = actual_deltas

        print(f"    RMSE: " + ", ".join(
            f"{c}={metrics[c]['rmse']:.4f}{COMPONENTS[c][1]}"
            for c in COMPONENTS
        ))

    # Also add incomplete configs to ranked table (with no metrics)
    for r in valid_results[top_k:]:
        arch_label = r.get("arch", "mlp")
        if r["name"] in WIDE_DEEP_NAMES:
            arch_label = "wide/deep"
        ranked.append((r["name"], arch_label, r["val_loss"], {c: {"rmse": 0, "mae": 0, "r2": 0, "unit": ""} for c in COMPONENTS}))

    # 4. Generate plots
    print("\n\nGenerating diagnostic plots...")

    # Plot 1: Sweep comparison (all configs)
    plot_sweep_comparison(results, figures_dir)

    # Plot 2: Per-component RMSE (top K)
    if top_metrics:
        plot_per_component_rmse(top_metrics, figures_dir)

    # Plot 3 & 4: Error histograms and predicted vs actual (best model)
    if best_pred_deltas is not None:
        plot_error_histograms(best_pred_deltas, best_actual_deltas, best_name, figures_dir)
        plot_predicted_vs_actual(best_pred_deltas, best_actual_deltas, best_name, figures_dir)

    # 5. Print ranked summary table
    print_ranked_table(ranked)

    # 6. Save analysis results JSON for downstream consumption
    analysis_path = output_base / "analysis_results.json"
    analysis_data = {
        "best_config": best_name,
        "best_val_loss": valid_results[0]["val_loss"],
        "n_completed": len(valid_results),
        "top_models": {},
    }
    for name, metrics in top_metrics.items():
        analysis_data["top_models"][name] = {
            comp: {
                "rmse": m["rmse"],
                "mae": m["mae"],
                "r2": m["r2"],
                "unit": m["unit"],
            }
            for comp, m in metrics.items()
        }
    with open(analysis_path, "w") as f:
        json.dump(analysis_data, f, indent=2)
    print(f"Analysis results saved to {analysis_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
