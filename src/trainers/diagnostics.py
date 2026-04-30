"""RL training diagnostics based on Andy Jones' debugging methodology.

Provides:
- TrainingDiagnostics: non-invasive diagnostic middleware that computes
  and logs additional metrics (explained variance, action stats, advantage
  stats, Q-value health, entropy) without modifying core trainer logic.
- check_alerts(): automated W&B alert checks for known RL failure signatures
  (gradient explosion, entropy collapse, Q-value divergence, reward collapse).
- compute_explained_variance(): one-liner for value function quality.
- compute_action_stats(): per-dimension action distribution health.
"""

from collections import deque
from typing import Dict, Optional

import torch
import torch.nn as nn


def compute_explained_variance(
    value_pred: torch.Tensor, value_target: torch.Tensor
) -> float:
    """Explained variance of value predictions: 1 - Var(residual) / Var(target).

    Returns:
        Float in (-inf, 1.0]. Values near 1.0 = good value function.
        Values near 0.0 = value function is no better than predicting mean.
        Negative = value function is anti-correlated (broken).
    """
    with torch.no_grad():
        y_pred = value_pred.flatten()
        y_true = value_target.flatten()
        var_y = y_true.var()
        if var_y < 1e-8:
            return 0.0
        return (1.0 - (y_true - y_pred).var() / var_y).item()


def compute_action_stats(actions: torch.Tensor) -> Dict[str, float]:
    """Compute per-dimension action distribution statistics.

    Args:
        actions: Action tensor [batch, action_dim]

    Returns:
        Dict with action_mean, action_std_mean, action_std_min,
        and per-dimension std for up to 10 dims.
    """
    with torch.no_grad():
        action_std = actions.std(dim=0)
        stats = {
            "action_mean": actions.mean().item(),
            "action_std_mean": action_std.mean().item(),
            "action_std_min": action_std.min().item(),
        }
        for i in range(min(actions.shape[-1], 10)):
            stats[f"action_dim{i}_std"] = action_std[i].item()
        return stats


def compute_advantage_stats(advantages: torch.Tensor) -> Dict[str, float]:
    """Compute advantage distribution statistics.

    Advantages should be approximately mean-zero when normalized.
    Large deviations indicate broken advantage computation.
    """
    with torch.no_grad():
        return {
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "advantage_abs_max": advantages.abs().max().item(),
        }


def compute_ratio_stats(
    new_log_prob: torch.Tensor, old_log_prob: torch.Tensor
) -> Dict[str, float]:
    """Compute importance sampling ratio statistics for PPO/MM-RKHS."""
    with torch.no_grad():
        ratio = torch.exp(new_log_prob - old_log_prob)
        return {
            "ratio_mean": ratio.mean().item(),
            "ratio_max": ratio.max().item(),
            "ratio_min": ratio.min().item(),
        }


def compute_q_value_stats(
    q1: torch.Tensor,
    q2: torch.Tensor,
    target_value: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute Q-value health statistics for SAC.

    Monitors Q-value spread (divergence between twin critics),
    absolute magnitude (divergence detection), and target statistics.
    """
    with torch.no_grad():
        stats = {
            "q_value_spread": abs(q1.mean().item() - q2.mean().item()),
            "q_max": max(q1.max().item(), q2.max().item()),
            "q_min": min(q1.min().item(), q2.min().item()),
        }
        if target_value is not None:
            stats["target_value_mean"] = target_value.mean().item()
            stats["target_value_std"] = target_value.std().item()
        return stats


def compute_log_prob_stats(log_prob: torch.Tensor) -> Dict[str, float]:
    """Compute log probability statistics (entropy proxy)."""
    with torch.no_grad():
        return {
            "log_prob_mean": log_prob.mean().item(),
            "log_prob_std": log_prob.std().item(),
            "entropy_proxy": -log_prob.mean().item(),
        }


def check_alerts(
    wandb_run,
    metrics: Dict[str, float],
    step: int,
    algorithm: str = "ppo",
) -> None:
    """Check metrics against known RL failure signatures and fire W&B alerts.

    Args:
        wandb_run: Active W&B run (or None if disabled).
        metrics: Dictionary of current training metrics.
        step: Current training step (total frames).
        algorithm: One of "ppo", "sac", "mmrkhs".
    """
    if wandb_run is None:
        return

    try:
        import wandb
    except ImportError:
        return

    alerts = []

    # --- Universal checks ---

    # NaN detection
    for key, val in metrics.items():
        if isinstance(val, float) and val != val:  # NaN check
            alerts.append((
                "NaN in metrics",
                wandb.AlertLevel.ERROR,
                f"Step {step}: {key} is NaN",
            ))

    # Gradient explosion
    grad_key = "actor_grad_norm" if algorithm == "sac" else "grad_norm"
    grad_val = metrics.get(grad_key, 0)
    if isinstance(grad_val, (int, float)) and grad_val > 1e4:
        alerts.append((
            "Gradient explosion",
            wandb.AlertLevel.ERROR,
            f"Step {step}: {grad_key}={grad_val:.2e}",
        ))

    # Entropy collapse (for on-policy methods)
    entropy = metrics.get("entropy_proxy", metrics.get("policy_entropy"))
    if entropy is not None and isinstance(entropy, (int, float)) and entropy < 0.01:
        alerts.append((
            "Entropy collapse",
            wandb.AlertLevel.WARN,
            f"Step {step}: entropy={entropy:.4f}. "
            "Policy may have collapsed to near-deterministic actions.",
        ))

    # Action std collapse (any dimension)
    action_std_min = metrics.get("action_std_min")
    if action_std_min is not None and action_std_min < 0.01:
        alerts.append((
            "Action dimension collapsed",
            wandb.AlertLevel.WARN,
            f"Step {step}: action_std_min={action_std_min:.4f}. "
            "One or more action dimensions have near-zero variance.",
        ))

    # --- SAC-specific checks ---
    if algorithm == "sac":
        for qkey in ("q1_mean", "q2_mean"):
            qval = metrics.get(qkey, 0)
            if isinstance(qval, (int, float)) and abs(qval) > 1000:
                alerts.append((
                    "Q-value divergence",
                    wandb.AlertLevel.WARN,
                    f"Step {step}: {qkey}={qval:.1f}. "
                    "Q-values growing unbounded — check target network updates.",
                ))

        spread = metrics.get("q_value_spread", 0)
        if isinstance(spread, (int, float)) and spread > 100:
            alerts.append((
                "Q-value twin divergence",
                wandb.AlertLevel.WARN,
                f"Step {step}: q_value_spread={spread:.1f}. "
                "Twin Q-networks have diverged significantly.",
            ))

    # --- PPO/MM-RKHS-specific checks ---
    if algorithm in ("ppo", "mmrkhs"):
        ev = metrics.get("explained_variance")
        if ev is not None and isinstance(ev, (int, float)) and ev < -0.5:
            alerts.append((
                "Value function anti-correlated",
                wandb.AlertLevel.WARN,
                f"Step {step}: explained_variance={ev:.3f}. "
                "Value predictions are worse than predicting the mean.",
            ))

        clip_frac = metrics.get("clip_fraction", 0)
        if isinstance(clip_frac, (int, float)) and clip_frac > 0.5:
            alerts.append((
                "Excessive PPO clipping",
                wandb.AlertLevel.WARN,
                f"Step {step}: clip_fraction={clip_frac:.3f}. "
                "Policy changed too much — consider reducing learning rate.",
            ))

        adv_max = metrics.get("advantage_abs_max", 0)
        if isinstance(adv_max, (int, float)) and adv_max > 100:
            alerts.append((
                "Advantage explosion",
                wandb.AlertLevel.WARN,
                f"Step {step}: advantage_abs_max={adv_max:.1f}. "
                "Advantages are extremely large — check reward scaling or GAE.",
            ))

    # Fire all alerts (W&B deduplicates by title with 5-min default cooldown)
    for title, level, text in alerts:
        wandb.alert(title=title, text=text, level=level, wait_duration=300)
