"""Shared Weights & Biases integration.

All W&B calls go through this module. Training loops should never
call ``wandb.*`` directly — use these helpers instead.

Functions:
    setup_run       — init W&B, define_metric axes, log config
    log_metrics     — route namespaced metrics (train/, episode/, timing/, system/)
    log_extra_params — one-time metadata (total_params, gpu_name, etc.)
    log_model_artifact — upload best checkpoint as versioned artifact
    end_run         — finish the active W&B run
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def setup_run(
    config,
    run_dir: Path,
    *,
    resume_id: Optional[str] = None,
) -> Any:
    """Initialize a W&B run with proper metric axes.

    Args:
        config: Dataclass config (must have ``wandb`` and ``name`` attrs).
        run_dir: Run output directory for W&B files.
        resume_id: Optional run ID for resuming.

    Returns:
        The ``wandb.Run`` object, or *None* if W&B is disabled.
    """
    if not getattr(config, "wandb", None) or not config.wandb.enabled:
        return None

    import wandb

    wandb_cfg = config.wandb
    init_kwargs = dict(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity or None,
        group=wandb_cfg.group or None,
        tags=wandb_cfg.tags or None,
        name=config.name,
        config=asdict(config),
        dir=str(run_dir),
    )
    if resume_id:
        init_kwargs["resume"] = "allow"
        init_kwargs["id"] = resume_id

    run = wandb.init(**init_kwargs)

    # Define independent x-axes so batch and episode metrics don't collide
    wandb.define_metric("batch/*", step_metric="batch/global_step")
    wandb.define_metric("train/*", step_metric="total_frames")
    wandb.define_metric("episode/*", step_metric="total_frames")
    wandb.define_metric("timing/*", step_metric="total_frames")
    wandb.define_metric("system/*", step_metric="total_frames")
    wandb.define_metric("tracking/*", step_metric="total_frames")
    wandb.define_metric("reward/*", step_metric="total_frames")
    wandb.define_metric("gradients/*", step_metric="total_frames")
    wandb.define_metric("diagnostics/*", step_metric="total_frames")
    wandb.define_metric("q_values/*", step_metric="total_frames")

    return run


def log_metrics(
    run,
    metrics: Dict[str, Any],
    step: int,
) -> None:
    """Log a dictionary of metrics to W&B.

    Metrics are logged as-is (keys should already be namespaced by the
    caller, e.g. ``train/loss_actor``, ``episode/mean_reward``).

    Args:
        run: W&B run object (or None to skip).
        metrics: Namespaced metric dictionary.
        step: Global step (total_frames).
    """
    if run is None:
        return
    # Inject total_frames so define_metric step references resolve
    metrics["total_frames"] = step
    run.log(metrics, step=step)


def log_extra_params(run, params: Dict[str, Any]) -> None:
    """Log one-time metadata to W&B config and summary.

    Call once after model init with params like total_params,
    trainable_params, gpu_name, num_envs.

    Args:
        run: W&B run object (or None to skip).
        params: Metadata dictionary.
    """
    if run is None:
        return
    # Update the run config (appears in the W&B run overview)
    run.config.update(params, allow_val_change=True)
    # Also log to summary for quick access in tables
    for k, v in params.items():
        run.summary[k] = v


def log_model_artifact(
    run,
    checkpoint_path: Path,
    artifact_name: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Upload a model checkpoint as a versioned W&B artifact.

    Call once per run at the end of training (not per improvement).

    Args:
        run: W&B run object (or None to skip).
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        artifact_name: Artifact name (e.g. ``"locomotion_elastica_forward"``).
        metadata: Optional metadata dict for the artifact.
    """
    if run is None:
        return
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return

    import wandb

    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        metadata=metadata or {},
    )
    artifact.add_file(str(checkpoint_path))
    run.log_artifact(artifact)


def end_run(run) -> None:
    """Finish the active W&B run.

    Args:
        run: W&B run object (or None to skip).
    """
    if run is None:
        return
    run.finish()


def _count_parameters(module: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable}


def collect_hardware_info(device: str) -> Dict[str, str]:
    """Collect static hardware info for one-time logging."""
    info: Dict[str, str] = {}
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        dev = torch.device(device)
        info["gpu_name"] = torch.cuda.get_device_name(dev)
        info["gpu_memory_total_gb"] = f"{torch.cuda.get_device_properties(dev).total_memory / 1e9:.1f}"
    try:
        import psutil
        info["cpu_count"] = str(psutil.cpu_count(logical=True))
        info["ram_total_gb"] = f"{psutil.virtual_memory().total / 1e9:.1f}"
    except ImportError:
        pass
    return info
