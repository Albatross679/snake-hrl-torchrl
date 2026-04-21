"""Shared metric logging utilities for trainers."""

import torch
import torch.nn as nn


def compute_grad_norm(model: nn.Module) -> float:
    """L2 norm of all gradients for a model."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5


def collect_system_metrics(device, counter: list, interval: int) -> dict:
    """Collect CPU/RAM and GPU metrics, throttled by interval.

    Args:
        device: Torch device (for GPU metrics)
        counter: Mutable list [int] used as call counter across invocations
        interval: Only collect every N calls

    Returns:
        Dictionary of system metrics (empty if throttled)
    """
    counter[0] += 1
    if counter[0] % interval != 0:
        return {}

    metrics = {}

    # CPU / RAM (lazy import, optional dependency)
    try:
        import psutil

        metrics["system/cpu_percent"] = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        metrics["system/ram_percent"] = mem.percent
        metrics["system/ram_used_gb"] = mem.used / (1024**3)
        metrics["system/ram_total_gb"] = mem.total / (1024**3)
    except ImportError:
        pass

    # GPU
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        dev = torch.device(device)
        metrics["system/gpu_memory_allocated_mb"] = (
            torch.cuda.memory_allocated(dev) / (1024**2)
        )
        metrics["system/gpu_memory_reserved_mb"] = (
            torch.cuda.memory_reserved(dev) / (1024**2)
        )

    return metrics
