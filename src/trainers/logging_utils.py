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


def log_system_metrics(
    writer,
    step: int,
    device,
    counter: list,
    interval: int,
) -> None:
    """Log CPU/RAM and GPU metrics to TensorBoard, throttled by interval.

    Args:
        writer: TensorBoard SummaryWriter
        step: Current training step
        device: Torch device (for GPU metrics)
        counter: Mutable list [int] used as call counter across invocations
        interval: Only log every N calls
    """
    counter[0] += 1
    if counter[0] % interval != 0:
        return

    # CPU / RAM (lazy import, optional dependency)
    try:
        import psutil

        writer.add_scalar("system/cpu_percent", psutil.cpu_percent(), step)
        mem = psutil.virtual_memory()
        writer.add_scalar("system/ram_percent", mem.percent, step)
        writer.add_scalar("system/ram_used_gb", mem.used / (1024**3), step)
        writer.add_scalar("system/ram_total_gb", mem.total / (1024**3), step)
    except ImportError:
        pass

    # GPU
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        dev = torch.device(device)
        writer.add_scalar(
            "system/gpu_memory_allocated_mb",
            torch.cuda.memory_allocated(dev) / (1024**2),
            step,
        )
        writer.add_scalar(
            "system/gpu_memory_reserved_mb",
            torch.cuda.memory_reserved(dev) / (1024**2),
            step,
        )
