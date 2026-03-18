"""VRAM cleanup utilities for sequential multi-config training."""

import gc

import torch


def cleanup_vram() -> None:
    """Free GPU memory between sequential training configs.

    Clears CUDA cache and runs garbage collection. Call this between
    sequential configs after deleting model, optimizer, and scheduler
    references.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
