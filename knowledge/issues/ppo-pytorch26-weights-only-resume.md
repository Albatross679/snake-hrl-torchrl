---
name: PyTorch 2.6 weights_only blocks checkpoint resume
description: torch.load defaults to weights_only=True in PyTorch 2.6, blocking checkpoint loading with custom config objects
type: issue
status: resolved
severity: low
subtype: compatibility
created: 2026-03-19
updated: 2026-03-19
tags: [ppo, pytorch, checkpoint, resume]
aliases: []
---

## Symptom

Resuming training with `--resume <checkpoint.pt>` fails with:
```
_pickle.UnpicklingError: Weights only load failed.
Unsupported global: GLOBAL choi2025.config.Choi2025PPOConfig was not an allowed global by default.
```

## Root Cause

PyTorch 2.6 changed `torch.load` default from `weights_only=False` to `weights_only=True`. Our checkpoints store the config dataclass object alongside model weights, which requires unpickling custom classes.

## Fix Applied

Added `weights_only=False` to `torch.load()` in `PPOTrainer.load_checkpoint()`. These are our own checkpoints, so trusting them is safe.

## Files Modified

- `src/trainers/ppo.py` line 685 — added `weights_only=False`
