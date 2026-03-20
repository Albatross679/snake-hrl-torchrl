---
name: PINN training script missing auto batch size
description: src/pinn/train_regularized.py uses hardcoded batch_size=4096 instead of auto-batch probing, resulting in 3% GPU utilization
type: issue
status: open
severity: medium
subtype: performance
created: 2026-03-18
updated: 2026-03-18
tags: [pinn, training, gpu-utilization, ml-checklist]
aliases: []
id: ISS-036
---

## Symptom

PINN regularized training (`src/pinn/train_regularized.py`) runs at only 3% GPU utilization and 366MB VRAM on a 16GB GPU. Each epoch takes ~3.5 minutes despite the model having only 666K parameters.

## Root Cause

The PINN training script uses a hardcoded `batch_size=4096`. The existing surrogate trainer (`papers/aprx_model_elastica/train_surrogate.py`) has an `auto_batch_size` feature that probes for the largest batch fitting 85% VRAM, but this was not ported to the PINN script.

With 3.9M training samples and a tiny MLP, batch_size=4096 means the GPU finishes each forward/backward pass almost instantly and then waits for CPU data loading — the GPU is starved.

## ML Checklist Violation

From `/tmp/ml-training-checklist.md`:
> **VRAM Management**: Auto batch size tuning or at minimum document why a fixed batch size is used.

Neither auto-batch nor a justification for the fixed size exists.

## Existing Implementation

`papers/aprx_model_elastica/train_surrogate.py` has `probe_auto_batch_size()` (line 141) which:
1. Starts at a candidate batch size
2. Runs a dummy forward+backward pass
3. Doubles until OOM or 85% VRAM target
4. Falls back to config default on failure

Config field: `auto_batch_size: bool = True` in `train_config.py`.

## Recommended Fix

1. Port `probe_auto_batch_size()` to `src/pinn/train_regularized.py` (or extract to a shared utility)
2. Add `auto_batch_size: bool = True` to the PINN config dataclass
3. Use probed batch size for DataLoader construction

## Files Affected

- `src/pinn/train_regularized.py` — needs auto-batch integration
- `papers/aprx_model_elastica/train_surrogate.py` — reference implementation
