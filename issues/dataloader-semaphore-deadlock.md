---
name: DataLoader semaphore deadlock during surrogate training
description: PyTorch DataLoader with num_workers=2 leaks semaphores and deadlocks after ~15 epochs on large dataset
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-11
updated: 2026-03-11
tags: [dataloader, multiprocessing, semaphore, deadlock, surrogate, sweep]
aliases: [dataloader-deadlock, semaphore-leak]
---

## Symptom

During the 15-config surrogate architecture sweep, config M1 (MLP, lr=1e-4, 256x3) trained successfully for 15 epochs then hung indefinitely. The process went to 0% CPU, 0% GPU, sleeping state, with no log output for 11+ minutes. On shutdown, Python reported:

```
resource_tracker: There appear to be 14 leaked semaphore objects to clean up at shutdown
```

Sweep script reported: `[M1] FAILED (rc=-10) in 41.3 min`

## Environment

- Dataset: 3.9M transitions (8.5 GB), FlatStepDataset
- Batch size: 117,760 (auto-detected) → ~33 batches/epoch
- DataLoader: `num_workers=2, pin_memory=True, drop_last=True`
- No `persistent_workers` set (defaults to `False`)
- Python 3.12.3, PyTorch (via torchrl 0.11.1)

## Root Cause

With `persistent_workers=False` (default), PyTorch's DataLoader spawns fresh worker subprocesses each time an iterator is created (once per epoch). Each worker pair allocates shared-memory semaphores for IPC. On Python 3.12 with `multiprocessing.resource_tracker`, these semaphores are tracked but not always cleaned up promptly between epochs, especially with large batch sizes and `WeightedRandomSampler`.

After ~15 epochs the accumulated leaked semaphores cause a deadlock: worker processes block waiting on semaphore acquisition while the main process blocks waiting on the worker queue.

## Fix

Add `persistent_workers=True` to all DataLoader instances in `aprx_model_elastica/train_surrogate.py`. This keeps worker processes alive across epochs, reusing the same semaphores instead of leaking new ones each epoch.

Applied to 4 DataLoader call sites (lines 479, 484, 488, 537).

## M1 Results (Pre-Deadlock)

Training was progressing well before the hang:

| Epoch | Train Loss | Val Loss | Best Val |
|-------|-----------|----------|----------|
| 1     | 1.6148    | 0.9906   | 0.9906   |
| 5     | 1.5438    | 0.9147   | 0.9147   |
| 10    | 1.2637    | 0.8861   | 0.8861   |
| 15    | 1.2149    | 0.8132   | 0.8132   |

The saved checkpoint (epoch 15, val_loss=0.813) is usable but incomplete — the model was still improving with patience=0/30.
