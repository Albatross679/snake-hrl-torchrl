---
name: M1 config killed by SIGUSR1 during Phase 3 sweep
description: M1 training terminated by signal 10 (SIGUSR1) at epoch 15/200, no metrics.json saved
type: issue
status: investigating
severity: medium
subtype: training
created: 2026-03-11
updated: 2026-03-11
tags: [sweep, phase-3, surrogate, signal]
aliases: []
---

# M1 Config Killed by SIGUSR1 During Sweep

## Symptom
Config M1 (MLP, lr=1e-4, hidden=256x3) failed with `rc=-10` (SIGUSR1) after 41.3 minutes (~15 epochs).

## Evidence
- **M1.log**: Training ran normally through epoch 15, val_loss decreasing (0.991 → 0.813)
- **Last log line**: `multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 14 leaked semaphore objects to clean up at shutdown`
- **Return code**: -10 (signal 10 = SIGUSR1 on Linux)
- **Saved files**: model.pt (800KB), config.json, normalizer.pt — NO metrics.json
- **System state**: 251GB RAM (49GB used), V100 16GB (2GB used), no OOM in dmesg/journalctl
- **Sweep runner**: No timeout or signal handling in sweep.py

## Impact
- M1 results incomplete — best val_loss=0.813 at epoch 15, was still improving
- Sweep continued correctly to M2 (sweep.py logs FAILED, moves to next config)
- 14 remaining configs unaffected so far

## Possible Causes
1. **Wandb internal signal** — wandb sometimes sends SIGUSR1 for sync management
2. **System/cgroup process management** — external monitoring killing long processes
3. **Semaphore leak** — 14 leaked semaphore objects from multiprocessing (DataLoader workers), could trigger resource tracker cleanup signal

## Monitoring Plan
- Watch M2 and subsequent configs for same failure pattern
- If SIGUSR1 recurs, investigate wandb signal handling and DataLoader num_workers
- If M1 is the only failure, likely a one-off external signal

## Resolution
- No action needed now — sweep continues, M1 has partial checkpoint
- If pattern repeats: add SIGUSR1 handler to train_surrogate.py to save metrics.json on signal
- M1 can be rerun after sweep if needed (already have model.pt from epoch 15)
