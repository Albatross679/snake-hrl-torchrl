# Issue: Training process killed silently by concurrent sweep

## Date: 2026-03-17

## Symptoms
- Process for `output/surrogate_20260317_171708` died after epoch 1
- No traceback in log — clean kill
- Another session started a hyperparameter sweep (M2, test_batch) that acquired GPU 0

## Root Cause
- GPU lock mechanism (`/tmp/gpu-task.lock`) likely killed our process when the sweep acquired the lock
- Our process was using GPU 0, same as the sweep

## Fix
- Restarted training on GPU 1 (`CUDA_VISIBLE_DEVICES=1`) to avoid conflicts with the sweep running on GPU 0
- New run dir: will be created fresh (old run `output/surrogate_20260317_171708` only had 1 epoch)
