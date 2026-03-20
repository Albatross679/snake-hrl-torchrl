---
name: surrogate-wandb-metric-naming-mismatch
description: W&B dashboard panels showed no data because metric names used train/ prefix instead of matching existing panel names
type: issue
status: resolved
severity: medium
subtype: training
created: 2026-03-16
updated: 2026-03-16
tags: [surrogate, wandb, metrics, dashboard]
aliases: []
---

## Symptom

W&B run page (run `esklpikh`) showed 6 empty chart panels with "There's no data for the selected runs. Try a different X axis setting. Current X axis: _step". The project workspace showed charts from a previous run but the current run's individual page was blank.

## Root Cause

The W&B project workspace had pre-configured dashboard panels looking for metrics named `val_loss`, `train_loss`, `train_val_gap`, `rollout_loss`, `lr`, `grad_norm`. However, the training code logged metrics with a `train/` prefix (`train/loss`, `train/val_loss`, `train/lr`, etc.) and did not log `grad_norm` or `train_val_gap` at all.

Mismatches:
| Dashboard panel | Code logged as | Status |
|---|---|---|
| `val_loss` | `train/val_loss` | Name mismatch |
| `train_loss` | `train/loss` | Name mismatch |
| `train_val_gap` | *(not logged)* | Missing metric |
| `rollout_loss` | `train/rollout_loss` | Name mismatch |
| `lr` | `train/lr` | Name mismatch |
| `grad_norm` | *(not logged)* | Missing metric |

## Fix

Modified `papers/aprx_model_elastica/train_surrogate.py`:

1. **Renamed W&B metrics** to match dashboard panels — removed `train/` and `tracking/` prefixes:
   - `train/loss` → `train_loss`
   - `train/val_loss` → `val_loss`
   - `train/rollout_loss` → `rollout_loss`
   - `train/lr` → `lr`
   - `tracking/best_val_loss` → `best_val_loss`
   - `tracking/patience_counter` → `patience_counter`
   - `timing/epoch_time_s` → `epoch_time_s`

2. **Added `grad_norm`** — captured return value from `torch.nn.utils.clip_grad_norm_()`, averaged over optimizer steps per epoch, logged as `grad_norm`.

3. **Added `train_val_gap`** — computed as `epoch_loss - val_loss`, logged as `train_val_gap`.

## Files Modified

- `papers/aprx_model_elastica/train_surrogate.py`
