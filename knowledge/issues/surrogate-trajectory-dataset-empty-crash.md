---
name: Trajectory dataset empty — rollout loss crash at epoch 21
description: TrajectoryDataset finds 0 valid windows because flat-format data has 1 step per episode
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-16
updated: 2026-03-16
tags: [surrogate, training, rollout-loss, dataset]
aliases: [trajectory-dataset-empty, rollout-loss-crash]
---

## Symptom

Training crashed at epoch 21 (start of rollout loss phase) with:

```
ValueError: num_samples should be a positive integer value, but got num_samples=0
```

The `DataLoader` for `TrajectoryDataset` received a dataset with 0 samples.

## Root Cause

`TrajectoryDataset` builds multi-step trajectory windows by grouping transitions
by `episode_id` and finding contiguous sequences of `rollout_length` (8) steps.

The Phase 02.2 flat-format data (`data/surrogate_rl_step_rel128/`) stores
independent single-step transitions — each transition has a unique `episode_id`
with exactly 1 step. No episode has multiple steps, so no valid windows of
length 8 can be constructed. Result: `len(traj_dataset) == 0`.

Additionally, `TrajectoryDataset` internally uses the deprecated `SurrogateDataset`
which adds per-file `episode_offset`, further preventing any cross-file episode
grouping.

## Secondary Crash: OOM at Epoch 21

After the babysitter's runtime guard fix, training restarted but crashed again
at epoch 21 with `torch.OutOfMemoryError`. The guard prevented the `ValueError`
but the `TrajectoryDataset.__init__` still loaded all data files and tried to
access `data["serpenoid_times"]` (a `KeyError` in the preprocessed rel128 data),
consuming GPU memory before the guard could fire.

## Fix Applied

1. **Runtime guard** (babysitter): `len(traj_dataset) == 0` check disables
   rollout loss at runtime — prevents `ValueError` but not OOM.
2. **Config default** (permanent fix): Set `rollout_loss_weight: float = 0.0`
   in `SurrogateTrainConfig` so the rollout code path is never entered.
3. **Display fix**: Console output now shows "Single-step loss only" when
   `rollout_loss_weight=0` instead of misleading "Rollout loss: epochs 21+".

Rationale: The data is Markov (single-step transitions), so rollout loss is
unnecessary. If multi-step rollout is needed in the future, the data collection
pipeline must produce multi-step episodes with contiguous `episode_id`s and
include `serpenoid_times` in the saved data.

## Files Modified

- `papers/aprx_model_elastica/train_surrogate.py` — runtime guard + display fix
- `papers/aprx_model_elastica/train_config.py` — `rollout_loss_weight` default 0.1 → 0.0
