---
name: surrogate-dataset-key-mismatch-crash
description: Training crashed at epoch 15 with KeyError because SurrogateDataset expected serpenoid_times/step_indices keys but data files have t_start/step_ids
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-16
updated: 2026-03-16
tags: [surrogate, dataset, crash, trajectory]
aliases: []
---

## Symptom

Training crashed at epoch 15 (when `TrajectoryDataset` loads for rollout loss) with:

```
KeyError: 'serpenoid_times'
```

Full traceback pointed to `dataset.py:167` in `SurrogateDataset.__init__()`.

## Root Cause

The `SurrogateDataset` class was written expecting batch files with keys `serpenoid_times` and `step_indices` (from the original data collection format). However, the actual batch files in `data/surrogate_rl_step_rel128/` (and `data/surrogate_rl_step/`) use different key names:

| Expected key | Actual key | Purpose |
|---|---|---|
| `serpenoid_times` | `t_start` | Simulation time for CPG phase computation |
| `step_indices` | `step_ids` | Step index within episode for trajectory ordering |

The crash only triggered at epoch 15 because `TrajectoryDataset` (which wraps `SurrogateDataset`) is lazily loaded before rollout loss begins at `rollout_start_epoch=20`.

## Fix

Modified `papers/aprx_model_elastica/dataset.py` to support both key conventions with fallback:

```python
serp_key = "serpenoid_times" if "serpenoid_times" in data else "t_start"
all_serp_times.append(data[serp_key])
step_key = "step_indices" if "step_indices" in data else "step_ids"
all_step_indices.append(data[step_key])
```

Applied to both the `.pt` and `.parquet` loading loops.

## Files Modified

- `papers/aprx_model_elastica/dataset.py`
