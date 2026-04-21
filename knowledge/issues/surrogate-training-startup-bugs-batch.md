---
name: surrogate-training-startup-bugs-batch
description: Five bugs encountered when launching surrogate training after workflow compliance overhaul
type: issue
status: resolved
severity: medium
subtype: training
created: 2026-03-16
updated: 2026-03-16
tags: [surrogate, training, startup, workflow-compliance]
aliases: []
---

# Surrogate Training Startup Bugs

Five bugs encountered in sequence when first launching surrogate model training after the ML workflow compliance overhaul. All resolved in-session.

## Bug 1: Watchdog multiprocessing pickle error

**Symptom:** `AttributeError: Can't get local object '_run_with_watchdog.<locals>._target'`

**Root Cause:** `_run_with_watchdog()` defined a local function `_target()` as the multiprocessing child process entry point. Python 3.12's default `spawn` start method requires picklable targets, and local functions can't be pickled.

**Fix:** Moved `_target` to module-level as `_watchdog_target()` and used `multiprocessing.get_context("fork")` explicitly.

**File:** `papers/aprx_model_elastica/train_surrogate.py`

---

## Bug 2: W&B adapter not a dataclass

**Symptom:** `TypeError: asdict() should be called on dataclass instances` in `wandb_utils.setup_run()`

**Root Cause:** `SurrogateTrainConfig` uses flat W&B fields (`wandb_enabled`, `wandb_project`, `wandb_entity`) instead of a nested `.wandb` object. The `_WandBAdapter` wrapper class was created to bridge this, but `wandb_utils.setup_run` calls `dataclasses.asdict(config)` on line 52, which fails on non-dataclass objects.

**Fix:** Replaced the adapter pattern with direct `wandb.init()` call in `train_surrogate.py`, passing `asdict(config)` (the actual dataclass) as the config dict. Also defined surrogate-specific `wandb.define_metric` axes using `epoch` as the step metric instead of `total_frames`.

**File:** `papers/aprx_model_elastica/train_surrogate.py`

---

## Bug 3: Dataset key mismatch (`serpenoid_times` vs `t_start`)

**Symptom:** `KeyError: 'serpenoid_times'` when loading preprocessed 128-dim data.

**Root Cause:** The deprecated `SurrogateDataset` class expects `serpenoid_times` and `step_indices` keys. The preprocessed data (from `preprocess_relative.py`) passes through the original collection format which uses `t_start` and `step_ids` keys.

**Fix:** Switched from deprecated `SurrogateDataset` to `FlatStepDataset`, which expects `t_start` and `step_ids`. Also updated `batch["serpenoid_time"]` references to `batch["t_start"]` in `compute_single_step_loss()` and `probe_auto_batch_size()`.

**Files:** `papers/aprx_model_elastica/train_surrogate.py`

---

## Bug 4: Missing `deltas` attribute on `FlatStepDataset`

**Symptom:** `AttributeError: 'FlatStepDataset' object has no attribute 'deltas'`

**Root Cause:** `SurrogateDataset` precomputed `self.deltas = self.next_states - self.states` during `__init__`. `FlatStepDataset` computes deltas per-item in `__getitem__` and doesn't store a bulk `deltas` tensor. The normalizer fitting code `normalizer.fit(train_dataset.states, train_dataset.deltas)` assumed the old interface.

**Fix:** Computed deltas inline: `deltas = train_dataset.next_states - train_dataset.states` before passing to `normalizer.fit()`.

**File:** `papers/aprx_model_elastica/train_surrogate.py`

---

## Bug 5: `total_mem` vs `total_memory` attribute name

**Symptom:** `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'. Did you mean: 'total_memory'?`

**Root Cause:** The correct PyTorch attribute is `total_memory`, not `total_mem`. This typo existed in two places.

**Fix:** Changed `total_mem` to `total_memory` in both files.

**Files:** `papers/aprx_model_elastica/train_surrogate.py`, `src/wandb_utils.py`

---

## Lessons Learned

- When switching dataset classes, audit all attribute access patterns (not just `__getitem__` return keys but also bulk attributes like `.deltas`, `.serpenoid_times`).
- When writing multiprocessing code, use module-level functions or explicit `fork` context to avoid pickle issues with Python 3.12+ spawn default.
- When creating adapter/wrapper classes for utility functions, check if the utility calls `asdict()` or other dataclass-specific functions.
- Always verify PyTorch attribute names against the actual API — `total_mem` vs `total_memory` is an easy mistake.
