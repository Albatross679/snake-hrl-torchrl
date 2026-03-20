---
name: Arch sweep A3/A5 restart failure — SurrogateDataset/OverlappingPairDataset format mismatch
description: A3 and A5 failed to restart because train_surrogate.py used the deprecated SurrogateDataset (expects 'states' key) while Phase 02.1 re-collected data with substep_states format. Root cause is a timing conflict between ongoing Phase 02.1 data collection and Phase 03.1 arch sweep.
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-10
updated: 2026-03-10
tags: [surrogate, arch-sweep, dataset, data-format, phase-ordering]
aliases: []
---

## Summary

After A3 and A5 crashed at epoch 25 (rollout DataLoader thread contention), restart attempts failed immediately with `KeyError: 'states'` or OpenBLAS thread exhaustion.

## Root Cause: Race Between Phase 02.1 Data Collection and Phase 03.1 Arch Sweep

Timeline:
- **13:05** — Original 5 arch sweep runs started. At this point `data/surrogate/` contained old Phase 1 format data (`states`/`next_states` keys). `SurrogateDataset` loaded them fine: 733k transitions.
- **13:44** — Phase 02.1-03 data collection began, overwriting `data/surrogate/` with new checkpoint format (`substep_states` key, `.pt` files via `OverlappingPairDataset` format).
- **13:52** — Commit `7554107` added `OverlappingPairDataset` to `dataset.py` and deprecated `SurrogateDataset`. But `train_surrogate.py` was NOT updated.
- **13:xx** — A3 and A5 crashed (thread contention). Restart attempted.
- **Restart** — By this point, `data/surrogate/` has new-format files. `SurrogateDataset` tries `data["states"]` → `KeyError`.

## Errors

**A3 restart:**
```
OpenBLAS blas_thread_init: pthread_create failed for thread 46 of 48
KeyboardInterrupt  (during scipy import, due to thread exhaustion without OPENBLAS_NUM_THREADS=1)
```

**A5 restart:**
```
KeyError: 'states'
  train_dataset = SurrogateDataset(...)
```

## Fix Applied

1. Updated `train_surrogate.py` to use `OverlappingPairDataset` (replacing deprecated `SurrogateDataset`)
2. A3/A5 not restarted yet — only 75k samples available in new format (collection ongoing; original had 733k)

## Status of Running Processes

- A1/A4/B1: Still running successfully on old-format data loaded into memory at 13:05
- A3/A5: Need restart once sufficient new-format data is available

## Next Steps

- Wait for data collection to accumulate ~700k+ transitions in new format
- Restart A3/A5 with env vars: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`
- Or accept partial results: A1/A4/B1 may complete before sufficient data exists for A3/A5
