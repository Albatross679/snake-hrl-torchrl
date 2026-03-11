---
created: 2026-03-10T00:58:14.774Z
title: Fix Numba thread pool deadlock in parallel data collection
area: general
files:
  - aprx_model_elastica/collect_data.py:33-35
  - aprx_model_elastica/collect_data.py:829
  - locomotion_elastica/env.py:220-283
  - issues/numba-thread-pool-deadlock-worker-stalls.md
---

## Problem

During surrogate data collection (16 workers, 1M transitions), 6 of 16 workers stalled due to Numba thread pool deadlocks. The collector sets `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` but does NOT set `NUMBA_NUM_THREADS`. Numba defaults to 48 (machine CPU count), creating 768 total threads across 16 workers. Combined with `forkserver` start method, this causes sporadic deadlocks in PyElastica's 22+ Numba JIT functions at episode boundaries (~1 stall per 9.6 worker-hours).

## Solution

1. **Quick fix**: Add `os.environ.setdefault("NUMBA_NUM_THREADS", "1")` at `collect_data.py:33-35` alongside existing thread-limiting env vars. PyElastica's Numba functions operate on a single 20-element rod — zero benefit from Numba parallelism.

2. **Better fix**: Also add `os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")` for a fork-safe threading backend.

3. **Long-term**: Consider `spawn` instead of `forkserver` start method to eliminate all fork-inherited state issues.

See `issues/numba-thread-pool-deadlock-worker-stalls.md` for full root cause analysis.
