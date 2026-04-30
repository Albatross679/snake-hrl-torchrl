---
name: ParallelEnv pthread exhaustion with 256 envs
description: 256 parallel envs cause OpenBLAS pthread_create failures and BrokenPipeError during initialization
type: issue
status: resolved
severity: medium
subtype: system
created: 2026-03-19
updated: 2026-03-19
tags: [ppo, choi2025, parallelenv, openblas, threading]
aliases: []
---

## Symptom

Launching `train_ppo.py` with `--num-envs 256` (or 64) causes hundreds of OpenBLAS `pthread_create failed` errors followed by `BrokenPipeError` and `BlockingIOError` during `ParallelEnv` initialization. The process either crashes or hangs indefinitely.

With 32 envs, training works fine.

## Root Cause

Each `ParallelEnv` worker subprocess spawns its own OpenBLAS thread pool (default 32 threads). With 256 workers: 256 x 32 = 8,192 threads, exceeding system limits. Even with 64 workers, the 2,048 threads combined with other processes caused pipe failures during multiprocessing init.

## Fix Applied

1. Set thread-limiting env vars at the top of `train_ppo.py` before any imports:
   ```python
   os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
   os.environ.setdefault("MKL_NUM_THREADS", "1")
   os.environ.setdefault("OMP_NUM_THREADS", "1")
   ```
2. Also pass these as shell env vars when launching.
3. Reverted to 32 envs (proven stable) since the DisMech physics runs on CPU — more envs doesn't help GPU utilization.

## Files Modified

- `papers/choi2025/train_ppo.py` — added thread-limiting env vars
