---
name: Numba thread pool deadlock causes sporadic worker stalls
description: PyElastica Numba JIT functions deadlock in forked workers due to unconstrained NUMBA_NUM_THREADS, causing 6 stalls across 3.5 hours of parallel data collection
type: issue
status: resolved
severity: medium
subtype: performance
created: 2026-03-10
updated: 2026-03-10
tags: [numba, deadlock, multiprocessing, pyelastica, surrogate, data-collection]
aliases: []
---

# Numba Thread Pool Deadlock Causes Sporadic Worker Stalls

## Summary

During surrogate data collection (16 workers, 1M transitions), 6 of 16 workers stalled and required auto-respawn. Root cause analysis reveals a Numba thread pool deadlock triggered by forked worker processes inheriting an oversized thread pool configuration.

## Stall Signature

Every stall follows an identical pattern:

| Poll | Transitions | Delta |
|------|------------|-------|
| t-4  | N          | 0     |
| t-3  | N          | 0     |
| t-2  | N+500      | 500   |
| t-1  | N+500      | 0     |
| t    | N+500      | 0     |
| ...  | N+500      | 0 (×6 consecutive → stall detected) |

Key observation: the freeze **always** occurs immediately after completing an episode (delta=500 = `max_episode_steps`), meaning the hang is at an episode boundary — during `env.reset()` → `_init_elastica()` or the first `_do_step()` of the new episode.

## Timeline

| Time (UTC)  | Worker | PID   | Stall# | Worker-hours elapsed |
|-------------|--------|-------|--------|----------------------|
| 21:49:06    | 7      | 45011 | 1      | ~16                  |
| 22:28:37    | 14     | 45018 | 2      | ~27                  |
| 23:29:39    | 10     | 45014 | 3      | ~43                  |
| 23:37:40    | 5      | 45009 | 4      | ~45                  |
| 00:14:11    | 9      | 45013 | 5      | ~55                  |
| 00:20:12    | 6      | 45010 | 6      | ~56                  |

Rate: ~1 stall per 9.6 worker-hours. No worker stalled twice. All auto-respawned successfully.

## Root Cause

### The deadlock chain

1. Collector uses `mp.set_start_method("forkserver")` (`collect_data.py:829`)
2. PyElastica is **heavily Numba JIT-compiled** — 22+ `@njit` functions including all core physics:
   - `_compute_internal_forces`, `_compute_internal_torques`
   - `_compute_geometry_from_state`, `_compute_all_dilatations`
   - `overload_operator_kinematic_numba`, `overload_operator_dynamic_numba`
   - Plus linalg helpers (`_batch_cross`, `_batch_dot`, `_batch_matvec`, etc.)
3. The collector sets `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` but **does NOT set `NUMBA_NUM_THREADS`**
4. Numba defaults to the machine's CPU count: **48 threads per worker**
5. 16 workers × 48 Numba threads = **768 threads** competing for 48 cores

### Why it deadlocks

- `forkserver` forks from a preloaded Python process; Numba's internal thread pool state (TBB/OpenMP backend) can become inconsistent after fork
- When a worker calls `env.reset()` → `_init_elastica()` → first `_do_step()`, the Numba thread pool is re-engaged for the new simulator instance
- With 48 Numba threads per worker under heavy contention, there is a low but non-zero probability of a thread pool deadlock (one thread waiting for a lock held by another blocked thread)
- The process hangs permanently in the JIT function — no progress, no crash, no exception

### Why it's random and non-repeating

- Deadlock requires a specific interleaving of thread pool operations — a classic low-probability race condition
- Respawned workers get a fresh process with a clean thread pool, so they don't re-stall
- 6 unique workers affected out of 16 — no pattern to which worker, consistent with random timing

## Evidence

- Numba version: 0.64.0
- Default threading layer: TBB (not fork-safe)
- `NUMBA_NUM_THREADS`: unset (defaults to 48)
- Multiprocessing start method: forkserver
- All stalls occur at exact episode boundaries (transition count is always a multiple of 500)
- No NaN discards, no crashes — purely a hang/deadlock

## Impact

- 6 stalls across 3.5 hours, each causing ~3 minutes of lost worker time before respawn
- Total lost time: ~18 worker-minutes out of ~56 worker-hours (0.5%)
- Auto-respawn handled all cases; zero data loss; FPS unaffected
- **Operational impact: low** (mitigated by existing respawn mechanism)
- **Latent risk: medium** (rate could increase with more workers or longer runs)

## Recommended Fixes

### 1. Quick fix: constrain Numba threads (high confidence)

Add to `collect_data.py` alongside existing thread-limiting env vars:

```python
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
```

PyElastica's Numba functions operate on a single 20-element rod — there is zero benefit from Numba's internal parallelism. This eliminates 768→16 total threads and removes the deadlock surface entirely.

### 2. Better fix: also set fork-safe threading layer

```python
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
```

The `workqueue` backend is simpler and fork-safe, unlike TBB/OpenMP. Combined with `NUMBA_NUM_THREADS=1`, this is belt-and-suspenders.

### 3. Long-term: consider `spawn` start method

`spawn` creates fully independent child processes with no inherited state. Slower startup (~2s per worker) but eliminates all fork-related state corruption. The current `forkserver` was chosen as a middle ground, but with the Numba fix above it should be sufficient.

## Related

- [surrogate-worker-7-stall-respawn.md](surrogate-worker-7-stall-respawn.md) — operational timeline of all 6 stalls
- `collect_data.py:33-35` — existing thread-limiting env vars (missing Numba)
- `collect_data.py:829` — `forkserver` start method
- `locomotion_elastica/env.py:220-283` — `_init_elastica()` (re-creates simulator each reset)
