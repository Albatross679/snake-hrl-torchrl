---
name: Numba thread pool deadlock causes worker stalls and data loss during surrogate collection
description: Numba thread pool deadlock in forked workers caused 6 stalls during 1M-transition collection; root cause diagnosed (NUMBA_NUM_THREADS unset), fix proposed but not yet applied; 19% data loss from respawned workers losing in-memory transitions
type: issue
status: open
severity: medium
subtype: performance
created: 2026-03-09
updated: 2026-03-16
tags: [surrogate, data-collection, worker-stall, numba, deadlock, multiprocessing, pyelastica, data-loss]
aliases: []
---

# Numba Thread Pool Deadlock: Worker Stalls and Data Loss

This issue consolidates the worker stall problem end-to-end: operational timeline, root cause analysis, data loss impact, and outstanding investigation items.

## Summary

During Phase 02.1 surrogate data collection (16 workers, 1M transitions, ~3.5 hours), 6 of 16 workers stalled and were auto-respawned. Root cause: Numba thread pool deadlock in forked workers due to unconstrained `NUMBA_NUM_THREADS`. A quick fix has been proposed but **not yet applied**.

## Stall Timeline

| Time (UTC)  | Worker | PID   | Stall# | Worker-hours elapsed |
|-------------|--------|-------|--------|----------------------|
| 21:49:06    | 7      | 45011 | 1      | ~16                  |
| 22:28:37    | 14     | 45018 | 2      | ~27                  |
| 23:29:39    | 10     | 45014 | 3      | ~43                  |
| 23:37:40    | 5      | 45009 | 4      | ~45                  |
| 00:14:11    | 9      | 45013 | 5      | ~55                  |
| 00:20:12    | 6      | 45010 | 6      | ~56                  |

Rate: ~1 stall per 9.6 worker-hours. No worker stalled twice. All auto-respawned successfully. FPS remained steady at ~57 throughout.

## Stall Signature

Every stall follows an identical pattern — the freeze **always** occurs at an episode boundary (transition count is a multiple of `max_episode_steps=500`), during `env.reset()` → `_init_elastica()` or the first `_do_step()` of the new episode:

| Poll | Transitions | Delta |
|------|------------|-------|
| t-2  | N+500      | 500   |
| t-1  | N+500      | 0     |
| t    | N+500      | 0     |
| ...  | N+500      | 0 (×6 consecutive → stall detected) |

## Root Cause

### The deadlock chain

1. Collector uses `mp.set_start_method("forkserver")` (`collect_data.py:829`)
2. PyElastica is **heavily Numba JIT-compiled** — 22+ `@njit` functions including all core physics (`_compute_internal_forces`, `_compute_internal_torques`, `_compute_geometry_from_state`, etc.)
3. The collector sets `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` but **does NOT set `NUMBA_NUM_THREADS`**
4. Numba defaults to the machine's CPU count: **48 threads per worker**
5. 16 workers × 48 Numba threads = **768 threads** competing for 48 cores

### Why it deadlocks

- `forkserver` forks from a preloaded Python process; Numba's internal thread pool state (TBB/OpenMP backend) can become inconsistent after fork
- When a worker calls `env.reset()` → `_init_elastica()` → first `_do_step()`, the Numba thread pool is re-engaged for the new simulator instance
- With 48 Numba threads per worker under heavy contention, there is a low but non-zero probability of a thread pool deadlock
- The process hangs permanently in the JIT function — no progress, no crash, no exception

### Why it's random and non-repeating

- Deadlock requires a specific interleaving of thread pool operations — a classic low-probability race condition
- Respawned workers get a fresh process with a clean thread pool, so they don't re-stall
- 6 unique workers affected out of 16 — no pattern to which worker, consistent with random timing

## Data Loss Impact

When a worker is respawned, all in-memory transitions that haven't been flushed to a batch file are lost:

- First 1M-transition run: 814,000 saved of 1,007,500 collected (**19.2% loss**)
- Workers that were respawned once lost 54–72% of their total transitions
- Workers that were never respawned lost ~0%
- With the current 50k-transition flush threshold, a single respawn can lose up to 49,999 transitions

## Evidence

- Numba version: 0.64.0
- Default threading layer: TBB (not fork-safe)
- `NUMBA_NUM_THREADS`: unset (defaults to 48)
- Multiprocessing start method: forkserver
- All stalls occur at exact episode boundaries
- No NaN discards, no crashes — purely a hang/deadlock

## Proposed Fixes (Not Yet Applied)

### 1. Quick fix: constrain Numba threads (high confidence)

Add to `collect_data.py` alongside existing thread-limiting env vars:

```python
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
```

PyElastica's Numba functions operate on a single 20-element rod — there is zero benefit from Numba's internal parallelism. This eliminates 768→16 total threads and removes the deadlock surface entirely.

### 2. Also set fork-safe threading layer

```python
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
```

The `workqueue` backend is simpler and fork-safe, unlike TBB/OpenMP.

### 3. Reduce data loss on respawn

- Reduce flush interval from 50k to 10k transitions (limits max loss to ~10k per respawn)
- Add a SIGTERM signal handler to flush the current buffer before worker exit
- Consider shared-memory or file-based ring buffer so the parent can recover data from dead workers

### 4. Long-term: consider `spawn` start method

`spawn` creates fully independent child processes with no inherited state. Slower startup (~2s per worker) but eliminates all fork-related state corruption.

## Investigation Still Needed

1. **Apply and validate the NUMBA_NUM_THREADS=1 fix** — run a multi-hour collection and confirm stall rate drops to zero
2. **Verify Phase 02.2 stall rate** — confirm whether `steps_per_run=1` (vs Phase 02.1's `steps_per_run=4`) changes stall characteristics
3. **Monitor longer runs** — 10+ GB collection runs (30+ hours) may surface new failure modes (memory leaks, file descriptor exhaustion)
4. **Check stall rate trend** — Phase 02.1 data showed slight acceleration (2 stalls in first 2h, 4 more in next 1.5h)

## Priority

Medium. The auto-respawn mechanism handles stalls operationally, but 19% data loss is significant. The `NUMBA_NUM_THREADS` fix is high-confidence and low-risk — apply before the next large-scale collection campaign.

## Related Files

- `collect_data.py:33-35` — existing thread-limiting env vars (missing Numba)
- `collect_data.py:829` — `forkserver` start method
- `locomotion_elastica/env.py:220-283` — `_init_elastica()` (re-creates simulator each reset)
- [surrogate-data-loss-on-worker-respawn.md](surrogate-data-loss-on-worker-respawn.md) — detailed data loss analysis (now incorporated here)
