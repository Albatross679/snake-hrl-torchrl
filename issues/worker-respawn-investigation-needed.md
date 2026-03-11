---
name: Worker respawn root cause investigation needed for longer collection runs
description: Consolidates three related worker-stall issues and frames the investigation needed when training completes and longer (10+ GB) collection runs are attempted
type: issue
status: open
severity: medium
subtype: performance
created: 2026-03-11
updated: 2026-03-11
tags: [surrogate, data-collection, worker-stall, investigation, numba]
aliases: []
---

# Worker Respawn Investigation Needed for Longer Collection Runs

## Overview

During Phase 02.1 surrogate data collection (1M transitions, 16 workers, ~3.5 hours), 6 workers stalled and were auto-respawned. The root cause has been diagnosed as a Numba thread pool deadlock in forked workers, and a quick fix has been proposed but **not yet applied**. This issue consolidates the diagnosis and frames what investigation is still needed when Phase 02.2 collection scales to 10 GB and beyond.

## Related Issues

This issue consolidates findings from three existing issues:

1. **[surrogate-worker-7-stall-respawn.md](surrogate-worker-7-stall-respawn.md)** -- Operational timeline documenting all 6 stalls during the Phase 02.1 run. Workers 7, 14, 10, 5, 9, and 6 each stalled once at episode boundaries and were auto-respawned. No worker stalled twice. FPS remained steady at ~57 throughout.

2. **[surrogate-data-loss-on-worker-respawn.md](surrogate-data-loss-on-worker-respawn.md)** -- Documents 19% data loss (193k of 1M transitions) caused by respawned workers losing in-memory transitions that had not been flushed to disk. With the current 50k-transition flush threshold, a single respawn can lose up to 49,999 transitions.

3. **[numba-thread-pool-deadlock-worker-stalls.md](numba-thread-pool-deadlock-worker-stalls.md)** -- Root cause analysis identifying Numba thread pool deadlock as the stall mechanism. `NUMBA_NUM_THREADS` is unset (defaults to 48), creating 768 total threads across 16 workers competing for 48 cores. Stalls occur at episode boundaries during `env.reset()` when Numba's thread pool re-engages under contention.

## What Has Been Diagnosed

- **Root cause:** Numba thread pool deadlock in forked workers. `NUMBA_NUM_THREADS` is unset, defaulting to 48 threads per worker. With 16 workers, this creates 768 threads competing for 48 CPU cores. The TBB threading backend is not fork-safe.
- **Stall rate:** ~1 stall per 9.6 worker-hours during Phase 02.1 (6 stalls across 3.5 hours with 16 workers).
- **Stall pattern:** Always occurs at episode boundaries (transition count is a multiple of `max_episode_steps=500`), during `env.reset()` or the first `_do_step()` of the new episode.
- **Quick fix proposed but NOT yet applied:** Set `NUMBA_NUM_THREADS=1` and `NUMBA_THREADING_LAYER=workqueue` in `collect_data.py` alongside existing thread-limiting env vars. PyElastica's Numba functions operate on a single 20-element rod with zero benefit from internal parallelism.
- **Data loss mechanism:** Respawned workers lose all in-memory transitions (up to 49,999 with the current 50k flush threshold). Workers that were respawned once lost 54-72% of their total transitions. Proposed mitigations: reduce flush interval to 10k, add SIGTERM handler for graceful flush.

## Investigation Still Needed

The following items should be investigated when training completes and longer collection runs are attempted (Phase 02.2 collection to 10 GB):

1. **Verify Phase 02.2 stall rate:** Confirm whether Phase 02.2 collection (currently running with `steps_per_run=1`) experiences the same ~1 stall per 9.6 worker-hours rate observed in Phase 02.1 (`steps_per_run=4`). The reduced steps per run may change stall characteristics.

2. **Apply and validate the NUMBA_NUM_THREADS=1 fix:** Add `os.environ.setdefault("NUMBA_NUM_THREADS", "1")` and `os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")` to `collect_data.py`. Run a multi-hour collection and measure whether the stall rate drops to zero.

3. **Implement data loss mitigation:** Reduce the flush interval from 50k to 10k transitions to limit maximum data loss per respawn from ~50k to ~10k transitions. Add a SIGTERM signal handler to flush the current buffer before worker exit.

4. **Monitor longer runs for new failure modes:** 10+ GB collection runs (targeting ~5M transitions at ~2055 bytes/transition) will run for 30+ hours. Monitor whether stall rates increase over time (the Phase 02.1 data showed a slight acceleration: 2 stalls in the first 2 hours, 4 more in the next 1.5 hours) or whether new failure modes emerge (memory leaks, file descriptor exhaustion, etc.).

5. **Check steps_per_run=1 vs steps_per_run=4 stall characteristics:** Phase 02.2 uses `steps_per_run=1` (one RL step per environment reset) while Phase 02.1 used `steps_per_run=4`. With more frequent resets, the stall surface (episode boundary deadlock) may be encountered more often per wall-clock hour -- or less often if shorter runs reduce thread pool contention buildup.

## Priority

Medium. The auto-respawn mechanism handles stalls operationally, but 19% data loss is significant for longer runs. The NUMBA_NUM_THREADS fix is high-confidence and low-risk -- it should be applied before the next large-scale collection campaign.

## When to Address

- **Before next large collection:** Apply the NUMBA_NUM_THREADS=1 fix (5 minutes of work, high confidence)
- **During Phase 02.2 10 GB collection:** Monitor stall rate and data loss
- **Before Phase 3 training data finalization:** Ensure data completeness by validating that the fix eliminates stalls or that data loss is within acceptable bounds
