---
name: Surrogate collector worker stalls during data collection
description: Workers 7 and 14 each stalled and were auto-respawned during surrogate data collection
type: issue
status: resolved
severity: medium
subtype: performance
created: 2026-03-09
updated: 2026-03-09
tags: [surrogate, data-collection, worker-stall]
aliases: []
---

# Surrogate Collector Worker 7 Stall and Respawn

## Timeline

- **21:49:06 UTC** — Worker 7 (PID 45011) flagged as stalled after 6 consecutive intervals with zero new transitions
- **21:49:06 UTC** — Worker 7 auto-respawned (reason: stall, respawn_count: 1)

## Impact

- Worker 7 lagging: 12,500 transitions vs fleet mean of 15,406 (18.9% below mean)
- No data loss — stalled worker contributed 0 transitions during stall period
- Overall FPS unaffected (rolling=56, steady)
- Worker is recovering post-respawn

## Context

- Collection target: 1,000,000 transitions
- Progress at time of stall: ~234k (23.4%)
- 16 workers, all other workers healthy
- Zero NaN discards throughout run

## Assessment

Single stall/respawn is within normal operating bounds for long-running PyElastica parallel collection. The auto-respawn mechanism handled it correctly.

## Resolution (Check 4, ~22:22 UTC)

Worker 7 fully recovered. Counter reset on respawn: 12,500 (pre) + 7,000 (post) = 19,500 total, matching fleet mean of 20,031. Max worker deviation back to 5.1%.

## Second stall: Worker 14 (Check 5, ~22:40 UTC)

- **22:28:37 UTC** — Worker 14 (PID 45018) stalled after 6 zero-delta intervals, auto-respawned
- Post-respawn: w14 at 23,000 total (recovered to within 5.4% of fleet mean 24,188)
- Pattern: 2 stalls across 16 workers over 1h55m of collection. Both resolved by auto-respawn with no data loss or FPS impact. Likely intermittent PyElastica simulation hangs.

## Third & fourth stalls: Workers 10, 5 (Check 8, ~23:40 UTC)

- **23:29:39 UTC** — Worker 10 (PID 45014) stalled, auto-respawned
- **23:37:40 UTC** — Worker 5 (PID 45009) stalled, auto-respawned
- Both recovered; fleet deviation 4.1% at mean 37,031
- Running total: 4 stalls / 4 respawns across 2h55m (~1 stall per 11.6 worker-hours)
- Rate increasing slightly (2 in first 1h55m, 2 more in next 40m) — could be coincidence or related to longer episode lengths as collection progresses
- FPS unaffected (rolling=57), all workers balanced post-recovery

## Fifth & sixth stalls: Workers 9, 6 (Check 10, ~00:25 UTC)

- **00:14:11 UTC** — Worker 9 (PID 45013) stalled, auto-respawned
- **00:20:12 UTC** — Worker 6 (PID 45010) stalled, auto-respawned
- Both recovered; fleet deviation 4.7% at mean 45,656
- Running total: 6 stalls / 6 respawns across 3h35m (~1 stall per 9.6 worker-hours)
- 6 of 16 workers have now stalled at least once — confirms random per-worker PyElastica hangs
- No worker has stalled twice; no NaN discards; FPS steady at 57

## Root Cause

Numba thread pool deadlock in forked workers. See [numba-thread-pool-deadlock-worker-stalls.md](numba-thread-pool-deadlock-worker-stalls.md) for full analysis.
