---
name: Stall detection false positives during PyElastica initialization
description: Health monitor killed workers as "stalled" during normal PyElastica startup because they showed zero progress during initialization
type: issue
status: resolved
severity: high
subtype: system
created: 2026-03-09
updated: 2026-03-09
tags: [surrogate, data-collection, health-monitoring, multiprocessing]
aliases: []
---

# Stall Detection False Positives During PyElastica Initialization

## Symptom

After implementing health monitoring with stall detection, the first data collection run killed all 16 workers within ~60 seconds of startup. The monitor log showed every worker flagged as "stalled" and respawned, creating an infinite kill-respawn cycle.

## Root Cause

PyElastica initialization is expensive — each worker must:
1. Fork via `forkserver` start method
2. Import PyElastica and NumPy
3. Construct Cosserat rod with material properties
4. Initialize RFT friction model
5. Run first simulation step to warm caches

With 16 workers competing for CPU on `forkserver`, this takes **>60 seconds**. The stall detection threshold was set to 2 intervals × 30 seconds = 60 seconds. Workers legitimately showed 0 transitions during init and were killed as stalled.

## Fix

Added a grace period check in the stall detection logic — only count zero-progress polls as stalls **after** a worker has produced at least 1 transition:

```python
# Before (broken): counted stalls from the start
if delta == 0:
    stall_counts[i] += 1

# After (fixed): grace period during init
if delta == 0 and current_wc > 0:  # Only after first transition
    stall_counts[i] += 1
elif delta > 0:
    stall_counts[i] = 0
```

Also increased `stall_intervals` from 2 to 6 (3 minutes at 30s poll) as additional safety margin in `collect_config.py`.

## Verification

After the fix, all 16 workers survived initialization and began producing transitions. Zero false stall detections across 39,000+ transitions.

## Lesson

When monitoring long-running worker processes, always account for initialization time. A grace period (wait for first output before enforcing liveness) is more robust than increasing timeouts alone.
