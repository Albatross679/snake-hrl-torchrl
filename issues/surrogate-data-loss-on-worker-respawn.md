---
name: 19% data loss from worker respawns during surrogate collection
description: Respawned workers lose in-memory transitions that haven't been flushed to disk, causing 193k of 1M transitions to be lost
type: issue
status: open
severity: medium
subtype: performance
created: 2026-03-10
updated: 2026-03-10
tags: [surrogate, data-collection, data-loss]
aliases: []
---

# Data Loss on Worker Respawn During Surrogate Collection

## Problem

When a worker is respawned (due to stall or shutdown death), all in-memory transitions that haven't been flushed to a batch file are lost. With the current 50k-transition flush threshold, a worker can lose up to 49,999 transitions on respawn.

## Impact

- First 1M-transition run: 814,000 saved of 1,007,500 collected (19.2% loss)
- Workers that were respawned once lost 54–72% of their total transitions
- Workers that were never respawned lost ~0%

## Root Cause

Batch files are written every 50,000 transitions. If a worker stalls at 45,000 transitions and is killed, those 45,000 transitions are lost. The respawned worker starts fresh.

## Proposed Fix

1. Reduce flush interval from 50k to 10k transitions (increases file count but limits max loss to ~10k per respawn)
2. Add a signal handler (SIGTERM) to flush the current buffer before exit
3. Consider shared-memory or file-based ring buffer so the parent can recover data from dead workers
