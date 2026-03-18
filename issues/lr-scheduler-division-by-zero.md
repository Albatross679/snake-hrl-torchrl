---
name: LR scheduler division by zero when total_frames < frames_per_batch
description: Linear LR scheduler divides by zero when total_frames is smaller than frames_per_batch, crashing smoke tests and short runs
type: issue
status: resolved
severity: medium
subtype: training
created: 2026-03-05T00:00:00
updated: 2026-03-05T00:00:00
tags: [bug, ppo, lr-scheduler, training]
aliases: []
---

# LR Scheduler Division by Zero

## Problem

When `total_frames < frames_per_batch`, the linear LR scheduler computation in `src/trainers/ppo.py` divides by zero:

```
total_updates = self.config.total_frames // self.config.frames_per_batch  # = 0
1.0 - step / total_updates  # ZeroDivisionError
```

This occurs during smoke tests or short runs where the batch size exceeds total frames.

## Root Cause

Integer division `total_frames // frames_per_batch` yields 0 when `total_frames < frames_per_batch`, which is then used as a divisor in the LR lambda.

## Fix

Added `max(1, ...)` guard:

```python
total_updates = max(1, self.config.total_frames // self.config.frames_per_batch)
```

File: `src/trainers/ppo.py:102`

## Status

Fixed.
