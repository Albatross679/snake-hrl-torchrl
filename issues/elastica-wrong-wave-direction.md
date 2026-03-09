---
id: 921d31f0-27c6-4863-8629-0396acc4f06d
name: elastica-wrong-wave-direction
description: Serpenoid wave direction reversal logic was incorrect
type: issue
created: 2026-03-06T00:00:00
updated: 2026-03-09T01:36:55
tags: [bug, locomotion, elastica, physics]
aliases: []
status: resolved
severity: high
subtype: physics
---

# Elastica Wrong Wave Direction

## Problem

The comment "Reverse curvature order so wave travels head→tail" was incorrect. The reversal was wrong because:
- The serpenoid `sin(k*s - ω*t)` propagates tail→head
- Reversing `[::-1]` doesn't change the wave direction, it mirrors the spatial pattern
- With `sin(k*s + ω*t)` (head→tail wave) the snake moves forward in some conditions

## Fix

Removed reversal, changed wave to `sin(k*s + ω*t)` with smooth per-substep curvature updates.

## Files Modified

- `locomotion_elastica/env.py`
