---
id: 8d2763de-29f5-49b3-8460-0c2bc466c580
name: elastica-amplitude-range-too-small
description: Amplitude range (0, 0.15) produced negligible bending with small rod radius
type: issue
created: 2026-03-06T00:00:00
updated: 2026-03-09T01:36:55
tags: [bug, physics, locomotion, elastica]
aliases: []
status: resolved
severity: medium
subtype: physics
---

# Elastica Amplitude Range Too Small

## Problem

With r=0.001m and amp=0.15, the curvature produced negligible bending:
- Max angle per segment: 0.15 × 0.025m = 0.00375 rad ≈ 0.2°
- Insufficient for serpentine locomotion

## Fix

Increased amplitude range from (0, 0.15) to (0, 5.0) after fixing rod radius to 0.02m.

## Files Modified

- `locomotion_elastica/config.py`
