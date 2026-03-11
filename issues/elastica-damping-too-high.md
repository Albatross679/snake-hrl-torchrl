---
name: elastica-damping-too-high
description: AnalyticalLinearDamper with damping_constant=0.1 completely freezes the rod
type: issue
created: 2026-03-06T00:00:00
updated: 2026-03-09T01:36:55
tags: [bug, locomotion, elastica, physics, damping]
aliases: []
status: resolved
severity: critical
subtype: physics
---

# Elastica Damping Too High

## Problem

`AnalyticalLinearDamper` with `damping_constant=0.1` completely freezes the rod — zero velocity, zero deformation. The threshold is ~0.008; above this, the rod cannot move.

This was one of five root causes for why the snake did not move at all during training (2M frames of Session 2 produced zero locomotion).

## Fix

Reduced `elastica_damping` from 0.1 to 0.002 in `locomotion_elastica/config.py`.

## Files Modified

- `locomotion_elastica/config.py`
