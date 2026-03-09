---
id: ed659234-ab19-4dfc-821f-cb1fb7662f83
name: elastica-curvature-not-updated-substeps
description: Serpenoid curvature computed once per RL step instead of per-substep
type: issue
created: 2026-03-06T00:00:00
updated: 2026-03-09T01:36:55
tags: [bug, locomotion, elastica, physics]
aliases: []
status: resolved
severity: high
subtype: physics
---

# Elastica Curvature Not Updated During Substeps

## Problem

The serpenoid curvature was computed once per RL step (dt_total=0.5s) and held constant for 500 internal substeps. This produced jerky, unrealistic rod dynamics instead of a smooth traveling wave.

## Fix

Rewritten `_step()` to compute curvature analytically at each internal substep, advancing `self._serpenoid._time` by `dt_sub` each step.

## Files Modified

- `locomotion_elastica/env.py`
