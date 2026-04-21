---
name: elastica-curvature-not-updated-substeps
description: Serpenoid curvature computed once per RL step instead of per-substep
type: issue
created: 2026-03-06T00:00:00
updated: 2026-03-09T11:25:23
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

### Phase 1 (2026-03-06)
Moved curvature computation from once per RL step (0.5s) to once per physics step (0.05s). Reduced phase jump from ~360° to ~20° per update.

### Phase 2 (2026-03-09)
Moved curvature computation to every PyElastica substep (0.001s). Phase jump now ~0.4° per update — near-continuous. Also applied same fix to `src/physics/elastica_snake_robot.py`. See [[unify-curvature-substep-frequency]].

## Files Modified

- `locomotion_elastica/env.py`
- `src/physics/elastica_snake_robot.py`
