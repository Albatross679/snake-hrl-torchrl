---
name: Serpenoid wave direction sign error (-omega*t instead of +omega*t)
description: Wave equation used sin(k*s - omega*t) producing tail-to-head propagation instead of the correct sin(k*s + omega*t) for head-to-tail; fixed in both locomotion_elastica/env.py and src/physics/cpg/action_wrapper.py
type: issue
status: resolved
severity: high
subtype: physics
created: 2026-03-06
updated: 2026-03-09
tags: [bug, locomotion, elastica, physics, serpenoid, wave-propagation, curvature, sign-error]
aliases: []
---

# Serpenoid Wave Direction Sign Error

## Problem

The serpenoid wave equation used the wrong sign convention for the temporal term in multiple locations, causing the curvature wave to propagate in the wrong direction.

### Correct vs incorrect

- **Incorrect:** `kappa(s, t) = A * sin(k*s - omega*t + phi)` — wave travels tail-to-head (anterior propagation)
- **Correct:** `kappa(s, t) = A * sin(k*s + omega*t + phi)` — wave travels head-to-tail (posterior propagation, creates forward thrust via anisotropic friction)

### Convention

- `joint_positions = np.linspace(0, 1, num_joints)`: s=0 is tail, s=1 is head
- `sin(k*s + omega*t)`: wave travels head-to-tail (correct for forward locomotion)
- `sin(k*s - omega*t)`: wave travels tail-to-head (would cause backward locomotion)

## Affected locations

### 1. locomotion_elastica/env.py

The env's `_step()` method had incorrect wave direction with a misleading comment ("Reverse curvature order so wave travels head→tail"). The `[::-1]` reversal doesn't change wave direction — it mirrors the spatial pattern. Fixed by removing reversal and changing to `sin(k*s + omega*t)` with smooth per-substep curvature updates.

### 2. src/physics/cpg/action_wrapper.py

Both `DirectSerpenoidTransform.step()` (line 341) and `DirectSerpenoidSteeringTransform.step()` (line 482) used `-omega*t` instead of `+omega*t`. These were in code paths not exercised by the main env (the env computes curvatures inline), but would produce incorrect behavior if called directly from `CPGEnvWrapper` or future code. Changed `-omega*t` to `+omega*t` in both methods.

## Fix

Changed all wave equations to use `+omega*t` for correct head-to-tail propagation across both files.

## Files changed

- `locomotion_elastica/env.py`
- `src/physics/cpg/action_wrapper.py`
