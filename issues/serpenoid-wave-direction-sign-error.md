---
name: Serpenoid wave direction sign error in transform classes
description: DirectSerpenoidTransform and DirectSerpenoidSteeringTransform used -omega*t instead of +omega*t, producing tail-to-head wave propagation (backward locomotion) instead of the correct head-to-tail direction
type: issue
status: resolved
severity: medium
subtype: physics
created: 2026-03-09
updated: 2026-03-09
tags: [serpenoid, wave-propagation, curvature, physics, sign-error]
aliases: []
---

## Problem

The serpenoid wave equation in `src/physics/cpg/action_wrapper.py` used the wrong sign convention for the temporal term, causing the curvature wave to propagate in the wrong direction.

### Affected methods

- `DirectSerpenoidTransform.step()` (line 341)
- `DirectSerpenoidSteeringTransform.step()` (line 482)

Both computed:

```
kappa(s, t) = A * sin(k*s - omega*t + phi)
```

This produces a wave with phase velocity `+omega/k` in the `+s` direction (tail to head), which is **anterior propagation** — the opposite of what serpentine locomotion requires.

### Correct equation

The actual simulation in `locomotion_elastica/env.py:542-548` correctly uses:

```
kappa(s, t) = A * sin(k*s + omega*t + phi)
```

This gives phase velocity `-omega/k` in the `+s` direction, meaning the wave propagates from **head to tail** (posteriorly). Posterior wave propagation is what creates forward thrust via anisotropic friction (c_n > c_t in RFT).

### Why it wasn't caught earlier

The env's `_step()` method computes curvatures **inline** and never calls `transform.step()`. The transform is only used for `denormalize_action()` and time tracking. So the bug existed in dead code paths.

However, these `step()` methods would produce incorrect behavior if ever called directly (e.g., from `CPGEnvWrapper` or future code).

## Fix

Changed `-omega*t` to `+omega*t` in both `DirectSerpenoidTransform.step()` and `DirectSerpenoidSteeringTransform.step()`.

### Convention

- `joint_positions = np.linspace(0, 1, num_joints)`: s=0 is tail, s=1 is head
- `sin(k*s + omega*t)`: wave travels head-to-tail (correct for forward locomotion)
- `sin(k*s - omega*t)`: wave travels tail-to-head (would cause backward locomotion)

## Files changed

- `src/physics/cpg/action_wrapper.py`
