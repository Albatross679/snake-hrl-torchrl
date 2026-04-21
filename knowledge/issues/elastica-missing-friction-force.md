---
name: elastica-missing-friction-force
description: AnisotropicRFTForce was never added to the PyElastica simulator
type: issue
created: 2026-03-06T00:00:00
updated: 2026-03-09T01:36:55
tags: [bug, locomotion, elastica, physics, friction]
aliases: []
status: resolved
severity: critical
subtype: physics
---

# Elastica Missing Friction Force

## Problem

The config defined RFT friction coefficients (`rft_ct`, `rft_cn`) but `_init_elastica()` never added any friction/drag force to the PyElastica simulator. Without anisotropic friction, serpentine undulation cannot produce net forward motion.

This was one of five root causes for why the snake did not move at all during training.

## Fix

Implemented `AnisotropicRFTForce` class (inherits `elastica.external_forces.NoForces`) that applies velocity-dependent anisotropic drag: `F_t = -c_t * v_t`, `F_n = -c_n * v_n`. Added to simulator in `_init_elastica()`.

## Files Modified

- `locomotion_elastica/env.py`
