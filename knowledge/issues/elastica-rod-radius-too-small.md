---
name: elastica-rod-radius-too-small
description: Rod radius 0.001m gave extremely small moment of inertia causing instability
type: issue
created: 2026-03-06T00:00:00
updated: 2026-03-09T01:36:55
tags: [bug, physics, locomotion, elastica, diagnosis]
aliases: []
status: resolved
severity: critical
subtype: physics
---

# Elastica Rod Radius Too Small

## Problem

The original rod radius of 0.001m gave an extremely small moment of inertia:
- I = π r⁴/4 = 7.85e-13 m⁴
- EI = 2e6 × 7.85e-13 = 1.57e-6 N·m²

This created a sharp instability boundary:
- **E ≤ 5e5**: Rod couldn't generate enough elastic force to deform → zero motion
- **E ≥ 1e6**: Rod snapped violently to target curvature → actual κ reached ±100 (vs target ±2.5), velocities reached 178 m/s → chaotic trajectory

Training sessions 1-6 showed the snake moving fast but in random/chaotic directions, never consistently toward the goal.

## Fix

Grid search over E ∈ {2e4, 1e5, 2e5, 5e5, 1e6, 2e6} × r ∈ {0.001, 0.005, 0.01, 0.02} found:
- **r=0.02, E=1e5**: Stable forward locomotion, snake reaches goal

## Key Physics Insight

The bending stiffness EI must be large enough that the elastic restoring force (F ≈ EI × κ / L) produces meaningful velocities when the rod deforms, but not so large that the simulation becomes unstable. The moment of inertia I ∝ r⁴ is extremely sensitive to radius — increasing r by 20× increases I by 160,000×.

## Files Modified

- `locomotion_elastica/config.py`
