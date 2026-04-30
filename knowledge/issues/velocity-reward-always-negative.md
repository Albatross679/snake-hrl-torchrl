---
name: velocity-reward-always-negative
description: Velocity-based reward function (v_g) was noisy and always negative
type: issue
created: 2026-03-06T00:00:00
updated: 2026-03-09T01:36:55
tags: [bug, locomotion, elastica, training, reward]
aliases: []
status: resolved
severity: high
subtype: training
---

# Velocity-Based Reward Always Negative

## Problem

The Liu 2023 reward `R = c_v * v_g + c_g * v_g * cos(θ_g) / dist` was:
- Tiny magnitude (±0.05 per step) — hard for PPO to learn from
- Negative whenever the snake overshoots or drifts laterally
- Proportional to an inherently noisy velocity estimate

Mean reward was always negative (v_g ≈ -0.16 m/s). PPO could not learn meaningful behavior.

## Fix

Replaced with distance-based potential reward:
`R = c_dist*(prev_dist - curr_dist) + c_align*cos(θ_g) + goal_bonus`

With c_dist=10, c_align=0.1, goal_bonus=100.

## Files Modified

- `locomotion_elastica/rewards.py`
