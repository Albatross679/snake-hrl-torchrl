---
name: Choi2025 physics parameters mismatched with paper
description: Rod radius 50x too thin and Young's modulus 5x too soft compared to paper Table A.3
type: issue
status: resolved
severity: critical
subtype: physics
created: 2026-03-19
updated: 2026-03-19
tags: [choi2025, physics, configuration, dismech]
aliases: []
---

## Symptom

PPO training for follow_target task plateaus at rolling100 ~31 instead of paper's ~100. Rod is too flexible and thin to effectively reach targets.

## Root Cause

Two physics parameters in `papers/choi2025/config.py` did not match the paper (Choi & Tong, 2025, Table A.3):

| Parameter | Implementation | Paper | Factor |
|-----------|---------------|-------|--------|
| Rod radius | 0.001 m | 0.05 m | 50x too thin |
| Young's modulus | 2 MPa | 10 MPa | 5x too soft |

The thin rod with low stiffness creates a near-limp structure that cannot reliably extend toward targets, capping achievable reward.

## Fix Applied

- `snake_radius`: 0.001 → 0.05
- `youngs_modulus`: 2e6 → 10e6

## Files Modified

- `papers/choi2025/config.py`: Lines 71, 76
