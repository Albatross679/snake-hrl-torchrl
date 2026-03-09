---
created: 2026-03-05T00:00:00
updated: 2026-03-05T00:00:00
tags: [torchrl, compatibility, v0.11, bug, ppo]
type: issue
status: resolved
severity: medium
subtype: compatibility
---

# TorchRL v0.11 ClipPPOLoss Parameter Renames

## Problem

TorchRL v0.11 renamed constructor parameters on `ClipPPOLoss`.

**Affected:** `src/trainers/ppo.py:77-84`

- `critic_coef` → `critic_coeff`
- `entropy_coef` → `entropy_coeff`

## Fix

Updated parameter names in PPO trainer to use the new spelling.
