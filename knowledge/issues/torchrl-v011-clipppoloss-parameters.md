---
name: TorchRL v0.11 ClipPPOLoss parameter renames (critic_coef → critic_coeff)
description: TorchRL v0.11 renamed critic_coef to critic_coeff and entropy_coef to entropy_coeff on ClipPPOLoss constructor
type: issue
status: resolved
severity: medium
subtype: compatibility
created: 2026-03-05T00:00:00
updated: 2026-03-05T00:00:00
tags: [torchrl, compatibility, v0.11, bug, ppo]
aliases: []
---

# TorchRL v0.11 ClipPPOLoss Parameter Renames

## Problem

TorchRL v0.11 renamed constructor parameters on `ClipPPOLoss`.

**Affected:** `src/trainers/ppo.py:77-84`

- `critic_coef` → `critic_coeff`
- `entropy_coef` → `entropy_coeff`

## Fix

Updated parameter names in PPO trainer to use the new spelling.
