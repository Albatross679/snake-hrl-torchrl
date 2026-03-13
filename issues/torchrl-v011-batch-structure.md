---
created: 2026-03-05T00:00:00
updated: 2026-03-05T00:00:00
description: TorchRL v0.11 moved done, reward, and episode_reward from batch root to batch next
tags: [torchrl, compatibility, v0.11, bug, ppo, batch]
type: issue
status: resolved
severity: high
subtype: compatibility
---

# TorchRL v0.11 Batch Structure Change

## Problem

TorchRL v0.11 stores `done`, `reward`, `episode_reward`, `step_count` under `batch["next"]`, not at the batch root. The PPO trainer was reading `batch["done"]` (always 0) instead of `batch["next"]["done"]`, causing episode metrics to never register completions.

**Affected:** `src/trainers/ppo.py` (episode metrics extraction)

## Fix

Changed all episode metric extraction to use `batch.get("next", batch)`.
