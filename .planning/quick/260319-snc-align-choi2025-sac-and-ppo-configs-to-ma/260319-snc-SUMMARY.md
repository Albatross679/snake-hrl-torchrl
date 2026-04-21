---
phase: quick-260319-snc
plan: 01
subsystem: choi2025-replication
tags: [config, sac, ppo, hyperparameters, paper-replication]
dependency_graph:
  requires: []
  provides: [aligned-choi2025-configs, soft-update-period-gating]
  affects: [papers/choi2025, src/configs/training, src/trainers/sac]
tech_stack:
  added: []
  patterns: [soft-update-period-gating, multi-substep-physics]
key_files:
  created: []
  modified:
    - src/configs/training.py
    - src/trainers/sac.py
    - papers/choi2025/config.py
    - papers/choi2025/env.py
decisions:
  - "soft_update_period defaults to 1 for backward compatibility"
  - "alpha set to 0.0 (not just auto_alpha=False) to fully disable entropy bonus"
  - "Mock physics only calls _apply_curvature on first substep to avoid double-stepping"
metrics:
  duration: "2 min"
  completed: "2026-03-19"
  tasks_completed: 2
  tasks_total: 2
  files_modified: 4
---

# Quick Task 260319-snc: Align Choi2025 SAC and PPO Configs to Match Paper

Aligned 7 confirmed hyperparameter mismatches between our choi2025 replication and the original paper (Table A.1), plus added soft_update_period infrastructure to SACConfig/SACTrainer.

## One-liner

SAC/PPO configs aligned to paper: dt=0.05, gravity ON, no entropy tuning, soft_update_period=8, 500 envs, multi-substep 10 Hz control.

## Changes Made

### Task 1: Add soft_update_period to SACConfig and implement in trainer

- Added `soft_update_period: int = 1` field to `SACConfig` (backward compatible default)
- Gated `_soft_update()` in `SACTrainer._update()` to fire every N critic updates using existing `_update_count`
- Commit: 3d32fac

### Task 2: Fix all choi2025 config mismatches

**Physics (Choi2025PhysicsConfig):**
- `dt`: 0.01 -> 0.05 (50ms substep)
- `enable_gravity`: False -> True
- `max_newton_iter_noncontact`: 15 -> 2
- `max_newton_iter_contact`: 25 -> 5

**SAC (Choi2025Config):**
- `auto_alpha`: True -> False (paper cites Yu et al. 2022)
- `alpha`: 0.2 -> 0.0 (no entropy bonus)
- `soft_update_period`: 1 -> 8 (target update every 8 critic updates)
- `num_envs`: 32 -> 500

**PPO (Choi2025PPOConfig):**
- `num_envs`: 1 -> 500 (match SAC parallelism)
- `total_frames`: 1M -> 5M (match SAC training budget)

**Environment (Choi2025EnvConfig + env.py):**
- Added `control_period: int = 2` (2 substeps at dt=0.05 = 10 Hz control)
- Replaced single physics step with multi-substep loop in `_step()`
- Mock backend only applies curvature on first substep (avoids double-stepping)
- Target movement scaled by `dt * num_substeps` for correct follow_target timing

- Commit: b1bd7d1

## Deviations from Plan

None -- plan executed exactly as written.

## Verification

All automated checks passed:
- SACConfig.soft_update_period exists with default 1
- Choi2025PhysicsConfig: dt=0.05, gravity=True, newton_iter=2/5
- Choi2025Config: auto_alpha=False, alpha=0.0, soft_update_period=8, num_envs=500
- Choi2025PPOConfig: num_envs=500, total_frames=5M
- Choi2025EnvConfig: control_period=2

## Self-Check: PASSED

All modified files exist and both commits verified:
- 3d32fac: soft_update_period infrastructure
- b1bd7d1: config alignment and multi-substep physics
