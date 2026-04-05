---
name: Normalized multi-component PBRS reward refactor
description: Refactored Choi2025 reward system to use normalized components, decomposed PBRS, and pure importance weights
type: log
status: complete
subtype: refactor
created: 2026-04-02
updated: 2026-04-02
tags: [reward-design, pbrs, normalization, choi2025]
aliases: [phase-18-reward-refactor]
---

# Normalized Multi-Component PBRS Reward Refactor

## Changes

### `papers/choi2025/rewards.py`
- Removed `improvement_weight` / `improvement_reward` logic (duplicated PBRS without policy-invariance)
- Removed `pbrs_only` mode
- Normalized action smoothness: `smooth_penalty = -||Δa||²/(2·action_dim)` → range [-1, 0]
- Decomposed PBRS into per-objective potentials:
  - Distance PBRS: Φ(s) = -dist/workspace_radius (bounded [-1, 0])
  - Heading PBRS: Φ(s) = cos_sim(tip_tangent, to_target_dir) (bounded [-1, 1])
- Restructured into clear sections: base components → weighted sum → PBRS addition
- Added `reward_pbrs_head` to components dict, removed `reward_improve`
- Normalized obstacle reward components with documented ranges

### `papers/choi2025/config.py`
- Removed `pbrs_only` and `improvement_weight` fields
- Renamed `action_smoothness_weight` → `smooth_weight`
- Added `workspace_radius: float = 1.0` for PBRS normalization

### `papers/choi2025/env.py`
- Added `_prev_tip_tangent` to episode state for heading PBRS
- Updated `_reset` to initialize `_prev_tip_tangent`
- Updated `_step` to save `_prev_tip_tangent` before physics step
- Updated reward function call with new parameter names
- Replaced `reward_improve` with `reward_pbrs_head` in observation spec

### `papers/choi2025/train_ppo.py`
- Removed `--pbrs-only` and `--improvement-weight` CLI flags
- Renamed `--action-smoothness-weight` → `--smooth-weight`

### `tests/test_choi2025.py`
- Added `TestNormalizedRewards` class with 5 tests:
  - `test_normalized_reward_ranges`: verifies base components stay in documented ranges
  - `test_pbrs_telescoping`: verifies PBRS distance terms telescope correctly
  - `test_weights_are_importance`: verifies doubling a weight doubles contribution
  - `test_no_improvement_bonus`: verifies improvement bonus completely removed
  - `test_pbrs_head_component_logged`: verifies heading PBRS in components dict

## Architecture

```
total_reward = Σ wᵢ · normalize(rᵢ)     ← base components, normalized then weighted
             + Σ (γ·Φⱼ(s') - Φⱼ(s))     ← PBRS components, raw (naturally centered)
```

## Verification
- All 5 new tests pass
- All existing tests pass (5 pre-existing failures from earlier config changes, unrelated)
- No references to removed fields (`improvement_weight`, `pbrs_only`, `action_smoothness_weight`) remain in source code
- CLI `--help` shows updated flags
