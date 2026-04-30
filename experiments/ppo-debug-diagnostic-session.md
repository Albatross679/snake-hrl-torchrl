---
name: PPO debug diagnostic session
description: Systematic 4-phase RL debug of PPO follow_target training — probe validation, obs normalization, entropy fix
type: experiment
status: complete
created: 2025-03-25
updated: 2025-03-25
tags: [ppo, debugging, diagnostics, observation-normalization, entropy]
aliases: []
---

## Objective

Diagnose why PPO training on `follow_target` task shows reward fluctuating 5–27 with no growth over 5M+ frames.

## Method

Followed systematic 4-phase RL debug process (Andy Jones methodology).

## Phase 1: Probe Environment Validation

All 5 probe environments PASS — trainer implementation is correct.

| Probe | Result | V(s)/Action |
|-------|--------|-------------|
| ProbeEnv1 (constant value) | PASS | V(0) = 1.0000 |
| ProbeEnv4 (policy gradient) | PASS | action_mean = 1.0615 |
| ProbeEnv5 (joint policy-value) | PASS | +1→+2.18, -1→-1.83 |

## Phase 2-3: Three Training Runs Compared

### Run 1: Baseline (no fixes)
- **Config**: entropy_coef=0.01, no obs norm
- **Frames**: ~900K
- **Result**: Rewards flat at ~15, EV declining (0.32→0.13), negative EV 13.7%
- **Diagnostics not logged** (code wasn't committed)

### Run 2: + Observation Normalization
- **Config**: entropy_coef=0.01, ObservationNorm added
- **Frames**: ~1.3M
- **Result**: Entropy collapsed 4.3→-0.6, rewards declined 20→17
- **Root cause**: Faster learning from normalized obs + weak entropy bonus = collapse

### Run 3: + Entropy Fix (entropy_coef=0.05)
- **Config**: entropy_coef=0.05, ObservationNorm
- **Frames**: ~1.8M
- **Result**: Entropy stabilized at 3.4 (no collapse), but rewards still flat at ~17

| Metric | Run 1 | Run 2 | Run 3 |
|--------|-------|-------|-------|
| Late EV | 0.13 | 0.12 | 0.17 |
| Late Entropy | ~5.0* | -0.59 | 3.44 |
| Reward Trend | +2.5 | -3.5 | -1.9 |
| EV Negative % | 13.7% | 17.1% | ~20% |

*Run 1 entropy was loss_entropy not entropy_proxy (different scale)

## Findings

### Bugs Fixed
1. **Diagnostic blind spot**: diagnostics.py uncommitted + averaging bug in _update()
2. **env.close() API**: missing **kwargs for TorchRL compat
3. **Observation scaling**: 148 dims with scales 0–60, 12 dead dims, 12 extreme dims

### Issues Identified
1. **Entropy collapse** (entropy_coef=0.01 too weak) → Fixed with 0.05
2. **Reward SNR = 0.06**: The `10.0 * improvement` bonus creates massive noise

### Phase 3 Decision Tree Path
```
entropy collapsed? → YES (Run 2) → Increase entropy_coef ✓
explained_variance < 0? → YES (22% of batches) → Check reward scale
All diagnostics healthy? → YES (Run 3) → Problem is reward function, not trainer
```

## Remaining Blocker

**Step reward SNR = 0.06** (mean=0.09, std=1.48). The value function cannot learn to predict returns when the signal is 6% of the noise. The `10.0 * improvement_bonus` in `compute_follow_target_reward()` is the primary noise source.

### Recommended Next Steps

1. Reduce improvement multiplier from 10.0 to 2.0 (5x SNR improvement)
2. OR add running reward normalization
3. OR remove improvement bonus entirely (rely on distance reward alone)

Note: The paper's reward was designed for SAC (replay buffer averages noise); PPO sees each sample once and is more sensitive to reward variance.
