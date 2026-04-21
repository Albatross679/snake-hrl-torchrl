---
name: Low reward signal-to-noise ratio in follow_target
description: The 10x improvement bonus in the reward function creates a reward SNR of 0.08, making value function learning difficult
type: issue
status: resolved
severity: medium
subtype: training
created: 2025-03-25
updated: 2025-03-25
tags: [ppo, reward, signal-noise, debugging]
aliases: []
---

## Problem

The `follow_target` reward function has a very low signal-to-noise ratio:

```python
reward = exp(-5.0 * dist) + 10.0 * (prev_dist - dist)
```

| Component | Range | Scale |
|-----------|-------|-------|
| `exp(-5 * dist)` | [0, 1] | Bounded, smooth |
| `10 * improvement` | unbounded | High variance, noisy |

Step-level statistics:
- **Mean reward**: 0.088
- **Std reward**: 1.14
- **SNR**: 0.08 (terrible — signal is 8% of noise)

The `10.0 * improvement` multiplier amplifies physics noise (small random movements) into large reward swings that dominate the useful distance signal.

## Impact

- Value function can't reliably predict returns
- Explained variance oscillates and goes negative
- 13.7% of batches had anti-correlated value predictions (EV < 0)

## Possible Fixes (not yet applied)

1. **Reduce improvement multiplier**: 10.0 → 2.0 (would improve SNR ~5x)
2. **Add reward normalization**: Running standardization of rewards
3. **Remove improvement bonus**: Rely solely on distance-based reward

Note: The reward function is from the paper's design (Choi & Tong, 2025). The paper used SAC which handles high variance better (replay buffer averages noise over many samples). PPO sees each sample once, making it more sensitive to reward noise.

## Resolution

Improvement bonus removed entirely. Reward simplified to `exp(-5.0 * dist)` only.

- SNR improved from 0.08 to 1.02 (13x improvement)
- Explained variance improved from 0.24 to 0.65 average (0.74 peak)
- EV negative % dropped from ~15% to 1.4%

See `experiments/ppo-clean-reward-diagnostic.md` and `logs/ppo-reward-simplification.md`.
