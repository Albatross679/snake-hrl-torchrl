---
name: MM-RKHS Adopt Notebook Mechanics
description: Added 5 opt-in notebook mechanics to MMRKHSTrainer for closing theory-practice gap
type: log
status: complete
subtype: feature
created: 2026-03-27
updated: 2026-03-27
tags: [mmrkhs, adaptive-schedules, kernel-correction, inner-mm-loop]
aliases: []
---

# MM-RKHS Adopt Notebook Mechanics

Added 5 opt-in mechanics from the reference MMRKHS.ipynb notebook to the neural MMRKHSTrainer:

1. **Adaptive eta** (`eta_schedule=True`): `eta_effective = eta * (k+1)^eta_exponent` grows the mirror descent step size over training
2. **Adaptive beta** (`beta_schedule=True`): `beta_effective = max|A| / sqrt(k+1)` scales the MMD penalty by advantage magnitude
3. **Inner MM iterations** (`inner_mm_iterations=N`): Fixed-point iterations per mini-batch step (notebook uses 2-3)
4. **Configurable exponent clip** (`exponent_clip=2.0`): Tighter log-ratio clamping (default 20.0 preserves current behavior)
5. **Kernel correction** (`kernel_correction=True`): Additive `||old_loc - new_loc||^2 * |A|` correction term

All mechanics are opt-in with backward-compatible defaults. New wandb metrics: `train/eta_effective`, `train/beta_effective`, `train/kernel_correction`.

## Files Modified

- `src/configs/training.py`: 7 new fields on MMRKHSConfig
- `src/trainers/mmrkhs.py`: _update() rewritten with gated mechanics, _global_batch_idx counter, updated _log_metrics
