---
name: DD-PINN trained in relative state space — physics residual mismatch
description: DD-PINN was training in 130D relative state space but physics residual operates on 124D raw state, causing lossy conversion and incorrect gradient flow
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-18
updated: 2026-03-18
tags: [pinn, physics-residual, state-space, dd-pinn]
aliases: [relative-state-pinn-bug]
---

## Affected Run

- **W&B run**: `o6ezkt8p` — [ddpinn on snake-hrl-pinn](https://wandb.ai/qifan_wen-ohio-state-university/snake-hrl-pinn/runs/o6ezkt8p)
- **Run dir**: `output/surrogate/ddpinn/ddpinn_20260318_133908`
- **Config**: `n_basis=5, lambda_phys=1.0, use_relative=True (implicit default)`

This run was launched before the fix and trains in 130D relative space. Results from this run should be treated as **baseline for comparison** against the corrected raw-state run.

## Symptom

The DD-PINN model was training on 130D relative state representations, but the Cosserat rod physics residual (`CosseratRHS`) expects and returns 124D raw state vectors (absolute node positions, velocities, yaw, angular velocity).

At the physics loss computation (train_pinn.py, formerly lines 620–621):

```python
g_dot_denorm = normalizer.denormalize_delta(g_dot.reshape(-1, 130)).reshape(...)
g_dot_raw = g_dot_denorm.reshape(-1, 130)[:, :RAW_STATE_DIM]  # truncate 130 → 124
```

This truncation drops the 6 extra relative-space dimensions (CoM position, heading sin/cos, etc.) and implicitly assumes the first 124 dimensions of the relative state derivative equal the raw state derivative — which is **mathematically incorrect** since the relative-to-raw transform is nonlinear.

## Root Cause

The relative state (130D) includes center-of-mass position, heading-frame-relative node positions, and heading angle decomposed into sin/cos. The transform `raw_to_relative` involves:
- Subtracting CoM from node positions
- Rotating into the heading frame
- Decomposing heading into sin/cos components

These are nonlinear transforms, so derivatives in relative space are **not** the same as derivatives in raw space. The Jacobian of the transform was not applied, making the physics residual comparison invalid.

## Fix Applied

Added a `use_relative: bool = False` config flag (defaulting to raw-state training). When `use_relative=False`:

1. **Data loading**: keeps the original 124D raw state vectors (no `raw_to_relative` conversion)
2. **Model**: built with `state_dim=124` — ansatz outputs 124D deltas directly
3. **Physics loss**: no conversion needed — model output and physics RHS both operate in 124D raw space
4. **Evaluation**: no `relative_to_raw` conversion needed for R2/RMSE metrics

The old relative-state mode is preserved via `--use-relative` CLI flag for backward compatibility.

## Files Modified

- `src/pinn/train_pinn.py`:
  - Added `use_relative: bool = False` to `DDPINNTrainConfig`
  - Added `--use-relative` CLI argument
  - Made `_load_data_and_normalizer()` conditional on `use_relative`
  - Made physics loss computation conditional (no conversion in raw mode)
  - Made evaluation metrics conditional
  - Updated model construction to use dynamic `state_dim`
  - Updated saved config to record actual `state_dim`

## Impact

- **Model size**: slightly smaller (124D vs 130D → `param_dim = 4 * 124 * 5 = 2480` vs 2600)
- **Physics loss**: now exact — no lossy dimension truncation or nonlinear transform ignored
- **Gradient flow**: physics gradients flow correctly through the raw state space
- **Expected result**: better physics-informed regularization, potentially better generalization
