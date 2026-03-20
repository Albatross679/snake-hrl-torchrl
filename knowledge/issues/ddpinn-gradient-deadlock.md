---
name: DD-PINN Gradient Deadlock from Zero Output Layer Init
description: DD-PINN model fails to learn because zero-initialized output layer causes zero ansatz parameters, leading to zero gradients throughout the network
type: issue
status: resolved
severity: critical
subtype: training
created: 2026-03-18
updated: 2026-03-18
tags: [ddpinn, ansatz, initialization, gradient-deadlock]
aliases: [ddpinn-zero-init-bug]
---

## Symptom

DD-PINN training ran for 10 epochs with absolutely no learning:
- `val_loss = 0.986203` identical across all 10 epochs
- `data_loss ~ 1.0` (model predicting the mean of normalized data)
- Physics weight = 0.0 (curriculum warmup), but data loss also not decreasing

The loss of 0.986 is exactly the variance of the normalized data, confirming the model predicts zero (the mean) and never escapes.

## Root Cause

**Gradient deadlock** caused by zero-initialization of the output layer (`nn.init.zeros_`).

The ansatz function is:
```
g_i(t) = alpha * exp(-delta * t) * [sin(beta * t + gamma) - sin(gamma)]
```

When the output layer is zero-initialized:
1. All ansatz parameters (alpha, beta, gamma, delta_raw) = 0
2. alpha = 0 makes the entire ansatz output = 0
3. Gradients w.r.t. all ansatz parameters are also exactly 0 (product rule: alpha * ... → 0 * ...)
4. Zero gradients propagate back through the chain rule, zeroing all upstream gradients
5. The optimizer step does nothing — the model is stuck permanently

The IC constraint `g(a, 0) = 0` is satisfied by the ansatz structure (`sin(gamma) - sin(gamma) = 0`) for ANY parameter values, so zero-init was never needed for correctness.

## Fix Applied

Changed output layer initialization from `nn.init.zeros_` to `nn.init.normal_(std=0.01)`:

```python
# Before:
nn.init.zeros_(self.output.weight)

# After:
nn.init.normal_(self.output.weight, std=0.01)
```

Small random init breaks the gradient deadlock while keeping initial predictions near-zero. Bias remains zero-initialized (fine since it doesn't cause the deadlock).

## Files Modified

- `src/pinn/models.py` — Line 100-103: Changed output layer weight initialization
