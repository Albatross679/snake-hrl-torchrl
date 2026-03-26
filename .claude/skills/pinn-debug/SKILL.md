---
name: pinn-debug
description: >
  Systematic PINN training debugger. Four-phase diagnostic process for identifying
  and fixing physics-informed neural network training failures. Use when (1) loss not
  decreasing or stuck, (2) physics loss dominating or data loss dominating,
  (3) gradient pathology between loss components, (4) NaN in training,
  (5) user says "debug PINN", "loss stuck", "why isn't it learning",
  (6) validate PINN pipeline before real training, (7) run probe PDEs,
  (8) user mentions "loss ratio", "NTK eigenvalues", "ReLoBRaLo", "spectral bias",
  "Fourier features", "collocation", "residual", or any PINN diagnostic metric,
  (9) pre-flight check before PINN training, (10) physics residual debugging,
  per-component violation analysis, nondimensionalization quality,
  (11) convergence rate mismatch, causality violation, collocation starvation,
  (12) ansatz numerical issues, DD-PINN-specific debugging.
  Works with DD-PINN (train_pinn.py) and physics-regularized surrogates (train_regularized.py).
---

# PINN Training Debugger

Systematic 4-phase diagnostic process for physics-informed neural network training failures. Do not tune hyperparameters until bugs are ruled out. Do not look at total loss alone -- total loss hides whether physics or data loss is decreasing.

## Phase 1: Probe PDE Validation

Run BEFORE any real training. Catches implementation bugs in seconds using simple PDEs with known analytical solutions.

**Probe PDEs location:** `src/pinn/probe_pdes.py`

| Probe | PDE | Tests | Pass Criterion | Failure Means |
|-------|-----|-------|-----------------|---------------|
| probe1_heat_1d | 1D heat: u_t = alpha * u_xx | Data fitting + optimizer | MSE < 1e-4 in ~500 epochs | Broken data loss or backprop |
| probe2_advection_1d | 1D advection: u_t + c * u_x = 0 | BC/IC enforcement + PDE residual | Residual < 1e-3, BC error < 1e-4 | Broken BC enforcement or residual computation |
| probe3_burgers_1d | 1D Burgers: u_t + u * u_x = nu * u_xx | Nonlinear PDE + loss balancing | Both losses decrease, ratio stays < 100:1 | Broken ReLoBRaLo or loss balancing |
| probe4_reaction_diffusion_1d | 1D reaction-diffusion: u_t = D * u_xx + k * u * (1-u) | Multi-scale + Fourier features | Captures both low and high-freq modes | Broken Fourier embedding or spectral bias |

If probe N fails but N-1 passes, the bug is in exactly the component N adds.

Run probes:
```python
from src.pinn.probe_pdes import run_probe_validation
run_probe_validation()
```

Run PDE system analysis (checks your actual PDE for nondimensionalization quality and stiffness):
```python
from src.pinn.probe_pdes import analyze_pde_system
print(analyze_pde_system())
```

Run tests:
```bash
python3 -m pytest tests/test_pinn_probes.py -x -q
```

## Phase 2: Dashboard Diagnostic Metrics

Start real training. Check these W&B metrics in priority order.

**Diagnostics module:** `src/pinn/diagnostics.py`

### Priority 1: Loss component balance
- `diagnostics/loss_ratio`: ratio of physics loss to data loss -- should stay in [0.1, 10]
  - Ratio > 100 -> Physics loss dominating. ReLoBRaLo not working or curriculum warmup too short. See Phase 3 branch 1.
  - Ratio < 0.01 -> Data overfitting, physics being ignored. See Phase 3 branch 2.
  - Oscillating wildly -> ReLoBRaLo weight instability. Increase alpha (more smoothing).

### Priority 2: Gradient health
- `diagnostics/grad_norm_data`: gradient norm from data loss alone
- `diagnostics/grad_norm_phys`: gradient norm from physics loss alone
- `diagnostics/grad_norm_ratio`: ratio of physics to data gradient norms -- should stay in [0.1, 10]
  - Ratio > 100 -> Gradient pathology. Physics gradients dominating parameter updates (NTK eigenvalue mismatch). Use per-loss gradient clipping.
  - Ratio ~= 0 -> Physics loss not producing gradients. Check requires_grad on collocation points, look for detach() bugs.

### Priority 3: Residual distribution
- `diagnostics/residual_mean`: mean absolute residual across collocation points
- `diagnostics/residual_max`: maximum absolute residual
- `diagnostics/residual_std`: standard deviation of residuals
- `diagnostics/residual_p95`: 95th percentile of absolute residuals
- Check: `diagnostics/residual_max` / `diagnostics/residual_mean` should be < 100
  - Ratio > 100 -> Residual concentrated in specific regions. Enable RAR or increase rar_fraction.
  - residual_mean not decreasing -> Physics loss stagnating. Check collocation coverage and curriculum warmup status.

### Priority 4: ReLoBRaLo weight health
- `diagnostics/relobralo_w_data`: current weight for data loss
- `diagnostics/relobralo_w_phys`: current weight for physics loss
- `diagnostics/relobralo_ratio`: ratio of physics weight to data weight -- should stay in [0.01, 100]
  - Ratio oscillating between extremes (0.001 to 1000) -> Temperature too low or alpha too small. Increase alpha from 0.999 to 0.9999, increase temperature from 1.0 to 2.0.
  - Ratio stuck at extreme -> ReLoBRaLo not adapting. Check that both losses are non-zero and finite.

### Priority 5: NTK eigenvalue spectrum
- `ntk/eigenvalue_max`: largest eigenvalue of the Neural Tangent Kernel
- `ntk/eigenvalue_min`: smallest eigenvalue
- `ntk/condition_number`: ratio of max to min eigenvalue -- should be < 1e6
- `ntk/spectral_decay_rate`: ratio of max eigenvalue to median eigenvalue
  - Condition number > 1e6 -> Severe convergence rate mismatch between loss components. NTK-based reweighting or per-loss gradient clipping needed.
  - Condition number growing over training -> Optimization becoming increasingly ill-conditioned.

### Priority 6: Per-component violations
- `diagnostics/violation_pos_x`, `diagnostics/violation_pos_y`: position RMSE per node
- `diagnostics/violation_vel_x`, `diagnostics/violation_vel_y`: velocity RMSE per node
- `diagnostics/violation_yaw`, `diagnostics/violation_omega_z`: angular RMSE per node
  - One component much worse than others -> Focus debugging on that component's PDE terms.
  - All components bad -> Systemic issue (nondimensionalization, network capacity, or loss balancing).

## Phase 3: Loss Not Decreasing -- Decision Tree

Follow this tree top-to-bottom. Stop at the first match.

```
loss_phys >> loss_data (ratio > 100)?
  -> Physics loss dominating. ReLoBRaLo not working or warmup too short.
     Check: Is ReLoBRaLo enabled? Is curriculum_warmup > 0?
     Fix: Increase curriculum_warmup from 0.15 to 0.3, verify ReLoBRaLo alpha,
     try fixed lambda_phys=0.01 as baseline test.

loss_data >> loss_phys (ratio > 100)?
  -> Data overfitting, physics being ignored.
     Check: Is lambda_phys > 0? Is n_collocation sufficient (>= 5000)?
     Fix: Increase lambda_phys, increase n_collocation, check collocation
     point coverage spans the full domain.

grad_norm_phys >> grad_norm_data (ratio > 100)?
  -> Gradient pathology: physics gradients dominating parameter updates.
     This is the NTK eigenvalue mismatch (Wang et al. NeurIPS 2021).
     Fix: Use per-loss gradient clipping. Clip physics gradient norm to
     match data gradient norm magnitude.

grad_norm_phys ~= 0?
  -> Physics loss not producing gradients.
     Check: CosseratRHS computation (is output differentiable?).
     Check: Collocation points have requires_grad=True.
     Check: No accidental .detach() or .item() in residual computation.
     Check: Physics residual is not returning constant zero.

residual_max >> residual_mean (ratio > 100)?
  -> Residual concentrated in specific regions. RAR should help.
     Check: Is use_rar=True?
     Fix: If RAR already enabled, increase rar_fraction from 0.1 to 0.2,
     decrease rar_interval from 20 to 10.

val_loss decreasing but loss_phys flat?
  -> Data fitting working but physics not being enforced.
     Check: Is curriculum warmup still active? Compare current epoch
     vs warmup_end = n_epochs * curriculum_warmup.
     Fix: If warmup ended and loss_phys still flat, physics residual
     computation may be broken. Run probe PDEs to isolate.

loss_data has sudden jumps?
  -> Learning rate too high, or ReLoBRaLo weight oscillation.
     Check: diagnostics/relobralo_ratio -- is it oscillating?
     Fix: Reduce lr. Increase ReLoBRaLo alpha from 0.999 to 0.9999
     (more exponential smoothing).

All metrics healthy but accuracy poor?
  -> Spectral bias. Network cannot represent high-frequency solution components.
     Check: Fourier feature sigma and n_fourier.
     Fix: If n_fourier=0, enable Fourier features. If already enabled,
     increase fourier_sigma (10 -> 30), increase n_fourier (128 -> 256).
     Follow Phase 4 sub-tree for detailed physics accuracy analysis.

Everything NaN?
  -> Division by zero in physics residual (dl=0 in Cosserat bending,
     norm=0 in tangent computation), log(0) in loss computation,
     or inf in state normalization.
     Check: NondimScales -- are reference scales nonzero?
     Check: Normalizer fit data -- was normalizer fitted on valid data?
     Check: Ansatz parameters -- is delta constrained positive (via softplus)?
```

## Phase 4: Physics Accuracy Poor -- Sub-Tree

When all training diagnostics are healthy but physical accuracy is poor. Follow tiers in order -- stop at the first match. Diagnose with analysis before tuning.

### Tier 1: PDE Residual Analysis (cheapest to check)

```
1.1 Per-equation residual decomposition.
    Log residuals separately for kinematic, bending, friction terms.
    One component > 90% of total residual?
      -> That physics term is hardest to learn. Weight it higher.

1.2 Residual spatial distribution.
    Plot |residual| vs collocation point position/time.
    Concentrated at boundaries?
      -> BC enforcement weak. Increase boundary collocation points.
    Concentrated at t=0?
      -> IC satisfaction failing. Check ansatz g(a,0)=0 property.
    Concentrated at late times?
      -> Causality violation. Increase curriculum_warmup or add causal weighting.

1.3 Collocation point coverage.
    Plot collocation points vs residual magnitude.
    High-residual regions under-sampled?
      -> RAR should fix this. If already enabled, increase rar_fraction.

1.4 Per-component error analysis.
    RMSE per state component (pos_x, vel_x, yaw, omega_z).
    Which component has highest error in physical units?
      -> Focus debugging on that component's PDE terms.
```

### Tier 2: Network Architecture (check second)

```
2.1 Fourier feature analysis.
    Compute FFT of predicted solution along spatial dimension.
    Missing high-frequency modes?
      -> Increase fourier_sigma or n_fourier.
    High-frequency noise but low-frequency wrong?
      -> Decrease fourier_sigma (spectral bias overcorrected).

2.2 Ansatz verification.
    Evaluate ansatz at t=0 for 1000 random inputs.
    Any |g(a,0)| > 1e-6?
      -> Ansatz IC satisfaction broken. Check DampedSinusoidalAnsatz.
    Evaluate dg/dt at t=0. Reasonable magnitudes?
      -> If too large, ansatz basis functions too aggressive.
         Clamp delta > 0 via softplus, clamp beta to [-100, 100].

2.3 Network capacity.
    Train on 10% of data. Achieves low training loss?
      Yes -> Network has capacity, training data or physics is the issue.
      No -> Network too small. Increase hidden_dim or n_layers.
```

### Tier 3: Physics Model Fidelity (hardest to fix)

```
3.1 CosseratRHS verification.
    Run CosseratRHS on known analytical states.
    Compare output to PyElastica reference.
    Discrepancy > 10%?
      -> CosseratRHS parameters wrong. Check elastic modulus, friction coefficients.

3.2 Nondimensionalization check.
    Run: analyze_pde_system() from src/pinn/probe_pdes.py.
    Are all terms in physics residual O(1)?
    Any term >> 1 or << 1?
      -> NondimScales needs adjustment. Adjust L_ref, t_ref, F_ref until
         all per_term_magnitudes are within [0.1, 10].

3.3 Stiff PDE detection.
    Compute condition number of Jacobian df/dx.
    Condition number > 1e6?
      -> Problem is stiff. Consider implicit time integration or
         sequence-to-sequence training with shorter time windows.
```

## Quick Symptom Lookup

| Symptom | Start At |
|---------|----------|
| Loss flat from epoch 0 | Phase 1 (probe PDEs) + Phase 3 (grad_norm_phys ~= 0?) |
| Physics loss drops, data loss stuck | Phase 3 (loss_phys >> loss_data) |
| Data loss drops, physics loss stuck | Phase 3 (loss_data >> loss_phys) |
| Loss oscillates wildly | Phase 3 (loss_data has sudden jumps?) |
| High loss but smooth convergence | Phase 4 (spectral bias) -- Tier 2 Fourier analysis |
| NaN after N epochs | Phase 3 (Everything NaN?) |
| Good loss but bad predictions | Phase 4 Tier 1 (per-component residual analysis) |
| Validation loss increasing | Overfitting -- reduce network capacity or increase regularization |

## Key Principle

Do NOT tune hyperparameters until implementation bugs are ruled out via probe PDEs.
Do NOT look at total loss alone. Total loss hides whether physics or data loss is decreasing.
Accuracy is a lagging indicator -- for PINNs: by the time total loss looks bad, a specific component (loss balance, gradients, residuals) already broke upstream.

## Failure Mode Reference

For detailed failure mode signatures with root causes, diagnostic metrics, and literature-backed remediation, see [failure-modes.md](references/failure-modes.md).
