# PINN Failure Mode Reference

Detailed diagnostic signatures for each PINN failure mode, with root causes, diagnostic metrics, and literature-backed remediation steps.

## 1. Convergence Rate Mismatch

**Category:** Optimization

**Signature:**
- `diagnostics/loss_ratio` > 10 (physics loss much larger than data loss)
- `diagnostics/grad_norm_ratio` > 10 (physics gradients dominating)
- `ntk/condition_number` > 1e4
- loss_phys drops fast while loss_data stagnates or increases

**Root Cause:**
The PDE operator amplifies network output through differentiation, producing large gradients that dominate BC/IC fitting gradients. The Neural Tangent Kernel has eigenvalues spanning many orders of magnitude, causing different loss components to converge at wildly different rates (Wang et al. NeurIPS 2021).

**Diagnostic Metrics:**
- `diagnostics/loss_ratio` -- primary indicator
- `diagnostics/grad_norm_ratio` -- confirms gradient-level mismatch
- `ntk/condition_number` -- quantifies the severity of convergence rate disparity

**Remediation:**
1. Verify ReLoBRaLo is enabled in config. If disabled, enable it.
2. Increase `curriculum_warmup` from 0.15 to 0.3 (trains on data-only first, gradually introduces physics).
3. Apply per-loss gradient clipping: clip physics gradient norm to match data gradient norm magnitude.
4. If ReLoBRaLo is enabled but ratio still > 100, increase `alpha` from 0.999 to 0.9999 for smoother weight updates.
5. As a diagnostic baseline test, try fixed `lambda_phys=0.01` to confirm the issue is loss balancing.

**Code Example:**
```python
# Check in W&B dashboard:
# diagnostics/grad_norm_ratio -- if > 100, gradient pathology confirmed
# diagnostics/loss_ratio -- if > 100, loss imbalance confirmed

# Quick fix: manual gradient clipping per loss term
grad_data = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Then separately compute and clip physics gradients
```

---

## 2. Spectral Bias

**Category:** Architecture

**Signature:**
- Smooth predictions that look qualitatively correct but have poor quantitative accuracy
- High error near boundaries, sharp gradients, or high-curvature regions
- `ntk/spectral_decay_rate` is high (eigenvalue spectrum drops off steeply)
- FFT of predicted solution vs true solution shows missing high-frequency components

**Root Cause:**
MLPs have a frequency response biased toward low frequencies. The network learns smooth, low-frequency components of the solution first, and may never capture high-frequency features without architectural intervention (Rahaman et al. ICML 2019). PDE solutions with boundary layers, shock waves, or multi-scale physics require high-frequency representation that standard MLPs cannot efficiently provide.

**Diagnostic Metrics:**
- `ntk/spectral_decay_rate` -- steep decay indicates network biased toward low frequencies
- Per-component RMSE -- high error in components with sharp spatial variation
- Visual: FFT of prediction vs FFT of reference solution

**Remediation:**
1. Verify Fourier feature embeddings are enabled (`n_fourier > 0` in config).
2. If enabled but accuracy still poor, increase `fourier_sigma` from 10 to 30 (shifts frequency response to higher frequencies).
3. Increase `n_fourier` from 128 to 256 (more frequency components in the embedding).
4. If high-frequency noise appears (overcorrection), decrease `fourier_sigma`.
5. For extreme multi-scale problems, consider multi-scale Fourier features with logarithmically spaced frequencies.

**Code Example:**
```python
# Diagnostic: compare frequency content
import torch.fft
pred_fft = torch.fft.rfft(predicted_solution, dim=-1)
true_fft = torch.fft.rfft(true_solution, dim=-1)
# Plot |pred_fft| vs |true_fft| -- missing peaks indicate spectral bias
```

---

## 3. Causality Violation

**Category:** Optimization

**Signature:**
- Residual high at t=0, low at later times
- IC error high despite overall loss decreasing
- `diagnostics/residual_temporal_uniformity` >> 10 (residual not uniformly distributed in time)
- val_loss good overall but initial condition satisfaction poor

**Root Cause:**
Standard PINN loss equally weights all collocation points in time. Without causal ordering, the optimizer can reduce total loss by fitting late-time points (which may have simpler structure) at the expense of early-time accuracy. This propagates errors forward in time since the solution at t depends on the solution at earlier times (Wang et al. CMAME 2024).

**Diagnostic Metrics:**
- `diagnostics/residual_temporal_uniformity` -- ratio of max to min temporal bin residual
- Per-time-bin residual plots -- should be roughly uniform, not concentrated at t=0
- IC error vs overall loss -- IC error should decrease at least as fast as total loss

**Remediation:**
1. Verify `curriculum_warmup > 0` in config (trains data-only first, gradually introduces physics collocation points starting from t=0).
2. Increase `curriculum_warmup` from 0.15 to 0.3 if causality violation persists.
3. Future: implement causal weighting (Wang et al. CMAME 2024) -- weight each temporal collocation point by exp(-epsilon * cumulative_residual_before_t).
4. As a workaround, train on shorter time windows first, then extend.

**Code Example:**
```python
# Diagnostic: bin residuals by time
t_bins = torch.linspace(0, t_max, 11)
for i in range(len(t_bins) - 1):
    mask = (t_colloc >= t_bins[i]) & (t_colloc < t_bins[i+1])
    bin_residual = residuals[mask].abs().mean()
    print(f"t=[{t_bins[i]:.2f}, {t_bins[i+1]:.2f}]: mean |residual| = {bin_residual:.4e}")
# If early bins >> late bins, causality violation confirmed
```

---

## 4. Collocation Point Starvation

**Category:** Optimization

**Signature:**
- Loss decreasing but per-component RMSE flat for specific components
- `diagnostics/residual_max` / `diagnostics/residual_mean` > 100
- Residual heatmap shows high-error regions with few collocation points

**Root Cause:**
Uniform or Sobol sampling does not account for solution structure. High-gradient regions (boundary layers, shock locations, sharp curvature) need more collocation points to provide adequate gradient signal, but uniform sampling under-represents these regions. The network has no gradient signal to learn the solution in under-sampled areas.

**Diagnostic Metrics:**
- `diagnostics/residual_max` / `diagnostics/residual_mean` ratio -- > 100 indicates concentration
- Per-component RMSE trends -- specific components staying flat while total loss decreases
- Collocation density vs residual magnitude scatter plot

**Remediation:**
1. Verify `use_rar=True` in config (Residual-based Adaptive Refinement).
2. If RAR already enabled, increase `rar_fraction` from 0.1 to 0.2 (more adaptive points per refinement step).
3. Decrease `rar_interval` from 20 to 10 (refine more frequently).
4. For persistent starvation, check that the adaptive sampling covers the full spatial domain (not just temporal).
5. Verify collocation points span the boundary regions where the solution has sharp gradients.

**Code Example:**
```python
# Diagnostic: compare residual vs collocation density
residual_mag = residuals.abs()
# If max(residual_mag) / mean(residual_mag) > 100, starvation likely
ratio = residual_mag.max() / residual_mag.mean()
print(f"Residual concentration ratio: {ratio:.1f}")
# If > 100, check where max residual occurs and verify collocation density there
```

---

## 5. Nondimensionalization Errors

**Category:** Physics

**Signature:**
- One residual component magnitude ~1e6 while another ~1e-3
- `diagnostics/relobralo_ratio` oscillating wildly (loss balancer fighting magnitude mismatch)
- `analyze_pde_system()` reports `nondim_quality="poor"`
- Per-term magnitudes in physics residual span many orders of magnitude

**Root Cause:**
Mixing SI units without proper scaling causes orders-of-magnitude differences in residual terms. Positions may be in meters (~1.0), angular velocities in rad/s (~10), and forces in Newtons (~1e-3 for thin rods). The loss balancer cannot compensate for 6+ orders of magnitude difference between residual components.

**Diagnostic Metrics:**
- Per-term residual magnitudes from `analyze_pde_system()`
- `diagnostics/relobralo_ratio` stability (wild oscillation indicates magnitude mismatch)
- Raw residual component values (before any weighting)

**Remediation:**
1. Run `analyze_pde_system()` from `src/pinn/probe_pdes.py` to get per-term magnitude analysis.
2. Check `per_term_magnitudes` output -- all terms should be O(1) after nondimensionalization.
3. Adjust `NondimScales` (`L_ref`, `t_ref`, `F_ref`) in `src/pinn/nondim.py` until all terms are within [0.1, 10].
4. Current project defaults: L_ref=1.0m, t_ref=0.5s, F_ref=E*I/L^2 (physics-based scaling).
5. After adjusting scales, re-run probe PDEs to verify nothing else broke.

**Code Example:**
```python
# Diagnostic: check nondimensionalization quality
from src.pinn.probe_pdes import analyze_pde_system
result = analyze_pde_system()
print(result)
# Look for: nondim_quality, per_term_magnitudes
# All terms should be O(1). If any term is > 100 or < 0.01, scales need adjustment.
```

---

## 6. Ansatz Numerical Issues

**Category:** DD-PINN-specific

**Signature:**
- NaN appearing after N epochs (not immediately)
- Extreme parameter values in W&B model histograms
- Loss suddenly jumps to inf then becomes NaN
- Ansatz evaluation at t=0 does not exactly equal zero

**Root Cause:**
The DD-PINN ansatz `x(a,t) = x0(a) + t * f(a) + g(a,t)` uses basis functions of the form `exp(-delta*t) * sin(beta*t + gamma)`. Numerical overflow occurs when delta is negative (due to softplus or clamp misconfiguration), causing exponential growth, or when beta is very large, causing high-frequency oscillations that amplify through differentiation.

**Diagnostic Metrics:**
- Ansatz parameter histograms in W&B (delta, beta, gamma distributions)
- `max |g(a,0)|` -- should be < 1e-6 (exact IC satisfaction)
- Loss trajectory -- sudden jump to inf followed by NaN is characteristic

**Remediation:**
1. Verify `g(a,0) = 0` by evaluating ansatz at t=0 for 1000 random inputs. If `max |g(a,0)| > 1e-6`, the IC satisfaction property is broken.
2. Constrain delta > 0 via `softplus` (not clamp, which has zero gradient at boundary).
3. Clamp beta to `[-100, 100]` to prevent extreme oscillation frequencies.
4. Monitor ansatz parameter statistics: if `delta_min < 0` or `beta_max > 100`, intervention needed.
5. Check that the ansatz t=0 cancellation is exact algebraically (not relying on numerical cancellation).

**Code Example:**
```python
# Diagnostic: verify ansatz IC satisfaction
a_random = torch.randn(1000, input_dim)
t_zero = torch.zeros(1000, 1)
g_at_zero = model.ansatz(a_random, t_zero)
max_violation = g_at_zero.abs().max().item()
print(f"Max |g(a,0)|: {max_violation:.2e}")
assert max_violation < 1e-6, f"Ansatz IC violation: {max_violation:.2e}"
```

---

## 7. ReLoBRaLo Weight Oscillation

**Category:** Optimization

**Signature:**
- `diagnostics/relobralo_ratio` oscillates between 0.001 and 1000 across epochs
- `loss_data` has sudden jumps correlated with weight changes
- Training loss not monotonically decreasing despite healthy gradients

**Root Cause:**
ReLoBRaLo temperature too low or alpha (exponential smoothing) too small, causing insufficient smoothing of the loss weight updates. Each epoch, the weights swing dramatically to compensate for the previous epoch's imbalance, creating a feedback oscillation that prevents stable training.

**Diagnostic Metrics:**
- `diagnostics/relobralo_ratio` trajectory (should stabilize, not oscillate)
- `diagnostics/relobralo_w_data` and `diagnostics/relobralo_w_phys` individual trajectories
- Correlation between weight jumps and loss_data jumps

**Remediation:**
1. Increase `alpha` from 0.999 to 0.9999 (stronger exponential smoothing).
2. Increase `temperature` from 1.0 to 2.0 (softer softmax in weight computation).
3. If oscillation persists, try fixed weights as a diagnostic baseline (lambda_data=1.0, lambda_phys=0.01).
4. Verify that both loss components are non-zero and finite before ReLoBRaLo computes weights.

**Code Example:**
```python
# Diagnostic: check weight stability
# In W&B, plot diagnostics/relobralo_ratio over training
# Healthy: ratio stabilizes within [0.1, 10] after initial transient
# Unhealthy: ratio oscillates wildly throughout training

# Quick fix in config:
# relobralo_alpha: 0.9999  (was 0.999)
# relobralo_temperature: 2.0  (was 1.0)
```

---

## Literature Citations

1. **Wang, S., Teng, Y., and Perdikaris, P.** "Understanding and Mitigating Gradient Flow Pathologies in Physics-Informed Neural Networks." *SIAM Journal on Scientific Computing*, 43(5), 2021. (Also presented at NeurIPS 2021 as "When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective.") -- NTK eigenvalue analysis, convergence rate mismatch, learning rate annealing.

2. **Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R., and Mahoney, M.** "Characterizing Possible Failure Modes in Physics-Informed Neural Networks." *NeurIPS 2021.* -- Failure mode taxonomy, curriculum regularization, progressive complexity probes.

3. **Bischof, R. and Kraus, M.** "Multi-Objective Loss Balancing for Physics-Informed Deep Learning." *arXiv:2110.09813*, 2021. -- ReLoBRaLo algorithm: Random Lookback with softmax-based loss weight adaptation.

4. **Wang, S., Sankaran, S., and Perdikaris, P.** "Respecting Causality is All You Need for Training Physics-Informed Neural Networks." *Computer Methods in Applied Mechanics and Engineering*, 2024. -- Causal training with temporal weighting, exponential residual accumulation.

5. **Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y., and Courville, A.** "On the Spectral Bias of Neural Networks." *ICML 2019.* -- Frequency-dependent learning dynamics, low-frequency bias of MLPs.

6. **Tancik, M., Srinivasan, P., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J., and Ng, R.** "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains." *NeurIPS 2020.* -- Random Fourier feature embeddings for overcoming spectral bias.
