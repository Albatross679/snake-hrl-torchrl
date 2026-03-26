# Phase 17: Design PINN Debugging Workflow and Diagnostic Pipeline - Research

**Researched:** 2026-03-26
**Domain:** Physics-Informed Neural Network diagnostics, training failure detection, systematic debugging
**Confidence:** HIGH

## Summary

PINN debugging is fundamentally different from RL debugging because failure modes arise from the interplay between PDE structure and neural network optimization rather than from environment-agent interaction loops. The established literature (Wang et al. NeurIPS 2021, Krishnapriyan et al. NeurIPS 2021, Bischof & Kraus 2021) identifies three core failure categories: (1) loss component convergence rate mismatch where PDE residual loss dominates boundary/initial condition loss, (2) spectral bias where the network cannot represent high-frequency solution components, and (3) causality violation where the network fits interior points before properly resolving initial/boundary conditions. These failure modes are interconnected -- spectral bias exacerbates convergence mismatch, and convergence mismatch causes causality violations.

Our codebase already has a DD-PINN implementation (Phase 13) with ReLoBRaLo loss balancing, Residual-based Adaptive Refinement (RAR), Fourier feature embeddings, curriculum warmup, and L-BFGS refinement. The current training loop logs `loss_data`, `loss_phys`, `phys_weight`, and `val_loss` to W&B, but lacks systematic diagnostic metrics (loss component ratios, gradient norms per loss term, residual spatial distribution, per-component physics violation magnitudes), probe PDEs for validation, automated failure detection, and decision trees for systematic fault isolation.

The PINN debug skill should mirror the RL debug skill structure: (1) probe PDE validation before real training, (2) dashboard diagnostic metrics with priority ordering, (3) decision tree for "loss not decreasing", and (4) physics-specific sub-tree for residual analysis. The project already has the infrastructure (W&B logging, dataclass configs, diagnostics module pattern in `src/trainers/diagnostics.py`) to build this without new dependencies.

**Primary recommendation:** Build a `src/pinn/diagnostics.py` module and `src/pinn/probe_pdes.py` suite mirroring the RL debug skill pattern, using only existing PyTorch and W&B APIs. Create a Claude Code skill at `.claude/skills/pinn-debug/` with the same 4-phase structure as the RL debug skill.

## Standard Stack

### Core (Already in Project)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | >=2.6.0 | Autodiff for PDE residuals, gradient hooks, NTK computation | Native `torch.autograd.functional.jacobian` for NTK; `register_hook` for per-loss-term gradients |
| wandb | >=0.18.0 | Metric logging, alerts, loss component dashboards | Already used; `wandb.alert()` for automated failure detection |
| scipy | >=1.14.0 | Analytical PDE solutions for probe problems, Sobol sampling | Already used in `collocation.py`; `scipy.stats.qmc.Sobol` |
| matplotlib | >=3.9.0 | Residual heatmaps, loss landscape plots, collocation density | Already available for diagnostic visualization |

### Supporting (No New Dependencies)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `torch.linalg.eigh` | PyTorch built-in | NTK eigenvalue spectrum computation | Diagnosing convergence rate mismatch |
| `wandb.alert()` | W&B built-in | Automated PINN failure alerts | Loss ratio explosion, residual stagnation |
| `torch.nn.utils.clip_grad_norm_` | PyTorch built-in | Per-loss-term gradient norm reporting | Already used in train_pinn.py |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom probe PDEs | DeepXDE benchmark suite | DeepXDE is a full framework (adds dependency, different API); custom probes are minimal and test exactly our DD-PINN pipeline |
| Custom NTK computation | Neural Tangents (Google JAX) | JAX dependency, incompatible with PyTorch; finite-width NTK via Jacobian is sufficient for diagnostics |
| Custom loss landscape viz | `loss-landscapes` PyPI package | Adds dependency for a feature needed only in debugging; 20 lines of custom code suffices |

## Architecture Patterns

### Recommended Project Structure
```
src/
  pinn/
    diagnostics.py      # PINN diagnostic middleware (NEW)
    probe_pdes.py        # Probe PDE problems for validation (NEW)
    train_pinn.py        # Existing (add diagnostic hooks)
    loss_balancing.py    # Existing ReLoBRaLo
    physics_residual.py  # Existing CosseratRHS
    collocation.py       # Existing collocation sampling
    ...

.claude/
  skills/
    pinn-debug/
      SKILL.md           # Main skill file (NEW) -- mirrors rl-debug/SKILL.md
      references/
        failure-modes.md # Detailed failure mode signatures (NEW)
```

### Pattern 1: Probe PDE Validation (Phase 1 of Debug Workflow)
**What:** A suite of progressively complex PDEs with known analytical solutions that test individual PINN components before training on the full Cosserat rod system.
**When to use:** Before any real PINN training run, analogous to RL probe environments.

Each probe tests one additional capability:

| Probe | PDE | Tests | Pass Criterion | Failure Means |
|-------|-----|-------|-----------------|---------------|
| ProbePDE1 | 1D heat equation: u_t = alpha * u_xx | Data fitting + optimizer | MSE < 1e-4 in ~500 epochs | Broken data loss or backprop |
| ProbePDE2 | 1D advection: u_t + c * u_x = 0 | BC/IC enforcement + PDE residual | Residual < 1e-3, BC error < 1e-4 | Broken BC enforcement or residual computation |
| ProbePDE3 | 1D Burgers: u_t + u * u_x = nu * u_xx | Nonlinear PDE + loss balancing | Both losses decrease, ratio stays < 100:1 | Broken ReLoBRaLo or loss balancing |
| ProbePDE4 | 1D reaction-diffusion: u_t = D * u_xx + k * u * (1-u) | Multi-scale + Fourier features | Captures both low and high-freq modes | Broken Fourier embedding or spectral bias |
| ProbePDE5 | 2D kinematic coupling (simplified Cosserat): dx/dt = v, dv/dt = f(x) | Full DD-PINN ansatz + physics residual | Ansatz IC satisfaction exact, physics residual < 1e-2 | Broken ansatz or CosseratRHS |

```python
# Source: Adapted from Krishnapriyan et al. NeurIPS 2021 probe methodology
class ProbePDE1:
    """1D heat equation with known solution."""
    name = "heat_1d"

    @staticmethod
    def analytical_solution(x, t, alpha=0.01):
        return torch.exp(-alpha * torch.pi**2 * t) * torch.sin(torch.pi * x)

    @staticmethod
    def pde_residual(u, u_t, u_xx, alpha=0.01):
        return u_t - alpha * u_xx

    @staticmethod
    def initial_condition(x):
        return torch.sin(torch.pi * x)

    @staticmethod
    def boundary_conditions():
        return {"x=0": 0.0, "x=1": 0.0}

    @staticmethod
    def pass_criterion(mse_vs_analytical):
        return mse_vs_analytical < 1e-4
```

### Pattern 2: Dashboard Diagnostic Metrics (Phase 2 of Debug Workflow)
**What:** A `PINNDiagnostics` class that computes and logs PINN-specific diagnostic metrics to W&B, checked in priority order.
**When to use:** Every PINN training run.

```python
# Source: Adapted from src/trainers/diagnostics.py pattern + Wang et al. 2021 NTK analysis
class PINNDiagnostics:
    """Non-invasive diagnostic layer for PINN trainers."""

    def __init__(self, wandb_run, config=None):
        self.wandb_run = wandb_run
        self._history = {
            "loss_ratio": deque(maxlen=100),
            "residual_norms": deque(maxlen=100),
            "grad_norms_data": deque(maxlen=100),
            "grad_norms_phys": deque(maxlen=100),
        }

    def compute_loss_ratio(self, loss_data, loss_phys):
        """Ratio of physics loss to data loss -- should stay in [0.1, 10]."""
        ratio = loss_phys.item() / max(loss_data.item(), 1e-10)
        self._history["loss_ratio"].append(ratio)
        return ratio

    def compute_per_loss_gradients(self, model, loss_data, loss_phys):
        """Separate gradient norms for each loss term."""
        # Data loss gradients
        model.zero_grad()
        loss_data.backward(retain_graph=True)
        grad_data = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        norm_data = grad_data.norm().item()

        # Physics loss gradients
        model.zero_grad()
        loss_phys.backward(retain_graph=True)
        grad_phys = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        norm_phys = grad_phys.norm().item()

        return {
            "diagnostics/grad_norm_data": norm_data,
            "diagnostics/grad_norm_phys": norm_phys,
            "diagnostics/grad_norm_ratio": norm_phys / max(norm_data, 1e-10),
        }

    def compute_residual_statistics(self, residuals):
        """Per-component residual statistics for spatial analysis."""
        return {
            "diagnostics/residual_mean": residuals.abs().mean().item(),
            "diagnostics/residual_max": residuals.abs().max().item(),
            "diagnostics/residual_std": residuals.std().item(),
            "diagnostics/residual_p95": residuals.abs().quantile(0.95).item(),
        }

    def compute_relobralo_health(self, weights):
        """Track ReLoBRaLo weight evolution."""
        return {
            "diagnostics/relobralo_w_data": weights[0].item(),
            "diagnostics/relobralo_w_phys": weights[1].item(),
            "diagnostics/relobralo_ratio": weights[1].item() / max(weights[0].item(), 1e-10),
        }
```

### Pattern 3: Decision Tree for "Loss Not Decreasing" (Phase 3 of Debug Workflow)
**What:** A systematic fault isolation tree, checked top-to-bottom, stopping at first match.
**When to use:** When total PINN loss stagnates or diverges.

```
loss_phys >> loss_data (ratio > 100)?
  -> Physics loss dominating. ReLoBRaLo not working or warmup too short.
     Increase curriculum_warmup, verify ReLoBRaLo alpha, try fixed lambda_phys=0.01.

loss_data >> loss_phys (ratio > 100)?
  -> Data overfitting, physics ignored.
     Increase lambda_phys, increase n_collocation, check collocation point coverage.

grad_norm_phys >> grad_norm_data (ratio > 100)?
  -> Gradient pathology: physics gradients dominating parameter updates.
     This is the NTK eigenvalue mismatch. Use per-loss gradient clipping.

grad_norm_phys ~= 0?
  -> Physics loss not producing gradients. Check CosseratRHS computation,
     verify collocation points have requires_grad=True, check for detach() bugs.

residual_max >> residual_mean (ratio > 100)?
  -> Residual concentrated in specific regions. RAR should help.
     If RAR already enabled, increase rar_fraction or decrease rar_interval.

val_loss decreasing but loss_phys flat?
  -> Data fitting working but physics not being enforced.
     Curriculum warmup still active? Check epoch vs warmup_end.

loss_data has sudden jumps?
  -> Learning rate too high, or ReLoBRaLo weight oscillation.
     Reduce lr, increase ReLoBRaLo alpha (more smoothing).

All metrics healthy but accuracy poor?
  -> Spectral bias. Check Fourier feature sigma, increase n_fourier.
     If already using Fourier features, try higher sigma (10 -> 30).
     Follow Phase 4 sub-tree.

Everything NaN?
  -> Division by zero in physics residual (dl=0, norm=0),
     log(0) in loss computation, inf in state normalization.
     Check NondimScales, check normalizer fit data.
```

### Pattern 4: Physics-Specific Sub-Tree (Phase 4 of Debug Workflow)
**What:** When all training diagnostics are healthy but physical accuracy is poor. Three tiers matching the RL debug skill structure.
**When to use:** After Phase 3 rules out optimizer/loss issues.

```
Tier 1: PDE Residual Analysis (cheapest to check)

1.1 Per-equation residual decomposition.
    Log residuals separately for kinematic, bending, friction terms.
    One component >90% of total residual?
      -> That physics term is hardest to learn. Weight it higher.

1.2 Residual spatial distribution.
    Plot |residual| vs collocation point position/time.
    Concentrated at boundaries? -> BC enforcement weak.
    Concentrated at t=0? -> IC satisfaction failing (check ansatz).
    Concentrated at late times? -> Causality violation (use causal weighting).

1.3 Collocation point coverage.
    Plot collocation points vs residual magnitude.
    High-residual regions under-sampled?
      -> RAR should fix this. If already enabled, increase rar_fraction.

1.4 Per-component error analysis.
    RMSE per state component (pos_x, vel_x, yaw, omega).
    Which component has highest error in physical units?
      -> Focus debugging on that component's PDE terms.

Tier 2: Network Architecture (check second)

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

2.3 Network capacity.
    Train on 10% of data. Achieves low training loss?
      Yes -> Network has capacity, training data is the issue.
      No -> Network too small. Increase hidden_dim or n_layers.

Tier 3: Physics Model Fidelity (hardest to fix)

3.1 CosseratRHS verification.
    Run CosseratRHS on known analytical states.
    Compare output to PyElastica reference.
    Discrepancy > 10%? -> CosseratRHS parameters wrong.

3.2 Nondimensionalization check.
    Are all terms in physics residual O(1)?
    Any term >> 1 or << 1? -> NondimScales needs adjustment.

3.3 Stiff PDE detection.
    Compute condition number of Jacobian df/dx.
    Condition number > 1e6? -> Problem is stiff. Use implicit methods
    or sequence-to-sequence training.
```

### Anti-Patterns to Avoid
- **Tuning hyperparameters before ruling out bugs:** Always run probe PDEs first. A broken CosseratRHS will never be fixed by adjusting lambda_phys.
- **Monitoring only total loss:** The total loss hides whether physics or data loss is actually decreasing. Always log both components separately.
- **Ignoring gradient norm ratio:** Even when losses look balanced, gradient magnitudes can be wildly mismatched, causing one loss to dominate parameter updates (Wang et al. 2021).
- **Fixed collocation points without monitoring coverage:** Residuals can concentrate in unsampled regions. Always compute residual statistics and use RAR.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Loss balancing | Custom fixed-weight scheduler | ReLoBRaLo (already in `src/pinn/loss_balancing.py`) | Dynamic adaptation based on loss statistics; handles magnitude differences automatically |
| Collocation refinement | Manual point placement | RAR (already in `src/pinn/collocation.py`) | Probability-weighted sampling concentrates points where residuals are high |
| NTK eigenvalue computation | Full NTK matrix for large models | Approximate via random projections or subsample parameters | Full NTK is O(n_params^2); use `torch.autograd.functional.jacobian` on parameter subsets |
| Analytical PDE solutions | Numerical integration for verification | scipy.integrate or known closed-form solutions | Analytical solutions are exact; numerical adds its own errors |
| Gradient monitoring | Custom backward hooks | `torch.nn.utils.clip_grad_norm_` return value + `wandb.log()` | Already available, already used in codebase |
| Loss landscape visualization | Full parameter-space sweep | 2D random-direction slicing (Li et al. 2018 method) | O(n_points) evaluations vs impossible full sweep; ~20 lines of code |

**Key insight:** The existing DD-PINN codebase already has most of the computational machinery (ReLoBRaLo, RAR, CosseratRHS, Fourier features). What's missing is the diagnostic interpretation layer -- the code that monitors these components and tells the user what's wrong when training fails.

## Common Pitfalls

### Pitfall 1: Convergence Rate Mismatch (Most Common)
**What goes wrong:** PDE residual loss converges much faster than BC/IC loss, causing the network to satisfy the PDE in a trivial way (e.g., predicting zero everywhere).
**Why it happens:** The PDE operator amplifies network output through differentiation, producing large gradients that dominate the BC/IC fitting gradients (Wang et al. 2021).
**How to avoid:** Monitor `diagnostics/grad_norm_ratio` (physics vs data gradient norms). If ratio > 10, ReLoBRaLo should be adjusting weights. If ratio > 100, intervene manually.
**Warning signs:** `loss_phys` drops quickly while `loss_data` stagnates or increases.

### Pitfall 2: Spectral Bias
**What goes wrong:** Network learns low-frequency solution components but fails to capture high-frequency modes (sharp gradients, boundary layers, wave fronts).
**Why it happens:** MLP frequency response is biased toward low frequencies (Rahaman et al. 2019). PDE solutions with multi-scale features require high-frequency representation.
**How to avoid:** Use Fourier feature embeddings (already in `src/pinn/models.py`). Monitor per-frequency error by computing FFT of predicted vs true solution.
**Warning signs:** Smooth predictions that look qualitatively correct but have poor quantitative accuracy, especially near boundaries or at high curvature regions.

### Pitfall 3: Causality Violation
**What goes wrong:** PINN fits the solution at interior space-time points before properly resolving initial conditions, leading to propagation of errors forward in time.
**Why it happens:** Standard loss equally weights all collocation points in time. Without causal ordering, the optimizer can reduce total loss by fitting late-time points at the expense of early-time accuracy.
**How to avoid:** Use curriculum warmup (already in `train_pinn.py`). Alternatively, implement causal weighting (Wang et al. 2022): weight each temporal collocation point by exp(-epsilon * cumulative_residual_before_t). Monitor `residual_vs_time` distribution.
**Warning signs:** High residuals at t=0 with low residuals at later times; or val_loss good overall but initial condition error high.

### Pitfall 4: Collocation Point Starvation
**What goes wrong:** Important regions of the domain (boundary layers, shock locations) have few collocation points, so the network has no gradient signal to learn the solution there.
**Why it happens:** Uniform or Sobol sampling doesn't account for solution structure. High-residual regions need more points but don't get them without adaptive refinement.
**How to avoid:** Enable RAR (already implemented). Monitor `residual_max / residual_mean` ratio -- if > 100, collocation distribution is poor.
**Warning signs:** Overall loss decreasing but per-component RMSE for specific state variables staying flat.

### Pitfall 5: Nondimensionalization Errors
**What goes wrong:** Physics residual terms have wildly different magnitudes, making loss balancing impossible regardless of algorithm.
**Why it happens:** Mixing SI units (positions in meters, velocities in m/s, forces in N) without proper scaling causes orders-of-magnitude differences in residual terms.
**How to avoid:** Verify all terms in CosseratRHS are O(1) after nondimensionalization. The existing `NondimScales` class should handle this, but check by printing raw residual component magnitudes.
**Warning signs:** One residual component (e.g., position) is 1e6 while another (e.g., angular velocity) is 1e-3.

### Pitfall 6: Ansatz Numerical Issues (DD-PINN Specific)
**What goes wrong:** Damped sinusoidal ansatz produces numerical overflow for large time values or extreme parameter combinations, causing NaN gradients.
**Why it happens:** `exp(-delta * t) * sin(beta * t + gamma)` can produce very large intermediate values if delta is negative (due to softplus or clamp misconfiguration) or beta is very large.
**How to avoid:** Monitor ansatz parameter statistics: `alpha_max`, `delta_min`, `beta_max`. Clamp parameters to physically reasonable ranges. Verify `g(a, 0) = 0` exactly (tolerance 1e-6).
**Warning signs:** NaN in loss after some training epochs; extreme values in model parameter histograms.

## Code Examples

### Probe PDE Runner
```python
# Source: Pattern from src/trainers/probe_envs.py adapted for PINNs
from src.pinn.probe_pdes import ALL_PROBES

def run_probe_validation(model_class, config):
    """Run all probe PDEs before real training."""
    results = {}
    for probe in ALL_PROBES:
        model = model_class(**probe.model_kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(probe.max_epochs):
            loss = probe.compute_loss(model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        passed = probe.check_pass(model)
        results[probe.name] = passed
        status = "PASS" if passed else "FAIL"
        print(f"  Probe {probe.name}: {status}")

    return results
```

### NTK Eigenvalue Diagnostic (Lightweight)
```python
# Source: Wang et al. "When and Why PINNs Fail to Train" (NeurIPS 2021)
def compute_ntk_eigenvalues(model, collocation_points, n_params_sample=500):
    """Approximate NTK eigenvalue spectrum for convergence analysis.

    Uses parameter subsampling for tractability on large models.
    """
    model.eval()
    # Sample a subset of parameters for tractability
    all_params = [p for p in model.parameters() if p.requires_grad]
    param_indices = torch.randperm(sum(p.numel() for p in all_params))[:n_params_sample]

    # Compute Jacobian of outputs w.r.t. sampled parameters
    def model_fn(params_flat):
        # ... reconstruct model with params_flat ...
        return model(collocation_points)

    J = torch.autograd.functional.jacobian(model_fn, params_flat)
    # NTK = J @ J.T
    K = J @ J.T
    eigenvalues = torch.linalg.eigvalsh(K)

    return {
        "ntk/eigenvalue_max": eigenvalues[-1].item(),
        "ntk/eigenvalue_min": eigenvalues[0].item(),
        "ntk/condition_number": (eigenvalues[-1] / max(eigenvalues[0], 1e-10)).item(),
        "ntk/spectral_decay_rate": (eigenvalues[-1] / eigenvalues[-len(eigenvalues)//2]).item(),
    }
```

### Residual Spatial Distribution Analysis
```python
# Source: Krishnapriyan et al. 2021 + existing collocation.py pattern
def analyze_residual_distribution(model, rhs, collocation_points, normalizer):
    """Compute per-region and per-component residual statistics."""
    model.eval()
    with torch.enable_grad():
        # Forward through physics
        residuals = compute_physics_residual(model, rhs, collocation_points, normalizer)

    # Temporal distribution: bin by time
    t_bins = torch.linspace(0, 0.5, 11)
    temporal_stats = {}
    for i in range(len(t_bins) - 1):
        mask = (collocation_points >= t_bins[i]) & (collocation_points < t_bins[i+1])
        if mask.sum() > 0:
            bin_residuals = residuals[mask]
            temporal_stats[f"residual_t{i}"] = bin_residuals.abs().mean().item()

    return {
        "diagnostics/residual_temporal_uniformity":
            max(temporal_stats.values()) / max(min(temporal_stats.values()), 1e-10),
        **{f"diagnostics/{k}": v for k, v in temporal_stats.items()},
    }
```

### W&B Alert Configuration for PINNs
```python
# Source: Pattern from src/trainers/diagnostics.py check_alerts()
def check_pinn_alerts(wandb_run, metrics):
    """Check PINN-specific failure conditions and fire W&B alerts."""
    import wandb

    alerts = [
        # Loss ratio explosion
        ("Loss ratio explosion",
         metrics.get("diagnostics/loss_ratio", 1.0) > 1000,
         wandb.AlertLevel.ERROR,
         "Physics loss is 1000x data loss. ReLoBRaLo may be failing."),

        # Gradient pathology
        ("Gradient pathology",
         metrics.get("diagnostics/grad_norm_ratio", 1.0) > 100,
         wandb.AlertLevel.WARN,
         "Physics gradients dominating parameter updates."),

        # NaN detection
        ("NaN in PINN metrics",
         any(v != v for v in metrics.values() if isinstance(v, float)),
         wandb.AlertLevel.ERROR,
         "NaN detected. Check physics residual computation."),

        # Residual concentration
        ("Residual concentration",
         metrics.get("diagnostics/residual_max", 0) /
         max(metrics.get("diagnostics/residual_mean", 1), 1e-10) > 100,
         wandb.AlertLevel.WARN,
         "Residual concentrated in specific regions. RAR may help."),

        # Physics loss stagnation
        ("Physics loss stagnation",
         metrics.get("diagnostics/phys_loss_stagnant_epochs", 0) > 20,
         wandb.AlertLevel.WARN,
         "Physics loss unchanged for 20 epochs. Check collocation coverage."),
    ]

    for name, condition, level, text in alerts:
        if condition:
            wandb.alert(title=name, text=text, level=level)
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed loss weights (lambda_phys = 1.0) | ReLoBRaLo / NTK-based adaptive weighting | 2021-2022 | 1-2 orders of magnitude accuracy improvement |
| Uniform collocation sampling | Residual-based Adaptive Refinement (RAR) | 2020-2021 | Concentrates compute on high-error regions |
| Raw MLP for PINNs | Fourier feature embeddings | 2020 (Tancik et al.) | Mitigates spectral bias for high-frequency solutions |
| Time-uniform loss weighting | Causal training (Wang et al. 2022) | 2022-2024 | 10-100x accuracy on chaotic/time-dependent PDEs |
| Adam-only optimization | Adam warmup + L-BFGS refinement | 2021-2023 | L-BFGS handles ill-conditioned loss landscape better near convergence |
| Standard PINN | DD-PINN with ansatz (Krauss et al. 2024) | 2024 | Exact IC satisfaction, closed-form time derivatives |
| NVIDIA Modulus | NVIDIA PhysicsNeMo | 2024 | Rebranded; added neural operator and diffusion model support |

**Deprecated/outdated:**
- Fixed lambda scheduling: replaced by ReLoBRaLo and NTK-based methods
- Manual collocation point placement: replaced by Sobol + RAR
- Plain MLP without Fourier features: known to fail on multi-scale PDEs

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest |
| Config file | `pytest.ini` / `pyproject.toml` |
| Quick run command | `python3 -m pytest tests/test_pinn_diagnostics.py -x -q` |
| Full suite command | `python3 -m pytest tests/ -x -q --timeout=120` |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PDIAG-01 | Probe PDEs pass on known-good DD-PINN | integration | `pytest tests/test_pinn_probes.py -x` | Wave 0 |
| PDIAG-02 | Diagnostics module computes correct metrics | unit | `pytest tests/test_pinn_diagnostics.py -x` | Wave 0 |
| PDIAG-03 | Alert system fires on known failure conditions | unit | `pytest tests/test_pinn_diagnostics.py::test_alerts -x` | Wave 0 |
| PDIAG-04 | Decision tree matches expected failure for known broken config | integration | `pytest tests/test_pinn_probes.py::test_decision_tree -x` | Wave 0 |
| PDIAG-05 | Skill file renders correctly as Claude Code skill | manual | Review `.claude/skills/pinn-debug/SKILL.md` | Wave 0 |

### Sampling Rate
- **Per task commit:** `python3 -m pytest tests/test_pinn_diagnostics.py -x -q`
- **Per wave merge:** `python3 -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_pinn_diagnostics.py` -- covers PDIAG-02, PDIAG-03
- [ ] `tests/test_pinn_probes.py` -- covers PDIAG-01, PDIAG-04
- [ ] `src/pinn/diagnostics.py` -- diagnostic middleware
- [ ] `src/pinn/probe_pdes.py` -- probe PDE suite

## Open Questions

1. **NTK computation cost for production use**
   - What we know: Full NTK is O(n_params^2 * n_collocation), infeasible for 512x4 models
   - What's unclear: How many sampled parameters are sufficient for meaningful diagnostics
   - Recommendation: Start with n_params_sample=500, run NTK diagnostics every 50 epochs (not every epoch). Flag as expensive diagnostic.

2. **Causal weighting integration**
   - What we know: Causal training (Wang et al. 2022) gives 10-100x accuracy on time-dependent PDEs. Our codebase has curriculum warmup but not full causal weighting.
   - What's unclear: Whether causal weighting interacts well with ReLoBRaLo or conflicts
   - Recommendation: Add causal weighting as a diagnostic recommendation in the decision tree, not as a default. If residual-vs-time analysis shows causality violation, suggest enabling it.

3. **DD-PINN-specific probe PDEs**
   - What we know: Standard PINN probes test vanilla PINN training. DD-PINN with ansatz has unique failure modes (ansatz parameter overflow, basis function interference).
   - What's unclear: Minimum set of probes that covers DD-PINN-specific issues
   - Recommendation: ProbePDE5 specifically tests the ansatz + physics residual pipeline. If time permits, add a ProbePDE6 for multi-basis interference.

## Sources

### Primary (HIGH confidence)
- Wang et al. "When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective" (NeurIPS 2021) - NTK eigenvalue analysis, convergence rate mismatch
- Krishnapriyan et al. "Characterizing Possible Failure Modes in PINNs" (NeurIPS 2021) - Failure mode taxonomy, curriculum regularization, probe PDEs
- Bischof & Kraus "Multi-Objective Loss Balancing for Physics-Informed Deep Learning" (arXiv:2110.09813) - ReLoBRaLo algorithm (already implemented in codebase)
- Wang et al. "Respecting Causality is All You Need for Training PINNs" (CMAME 2024) - Causal training, temporal weighting
- Existing codebase: `src/pinn/` (DD-PINN implementation), `src/trainers/diagnostics.py` (RL diagnostic pattern), `.claude/skills/rl-debug/` (skill template)

### Secondary (MEDIUM confidence)
- Muller & Zeinhofer "Challenges in Training PINNs: A Loss Landscape Perspective" (arXiv:2402.01868) - NysNewton-CG optimizer, ill-conditioning analysis
- Daw et al. "Mitigating Propagation Failures in Physics-informed Neural Networks" (ICML 2023) - Propagation failure characterization
- PINNacle benchmark (NeurIPS 2024) - 20+ PDE benchmark suite for systematic evaluation

### Tertiary (LOW confidence)
- DeepXDE and NVIDIA PhysicsNeMo diagnostic capabilities - referenced but not deeply verified against current versions

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - all libraries already in project, no new dependencies needed
- Architecture: HIGH - directly mirrors existing RL debug skill with well-documented PINN failure mode literature
- Pitfalls: HIGH - extensively documented in 4+ NeurIPS/ICML papers with reproducible failure demonstrations
- Probe PDEs: MEDIUM - adapted from literature but specific probe selection for DD-PINN needs validation
- NTK diagnostics: MEDIUM - well-studied theoretically but practical cost/value tradeoff for our model size unclear

**Research date:** 2026-03-26
**Valid until:** 2026-05-26 (PINN methodology is stable; no rapid ecosystem changes expected)
