# Phase 13: Implement PINN and DD-PINN Surrogate Models - Research

**Researched:** 2026-03-17
**Domain:** Physics-informed neural networks for Cosserat rod dynamics surrogate modeling
**Confidence:** MEDIUM-HIGH (overall; varies by sub-approach)

## Summary

This phase concerns adding physics-informed training to the existing data-driven surrogate model pipeline for snake robot Cosserat rod dynamics. The project already has an extensive feasibility analysis (knowledge/pinn-ddpinn-snake-locomotion-feasibility.md) that established three possible approaches: (A) full DD-PINN, (B) physics loss regularizer on the existing surrogate, and (C) KNODE hybrid. This research validates that prior analysis, updates it with current ecosystem information, and provides prescriptive implementation guidance.

**Key ecosystem update since the feasibility analysis:** No DD-PINN open-source code has been released as of March 2026. The Krauss/Licher group (Leibniz Hannover) has published two DD-PINN papers (arXiv:2408.14951, arXiv:2508.12681) but no public repository exists. The tdcr-pinn repository (Martin-Bensch, ICRA 2024) handles only *static* Cosserat rods, not dynamic. The StableCosseratRods SIGGRAPH 2025 repository is C++/GLSL (not PyTorch, not differentiable for PINN use). A 2024 paper by Li et al. demonstrates PINNs for friction-involved nonsmooth dynamics, providing evidence that friction can be handled in PINN losses -- but their system is vastly simpler than ours (1-2 DOF vs 124 states).

**Primary recommendation:** Implement the phase in three tiers: (1) physics loss regularizer on the existing data-driven surrogate (1-2 weeks, high confidence), (2) DD-PINN ansatz prototype without friction (3-4 weeks, medium confidence), (3) DD-PINN with simplified friction (4-6 weeks, research-grade, low confidence of full success). Tier 1 is the minimum deliverable. Tiers 2-3 are stretch goals.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.10.0+cu128 | Autodiff backbone, existing stack | Already installed; all surrogate code uses it |
| torchdiffeq | 0.2.5 | Neural ODE integration (KNODE variant, adjoint backprop) | Mature, GPU-native, O(1)-memory adjoint; 6.4k GitHub stars |
| DeepXDE | 1.15.0 | PINN reference implementation, loss diagnostics | Most mature PINN library; PyTorch backend; ODE system examples |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| rbischof/relative_balancing | latest | ReLoBRaLo adaptive loss weighting | When implementing multi-term PINN loss (physics + data) |
| scipy.stats.qmc | (bundled) | Latin Hypercube / Sobol collocation sampling | For generating collocation points in temporal domain |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Custom PyTorch PINN loop | DeepXDE | DeepXDE adds overhead but provides loss diagnostics; for our custom DD-PINN ansatz, custom loop is better |
| torchdiffeq | diffrax (JAX) | Would require rewriting in JAX; not compatible with existing stack |
| NVIDIA PhysicsNeMo | Custom | Massive dependency for minimal benefit at this system scale |

**Installation:**
```bash
pip install torchdiffeq==0.2.5 deepxde==1.15.0
```

**Version verification:** torchdiffeq 0.2.5 confirmed on PyPI (latest as of 2026-03-17). DeepXDE 1.15.0 confirmed on PyPI (released 2025-12-05). PyTorch 2.10.0+cu128 already installed.

## Architecture Patterns

### Recommended Project Structure
```
papers/aprx_model_elastica/
    model.py                  # Existing: SurrogateModel, ResidualSurrogateModel, TransformerSurrogateModel
    model_pinn.py             # NEW: PhysicsRegularizedSurrogate, DDPINNSurrogate, SinusoidalAnsatz
    physics_residual.py       # NEW: Differentiable Cosserat rod physics (partial f_SSM)
    loss_pinn.py              # NEW: Physics loss functions + ReLoBRaLo balancing
    train_pinn.py             # NEW: Training loop with physics loss + collocation sampling
    train_config.py           # EXTEND: Add PINNTrainConfig dataclass
tests/
    test_pinn_ansatz.py       # NEW: Ansatz forward/backward, IC enforcement, derivative accuracy
    test_physics_residual.py  # NEW: Compare PyTorch f_SSM against PyElastica outputs
    test_pinn_training.py     # NEW: Smoke test physics-regularized training loop
```

### Pattern 1: Physics Loss Regularizer (Tier 1 -- PRIMARY)
**What:** Add soft physics constraints to the existing data-driven training loop. No architectural change to the surrogate model. Only the loss function changes.
**When to use:** First step. Always implement this before attempting DD-PINN.
**Example:**
```python
# Source: Derived from existing knowledge/pinn-ddpinn-snake-locomotion-feasibility.md
class PhysicsRegularizer(nn.Module):
    """Soft physics constraints for existing delta-prediction surrogate.

    Three constraint types:
    1. Velocity-position consistency: delta_pos ~ avg_vel * dt
    2. Curvature-moment consistency: omega_dot ~ B*(kappa - kappa_rest) / (rho*I*ds)
    3. Angular velocity-yaw consistency: delta_psi ~ omega * dt
    """

    def __init__(self, rod_params: dict, dt: float = 0.5):
        super().__init__()
        self.dt = dt
        self.ds = rod_params['length'] / rod_params['n_elements']
        self.B = rod_params['bending_stiffness']
        self.rho_I = rod_params['density'] * rod_params['second_moment_of_area']

    def forward(self, state: torch.Tensor, delta_pred: torch.Tensor,
                action: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Compute physics regularization loss.

        Uses REL_* slice constants from state.py for index mapping.
        Returns scalar loss (sum of all constraint violations).
        """
        next_state = state + delta_pred

        # 1. Velocity-position consistency (positions 0-41, velocities 42-83)
        # In relative coords: REL_POS_X, REL_POS_Y, REL_VEL_X, REL_VEL_Y
        vel_x = state[..., 48:69]  # REL_VEL_X
        vel_y = state[..., 69:90]  # REL_VEL_Y
        delta_vel_x = delta_pred[..., 48:69]
        delta_vel_y = delta_pred[..., 69:90]
        delta_pos_x = delta_pred[..., 6:27]   # REL_POS_X
        delta_pos_y = delta_pred[..., 27:48]  # REL_POS_Y

        avg_vel_x = vel_x + 0.5 * delta_vel_x
        avg_vel_y = vel_y + 0.5 * delta_vel_y
        loss_kinematic = (
            F.mse_loss(delta_pos_x, avg_vel_x * self.dt) +
            F.mse_loss(delta_pos_y, avg_vel_y * self.dt)
        )

        # 2. Angular velocity-yaw consistency
        omega = state[..., 110:130]  # REL_OMEGA_Z
        delta_psi = delta_pred[..., 90:110]  # REL_YAW
        delta_omega = delta_pred[..., 110:130]
        avg_omega = omega + 0.5 * delta_omega
        loss_angular = F.mse_loss(delta_psi, avg_omega * self.dt)

        # 3. Curvature-moment consistency (simplified -- no friction)
        psi_next = state[..., 90:110] + delta_psi
        kappa = (psi_next[..., 1:] - psi_next[..., :-1]) / self.ds
        kappa_rest = self._compute_rest_curvature(action, phase)
        moment = self.B * (kappa - kappa_rest[..., :-1])
        omega_dot = delta_omega / self.dt
        loss_moment = F.mse_loss(
            omega_dot[..., 1:-1] * self.rho_I,
            (moment[..., 1:] - moment[..., :-1]) / self.ds
        )

        return loss_kinematic + loss_angular + loss_moment

    def _compute_rest_curvature(self, action, phase):
        """Extract rest curvature from action + per-element phase encoding."""
        # phase encoding: (B, 60) = 20 elements x (sin, cos, kappa)
        # kappa component is at indices 2, 5, 8, ..., 59
        kappa_rest = phase[..., 2::3]  # (B, 20) rest curvature per element
        return kappa_rest
```

### Pattern 2: DD-PINN Ansatz (Tier 2 -- STRETCH)
**What:** Sinusoidal ansatz that decouples time from the neural network. Network outputs ansatz parameters; time enters only through closed-form sin/cos evaluation.
**When to use:** After Tier 1 proves physics loss helps; if higher accuracy or fewer data points are needed.
**Example:** See the SinusoidalAnsatz class in the existing feasibility analysis (knowledge/pinn-ddpinn-snake-locomotion-feasibility.md, Code Examples section). That implementation is correct and ready to use. Key parameters for our system:
- state_dim = 130 (relative state, not raw 124)
- n_basis = 5 (matching DD-PINN paper; increase to 7 if underfitting)
- NN output dim = 3 * 130 * 5 = 1,950 (alpha, beta, gamma)
- With damping: 4 * 130 * 5 = 2,600

### Pattern 3: Loss Balancing with ReLoBRaLo
**What:** Adaptive loss weighting using Relative Loss Balancing with Random Lookback.
**When to use:** Whenever combining data loss + physics loss (both Tier 1 and Tier 2).
**Example:**
```python
# Source: rbischof/relative_balancing (GitHub)
import torch

class ReLoBRaLo:
    """Relative Loss Balancing with Random Lookback.

    Adaptively weights multiple loss terms based on their relative
    rate of change, with a random lookback to prevent oscillation.
    """

    def __init__(self, n_losses: int, alpha: float = 0.999, temperature: float = 1.0):
        self.n_losses = n_losses
        self.alpha = alpha  # EMA smoothing factor
        self.temperature = temperature
        self.prev_losses = None
        self.weights = torch.ones(n_losses)

    def update(self, losses: list[torch.Tensor]) -> torch.Tensor:
        """Return balanced weights for current losses.

        Args:
            losses: List of scalar loss tensors (detached).
        Returns:
            Tensor of weights (n_losses,).
        """
        current = torch.tensor([l.item() for l in losses])

        if self.prev_losses is None:
            self.prev_losses = current.clone()
            return self.weights

        # Relative change
        ratios = current / (self.prev_losses + 1e-8)

        # Softmax with temperature
        weights = torch.softmax(ratios / self.temperature, dim=0) * self.n_losses

        # EMA smoothing
        self.weights = self.alpha * self.weights + (1 - self.alpha) * weights
        self.prev_losses = current.clone()

        return self.weights
```

### Anti-Patterns to Avoid
- **Using PyElastica or DisMech as f_SSM:** Neither is differentiable. PyElastica is NumPy+Numba; DisMech is C++/pybind11. Attempting torch.autograd through them will silently fail or crash with `RuntimeError: Can't call numpy() on Tensor that requires grad`.
- **Standard PINN (time as network input) for 124-state system:** DD-PINN strictly dominates. The autodiff overhead of computing dx/dt through a network with 124+ outputs at 250K+ collocation points is prohibitive. Use DD-PINN's closed-form derivative instead.
- **Full physics loss without curriculum:** Starting training with both data and physics loss from epoch 0 causes catastrophic loss competition. Always use a curriculum: data-only first, then ramp physics loss.
- **Reimplementing full f_SSM as first step:** Building the full differentiable Cosserat rod equations (including RFT friction) is 2-4 weeks of work with uncertain benefit. Start with partial physics (kinematic consistency, moment balance) which requires zero f_SSM code.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ODE integration in KNODE | Custom Euler/RK4 loop | torchdiffeq.odeint_adjoint | Adjoint method, adaptive stepping, numerical stability, GPU support |
| PINN collocation sampling | Custom random sampling | scipy.stats.qmc.LatinHypercube or Sobol | Better space-filling, proven for PINNs |
| Loss balancing | Manual lambda tuning | ReLoBRaLo (rbischof/relative_balancing) | Adaptive, near-zero overhead, handles magnitude differences |
| Sinusoidal ansatz derivatives | Torch autograd through sin() | Closed-form analytic expression | The whole point of DD-PINN is avoiding autograd for time derivatives |
| State normalization | New normalizer | Existing StateNormalizer in state.py | Already handles mean/std for all 130 relative state dims |
| Cosserat rod constitutive law | Derive from papers | Extract from PyElastica source (CosseratRod._compute_internal_forces) | Validated equations; translate NumPy to PyTorch, don't reinvent |

**Key insight:** The engineering effort is NOT in the PINN framework or the ansatz -- it is in (a) formulating which physics constraints are useful without a full f_SSM, and (b) loss balancing. No library solves these for you.

## Common Pitfalls

### Pitfall 1: Assuming PyElastica Is Differentiable
**What goes wrong:** Attempting to call PyElastica's integrator inside a PyTorch computation graph.
**Why it happens:** Developers assume "Python = differentiable". PyElastica uses NumPy arrays internally with Numba JIT, which breaks autograd.
**How to avoid:** The physics regularizer (Tier 1) avoids this entirely by using only algebraic constraints in PyTorch. For Tier 2 DD-PINN, the f_SSM must be pure PyTorch.
**Warning signs:** `RuntimeError: Can't call numpy() on Tensor that requires grad` or gradients silently equal to zero.

### Pitfall 2: Loss Balancing Catastrophe
**What goes wrong:** Data loss and physics loss compete. One dominates training while the other stagnates. Total loss decreases but validation error gets worse.
**Why it happens:** Physics residual magnitude (in Cosserat rod units: N*m, rad/s^2) is completely different from MSE on normalized deltas (dimensionless, ~0.01). Equal weighting is almost always wrong.
**How to avoid:** Use ReLoBRaLo for adaptive weighting. Additionally, use a curriculum: train data-only for first 20% of epochs, then ramp lambda_phys linearly from 0 to target over next 30%, then hold.
**Warning signs:** One loss component flatlines while total loss decreases. Validation MSE increases despite total training loss decreasing.

### Pitfall 3: Spectral Bias with Angular Velocity
**What goes wrong:** The PINN learns position and velocity components accurately but fails on omega_z (angular velocity), which has the highest frequency content.
**Why it happens:** Standard MLP activations preferentially learn low-frequency functions. The omega_z component oscillates at CPG frequency (0.5-3 Hz) -- the same spectral bias that gives the current surrogate omega_z R^2 = 0.23.
**How to avoid:** For Tier 1 (regularizer): the physics loss on curvature-moment consistency directly targets omega_z accuracy, which may help. For Tier 2 (DD-PINN): the sinusoidal ansatz naturally captures periodicity -- this is DD-PINN's key advantage.
**Warning signs:** Good total loss, poor omega_z R^2 on validation.

### Pitfall 4: Ignoring Friction Makes Physics Loss Misleading
**What goes wrong:** A physics regularizer that models only internal elastic forces (moment balance) ignores RFT friction, which dominates net locomotion. The model satisfies the physics loss but produces non-physical trajectories.
**Why it happens:** Internal forces are algebraic and easy to implement. Friction requires velocity-dependent, direction-dependent computation that is hard to differentiate.
**How to avoid:** Do NOT include a full momentum balance (F = ma) without friction. Instead, use only self-consistency constraints (velocity-position, angular velocity-yaw) which are valid regardless of friction. The moment balance constraint should be weighted lower than kinematic consistency.
**Warning signs:** Physics loss near zero, validation RMSE on position components unchanged or worse.

### Pitfall 5: Collocation Point Budget Too Low
**What goes wrong:** Using too few collocation points for the temporal domain. The physics loss appears satisfied at sampled points but is violated between them.
**Why it happens:** For a 0.5s time horizon with dynamics at 0.5-3 Hz, the Nyquist criterion requires >= 6 Hz temporal resolution, i.e., >= 3 collocation points per cycle. With 1.5 cycles max, that is >= 5 points. But for the physics loss to meaningfully constrain, much more are needed.
**How to avoid:** For DD-PINN: use N_c = 100-500 collocation points per training sample over [0, 0.5s]. This is cheap because ansatz evaluation is O(n_g) per point with no network forward pass. For physics regularizer (Tier 1): N/A -- regularizer operates on single-step predictions.
**Warning signs:** Physics loss << data loss from the start (too easy to satisfy). Increasing N_c causes physics loss to suddenly jump.

## Code Examples

### Example 1: Curriculum Training with Physics Loss Ramp
```python
# Source: Adapted from Wang & Perdikaris (2024), "Challenges in Training PINNs"
def get_physics_weight(epoch: int, total_epochs: int, max_weight: float = 0.1) -> float:
    """Curriculum schedule: data-only -> ramp physics -> hold.

    Schedule:
        epochs [0, 0.2*total):      lambda_phys = 0.0  (data-only)
        epochs [0.2*total, 0.5*total): lambda_phys ramps linearly to max_weight
        epochs [0.5*total, total):   lambda_phys = max_weight
    """
    warmup_end = int(0.2 * total_epochs)
    ramp_end = int(0.5 * total_epochs)

    if epoch < warmup_end:
        return 0.0
    elif epoch < ramp_end:
        progress = (epoch - warmup_end) / (ramp_end - warmup_end)
        return max_weight * progress
    else:
        return max_weight
```

### Example 2: Collocation Point Sampling for DD-PINN
```python
# Source: scipy.stats.qmc documentation + PINN best practices
from scipy.stats import qmc

def sample_collocation_points(
    n_points: int,
    t_start: float = 0.0,
    t_end: float = 0.5,
    method: str = "sobol",
) -> torch.Tensor:
    """Sample collocation points in [t_start, t_end] for DD-PINN.

    Args:
        n_points: Number of collocation points.
        t_start: Start of temporal domain.
        t_end: End of temporal domain.
        method: "sobol", "lhs", or "uniform".

    Returns:
        (n_points,) tensor of sorted collocation times.
    """
    if method == "sobol":
        sampler = qmc.Sobol(d=1, scramble=True)
        points = sampler.random(n_points)
    elif method == "lhs":
        sampler = qmc.LatinHypercube(d=1)
        points = sampler.random(n_points)
    elif method == "uniform":
        points = torch.rand(n_points, 1).numpy()
    else:
        raise ValueError(f"Unknown method: {method}")

    # Scale to [t_start, t_end]
    t = torch.tensor(
        qmc.scale(points, t_start, t_end).flatten(),
        dtype=torch.float32,
    ).sort().values

    return t
```

### Example 3: DD-PINN Training Step
```python
# Source: Derived from DD-PINN paper (arXiv:2408.14951) + existing train_surrogate.py
def ddpinn_train_step(
    model: nn.Module,       # NN that outputs ansatz parameters
    ansatz: SinusoidalAnsatz,
    optimizer: torch.optim.Optimizer,
    batch: dict,
    t_collocation: torch.Tensor,  # (N_c,)
    f_ssm_partial,          # Partial physics RHS (internal forces only)
    lambda_phys: float,
    rod_params: dict,
) -> dict:
    """One DD-PINN training step with data + physics loss."""
    optimizer.zero_grad()

    x0 = batch['state']          # (B, 130) relative state
    action = batch['action']     # (B, 5)
    phase = batch['phase']       # (B, 60)
    target = batch['next_state'] # (B, 130) target next state

    # NN forward: predict ansatz parameters
    nn_input = torch.cat([x0, action, phase], dim=-1)
    a_params = model(nn_input)  # (B, 3*m*n_g)

    # Ansatz evaluation at t=0.5s (end of RL step)
    t_final = torch.tensor([0.5], device=x0.device)
    g_final = ansatz(a_params, t_final)  # (B, 1, m)
    x_pred = g_final.squeeze(1) + x0     # (B, m) predicted next state

    # Data loss
    loss_data = F.mse_loss(x_pred, target)

    # Physics loss at collocation points
    if lambda_phys > 0:
        g_dot = ansatz.time_derivative(a_params, t_collocation)  # (B, N_c, m)
        g_vals = ansatz(a_params, t_collocation)                  # (B, N_c, m)
        x_hat = g_vals + x0.unsqueeze(1)                          # (B, N_c, m)

        # Partial physics RHS (internal moments only -- no friction)
        x_dot_phys = f_ssm_partial(x_hat, action, phase, rod_params)  # (B, N_c, m)

        # Physics residual: only on components where f_SSM is defined
        # Kinematic components (dx/dt = v) are exact
        # Moment components (domega/dt = ...) are partial
        loss_phys = F.mse_loss(g_dot, x_dot_phys)
    else:
        loss_phys = torch.tensor(0.0, device=x0.device)

    loss = loss_data + lambda_phys * loss_phys
    loss.backward()
    optimizer.step()

    return {
        'loss': loss.item(),
        'loss_data': loss_data.item(),
        'loss_phys': loss_phys.item(),
        'lambda_phys': lambda_phys,
    }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Vanilla PINN (time as input) | DD-PINN (time in ansatz) | Aug 2024 (Krauss et al.) | 5-38x training speedup; scales to 72+ states |
| Manual loss weights | ReLoBRaLo / NTK-based adaptive | 2022-2025 | Eliminates lambda tuning; improves convergence |
| Uniform random collocation | Adaptive / failure-informed sampling | 2023-2024 | Better accuracy with fewer points |
| Standard MLP for PINN | Fourier feature encoding | 2020-present | Mitigates spectral bias; critical for oscillatory dynamics |
| Equal-weight loss terms | Curriculum training (data first) | 2023-present | Prevents physics loss from corrupting early learning |
| PINN for static Cosserat (Bensch) | DD-PINN for dynamic Cosserat (Licher) | 2024-2025 | Dynamic control at 70 Hz demonstrated |

**Deprecated/outdated:**
- Standard PINN for large ODE systems (use DD-PINN instead -- strictly better)
- Equal loss weighting (always use adaptive or curriculum)
- Uniform collocation sampling (use quasi-random or adaptive)

## Open Questions

1. **Does the physics regularizer actually improve omega_z R^2?**
   - What we know: The current surrogate has omega_z R^2 = 0.23 (very poor). Kinematic consistency and moment balance target exactly this component.
   - What's unclear: Whether soft constraints without friction can improve a quantity that is dominated by friction effects at the RL-step timescale (0.5s).
   - Recommendation: This is the key experiment. If Tier 1 does NOT improve omega_z, Tier 2 (DD-PINN) is unlikely to help either without friction in the physics loss.

2. **Can RFT friction be smoothly differentiated in PyTorch?**
   - What we know: The existing friction.py uses sigmoid regularization at low speed. The Li et al. 2024 paper demonstrates PINNs with friction but on 1-2 DOF systems. Our system has 21 nodes each with their own friction.
   - What's unclear: Whether the gradients through 21-node RFT friction are numerically stable for PINN training at scale.
   - Recommendation: If Tier 2 succeeds without friction, attempt a simplified isotropic drag (F = -c*v, fully smooth) as a first approximation before attempting full anisotropic RFT.

3. **What is the right n_basis for the DD-PINN ansatz?**
   - What we know: The DD-PINN paper uses n_g=5 for 72-state systems. Our system has higher-frequency content (CPG at 0.5-3 Hz over 0.5s window).
   - What's unclear: Whether n_g=5 provides enough basis functions for our 130-state relative system.
   - Recommendation: Start with n_g=5, sweep {3, 5, 7, 10}. The NN output dimension scales linearly with n_g, so this is a direct cost-accuracy tradeoff.

4. **Should the DD-PINN predict in raw (124-dim) or relative (130-dim) space?**
   - What we know: The current surrogate uses 130-dim relative coordinates (CoM-subtracted). The DD-PINN ansatz works in state space.
   - What's unclear: Whether the CoM-relative transform (which involves a mean operation) interacts poorly with the sinusoidal ansatz.
   - Recommendation: Use relative coordinates. The CoM translation is a simple offset that the ansatz can absorb into the alpha parameters.

5. **Has any DD-PINN code been released since March 2026?**
   - What we know: No code found as of 2026-03-17. The Krauss/Licher group published arXiv:2508.12681v2 (Jan 2026 update) with no code link.
   - Recommendation: Check again before starting Tier 2. If code appears, it dramatically reduces implementation effort.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest 8.0+ |
| Config file | pyproject.toml [tool.pytest.ini_options] |
| Quick run command | `pytest tests/test_pinn_ansatz.py tests/test_physics_residual.py -x -v` |
| Full suite command | `pytest tests/ -v --tb=short` |

### Phase Requirements -> Test Map

Phase 13 has TBD requirements. Based on the implementation tiers, the following test map captures the core behaviors:

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PINN-01 | PhysicsRegularizer computes valid scalar loss from state+delta+action+phase | unit | `pytest tests/test_physics_residual.py::test_regularizer_forward -x` | No -- Wave 0 |
| PINN-02 | PhysicsRegularizer gradients flow to model parameters | unit | `pytest tests/test_physics_residual.py::test_regularizer_gradients -x` | No -- Wave 0 |
| PINN-03 | SinusoidalAnsatz satisfies g(a, 0) = 0 for any a | unit | `pytest tests/test_pinn_ansatz.py::test_ansatz_ic -x` | No -- Wave 0 |
| PINN-04 | SinusoidalAnsatz.time_derivative matches finite-difference check | unit | `pytest tests/test_pinn_ansatz.py::test_ansatz_derivative_accuracy -x` | No -- Wave 0 |
| PINN-05 | DD-PINN training step runs without error (smoke test) | smoke | `pytest tests/test_pinn_training.py::test_ddpinn_smoke -x` | No -- Wave 0 |
| PINN-06 | Physics-regularized training improves omega_z R^2 vs baseline | integration | `pytest tests/test_pinn_training.py::test_physics_reg_improves_omega -x` | No -- Wave 0 |
| PINN-07 | ReLoBRaLo balances data+physics loss within 10x of each other | unit | `pytest tests/test_pinn_training.py::test_relobralo_balance -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_pinn_ansatz.py tests/test_physics_residual.py -x -v`
- **Per wave merge:** `pytest tests/ -v --tb=short`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_pinn_ansatz.py` -- covers PINN-03, PINN-04
- [ ] `tests/test_physics_residual.py` -- covers PINN-01, PINN-02
- [ ] `tests/test_pinn_training.py` -- covers PINN-05, PINN-06, PINN-07
- [ ] `pip install torchdiffeq==0.2.5` -- if Neural ODE / KNODE variant is pursued

## Sources

### Primary (HIGH confidence)
- **Existing project analysis:** knowledge/pinn-ddpinn-snake-locomotion-feasibility.md -- extensive feasibility study, code examples, architecture analysis
- **Existing project analysis:** knowledge/neural-ode-pde-approximation-survey.md -- comprehensive survey of PINN/Neural ODE/operator methods
- **Existing project analysis:** knowledge/knode-cosserat-hybrid-surrogate-report.md -- KNODE alternative analysis
- **DD-PINN original:** Krauss et al. "Domain-decoupled Physics-informed Neural Networks." [arXiv:2408.14951](https://arxiv.org/abs/2408.14951) (Aug 2024)
- **DD-PINN Cosserat application:** Licher et al. "Adaptive MPC of a Soft Continuum Robot Using a PINN Based on Cosserat Rod Theory." [arXiv:2508.12681](https://arxiv.org/abs/2508.12681) (Aug 2025, updated Jan 2026)
- **PyPI verified:** torchdiffeq 0.2.5, DeepXDE 1.15.0, PyTorch 2.10.0+cu128

### Secondary (MEDIUM confidence)
- **ReLoBRaLo:** Bischof & Kraus. "Multi-Objective Loss Balancing for Physics-Informed Deep Learning." [arXiv:2110.09813](https://arxiv.org/abs/2110.09813). Code: [github.com/rbischof/relative_balancing](https://github.com/rbischof/relative_balancing)
- **PINN with friction:** Li et al. "Physics-informed neural networks for friction-involved nonsmooth dynamics." [Nonlinear Dynamics 112, 7159-7183 (2024)](https://link.springer.com/article/10.1007/s11071-024-09350-z)
- **tdcr-pinn (static Cosserat):** Bensch et al., ICRA 2024. Code: [github.com/Martin-Bensch/tdcr-pinn](https://github.com/Martin-Bensch/tdcr-pinn) -- static only, not applicable to dynamic problem
- **Adaptive collocation sampling:** [arXiv:2501.07700](https://arxiv.org/html/2501.07700) -- QR-DEIM adaptive collocation points
- **Causal training:** Wang & Perdikaris (2024). "Challenges in Training PINNs." [arXiv:2402.01868](https://arxiv.org/pdf/2402.01868)

### Tertiary (LOW confidence -- needs validation)
- No published DD-PINN open-source code found as of 2026-03-17 (confirmed by WebSearch)
- No published results on PINNs with RFT friction for locomotion (confirmed by WebSearch)
- Claim that "physics regularizer helps omega_z" is hypothetical -- requires experimental validation
- The StableCosseratRods (SIGGRAPH 2025) codebase is C++/GLSL, not usable as differentiable PyTorch backbone

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified on PyPI, compatible with existing project
- Architecture (Tier 1 physics regularizer): HIGH -- algebraic constraints well-understood, no external dependencies
- Architecture (Tier 2 DD-PINN ansatz): MEDIUM -- paper is clear but no reference implementation exists
- Architecture (Tier 3 DD-PINN with friction): LOW -- unsolved research problem at this scale
- Pitfalls: HIGH -- drawn from extensive existing analysis + verified literature
- Loss balancing: MEDIUM-HIGH -- ReLoBRaLo is published and has code, but untested on this specific system

**Research date:** 2026-03-17
**Valid until:** 2026-04-17 (30 days -- check for DD-PINN code releases before implementing)
