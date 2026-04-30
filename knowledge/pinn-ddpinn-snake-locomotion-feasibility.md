---
name: pinn-ddpinn-snake-locomotion-feasibility
description: Feasibility analysis of PINN and DD-PINN for snake robot locomotion with Elastica/DisMech physics
type: knowledge
created: 2026-03-17T00:00:00
updated: 2026-03-17T00:00:00
tags: [pinn, dd-pinn, cosserat-rod, surrogate, elastica, dismech, feasibility, rft, friction]
aliases: [pinn-feasibility, dd-pinn-snake-feasibility]
---

# PINN / DD-PINN Feasibility for Snake Robot Locomotion

## Executive Summary

**Recommendation: CONDITIONAL NO-GO for DD-PINN; QUALIFIED GO for simplified PINN physics loss as a training regularizer.**

Confidence: HIGH

Adding a full DD-PINN surrogate to this snake locomotion project is **not feasible in the near term** due to three fundamental blockers:

1. **No DD-PINN code exists.** The Krauss/Licher DD-PINN papers (arXiv:2408.14951, arXiv:2508.12681) have no open-source implementation as of March 2026. Implementing from scratch requires building the ansatz framework, the physics residual evaluator, and the training loop -- estimated 4-8 weeks of development.

2. **The physics residual requires a differentiable f_SSM.** The DD-PINN physics loss evaluates ||g_dot(a, t) - f_SSM(x_hat, u)||^2 at collocation points. For our system, f_SSM must encode the Cosserat rod equations + RFT friction + CPG actuation in differentiable PyTorch. PyElastica (NumPy+Numba) and DisMech (C++/pybind11) are NOT differentiable. Writing f_SSM from scratch in PyTorch is the core engineering challenge.

3. **RFT friction is discontinuous.** The DD-PINN paper handles pneumatic pressure (smooth, continuous input) with no external forces. Our system has anisotropic RFT friction (velocity-direction-dependent, regularized but still stiff) and Coulomb/Stribeck ground contact (barrier function). Including these in a PINN physics loss is a largely unsolved research problem.

**However**, a more practical path exists: use a **simplified physics loss as a regularizer** for the existing data-driven surrogate. This does not require a full f_SSM -- only soft physics constraints (energy conservation, momentum consistency, curvature-force relationships). This is lower effort, lower risk, and compatible with the existing training pipeline.

**Primary recommendation:** Continue with the existing data-driven surrogate architecture (which is working and under active sweep). If physics regularization is desired, add a lightweight curvature-force consistency loss rather than attempting a full PINN/DD-PINN. The KNODE-Cosserat hybrid approach (documented in knowledge/knode-cosserat-hybrid-surrogate-report.md) remains the most practical physics-informed alternative.

---

## Standard Stack

### If Proceeding with PINN Physics Loss (Regularizer Approach)

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| PyTorch | 2.x (existing) | Autodiff for physics residual | Already in stack, native autograd |
| torchdiffeq | ~0.2.4 | Neural ODE integration (for KNODE variant) | Mature, PyTorch-native, GPU support |

### If Attempting Full DD-PINN (NOT recommended)

| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| DeepXDE | 1.15.x | PINN framework with PyTorch backend | Most mature PINN library, ODE system examples |
| Custom PyTorch | - | DD-PINN ansatz implementation | No existing library implements DD-PINN |

### Libraries NOT Recommended

| Library | Why Not |
|---------|---------|
| NVIDIA Modulus/PhysicsNeMo | Overkill for this scale; heavy dependency |
| DeepXDE for DD-PINN | DeepXDE implements standard PINNs, not DD-PINN |
| JAX-based frameworks | Would require rewriting entire training pipeline |

---

## Architecture Patterns

### Pattern 1: DD-PINN Ansatz Architecture (Reference Only)

The DD-PINN separates the neural network from time:

```
Input: (x_0, u) --> NN --> a = (alpha, beta, gamma) --> g(a, t) --> x_hat = g(a, t) + x_0
```

For our 124-state system with n_g=5 basis functions per state:
- NN output dimension: 3 * 124 * 5 = 1,860 parameters (alpha, beta, gamma)
- With damping: 4 * 124 * 5 = 2,480 parameters
- Ansatz: g_j(t) = sum_i alpha_ij * [sin(beta_ij * t + gamma_ij) - sin(gamma_ij)]
- Closed-form derivative: g_dot_j(t) = sum_i alpha_ij * beta_ij * cos(beta_ij * t + gamma_ij)

**Problem for our system:** The NN output is 1,860+ dimensions to predict 124 state dimensions. This is a much harder regression problem than the current delta prediction (124-dim output). The DD-PINN paper works with 72 states and simpler dynamics.

### Pattern 2: Physics Loss Regularizer (RECOMMENDED if pursuing physics)

Add a soft physics constraint to the existing data-driven training:

```python
# Existing data loss
loss_data = MSE(model(state, action, phase), delta_target)

# Physics regularizer: curvature-force consistency
# The internal moment is proportional to curvature deviation from rest
kappa_pred = extract_curvatures(state + model(state, action, phase))
kappa_rest = compute_rest_curvature(action, phase)
moment_residual = B * (kappa_pred - kappa_rest)  # Should drive angular acceleration
omega_dot_pred = extract_angular_accel(model(state, action, phase))
loss_physics = MSE(omega_dot_pred, moment_residual / (rho * I))  # Simplified moment balance

loss = loss_data + lambda_phys * loss_physics
```

This does NOT require a full differentiable simulator -- just physically motivated soft constraints.

### Pattern 3: KNODE Hybrid (Documented Separately)

The most practical physics-informed approach, already analyzed in detail in `knowledge/knode-cosserat-hybrid-surrogate-report.md` and `report/knode-cosserat-hybrid-surrogate.tex`.

### Anti-Patterns to Avoid

- **Full PINN from scratch for 124-state dynamic ODE with friction**: The problem dimensionality, stiffness, and non-smooth friction make this a multi-month research project with uncertain outcome.
- **Using PyElastica/DisMech as f_SSM directly**: These are not differentiable. Cannot compute gradients of the physics residual through them.
- **Standard PINN (time as network input)**: For a 124-state system, autodiff through the network to compute dx/dt is expensive and suffers from the exact problems DD-PINN was designed to solve. Do not use vanilla PINNs for this scale.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PINN training loop | Custom training loop with collocation sampling | DeepXDE (if using standard PINN) | Loss balancing, adaptive sampling, diagnostics already solved |
| ODE integration in NN | Custom Euler/RK4 in PyTorch | torchdiffeq (if using Neural ODE/KNODE) | Adjoint method, adaptive stepping, numerical stability |
| Cosserat rod equations in PyTorch | Re-derive from scratch | Extract from PyElastica source + simplify | PyElastica's equations are validated; translate, don't reinvent |
| Ansatz function library | Custom sin/cos manipulation | Standard PyTorch tensor ops | The ansatz math is elementary; no library needed |

**Key insight:** The engineering effort is NOT in the PINN framework or ansatz -- it is in formulating the physics residual f_SSM in differentiable PyTorch. This is the bottleneck that no library solves for you.

---

## Common Pitfalls

### Pitfall 1: Assuming PyElastica Is Differentiable
**What goes wrong:** Attempting to use PyElastica's `PositionVerlet` or `PEFRL` integrator in a PyTorch autograd computation graph.
**Why it happens:** PyElastica is NumPy + Numba. It cannot participate in PyTorch's autograd.
**How to avoid:** Accept that f_SSM must be reimplemented in PyTorch. Alternatively, use the physics residual only as a data-generation mechanism (train on PyElastica trajectories) rather than in the loss function.
**Warning signs:** `RuntimeError: Can't call numpy() on Tensor that requires grad` or silently broken gradients.

### Pitfall 2: Ignoring Friction in the Physics Loss
**What goes wrong:** Building a physics loss that models only internal elastic forces but ignores RFT friction. The resulting surrogate may satisfy the "physics" loss perfectly but be wildly inaccurate because friction dominates locomotion dynamics.
**Why it happens:** Internal forces (constitutive law) are smooth and easy to differentiate. Friction is velocity-dependent, direction-dependent, and stiff.
**How to avoid:** Either (a) include a regularized friction model in the physics loss, or (b) use a data loss for the friction-dominated dynamics and physics loss only for the internal elastic dynamics.
**Warning signs:** Good physics loss, bad validation error. Snake "slides" without friction effects.

### Pitfall 3: Loss Balancing Catastrophe
**What goes wrong:** The physics loss and data loss compete, with one dominating training and the other stagnating.
**Why it happens:** For a 124-state system, the physics residual has very different magnitude than the data loss. The gradients from physics and data losses conflict.
**How to avoid:** Use adaptive loss weighting (e.g., Wang et al.'s learning rate annealing). Start with lambda_phys = 0 and gradually increase it (curriculum approach). Monitor individual loss components.
**Warning signs:** Total loss decreases but one component flatlines or increases.

### Pitfall 4: Spectral Bias with High-Frequency Dynamics
**What goes wrong:** The PINN preferentially learns low-frequency dynamics (slow drift) and fails to capture high-frequency oscillations (CPG-driven vibrations at 0.5-3 Hz).
**Why it happens:** Standard MLP activations have frequency-dependent learning rates that favor low frequencies (the spectral bias problem).
**How to avoid:** Use Fourier feature encoding (already analyzed in knowledge/periodic-pattern-learning-surrogate-research.md). For DD-PINN specifically, the sinusoidal ansatz naturally handles periodicity -- this is one of DD-PINN's genuine advantages.
**Warning signs:** Position predictions are accurate but angular velocity predictions are poor (exactly the current surrogate's omega_z R^2=0.23 problem).

### Pitfall 5: Collocation Point Budget for 124-State System
**What goes wrong:** Using too few collocation points for the physics loss, leading to the physics constraint being trivially satisfied at sample points but violated everywhere else.
**Why it happens:** A 124-state system needs many collocation points to cover the time domain adequately. DD-PINN's advantage (cheap collocation) helps, but the physics evaluation at each point is expensive if f_SSM is complex.
**How to avoid:** DD-PINN uses 250K-1M collocation points for 72 states. For our 124 states, expect similar or higher requirements. Budget GPU memory accordingly.

---

## Feasibility Analysis by Approach

### Approach A: Full DD-PINN Surrogate

**Feasibility: LOW (2-3 months, uncertain outcome)**

| Requirement | Status | Difficulty |
|-------------|--------|------------|
| DD-PINN framework code | Must build from scratch | MEDIUM -- the architecture is well-described in the paper |
| Ansatz function (sinusoidal) | Straightforward to implement | LOW |
| f_SSM in differentiable PyTorch | Must rewrite Cosserat rod equations | HIGH -- 2-4 weeks to get right |
| RFT friction in f_SSM | Must implement differentiable RFT | HIGH -- direction-dependent, stiff |
| Stribeck/barrier contact in f_SSM | Must implement smooth differentiable contact | HIGH -- softplus barrier exists but needs PyTorch port |
| CPG rest curvature in f_SSM | Straightforward (sin function) | LOW |
| Training pipeline | Must build collocation sampling, adaptive weighting | MEDIUM |
| Validation against PyElastica | Need trajectory comparisons | LOW |

**The hard part:** Reimplementing the full Cosserat rod dynamics + RFT friction + contact in differentiable PyTorch. This is ~2000 lines of physics code that must match PyElastica's behavior exactly. The existing PyElastica code uses NumPy and Numba with specific numerical tricks (e.g., the barrier function in friction.py) that need careful translation.

**What makes our system harder than the DD-PINN paper's system:**
1. Our system has 124 states (vs 72) -- larger ansatz output
2. Our system has RFT friction (vs no external forces)
3. Our system has ground contact (vs free space)
4. Our actuation is rest curvature (vs pneumatic pressure)
5. Our system has 500 substeps per RL step (vs single-step prediction)

### Approach B: Standard PINN with f_SSM

**Feasibility: LOW-MEDIUM (same f_SSM problem, plus autodiff overhead)**

Same as Approach A for f_SSM difficulty, but additionally:
- Requires autodiff through the network for dx/dt (the problem DD-PINN solves)
- Loss balancing is harder (3 terms instead of 2)
- Initial condition is approximate (not exact as in DD-PINN)
- Training is ~5-38x slower than DD-PINN per the paper

**Verdict:** If you're going to implement f_SSM, you should use DD-PINN rather than standard PINN. DD-PINN strictly dominates for this use case.

### Approach C: Physics Loss Regularizer on Existing Surrogate

**Feasibility: MEDIUM-HIGH (1-2 weeks, likely to help)**

Instead of a full physics residual, add targeted physics constraints:

1. **Curvature-moment consistency:** The bending moment m = B(kappa - kappa_rest) should produce angular acceleration omega_dot = m / (rho * I). This is a local, algebraic constraint that does NOT require a full f_SSM -- just the constitutive law.

2. **Kinematic consistency:** The predicted velocity should be consistent with the predicted position change: v_{t+1} * dt approx delta_position. This is a self-consistency check.

3. **Energy bound:** The total kinetic + elastic energy should change by an amount consistent with the work done by friction and actuation. This prevents non-physical energy creation.

**Implementation:**

```python
def physics_regularizer(state, action, phase, delta_pred, dt=0.5):
    """Lightweight physics constraints -- no full f_SSM needed."""

    # Extract predicted next-step quantities
    next_state = state + delta_pred

    # 1. Curvature-moment consistency
    psi = state[..., 84:104]  # element yaws (20,)
    psi_next = next_state[..., 84:104]
    omega = state[..., 104:124]  # angular velocities (20,)
    omega_next = next_state[..., 104:124]

    # Discrete curvature: kappa_i = (psi_{i+1} - psi_i) / ds
    ds = rod_length / n_elements  # known constant
    kappa = (psi[..., 1:] - psi[..., :-1]) / ds  # (19,)
    kappa_next = (psi_next[..., 1:] - psi_next[..., :-1]) / ds

    # Rest curvature from action + phase (known analytically)
    kappa_rest = compute_rest_curvature_from_phase(action, phase)  # (20,)

    # Moment balance: rho*I * omega_dot = B * (kappa - kappa_rest) / ds + ...
    omega_dot_pred = (omega_next - omega) / dt
    moment = B * (kappa_next - kappa_rest[..., :-1])  # simplified

    # Soft constraint: omega_dot should be correlated with moment/inertia
    loss_moment = F.mse_loss(
        omega_dot_pred[..., :-1] * rho_I,
        moment / ds
    )

    # 2. Velocity-position consistency
    pos_x = state[..., 0:21]
    vel_x = state[..., 42:63]
    delta_x = delta_pred[..., 0:21]
    delta_vx = delta_pred[..., 42:63]
    # Average velocity * dt should approximate position change
    avg_vel = vel_x + 0.5 * delta_vx
    loss_kinematic = F.mse_loss(delta_x, avg_vel * dt)

    return loss_moment + loss_kinematic
```

This requires NO reimplementation of f_SSM. The physics knowledge enters through soft constraints derived from the constitutive law and kinematics.

### Approach D: KNODE Hybrid (Separately Documented)

**Feasibility: MEDIUM (3-6 weeks, good evidence from literature)**

See `knowledge/knode-cosserat-hybrid-surrogate-report.md` for full analysis. Key advantage: the physics backbone handles kinematics and time integration; the NN only corrects internal forces. This is the most principled physics-informed approach that remains practical.

---

## Comparison: PINN vs Current Data-Driven Surrogate

| Aspect | Current Data-Driven | DD-PINN (hypothetical) | Physics Regularizer | KNODE Hybrid |
|--------|---------------------|------------------------|--------------------|----|
| **Implementation effort** | Done | 2-3 months | 1-2 weeks | 3-6 weeks |
| **f_SSM required?** | No | Yes (full) | No (partial) | Partial (simplified) |
| **Friction handling** | Learned from data | Must be in physics loss | Not needed in loss | In physics backbone |
| **Training data** | Large dataset | Less needed (physics supervises) | Same as current | Less needed |
| **Training time** | Fast (~hours) | Slow (autodiff overhead) | Slightly slower | Slower (ODE solve) |
| **Inference speed** | Fast (single MLP pass) | Fast (ansatz eval) | Same as current | Slower (ODE integration) |
| **Accuracy potential** | Good (with enough data) | Potentially better OOD | Modestly better | Best for generalization |
| **Physics consistency** | Not guaranteed | At collocation points | Soft constraints | Hard constraints on kinematics |
| **Risk** | Low (proven) | High (no code, hard problem) | Low | Medium |

**Bottom line:** The current data-driven surrogate is the right approach for now. Physics regularization (Approach C) is the lowest-risk way to add physics knowledge. DD-PINN (Approach A) is a research project, not an engineering task.

---

## Detailed Technical Analysis

### Can We Formulate PyElastica Equations as a Physics Loss?

**Partially, with significant effort.**

The PyElastica Cosserat rod equations in 2D reduce to:

```
rho * A * x_ddot_i = (n_{i+1} - n_i) / ds + f_friction_i + f_gravity_i
rho * I * omega_dot_i = (m_{i+1} - m_i) / ds + cross(r_s, n_i) + tau_cpg_i
```

where:
- n = S * (epsilon - epsilon_0) is the internal force from the constitutive law
- m = B * (kappa - kappa_0) is the internal moment
- epsilon = stretch strain, kappa = curvature (both derived from positions and angles)
- f_friction_i = -ct * v_tangential - cn * v_normal (RFT)

**What is differentiable:**
- Constitutive law: n = S * (epsilon - epsilon_0) -- purely algebraic, trivially differentiable
- Moment balance: m = B * (kappa - kappa_rest) -- algebraic
- Kinematic relations: epsilon from positions, kappa from angles -- differentiable
- CPG rest curvature: sin function -- differentiable

**What is problematic:**
- RFT friction: direction-dependent (v_tangential vs v_normal decomposition uses tangent directions derived from positions). The tangent computation involves normalization (division by norm), which has gradient singularities at zero length.
- Barrier contact: softplus-based, continuous and differentiable -- actually fine in PyTorch
- Time integration: PositionVerlet is a symplectic integrator with specific half-step structure. Differentiating through it is possible but requires careful implementation.

### Can Gradients Flow Through DisMech?

**No.** DisMech is C++ with pybind11 bindings. PyTorch autograd cannot differentiate through C++ code unless it is wrapped as a custom autograd Function with hand-derived gradients. This would require:
1. Computing the Jacobian of DisMech's implicit Euler solve analytically
2. Implementing a backward() method that applies this Jacobian
3. This is a research project in itself (adjoint method for implicit Euler)

**Verdict:** DisMech is off the table for PINN physics losses.

### What Would the Physics Residual Look Like?

For DD-PINN with our system, the physics residual at collocation point t_k is:

```python
def physics_residual(a_params, x_0, u, t_k, rod_params):
    """Compute ||g_dot(a, t_k) - f_SSM(x_hat_k, u_k)||^2."""

    # Ansatz prediction at t_k
    x_hat_k = ansatz(a_params, t_k) + x_0  # (124,)

    # Ansatz time derivative (closed form)
    x_dot_hat_k = ansatz_dot(a_params, t_k)  # (124,)

    # Physics model evaluation (THIS IS THE HARD PART)
    x_dot_physics = f_SSM(x_hat_k, u_k, rod_params)  # (124,)

    residual = x_dot_hat_k - x_dot_physics
    return (residual ** 2).sum()
```

The `f_SSM` function must compute all 124 time derivatives from first principles:
- 42 position derivatives = velocities (trivial, from state)
- 42 velocity derivatives = accelerations (requires force computation)
- 20 angle derivatives = angular velocities (from state)
- 20 angular velocity derivatives = angular accelerations (requires moment computation)

The velocity derivatives (accelerations) require computing:
1. Internal forces from strains (differentiable)
2. RFT friction forces (stiff, direction-dependent)
3. Gravity (trivial)
4. Mass matrix inversion (diagonal for DER, trivial)

**Total effort to implement f_SSM in PyTorch: estimated 1500-2500 lines of code, 2-4 weeks.**

### Is the Sinusoidal Ansatz Appropriate for Snake Locomotion?

**Yes, with caveats.**

The snake's locomotion is driven by a CPG with sinusoidal rest curvature. The body undergoes periodic oscillations during each RL step. A sinusoidal ansatz is a natural fit:

- **Positions:** Nearly sinusoidal lateral oscillation (traveling wave along body)
- **Velocities:** Derivative of sinusoidal position = cosine = sinusoidal
- **Angles (yaw):** Directly related to body curvature, which is sinusoidally forced
- **Angular velocities:** Derivative of angles, again sinusoidal

**Caveats:**
1. The actual motion is not purely sinusoidal due to nonlinear friction and finite-amplitude effects. The ansatz would need enough basis functions (n_g >= 5) to capture harmonics.
2. Transient behavior at step boundaries (when the RL action changes) is not well-represented by steady-state sinusoids. A damped sinusoidal ansatz (with exponential decay) helps.
3. The 0.5s control interval may contain fractional cycles (0.25 to 1.5 cycles). The ansatz must handle both "within one cycle" and "multiple cycles" regimes.

---

## Implementation Path (If Proceeding)

### Phase 1: Physics Regularizer (Recommended, 1-2 weeks)

1. Implement `compute_rest_curvature_from_phase()` in PyTorch (extract from existing state.py)
2. Implement curvature-moment consistency loss
3. Implement velocity-position consistency loss
4. Add to existing training loop with lambda_phys starting at 0.0
5. Sweep lambda_phys in {0.001, 0.01, 0.1, 1.0}
6. Evaluate: does physics regularization improve omega_z R^2?

### Phase 2: DD-PINN Prototype (Only if Phase 1 shows promise, 4-6 weeks)

1. Implement DD-PINN ansatz in PyTorch (sinusoidal + optional damping)
2. Implement simplified f_SSM: internal forces only, no friction
3. Train DD-PINN on small dataset with physics loss
4. Compare accuracy vs data-driven surrogate on frictionless test cases
5. Add RFT friction to f_SSM (regularized, differentiable version)
6. Full comparison on locomotion data

### Phase 3: Full DD-PINN with Friction (Research, 4-8 weeks)

1. Implement full f_SSM with Stribeck friction in PyTorch
2. Validate f_SSM against PyElastica outputs (forward simulation comparison)
3. Train DD-PINN with full physics + data loss
4. Integrate as drop-in replacement for existing surrogate
5. Benchmark: accuracy, training time, inference speed

### Do NOT Proceed With

- Differentiable PyElastica (does not exist, would be a major project)
- Differentiable DisMech (C++, not practical)
- Standard PINN for this system (DD-PINN strictly dominates)
- FNO for this problem (architectural mismatch -- see knowledge/periodic-pattern-learning-surrogate-research.md)

---

## Code Examples

### DD-PINN Ansatz Function (PyTorch)

```python
import torch
import torch.nn as nn
import math


class SinusoidalAnsatz(nn.Module):
    """DD-PINN sinusoidal ansatz: g(a, t) with g(a, 0) = 0.

    For each state dimension j, the ansatz is:
        g_j(t) = sum_i alpha_ij * [sin(beta_ij * t + gamma_ij) - sin(gamma_ij)]

    The NN outputs a = (alpha, beta, gamma) of shape (B, 3 * m * n_g).
    """

    def __init__(self, state_dim: int, n_basis: int = 5):
        super().__init__()
        self.state_dim = state_dim  # m = 124
        self.n_basis = n_basis       # n_g = 5
        self.param_dim = 3 * state_dim * n_basis  # Output dim of NN

    def forward(
        self,
        params: torch.Tensor,  # (B, 3*m*n_g) from NN
        t: torch.Tensor,        # (N_c,) collocation times
    ) -> torch.Tensor:
        """Evaluate ansatz at collocation times.

        Returns: (B, N_c, m) predicted state deviations from x_0.
        """
        B = params.shape[0]
        N_c = t.shape[0]
        m = self.state_dim
        n_g = self.n_basis

        # Unpack parameters
        alpha = params[:, :m*n_g].reshape(B, m, n_g)         # (B, m, n_g)
        beta = params[:, m*n_g:2*m*n_g].reshape(B, m, n_g)   # (B, m, n_g)
        gamma = params[:, 2*m*n_g:].reshape(B, m, n_g)       # (B, m, n_g)

        # Expand for broadcasting: (B, 1, m, n_g) and (1, N_c, 1, 1)
        t_exp = t[None, :, None, None]       # (1, N_c, 1, 1)
        alpha_exp = alpha[:, None, :, :]     # (B, 1, m, n_g)
        beta_exp = beta[:, None, :, :]       # (B, 1, m, n_g)
        gamma_exp = gamma[:, None, :, :]     # (B, 1, m, n_g)

        # g_j(t) = sum_i alpha_ij * [sin(beta_ij*t + gamma_ij) - sin(gamma_ij)]
        phase = beta_exp * t_exp + gamma_exp  # (B, N_c, m, n_g)
        basis = torch.sin(phase) - torch.sin(gamma_exp)  # (B, N_c, m, n_g)
        g = (alpha_exp * basis).sum(dim=-1)  # (B, N_c, m)

        return g

    def time_derivative(
        self,
        params: torch.Tensor,  # (B, 3*m*n_g)
        t: torch.Tensor,        # (N_c,)
    ) -> torch.Tensor:
        """Closed-form time derivative of ansatz.

        g_dot_j(t) = sum_i alpha_ij * beta_ij * cos(beta_ij*t + gamma_ij)

        Returns: (B, N_c, m)
        """
        B = params.shape[0]
        m = self.state_dim
        n_g = self.n_basis

        alpha = params[:, :m*n_g].reshape(B, m, n_g)
        beta = params[:, m*n_g:2*m*n_g].reshape(B, m, n_g)
        gamma = params[:, 2*m*n_g:].reshape(B, m, n_g)

        t_exp = t[None, :, None, None]
        alpha_exp = alpha[:, None, :, :]
        beta_exp = beta[:, None, :, :]
        gamma_exp = gamma[:, None, :, :]

        phase = beta_exp * t_exp + gamma_exp
        g_dot = (alpha_exp * beta_exp * torch.cos(phase)).sum(dim=-1)

        return g_dot
```

### Simplified Physics Residual (Internal Forces Only, No Friction)

```python
def simplified_cosserat_f_ssm(
    state: torch.Tensor,   # (B, 124)
    action: torch.Tensor,  # (B, 5)
    phase: torch.Tensor,   # (B, 60)
    rod_params: dict,
) -> torch.Tensor:
    """Simplified Cosserat rod RHS -- internal forces only.

    This omits friction (the hard part) and serves as a partial physics constraint.

    Returns: (B, 124) time derivatives.
    """
    B = state.shape[0]

    # Unpack state
    pos_x = state[:, 0:21]      # (B, 21)
    pos_y = state[:, 21:42]     # (B, 21)
    vel_x = state[:, 42:63]     # (B, 21)
    vel_y = state[:, 63:84]     # (B, 21)
    psi = state[:, 84:104]      # (B, 20) element yaws
    omega_z = state[:, 104:124] # (B, 20) angular velocities

    ds = rod_params['length'] / rod_params['n_elements']
    B_bend = rod_params['bending_stiffness']
    rho_A = rod_params['density'] * rod_params['cross_section_area']
    rho_I = rod_params['density'] * rod_params['second_moment_of_area']

    # Discrete curvature: kappa_i = (psi_{i+1} - psi_i) / ds
    kappa = (psi[:, 1:] - psi[:, :-1]) / ds  # (B, 19)

    # Rest curvature from CPG (extract from phase encoding)
    # kappa_rest = A * sin(k*s_j + omega*t + phi)
    kappa_rest = extract_rest_curvature(action, phase, rod_params)  # (B, 20)

    # Internal moment: m_i = B * (kappa_i - kappa_rest_i)
    # Moment at interior elements
    kappa_rest_interior = 0.5 * (kappa_rest[:, :-1] + kappa_rest[:, 1:])  # (B, 19)
    moment = B_bend * (kappa - kappa_rest_interior)  # (B, 19)

    # Angular acceleration from moment balance (simplified)
    # omega_dot_i = (m_{i+1} - m_{i-1}) / (2*ds) / (rho*I)
    # For boundary elements, use one-sided differences
    omega_dot = torch.zeros(B, 20, device=state.device)
    omega_dot[:, 1:-1] = (moment[:, 1:] - moment[:, :-1]) / ds / rho_I

    # Position derivatives = velocities
    dx_dt = torch.cat([
        vel_x,       # d(pos_x)/dt
        vel_y,       # d(pos_y)/dt
        torch.zeros(B, 21, device=state.device),  # d(vel_x)/dt = accel (TODO: needs forces)
        torch.zeros(B, 21, device=state.device),  # d(vel_y)/dt = accel (TODO: needs forces)
        omega_z,     # d(psi)/dt = omega_z
        omega_dot,   # d(omega_z)/dt
    ], dim=-1)

    return dx_dt
```

Note: The `torch.zeros` for linear accelerations highlights the key gap -- computing accelerations requires the full force balance including friction, which is the hard part.

---

## Open Questions

1. **Is a partial physics loss (internal forces only) useful, or does omitting friction make it misleading?**
   - What we know: Internal forces dominate short-timescale dynamics; friction dominates long-timescale behavior (net locomotion).
   - What's unclear: Whether a partial physics constraint helps or hurts at the RL-step timescale (0.5s).
   - Recommendation: Experiment with the physics regularizer (Approach C) to find out empirically.

2. **Can RFT friction be made smoothly differentiable?**
   - What we know: The existing implementation uses sigmoid regularization at low speed. The tangent direction computation is smooth except at zero-length elements (which don't occur in practice).
   - What's unclear: Whether the friction gradients are numerically stable enough for PINN training.
   - Recommendation: If attempting DD-PINN, start with a simplified isotropic drag (F = -c*v) before adding anisotropic RFT.

3. **What is the right ansatz for the damping-dominated regime?**
   - What we know: The DD-PINN paper uses sinusoidal ansatz (optionally with exponential damping). Snake locomotion has significant damping from friction.
   - What's unclear: Whether the sinusoidal ansatz can represent the over-damped dynamics at low frequencies.
   - Recommendation: Use the damped variant (exponential * sinusoidal) with n_g >= 5 to provide enough representational capacity.

4. **Has any DD-PINN code been released since our last check?**
   - What we know: As of March 2026, no open-source DD-PINN implementation exists. The Krauss/Licher group at Leibniz Hannover has published papers but not code.
   - Recommendation: Check periodically; the publication of arXiv:2502.01916 (generalizable PINN surrogates) suggests the field is active.

5. **Would a simplified linearized Cosserat model work as a physics backbone?**
   - What we know: KNODE-Cosserat uses simplified physics + NN correction. A linearized rod model (small curvature, small deformation) has a closed-form solution that could serve as the ansatz.
   - What's unclear: Whether the linearization is accurate enough for the large-amplitude serpentine motion of the snake.
   - Recommendation: Test on straight-rod trajectories first (where linearization is valid) before extending to full locomotion.

---

## Sources

### Primary (HIGH confidence)

- **DD-PINN original paper**: Krauss, Habich, Bartholdt, Seel, Schappler. "Domain-decoupled Physics-informed Neural Networks with Closed-form Gradients for Fast Model Learning of Dynamical Systems." [arXiv:2408.14951](https://arxiv.org/abs/2408.14951)
- **DD-PINN Cosserat application**: Licher, Bartholdt, Krauss, Habich, Seel, Schappler. "Adaptive Model-Predictive Control of a Soft Continuum Robot Using a Physics-Informed Neural Network Based on Cosserat Rod Theory." [arXiv:2508.12681](https://arxiv.org/abs/2508.12681)
- **Generalizable PINN surrogates**: "Generalizable and Fast Surrogates: Model Predictive Control of Articulated Soft Robots using Physics-Informed Neural Networks." [arXiv:2502.01916](https://arxiv.org/abs/2502.01916)
- **KNODE-Cosserat**: Hsieh et al. "Knowledge-based Neural Ordinary Differential Equations for Cosserat Rod-based Soft Robots." [arXiv:2408.07776](https://arxiv.org/abs/2408.07776)
- **DeepXDE ODE system example**: [deepxde.readthedocs.io](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/ode.system.html)
- **PyElastica documentation**: [cosseratrods.org](https://www.cosseratrods.org/software/pyelastica/)
- **Project-internal**: friction.py, elastica_snake_robot.py, model.py, existing knowledge files

### Secondary (MEDIUM confidence)

- **Bensch et al.**: "Physics-Informed Neural Networks for Continuum Robots: Towards Fast Approximation of Static Cosserat Rod Theory." ICRA 2024. [ResearchGate](https://www.researchgate.net/publication/382979679)
- **Physics-Informed Split Koopman Operators**: 2025. [ResearchGate](https://www.researchgate.net/publication/395226187)
- **PINNs-Torch**: PyTorch PINN implementation. [GitHub](https://github.com/rezaakb/pinns-torch)

### Tertiary (LOW confidence -- needs validation)

- No published results on PINNs with RFT friction for locomotion found as of March 2026.
- No published DD-PINN code found as of March 2026.
- The claim that "PyElastica is not differentiable" is based on code inspection (NumPy+Numba); there may be unreleased work on a JAX port.
