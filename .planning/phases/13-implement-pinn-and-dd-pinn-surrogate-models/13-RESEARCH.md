# Phase 13: Implement PINN and DD-PINN Surrogate Models - Research

**Researched:** 2026-03-17
**Domain:** Physics-Informed Neural Networks (PINNs) for PDE/ODE Systems -- General Methodology and Ecosystem
**Confidence:** HIGH (core methodology, standard stack) / MEDIUM (SOTA 2025 advances, project-specific application)

## Summary

Physics-Informed Neural Networks (PINNs) embed PDE/ODE residuals into the neural network loss function via automatic differentiation, enabling meshfree PDE solving with sparse data. The field has matured significantly since Raissi et al. (2019), with critical advances in loss balancing (NTK-based weighting, GradNorm, ReLoBRaLo), causal training for time-dependent problems, domain decomposition (XPINN, FBPINN, DD-PINN), spectral bias mitigation (Fourier features, SIREN), and separable architectures for high-dimensional problems.

The standard training recipe is well-established: Adam optimizer (lr=1e-3) for 10-20K epochs followed by L-BFGS-B refinement, with adaptive collocation point sampling and multi-term loss balancing. The PINNacle benchmark (NeurIPS 2024) systematically evaluated ~10 methods across 20+ PDEs, finding that NTK-based loss weighting and domain decomposition are the most consistently beneficial techniques. For ODE systems specifically (the snake robot's case after spatial discretization), DD-PINN with sinusoidal ansatz provides exact initial condition satisfaction and closed-form time derivatives, avoiding the expensive autodiff-through-network computation of vanilla PINNs.

The project already has extensive feasibility analysis for the specific snake robot application (see `knowledge/pinn-ddpinn-snake-locomotion-feasibility.md`). This research document covers the GENERAL PINN methodology, ecosystem, and best practices beyond the project-specific context, providing the planner with prescriptive guidance on how PINNs work, what libraries to use, and what pitfalls to avoid.

**Primary recommendation:** Use DeepXDE 1.15.0 (PyTorch backend) for standard PINN experimentation and prototyping. For the project-specific DD-PINN, implement custom PyTorch code since no DD-PINN library exists. Start with a physics regularizer approach (1-2 weeks) before attempting full PINN/DD-PINN (4-8 weeks). Use the two-phase optimizer (Adam then L-BFGS), adaptive loss balancing (ReLoBRaLo or NTK-based), and Fourier features for spectral bias mitigation.

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| DeepXDE | 1.15.0 | PINN framework for forward/inverse PDE/ODE problems | Most mature PINN library (Lu Lu, Yale); 5 backends; ODE system examples; SIAM Review paper; published 2025-12-05 on PyPI |
| PyTorch | 2.x (existing) | Autodiff backbone for physics residuals and custom PINN code | Already in project stack; native autograd for PDE residuals |
| torchdiffeq | 0.2.4+ | Neural ODE integration for KNODE-style hybrids | O(1) memory adjoint; GPU-native; mature (6.4K stars) |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| PINNs-Torch | latest | High-performance PyTorch PINN with CUDA Graphs | When DeepXDE overhead is too high; up to 9x faster than TF v1 |
| neuraloperator | 2.0.0 | FNO/TFNO for parametric PDE surrogates | When learning operator mappings across parameter families |
| scipy.stats.qmc | (bundled) | Latin Hypercube / Sobol collocation sampling | For generating collocation points in spatiotemporal domains |
| wandb | existing | Experiment tracking for PINN training | Track multi-term losses, collocation sampling, convergence |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| DeepXDE | NVIDIA PhysicsNeMo (formerly Modulus) | PhysicsNeMo is production-grade but heavyweight; overkill for research-scale single-GPU work. Import path changed from `import modulus` to `import physicsnemo` in 2025 |
| DeepXDE | Custom PyTorch | More control but must hand-roll collocation sampling, loss balancing, diagnostics |
| DeepXDE | NeuralPDE.jl (Julia) | Strong for PDEs but requires Julia; incompatible with existing PyTorch pipeline |
| torchdiffeq | diffrax (JAX) | diffrax is more feature-rich but requires JAX ecosystem migration |
| Standard MLP | PIKANs (Kolmogorov-Arnold Networks) | PIKANs show promise with fewer params and noise robustness, but less mature; start with MLP |

**Installation:**
```bash
pip install deepxde torchdiffeq
# Set PyTorch backend for DeepXDE
export DDEBACKEND=pytorch
# Or set in ~/.deepxde/config.json: {"backend": "pytorch"}
```

## Architecture Patterns

### Recommended Project Structure
```
src/
  pinn/
    __init__.py
    ansatz.py           # DD-PINN sinusoidal ansatz (g(a,t))
    physics_residual.py # f_SSM: differentiable Cosserat rod RHS
    loss_balancing.py   # NTK-based / GradNorm / ReLoBRaLo adaptive weighting
    collocation.py      # Adaptive sampling strategies (RAR, R3, Sobol)
    models.py           # PINN network architectures (modified MLP, Fourier features)
    train_pinn.py       # Training loop with Adam + L-BFGS two-phase optimizer
    regularizer.py      # Physics loss regularizer for existing surrogate
    utils.py            # Nondimensionalization, state extraction
```

### Pattern 1: Vanilla PINN for PDE/ODE Systems

**What:** Network u_theta(x,t) directly approximates the PDE solution. Physics residual computed via autodiff: L[u_theta] - f = 0 at collocation points. Total loss = L_data + lambda_PDE * L_PDE + lambda_BC * L_BC + lambda_IC * L_IC.

**When to use:** Forward PDE problems, inverse parameter estimation, benchmark problems, initial prototyping.

**Example:**
```python
# Source: DeepXDE documentation (https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/ode.system.html)
import deepxde as dde

def ode_system(x, y):
    """Physics residual for Lotka-Volterra ODE system."""
    dy_dt = dde.grad.jacobian(y, x)  # autodiff: dy/dx where x=t
    alpha, beta, delta, gamma = 1.0, 0.1, 0.02, 0.5
    residual_1 = dy_dt[:, 0:1] - (alpha * y[:, 0:1] - beta * y[:, 0:1] * y[:, 1:2])
    residual_2 = dy_dt[:, 1:2] - (delta * y[:, 0:1] * y[:, 1:2] - gamma * y[:, 1:2])
    return [residual_1, residual_2]

geom = dde.geometry.TimeDomain(0, 25)
ic1 = dde.icbc.IC(geom, lambda x: 10, lambda _, on_initial: on_initial, component=0)
ic2 = dde.icbc.IC(geom, lambda x: 1, lambda _, on_initial: on_initial, component=1)
data = dde.data.PDE(geom, ode_system, [ic1, ic2], num_domain=2500, num_boundary=2)
net = dde.nn.FNN([1] + [64] * 3 + [2], "tanh", "Glorot normal")

model = dde.Model(data, net)
model.compile("adam", lr=1e-3)
model.train(epochs=15000)
model.compile("L-BFGS")
model.train()
```

### Pattern 2: DD-PINN with Sinusoidal Ansatz (for ODE Systems)

**What:** Separate the neural network from time evolution. NN predicts ansatz parameters (alpha, beta, gamma); ansatz provides closed-form time evolution: g_j(t) = sum_i alpha_ij * [sin(beta_ij*t + gamma_ij) - sin(gamma_ij)]. This guarantees g(a, 0) = 0 (exact IC satisfaction) and provides closed-form time derivative g_dot without autodiff.

**When to use:** Dynamic ODE systems where: (a) initial conditions must be exactly satisfied, (b) dynamics are periodic/oscillatory, (c) autodiff overhead is prohibitive. DD-PINN is 5-38x faster than vanilla PINN for ODE systems.

**Key advantage:** No autodiff through the network for time derivatives -- the ansatz provides g_dot(a,t) in closed form.

```python
import torch
import torch.nn as nn

class SinusoidalAnsatz(nn.Module):
    """DD-PINN sinusoidal ansatz: g(a, t) with g(a, 0) = 0.

    For each state dimension j, the ansatz is:
        g_j(t) = sum_i alpha_ij * [sin(beta_ij*t + gamma_ij) - sin(gamma_ij)]

    The NN outputs a = (alpha, beta, gamma) of shape (B, 3 * m * n_g).
    """
    def __init__(self, state_dim: int, n_basis: int = 5):
        super().__init__()
        self.state_dim = state_dim
        self.n_basis = n_basis
        self.param_dim = 3 * state_dim * n_basis

    def forward(self, params: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate ansatz. Returns (B, N_c, m) state deviations from x_0."""
        B, m, n_g = params.shape[0], self.state_dim, self.n_basis
        alpha = params[:, :m*n_g].reshape(B, m, n_g)
        beta  = params[:, m*n_g:2*m*n_g].reshape(B, m, n_g)
        gamma = params[:, 2*m*n_g:].reshape(B, m, n_g)

        t_exp = t[None, :, None, None]
        phase = beta[:, None, :, :] * t_exp + gamma[:, None, :, :]
        g = (alpha[:, None, :, :] * (torch.sin(phase) - torch.sin(gamma[:, None, :, :]))).sum(-1)
        return g  # (B, N_c, m)

    def time_derivative(self, params: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Closed-form g_dot. NO autodiff needed."""
        B, m, n_g = params.shape[0], self.state_dim, self.n_basis
        alpha = params[:, :m*n_g].reshape(B, m, n_g)
        beta  = params[:, m*n_g:2*m*n_g].reshape(B, m, n_g)
        gamma = params[:, 2*m*n_g:].reshape(B, m, n_g)

        t_exp = t[None, :, None, None]
        phase = beta[:, None, :, :] * t_exp + gamma[:, None, :, :]
        g_dot = (alpha[:, None, :, :] * beta[:, None, :, :] * torch.cos(phase)).sum(-1)
        return g_dot  # (B, N_c, m)
```

### Pattern 3: Physics Loss Regularizer on Existing Surrogate

**What:** Add soft physics constraints to the existing data-driven MLP/Transformer training loop. No architectural change. Only the loss function changes.

**When to use:** Lowest-risk physics incorporation. First step before attempting full PINN/DD-PINN.

```python
def physics_regularizer(state, delta_pred, dt=0.5):
    """Soft physics constraints -- no full differentiable simulator needed.

    Three constraint types that are valid for ANY dynamical system:
    1. Velocity-position consistency: delta_pos ~ avg_vel * dt
    2. Angular velocity-yaw consistency: delta_psi ~ avg_omega * dt
    3. Curvature-moment consistency (system-specific, partial)
    """
    next_state = state + delta_pred

    # 1. Kinematic consistency (valid for any system with pos/vel state)
    vel_x = state[..., 42:63]
    delta_x = delta_pred[..., 0:21]
    delta_vx = delta_pred[..., 42:63]
    loss_kin = F.mse_loss(delta_x, (vel_x + 0.5 * delta_vx) * dt)

    # 2. Angular kinematic consistency
    omega = state[..., 104:124]
    delta_psi = delta_pred[..., 84:104]
    delta_omega = delta_pred[..., 104:124]
    loss_ang = F.mse_loss(delta_psi, (omega + 0.5 * delta_omega) * dt)

    return loss_kin + loss_ang

# Training integration:
loss_data = F.mse_loss(model(state, action, phase), delta_target)
loss_phys = physics_regularizer(state, model(state, action, phase))
loss = loss_data + lambda_phys * loss_phys  # sweep lambda_phys in {0.001, 0.01, 0.1}
```

### Pattern 4: Modified MLP with Fourier Features (Spectral Bias Mitigation)

**What:** Replace raw coordinate inputs with Fourier feature embedding to overcome spectral bias -- the tendency of standard MLPs to learn low-frequency functions first and struggle with high-frequency content.

**When to use:** Any PINN problem with multi-scale or oscillatory solutions. Reported 30% error reduction in high-frequency regimes.

```python
class FourierFeatureEmbedding(nn.Module):
    """Random Fourier features for spectral bias mitigation.

    Source: Tancik et al. (2020), "Fourier Features Let Networks Learn
    High Frequency Functions in Low Dimensional Domains"
    """
    def __init__(self, input_dim, n_features=256, sigma=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(input_dim, n_features) * sigma,
                              requires_grad=False)

    def forward(self, x):
        proj = 2 * torch.pi * x @ self.B
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class ModifiedMLP(nn.Module):
    """Modified MLP for PINNs: Fourier features + residual connections.

    This addresses two key PINN failure modes:
    1. Spectral bias (via Fourier features)
    2. Vanishing gradients in deep networks (via residual connections)
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256, n_layers=4,
                 n_fourier=128, sigma=10.0):
        super().__init__()
        self.fourier = FourierFeatureEmbedding(input_dim, n_fourier, sigma)
        feat_dim = 2 * n_fourier
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.tanh(self.input_proj(self.fourier(x)))
        for layer in self.layers:
            h = h + torch.tanh(layer(h))  # Residual connection
        return self.output(h)
```

### Pattern 5: Two-Phase Optimizer (Adam + L-BFGS)

**What:** The standard PINN training recipe: Adam for global exploration, then L-BFGS for local refinement. This is the most consistently recommended approach across PINN literature.

**When to use:** Always. This is the default training strategy.

```python
# Phase 1: Adam for global exploration (10K-20K epochs)
model.compile("adam", lr=1e-3)
model.train(epochs=20000)

# Phase 2: L-BFGS for local refinement (until convergence)
model.compile("L-BFGS")
model.train()
```

### Pattern 6: Causal Training for Time-Dependent Problems

**What:** Assign temporal weights that enforce sequential learning: the model must converge at early times before being allowed to learn at later times.

**When to use:** Any time-dependent PDE/ODE where vanilla PINN shows propagation failure (good loss at late times but poor accuracy at early times).

```python
def causal_weights(t_colloc, residuals, epsilon=100.0):
    """Compute causal weights following Wang & Perdikaris (2024).

    w_i = exp(-epsilon * sum_{j<i} r_j^2)

    This forces the model to achieve low residual at early times
    before it can reduce residual at later times.
    """
    sorted_idx = torch.argsort(t_colloc)
    sorted_residuals = residuals[sorted_idx]

    cumsum = torch.cumsum(sorted_residuals ** 2, dim=0)
    weights = torch.exp(-epsilon * cumsum)

    # Unsort back to original order
    inverse_idx = torch.argsort(sorted_idx)
    return weights[inverse_idx]
```

### Anti-Patterns to Avoid

- **Using vanilla PINNs on high-dimensional ODE systems without DD-PINN:** Autodiff through the network for dx/dt is 5-38x slower than DD-PINN's closed-form derivatives. Use DD-PINN for any ODE system with >10 states.
- **Equal weighting of loss terms:** Naive lambda=1.0 for all loss components almost always fails. Use adaptive weighting (ReLoBRaLo, NTK-based, GradNorm) from the start.
- **Uniform collocation point sampling:** Wastes capacity on easy regions. Use adaptive (R3, failure-informed, RAR) or at minimum Latin Hypercube / Sobol sampling.
- **Ignoring nondimensionalization:** Raw physical units create loss terms with wildly different magnitudes. Nondimensionalize all inputs and outputs before PINN training.
- **Standard PINN for time-dependent problems without causal training:** Vanilla PINNs can satisfy PDE residuals at later times while getting early times wrong. Use causal weighting or time-marching.
- **Starting with full physics loss + data loss simultaneously:** Always use a curriculum: data-only first, then ramp physics loss gradually.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| PINN training loop | Custom collocation + residual computation | DeepXDE | Adaptive sampling, loss balancing, multiple BC types already solved |
| Loss balancing | Manual lambda tuning | NTK-based weighting (DeepXDE built-in) or ReLoBRaLo | Manual tuning fails for >2 loss terms; adaptive methods proven by PINNacle |
| Collocation point sampling | Uniform random | RAR (Residual-based Adaptive Refinement) in DeepXDE, or scipy.stats.qmc.Sobol | Adaptive sampling places points where residual is high |
| ODE integration in NN | Custom RK4 in PyTorch | torchdiffeq | Adjoint method, adaptive stepping, numerical stability |
| Fourier feature embedding | Custom sin/cos mapping | Use DeepXDE's built-in multi-scale Fourier or Pattern 4 above | Correct initialization of frequency matrix sigma matters |
| PINN benchmarking | Custom evaluation metrics | PINNacle benchmark suite | Standardized comparison across methods, 20+ PDE problems |
| Inverse problem setup | Custom parameter-as-variable | DeepXDE's `dde.Variable` API | Handles gradient flow to unknown parameters automatically |
| Ansatz time derivatives | Torch autograd through sin() | Closed-form analytic expression | The whole point of DD-PINN is avoiding autograd for time derivatives |

**Key insight:** The PINN *framework* (DeepXDE) handles the training mechanics. For this project, the hard part is formulating f_SSM -- the differentiable physics residual for Cosserat rods. No library solves that for you.

## Common Pitfalls

### Pitfall 1: Spectral Bias (Low-Frequency Preference)
**What goes wrong:** The PINN learns smooth, low-frequency solutions and fails to capture high-frequency oscillatory dynamics.
**Why it happens:** Standard MLP activations have frequency-dependent convergence rates that strongly favor low frequencies. Mathematically characterized through the Neural Tangent Kernel (NTK).
**How to avoid:** (1) Use Fourier feature inputs with sigma tuned to the problem's frequency range. (2) Use SIREN (sinusoidal activation: sin(omega_0 * Wx + b)) with omega_0=30. (3) For DD-PINN, the sinusoidal ansatz naturally captures oscillatory dynamics. (4) Use ASPEN (Adaptive Spectral Physics-Enabled Network) with learnable Fourier features.
**Warning signs:** Position predictions accurate but velocity/acceleration predictions poor. Smooth predictions where sharp features should exist. 30%+ error improvement reported when adding Fourier features.

### Pitfall 2: Loss Imbalance / Gradient Pathology
**What goes wrong:** PDE residual loss and data/BC/IC losses converge at different rates. One loss component dominates while others stagnate.
**Why it happens:** Different loss terms produce gradients of wildly different magnitudes. PDE residuals involve second derivatives (stiffer gradients) while data loss is first-order.
**How to avoid:** Use NTK-based adaptive weighting (Wang et al., 2021) -- PINNacle benchmark found this most consistently effective. Or ReLoBRaLo (Relative Loss Balancing with Random Lookback). Or GradNorm. Always use adaptive weighting, never manual lambdas.
**Warning signs:** Total loss decreases but one component flatlines or oscillates. Physics loss near zero but data loss high (or vice versa).

### Pitfall 3: Propagation Failure in Time-Dependent Problems
**What goes wrong:** PINN minimizes residuals at later times without properly solving earlier times. Solution satisfies PDE pointwise but violates temporal causality.
**Why it happens:** Optimization landscape allows "shortcut" solutions at later times.
**How to avoid:** Causal training (Wang & Perdikaris, 2024): temporal weights that exponentially decay with accumulated residual at earlier times. Or time-marching: solve [0, T/k] then [T/k, 2T/k] etc.
**Warning signs:** Low total loss but poor accuracy at early times. Solution "appears" at t>0 before properly evolving from IC.

### Pitfall 4: Stiff Systems Causing Training Instability
**What goes wrong:** For stiff PDEs/ODEs, training becomes extremely unstable or converges to incorrect solutions.
**Why it happens:** Stiff systems require representing dynamics at very different timescales simultaneously. Optimization landscape becomes ill-conditioned.
**How to avoid:** (1) Nondimensionalize to reduce stiffness ratio. (2) Curriculum learning: start simple, gradually increase stiffness. (3) Exact enforcement of initial conditions (critical for stiff systems -- not soft enforcement). (4) Sequence-to-sequence formulation instead of whole-domain prediction.
**Warning signs:** Loss oscillates wildly. Different random seeds give very different results. NFE explosion in Neural ODE context.

### Pitfall 5: Non-Differentiable Physics Simulators
**What goes wrong:** Attempting to compute PDE residuals using simulators that don't participate in PyTorch autograd (e.g., NumPy-based, C++ based, Fortran-based).
**Why it happens:** PINN physics loss requires gradients of the residual with respect to network parameters. These gradients must flow through the physics computation.
**How to avoid:** Reimplement the physics residual in pure PyTorch. Or use physics regularizer approach (Pattern 3) which needs only algebraic constraints, not a full differentiable simulator.
**Warning signs:** `RuntimeError: Can't call numpy() on Tensor that requires grad`. Gradients silently equal to zero. Loss decreasing but physics wrong.

### Pitfall 6: Collocation Point Budget
**What goes wrong:** Too few collocation points. Physics constraint satisfied at sample points but violated between them.
**Why it happens:** High-dimensional systems need proportionally more collocation points. Uniform sampling wastes budget on easy regions.
**How to avoid:** Start with 10K-50K collocation points. Use adaptive refinement (RAR) to concentrate in high-residual regions. DD-PINN paper uses 250K-1M for 72-state systems. DD-PINN collocation is cheap (no network forward pass, just ansatz evaluation).
**Warning signs:** Physics loss very low but validation error high. Adding more collocation points significantly changes the solution.

## PINN Methodology by PDE Type

### Elliptic PDEs (Poisson, Laplace, Elastostatics)
- **Character:** Steady-state, boundary value problems. Solution is smooth. Information propagates in all directions.
- **PINN approach:** Standard PINN works well. Collocation in spatial domain only.
- **Key tricks:** Domain decomposition (FBPINN/XPINN) for complex geometries. Moderate-depth networks (3-4 hidden layers). No causal training needed.
- **Difficulty for PINNs:** LOW -- this is where PINNs work best.

### Parabolic PDEs (Heat Equation, Diffusion)
- **Character:** Time-dependent, diffusive. Solution smooths over time. Information propagates at infinite speed.
- **PINN approach:** Standard PINN with causal training. Moderate difficulty.
- **Key tricks:** Time-marching to prevent propagation failure. Fourier features for multi-scale diffusion. Curriculum learning on diffusion coefficient.
- **Difficulty for PINNs:** MEDIUM -- causal training essential.

### Hyperbolic PDEs (Wave Equation, Advection)
- **Character:** Time-dependent, wave propagation. Can develop shocks/discontinuities. Information propagates at finite speed along characteristics.
- **PINN approach:** HARDEST for PINNs. Requires causal training + adaptive sampling + conservation-form losses.
- **Key tricks:** Viscosity solution regularization. R3 sampling near shock fronts. Weighted loss emphasizing conservation. CPINN along characteristics.
- **Difficulty for PINNs:** HIGH -- shocks and discontinuities fundamentally challenge smooth NN approximation.

### ODE Systems (Primary Case for Snake Robot)
- **Character:** After spatial discretization, PDEs become large ODE systems. Time is the only independent variable.
- **PINN approach:** DD-PINN strongly preferred over vanilla PINN. Closed-form time derivatives avoid autodiff overhead. Exact IC satisfaction.
- **Key tricks:** Sinusoidal ansatz for oscillatory dynamics. Curriculum on physics loss. Adaptive loss balancing.
- **Difficulty for PINNs:** MEDIUM -- DD-PINN makes this tractable.

### Stiff / Multi-Scale Systems
- **Character:** Multiple timescales (fast oscillations + slow drift). High condition number in the Jacobian.
- **PINN approach:** Curriculum learning + nondimensionalization + exact IC enforcement.
- **Key tricks:** Separable PINNs for computational tractability. Stiff-PINN techniques (analytical enrichment, stability-optimized time-stepping). Sequence-to-sequence formulation.
- **Difficulty for PINNs:** HIGH -- active research area.

## Domain Decomposition Methods

### XPINN (Extended PINN, Jagtap & Karniadakis 2020)
- **How:** Decompose spatial/temporal domain into arbitrary subdomains, each with its own small network. Interface conditions enforced via penalty terms.
- **Advantage:** Arbitrary decomposition shape, space+time parallelization, applicable to any PDE type.
- **Disadvantage:** Interface penalty weight tuning. Can have discontinuities at interfaces.
- **Best for:** Complex geometry problems, multi-physics problems with different physics in different regions.

### FBPINN (Finite Basis PINN, Moseley et al. 2023)
- **How:** Overlapping domain decomposition with smooth window functions. Each subdomain network's output is windowed, then summed globally.
- **Advantage:** Overlapping regions provide natural information exchange without hard interface constraints. Multilevel variants (2024) handle multi-scale problems. Consistently outperforms vanilla PINNs.
- **Disadvantage:** Overlap size is a hyperparameter. More networks = more parameters.
- **Best for:** Multi-scale problems, high-frequency solutions, problems where vanilla PINNs fail due to spectral bias.

### CPINN (Conservative PINN, Jagtap et al. 2020)
- **How:** Decompose along characteristics of conservation laws. Enforce flux continuity at interfaces.
- **Advantage:** Respects physical conservation at interfaces. Natural for conservation-law PDEs.
- **Disadvantage:** Only applicable to conservation-law PDEs. Cannot be applied to general PDEs.
- **Best for:** Fluid dynamics (Euler, Navier-Stokes), conservation laws with shocks.

### DD-PINN (Domain-Decoupled PINN, Krauss et al. 2024)
- **How:** For ODE systems: decouple the NN from time via a parametric ansatz. NN maps (x_0, u) to ansatz parameters; ansatz provides x(t) with exact IC.
- **Advantage:** 5-38x faster than vanilla PINN. Exact IC satisfaction. No autodiff for time derivatives. Collocation is cheap (only ansatz evaluation).
- **Disadvantage:** Ansatz must match solution character (sinusoidal for oscillatory, exponential for diffusive). Limited to ODE systems (not spatial PDEs). No open-source implementation as of March 2026.
- **Best for:** Large ODE systems from spatially-discretized PDEs, oscillatory dynamics, control problems. Demonstrated on 72-state Cosserat rod system with 44,000x speedup over direct simulation.

### DADD-PINN (Dual Adaptive Domain Decomposition, 2025)
- **How:** Automatically adapts subdomain boundaries and number of subdomains during training based on residual distribution.
- **Advantage:** No manual subdomain specification needed. Adapts to problem complexity.
- **Disadvantage:** Very new (2025), limited validation.

### When to Decompose
- Complex geometry with heterogeneous regions -> XPINN or FBPINN
- Multi-scale temporal dynamics -> time-domain XPINN or Multilevel FBPINN
- Large ODE system from spatial discretization -> DD-PINN (Krauss et al.)
- Conservation law with shocks -> CPINN
- Unknown problem structure -> DADD-PINN (automatic)

## Inverse Problems with PINNs

PINNs are naturally suited for inverse problems (parameter estimation) because unknown parameters can be treated as trainable variables alongside network weights.

**Workflow in DeepXDE:**
```python
import deepxde as dde

# Unknown parameter (initial guess)
C = dde.Variable(1.0)

def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, j=0)
    return dy_t - C * dy_xx  # C is trainable

# Add observed data as boundary condition
observe = dde.icbc.PointSetBC(observe_x, observe_y, component=0)
data = dde.data.TimePDE(geomtime, pde, [observe], ...)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=[C])
model.train(epochs=30000)
# C converges to true value
```

**Applications for this project:** Estimate rod stiffness parameters from trajectory data. Identify friction coefficients. Calibrate CPG model parameters.

## SOTA Advances (2024-2026)

| Method | Year | Key Contribution | Impact |
|--------|------|------------------|--------|
| Causal PINNs | 2024 | Temporal causality weights prevent propagation failure | Essential for time-dependent problems |
| Separable PINNs (SPINNs) | 2023-24 | Factorized networks using forward-mode AD, 60x speedup | Linear scaling with dimensionality |
| PIKANs | 2024 | Kolmogorov-Arnold Networks for PINNs | Fewer parameters, more robust to noise |
| PINNacle Benchmark | 2024 (NeurIPS) | Systematic comparison of ~10 methods on 20+ PDEs | NTK weighting + domain decomposition = best combo |
| FRES | 2024 | Dynamic Fourier feature generation | 30% error reduction in high-frequency regimes |
| R3 Sampling | 2023 (ICML) | Retain-Resample-Release adaptive collocation | Low-overhead adaptive sampling |
| Stiff-PINN | 2024-25 | Curriculum regularization, sequence-to-sequence | Robust on stiff chemical kinetics, boundary layers |
| DADD-PINN | 2025 | Automatic domain decomposition adaptation | Eliminates manual subdomain design |
| DD-PINN for Cosserat | 2025 | Dynamic Cosserat rod control at 70 Hz | 44,000x speedup over direct simulation |
| ASPEN | 2025 | Adaptive Spectral Physics-Enabled Network | Learnable Fourier features, auto-tunes spectral basis |
| Bayesian CPINN | 2025 | Uncertainty quantification in domain decomposition | Robust to 15% data noise |

### Deprecated / Outdated Approaches
- **Equal-weight loss terms:** Superseded by NTK-based / GradNorm / ReLoBRaLo adaptive weighting
- **Uniform random collocation:** Superseded by RAR, R3, failure-informed sampling
- **Vanilla PINN on time-dependent problems without causal weighting:** Superseded by causal PINNs
- **NVIDIA Modulus (old name):** Renamed to PhysicsNeMo in 2025; import path changed from `import modulus` to `import physicsnemo`
- **TensorFlow 1.x for PINNs:** DeepXDE still supports it but PyTorch backend is now preferred

## Code Examples

### Example 1: DeepXDE Inverse Problem (Parameter Estimation)
```python
# Source: https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse.html
import deepxde as dde
import numpy as np

C = dde.Variable(1.0)  # Unknown diffusion coefficient

def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, j=1)
    dy_xx = dde.grad.hessian(y, x, j=0)
    return dy_t - C * dy_xx

geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

observe_x = np.array([[0.25, 0.5], [0.5, 0.5], [0.75, 0.5]])
observe_y = np.array([[0.3], [0.2], [0.1]])
observe = dde.icbc.PointSetBC(observe_x, observe_y, component=0)

data = dde.data.TimePDE(geomtime, pde, [observe],
                         num_domain=2500, num_boundary=100,
                         num_initial=160, num_test=2500)
net = dde.nn.FNN([2] + [32] * 3 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)
model.compile("adam", lr=0.001, external_trainable_variables=[C])
model.train(epochs=30000)
print(f"Estimated C = {C.detach().numpy():.4f}")
```

### Example 2: Curriculum Training with Physics Loss Ramp
```python
# Source: Adapted from Wang & Perdikaris (2024)
def get_physics_weight(epoch, total_epochs, max_weight=0.1):
    """Curriculum: data-only -> ramp physics -> hold."""
    warmup_end = int(0.2 * total_epochs)
    ramp_end = int(0.5 * total_epochs)
    if epoch < warmup_end:
        return 0.0
    elif epoch < ramp_end:
        return max_weight * (epoch - warmup_end) / (ramp_end - warmup_end)
    else:
        return max_weight
```

### Example 3: Adaptive Loss Balancing (ReLoBRaLo)
```python
class ReLoBRaLo:
    """Relative Loss Balancing with Random Lookback.

    Adaptively weights multiple loss terms based on relative rate of change.
    """
    def __init__(self, n_losses, alpha=0.999, temperature=1.0):
        self.alpha = alpha
        self.temperature = temperature
        self.prev_losses = None
        self.weights = torch.ones(n_losses)

    def update(self, losses):
        current = torch.tensor([l.item() for l in losses])
        if self.prev_losses is None:
            self.prev_losses = current.clone()
            return self.weights
        ratios = current / (self.prev_losses + 1e-8)
        weights = torch.softmax(ratios / self.temperature, dim=0) * len(losses)
        self.weights = self.alpha * self.weights + (1 - self.alpha) * weights
        self.prev_losses = current.clone()
        return self.weights
```

### Example 4: Collocation Point Sampling
```python
from scipy.stats import qmc

def sample_collocation(n_points, t_start=0.0, t_end=0.5, method="sobol"):
    """Quasi-random collocation points for PINN temporal domain."""
    if method == "sobol":
        sampler = qmc.Sobol(d=1, scramble=True)
    elif method == "lhs":
        sampler = qmc.LatinHypercube(d=1)
    points = sampler.random(n_points)
    t = torch.tensor(qmc.scale(points, t_start, t_end).flatten(), dtype=torch.float32)
    return t.sort().values
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Equal loss weights | NTK-based / GradNorm / ReLoBRaLo adaptive | 2021-2024 | 10-100x accuracy improvement |
| Uniform collocation | RAR / R3 adaptive sampling | 2023-2024 | 2-10x efficiency gain |
| Vanilla PINN for time PDEs | Causal PINNs | 2024 | Eliminates propagation failure |
| Standard MLP | Modified MLP + Fourier features | 2022-2024 | Resolves spectral bias |
| Single-network PINN | Domain decomposition (XPINN/FBPINN) | 2023-2025 | Complex geometries, parallelizable |
| MLP-only | PIKANs (Kolmogorov-Arnold) | 2024 | Fewer parameters, more robust |
| Vanilla PINN for ODE systems | DD-PINN (Krauss et al.) | Aug 2024 | 5-38x training speedup |
| Adam only | Adam + L-BFGS two-phase | Established | Consistently best optimizer strategy |
| NVIDIA Modulus | NVIDIA PhysicsNeMo | 2025 | Renamed; new import paths |

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (existing in project) |
| Config file | tests/ directory with existing test files |
| Quick run command | `python -m pytest tests/test_pinn.py -x -q --timeout=60` |
| Full suite command | `python -m pytest tests/ -v --timeout=120` |

### Phase Requirements -> Test Map

Phase 13 requirements are TBD in ROADMAP. Based on research, the following test map covers core PINN behaviors:

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PINN-01 | Physics regularizer computes valid scalar loss | unit | `pytest tests/test_pinn.py::test_physics_regularizer -x` | No -- Wave 0 |
| PINN-02 | Physics regularizer gradients flow to model parameters | unit | `pytest tests/test_pinn.py::test_regularizer_gradients -x` | No -- Wave 0 |
| PINN-03 | DD-PINN ansatz satisfies g(a,0)=0 for any params | unit | `pytest tests/test_pinn.py::test_ansatz_ic -x` | No -- Wave 0 |
| PINN-04 | DD-PINN time derivative matches finite differences | unit | `pytest tests/test_pinn.py::test_ansatz_derivative -x` | No -- Wave 0 |
| PINN-05 | Fourier features improve high-freq reconstruction | unit | `pytest tests/test_pinn.py::test_fourier_features -x` | No -- Wave 0 |
| PINN-06 | DeepXDE ODE system setup runs without error | smoke | `pytest tests/test_pinn.py::test_deepxde_smoke -x` | No -- Wave 0 |
| PINN-07 | Loss balancing keeps terms within 10x of each other | integration | `pytest tests/test_pinn.py::test_loss_balancing -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_pinn.py -x -q --timeout=60`
- **Per wave merge:** `python -m pytest tests/ -v --timeout=120`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_pinn.py` -- covers PINN-01 through PINN-07
- [ ] `src/pinn/__init__.py` -- package structure
- [ ] Framework install: `pip install deepxde torchdiffeq` -- verify in environment

## Open Questions

1. **Has DD-PINN code been released since this research date?**
   - What we know: As of 2026-03-17, no open-source DD-PINN implementation exists (Krauss/Licher group at Leibniz Hannover).
   - What's unclear: Whether code accompanied arXiv:2502.01916 (generalizable PINN surrogates).
   - Recommendation: Check before implementation. The architecture is well-described enough to implement from scratch.

2. **Is a partial physics loss (without friction) useful for the snake system?**
   - What we know: Internal elastic forces dominate short-timescale dynamics; friction dominates locomotion.
   - What's unclear: Whether kinematic consistency constraints alone help at the 0.5s RL-step timescale.
   - Recommendation: This is the key experiment. The physics regularizer (Pattern 3) is the cheapest test.

3. **PIKANs vs modified MLP for this problem scale?**
   - What we know: PIKANs show promise with fewer parameters and noise robustness on benchmarks.
   - What's unclear: Maturity of PyTorch PIKAN implementations for 100+ state ODE systems.
   - Recommendation: Start with modified MLP (proven), explore PIKANs as stretch goal.

4. **What is the right ansatz for damping-dominated dynamics?**
   - What we know: DD-PINN uses sinusoidal ansatz. Snake locomotion has significant friction damping.
   - What's unclear: Whether sinusoidal ansatz with n_g>=5 basis functions can capture overdamped dynamics.
   - Recommendation: Use damped variant (exponential * sinusoidal) with n_g in {5, 7, 10}.

5. **Can Separable PINNs scale to 124-state ODE systems?**
   - What we know: SPINNs achieve 60x speedup by factorizing along dimensions.
   - What's unclear: Whether factorization is meaningful for ODE systems (only 1 independent variable: time).
   - Recommendation: SPINNs are most useful for spatial PDEs, not ODE systems. DD-PINN is the right approach for our case.

## Sources

### Primary (HIGH confidence)
- [DeepXDE GitHub](https://github.com/lululxvi/deepxde) - v1.15.0 on PyPI (2025-12-05)
- [DeepXDE ODE system example](https://deepxde.readthedocs.io/en/latest/demos/pinn_forward/ode.system.html)
- [DeepXDE inverse problem demos](https://deepxde.readthedocs.io/en/latest/demos/pinn_inverse.html)
- [PINNacle benchmark (NeurIPS 2024)](https://github.com/i207M/PINNacle) - 10 methods, 20+ PDEs
- [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) - renamed from Modulus, v25.08
- [Wang & Perdikaris, Respecting causality for training PINNs (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0045782524000690)
- [Separable PINNs (NeurIPS 2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/4af827e7d0b7bdae6097d44977e87534-Paper-Conference.pdf)

### Secondary (MEDIUM confidence)
- [From PINNs to PIKANs survey (2024)](https://arxiv.org/html/2410.13228v1) - comprehensive survey of recent PINN variants
- [PINNs-Torch](https://github.com/rezaakb/pinns-torch) - PyTorch PINN with CUDA Graphs, up to 9x speedup
- [Multilevel FBPINN (2024)](https://www.sciencedirect.com/science/article/pii/S0045782524003724) - domain decomposition
- [DADD-PINN (2025)](https://www.mdpi.com/2227-7390/14/4/744) - dual adaptive domain decomposition
- [Stiff-PINN approaches](https://www.emergentmind.com/topics/stiff-pinn) - multi-scale stiff systems
- [Two-Phase Optimization for PINNs (2024)](https://arxiv.org/html/2409.07296v1) - Adam then L-BFGS
- [Spectral bias analysis and mitigation (2025)](https://arxiv.org/html/2602.19265v1)
- [Physics-informed NNs comprehensive review (2025)](https://link.springer.com/article/10.1007/s10462-025-11322-7)
- [PINN with weighted loss for hyperbolic conservation laws (2025)](https://www.nature.com/articles/s41598-025-34263-1)
- [Multi-Objective Loss Balancing (2025)](https://www.sciencedirect.com/science/article/pii/S0045782525001860)

### Tertiary (LOW confidence -- needs validation)
- PIKANs for large ODE systems -- limited evidence beyond benchmark problems
- No DD-PINN open-source code found as of March 2026
- ASPEN (learnable Fourier features) -- single paper, 2025

### Project-Internal (HIGH confidence)
- `knowledge/pinn-ddpinn-snake-locomotion-feasibility.md` - detailed system-specific PINN/DD-PINN feasibility analysis
- `knowledge/neural-ode-pde-approximation-survey.md` - broad survey of PINN/Neural ODE/operator methods
- `knowledge/knode-cosserat-hybrid-surrogate-report.md` - KNODE alternative approach

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - DeepXDE and torchdiffeq well-established, versions verified on PyPI
- Architecture patterns: HIGH - DD-PINN, vanilla PINN, physics regularizer all well-documented in literature and project knowledge base
- Training tricks (loss balancing, causal, curriculum): HIGH - PINNacle benchmark provides systematic evidence; community consensus
- PDE-type-specific guidance: HIGH - well-established in numerical PDE literature
- Domain decomposition methods: MEDIUM-HIGH - XPINN/FBPINN well-documented; DD-PINN proven but no public code
- SOTA advances (PIKANs, ASPEN, DADD-PINN): MEDIUM - recent papers, limited reproduction evidence
- Pitfalls: HIGH - failure modes well-characterized in literature (Wang & Perdikaris, PINNacle, project feasibility study)

**Research date:** 2026-03-17
**Valid until:** 2026-04-17 (30 days -- stable field with incremental advances; check for DD-PINN code releases)
