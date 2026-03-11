# Phase 12: Explore Hamiltonian or Lagrangian Neural Networks for Snake Dynamics - Research

**Researched:** 2026-03-11
**Domain:** Physics-informed neural networks for dissipative mechanical systems
**Confidence:** MEDIUM

## Summary

Hamiltonian and Lagrangian neural networks (HNN/LNN) encode physical structure -- energy conservation, symplectic dynamics, dissipation -- directly into the network architecture. For the snake robot system, the key challenge is that the dynamics are **dissipative** (RFT friction c_t=0.01, c_n=0.05) and **actuated** (5-dim CPG actions), ruling out vanilla HNN/LNN which assume conservative, autonomous systems. Three viable architecture families exist for this setting:

1. **Dissipative SymODEN** -- a port-Hamiltonian neural ODE that jointly learns H (Hamiltonian), D (dissipation matrix), and g(q) (control input matrix). This is the most directly applicable architecture because it was designed for controlled dissipative mechanical systems. It has been tested on pendulum, cartpole, and acrobot (2-4 DOF).

2. **Dissipative Hamiltonian Neural Networks (D-HNN)** -- learns separate H and Rayleigh dissipation function D as two scalar-valued networks. Tested on 2D phase-space systems. Does NOT natively handle control inputs (no actuation term).

3. **KNODE-Cosserat** -- a hybrid approach specifically designed for Cosserat rod robots that augments a physics ODE with a neural correction term. This is the most directly relevant to our snake system but requires a differentiable physics baseline.

**Primary recommendation:** Use the **Dissipative SymODEN / port-Hamiltonian** formulation as the primary architecture, adapted for the snake's generalized coordinates. This handles dissipation, control inputs, and learns interpretable energy components (kinetic energy via M(q), potential energy via V(q), dissipation via D(q)). Implement from scratch in PyTorch using `torchdiffeq` for ODE integration, following the SymODEN equations but adapted for the snake's state space.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torchdiffeq | 0.2.5 | Differentiable ODE solvers (odeint, odeint_adjoint) | De facto standard for neural ODEs in PyTorch; GPU support, O(1) memory adjoint |
| torch | 2.10.0 | Neural network framework, autograd for Hessians/Jacobians | Already installed; torch.func for efficient Jacobian/Hessian computation |
| scipy | 1.17.1 | Baseline numerical integration for comparison | Already installed |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| wandb | 0.25.0 | Experiment tracking | All training runs |
| matplotlib | (installed) | Energy landscape visualization, phase portraits | Diagnostic plots |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| torchdiffeq | torchode | torchode has better batched parallel ODE solving but less community adoption; stick with torchdiffeq for ecosystem compatibility |
| Custom HNN | neuraloperator or DeepXDE | Overkill for this exploration; those target PDE operators, not ODE systems |
| Port-Hamiltonian | Plain neural ODE | Loses physics structure; already have MLP baseline in Phase 3 |

**Installation:**
```bash
pip install torchdiffeq==0.2.5
```

## Architecture Patterns

### Recommended Project Structure
```
aprx_model_elastica/
├── model.py                    # Existing MLP/Residual/Transformer surrogates
├── hnn_model.py                # NEW: HNN/LNN/port-Hamiltonian models
├── hnn_train.py                # NEW: Training loop for physics-structured models
├── hnn_config.py               # NEW: Config dataclasses for HNN variants
├── hnn_coordinates.py          # NEW: Coordinate transforms (rod state <-> generalized coords)
├── train_config.py             # Existing configs (extend for HNN)
├── dataset.py                  # Existing FlatStepDataset (reuse)
└── validate_data.py            # Existing
```

### Pattern 1: Port-Hamiltonian Neural ODE (Dissipative SymODEN)

**What:** Learn three neural networks that parameterize the port-Hamiltonian system:
- M_theta(q): positive-definite mass/inertia matrix (via Cholesky: M = L L^T)
- V_theta(q): scalar potential energy
- D_theta(q): positive semi-definite dissipation matrix (via Cholesky: D = L_d L_d^T)
- g_theta(q): control input coupling matrix

The equations of motion are:
```
dq/dt = M^{-1}(q) p
dp/dt = -dV/dq + (dM^{-1}/dq) p - D(q) M^{-1}(q) p + g(q) u
```

Where H(q,p) = 0.5 p^T M^{-1}(q) p + V(q).

**When to use:** Primary architecture for this phase. Handles dissipation and control natively.

**Example:**
```python
# Source: Adapted from SymODEN (Zhong et al., ICLR 2020) and Dissipative SymODEN
import torch
import torch.nn as nn
from torchdiffeq import odeint

class PortHamiltonianODE(nn.Module):
    """Port-Hamiltonian neural ODE for dissipative controlled systems.

    State: (q, p) in generalized coordinates
    Control: u (CPG action parameters)
    """
    def __init__(self, q_dim: int, u_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q_dim = q_dim
        self.u_dim = u_dim

        # Mass matrix network: q -> lower-triangular L, M = L L^T
        self.mass_net = nn.Sequential(
            nn.Linear(q_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, q_dim * (q_dim + 1) // 2),  # lower-tri elements
        )

        # Potential energy network: q -> scalar V
        self.potential_net = nn.Sequential(
            nn.Linear(q_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Dissipation matrix network: q -> lower-tri L_d, D = L_d L_d^T
        self.dissipation_net = nn.Sequential(
            nn.Linear(q_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, q_dim * (q_dim + 1) // 2),
        )

        # Control input matrix: q -> g(q) of shape (q_dim, u_dim)
        self.input_net = nn.Sequential(
            nn.Linear(q_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, q_dim * u_dim),
        )

    def _get_mass_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Returns positive-definite mass matrix M(q) via Cholesky."""
        L_flat = self.mass_net(q)  # (B, q_dim*(q_dim+1)//2)
        L = torch.zeros(q.shape[0], self.q_dim, self.q_dim, device=q.device)
        idx = torch.tril_indices(self.q_dim, self.q_dim)
        L[:, idx[0], idx[1]] = L_flat
        # Ensure positive diagonal
        diag_idx = range(self.q_dim)
        L[:, diag_idx, diag_idx] = torch.abs(L[:, diag_idx, diag_idx]) + 1e-4
        return L @ L.transpose(-1, -2)  # M = L L^T

    def forward(self, t, state, u=None):
        """ODE right-hand side: d[q,p]/dt."""
        q = state[..., :self.q_dim]
        p = state[..., self.q_dim:]

        M = self._get_mass_matrix(q)
        M_inv = torch.linalg.inv(M)

        # dq/dt = M^{-1} p
        dqdt = torch.einsum('bij,bj->bi', M_inv, p)

        # dV/dq via autograd
        q_grad = q.detach().requires_grad_(True)
        V = self.potential_net(q_grad).sum()
        dVdq = torch.autograd.grad(V, q_grad, create_graph=True)[0]

        # Dissipation: D(q) M^{-1} p
        D = self._get_dissipation_matrix(q)
        dissipation = torch.einsum('bij,bj->bi', D @ M_inv, p)

        # Control: g(q) u
        if u is not None:
            g = self.input_net(q).view(-1, self.q_dim, self.u_dim)
            control = torch.einsum('bij,bj->bi', g, u)
        else:
            control = 0.0

        dpdt = -dVdq - dissipation + control

        return torch.cat([dqdt, dpdt], dim=-1)
```

### Pattern 2: D-HNN (Dissipative Hamiltonian Neural Network)

**What:** Two scalar networks: H_theta(q,p) for conservative dynamics, D_theta(q,p) for dissipation. Simpler than port-Hamiltonian but does NOT handle control inputs.
**When to use:** As a comparison/ablation -- train on autonomous trajectories (no action conditioning) to study whether the snake's free dynamics are well-captured by Hamiltonian structure.

### Pattern 3: KNODE-Style Hybrid

**What:** Use the existing MLP surrogate as a "physics model" and add a neural ODE correction term. Not strictly HNN/LNN but leverages the same ODE integration framework.
**When to use:** If port-Hamiltonian approach struggles with 124-dim state, this provides a fallback that still uses neural ODE machinery.

### Critical Design Decision: Coordinate Reduction

The snake has a 124-dim state but this is NOT 124 independent generalized coordinates. The state is:
- pos_x(21), pos_y(21): node positions (constrained by rod inextensibility)
- vel_x(21), vel_y(21): node velocities
- yaw(20): element orientations
- omega_z(20): angular velocities

For HNN/LNN, we need generalized coordinates (q) and momenta (p):
- **q candidates**: yaw(20) are the natural generalized coordinates for a planar Cosserat rod (20 element orientations), plus head position (x, y) = 22 DOF
- **p candidates**: omega_z(20) relate to angular momentum, plus head linear momentum (2) = 22 DOF
- Total phase space: 44 dimensions (22 q + 22 p)

This coordinate reduction from 124 to 44 dimensions is ESSENTIAL. Position nodes are derived from yaw angles + head position via the rod's geometry (integration along arc length). This is the single most important architectural decision.

### Anti-Patterns to Avoid
- **Treating all 124 state dims as independent generalized coordinates:** The node positions are NOT independent -- they are constrained by the rod geometry. HNN/LNN on 124 dims would violate the constraint structure and fail.
- **Ignoring dissipation:** Vanilla HNN assumes energy conservation. Snake friction dissipates ~30-60% of kinetic energy per cycle. A vanilla HNN will produce growing trajectory errors.
- **Using raw yaw angles as coordinates:** Yaw angles wrap around 2pi. Use (cos(yaw), sin(yaw)) embedding (as SymODEN recommends for angular coordinates on S^1).
- **Training on single-step predictions only:** HNN/LNN should be trained on trajectory segments (multi-step) to leverage ODE integration structure. Single-step MSE training undermines the physics inductive bias.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ODE integration with backprop | Custom Euler/RK4 with manual gradients | `torchdiffeq.odeint` or `odeint_adjoint` | Adaptive step size, GPU support, O(1) memory adjoint method |
| Positive-definite matrix parameterization | Softplus on diagonal of arbitrary matrix | Cholesky factorization (L L^T) | Cholesky guarantees PD structure; softplus-on-diagonal does not |
| Jacobian/Hessian computation | Manual differentiation of energy functions | `torch.func.jacrev`, `torch.func.hessian` | PyTorch 2.x functional transforms are vectorized and efficient |
| Coordinate transforms (rod geometry) | Ad-hoc position reconstruction | Dedicated `hnn_coordinates.py` module with unit tests | Rod geometry reconstruction from yaw angles is error-prone; needs verified once |
| Symplectic integrator | Custom leapfrog from scratch | Use `torchdiffeq` with `method='dopri5'` (adaptive) | Symplectic integrators are tricky to implement correctly with variable mass matrices |

**Key insight:** The value of HNN/LNN is the inductive bias from physics structure, not custom ODE solvers. Use battle-tested integrators and focus engineering effort on the coordinate representation and network architecture.

## Common Pitfalls

### Pitfall 1: Hessian Computation Bottleneck
**What goes wrong:** LNN requires computing the Hessian of the Lagrangian w.r.t. velocities (d^2 L / d qdot^2) at every integration step. For q_dim=22, this is a 22x22 Hessian -- expensive.
**Why it happens:** The Euler-Lagrange equation requires solving M(q) * qddot = f(q, qdot), where M = d^2L/dqdot^2.
**How to avoid:** Use the HNN/port-Hamiltonian formulation instead of LNN. HNN uses first-order ODEs (dq/dt, dp/dt) requiring only Jacobians, not Hessians. If LNN is desired, use `torch.func.hessian` with batched vmap.
**Warning signs:** Training time per epoch > 10x the MLP baseline.

### Pitfall 2: ODE Integration Instability During Training
**What goes wrong:** Adaptive ODE solver (dopri5) takes thousands of steps or diverges, causing OOM or NaN gradients.
**Why it happens:** Untrained networks produce wildly wrong dynamics; adaptive solver tries to resolve apparent stiffness.
**How to avoid:** (1) Start with short integration horizons (1-2 steps, dt=0.01s) and gradually increase. (2) Use `odeint` with `options={'max_num_steps': 1000}` to cap solver steps. (3) Warm up with single-step MSE loss before multi-step trajectory loss.
**Warning signs:** `odeint` calls taking >1s per batch, NaN in gradients.

### Pitfall 3: Angular Coordinate Wrapping
**What goes wrong:** Yaw angles that cross 0/2pi boundary cause discontinuities in the learned energy landscape.
**Why it happens:** Network sees a sharp jump in input when angle wraps.
**How to avoid:** Embed angles as (cos(yaw), sin(yaw)) pairs. This maps S^1 -> R^2 continuously. The generalized coordinate dimension becomes 22 (head x,y) + 20*2 (cos/sin of yaw) = 62 for position-like coordinates, but the true DOF remains 22.
**Warning signs:** Large prediction errors near yaw = +/- pi.

### Pitfall 4: Mass Matrix Ill-Conditioning
**What goes wrong:** Learned mass matrix M(q) becomes near-singular, causing M^{-1} to explode.
**Why it happens:** Cholesky diagonal elements L_ii approach zero during training.
**How to avoid:** Add a minimum diagonal: `L_diag = softplus(L_diag_raw) + eps` with eps >= 1e-3. Also consider initializing M close to identity.
**Warning signs:** Large values in M^{-1}, gradient explosion, loss spikes.

### Pitfall 5: Dissipation Dominance
**What goes wrong:** Network learns to put all dynamics into the dissipation term D, making H trivial (constant). Loses the benefit of Hamiltonian structure.
**Why it happens:** Dissipation term is more flexible (no symplectic constraint), so optimizer finds it easier to fit.
**How to avoid:** (1) Pre-train H on short conservative segments (low-friction regime). (2) Add regularization: penalize ||grad_D|| / ||grad_H|| ratio. (3) Monitor H and D contributions separately during training.
**Warning signs:** H stays near-constant while D gradient dominates the dynamics.

### Pitfall 6: Scale Mismatch Between q and p
**What goes wrong:** Position coordinates (meters) and momentum coordinates (kg*m/s) have very different scales, causing optimization difficulties.
**Why it happens:** Physical units differ by orders of magnitude.
**How to avoid:** Normalize q and p independently before feeding to energy networks. Use the existing `StateNormalizer` adapted for generalized coordinates.
**Warning signs:** One set of coordinates dominates the loss.

## Code Examples

### Coordinate Transform: Rod State to Generalized Coordinates

```python
# Source: Project-specific — derived from Cosserat rod geometry
import torch
import math

def rod_state_to_generalized(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert 124-dim rod state to (q, p) generalized coordinates.

    Args:
        state: (B, 124) — [pos_x(21), pos_y(21), vel_x(21), vel_y(21), yaw(20), omega_z(20)]

    Returns:
        q: (B, 22) — [head_x, head_y, yaw_0, ..., yaw_19]
        p: (B, 22) — [head_px, head_py, L_0, ..., L_19]  (angular momenta)
    """
    pos_x = state[:, 0:21]    # node x positions
    pos_y = state[:, 21:42]   # node y positions
    vel_x = state[:, 42:63]   # node x velocities
    vel_y = state[:, 63:84]   # node y velocities
    yaw = state[:, 84:104]    # element orientations (20)
    omega_z = state[:, 104:124]  # angular velocities (20)

    # Generalized positions: head position + yaw angles
    q = torch.cat([pos_x[:, 0:1], pos_y[:, 0:1], yaw], dim=-1)  # (B, 22)

    # Generalized momenta: head momentum + angular momentum
    # (These are proportional to velocities; the mass matrix M(q) relates them)
    # For initial implementation, use velocities as proxy (M will learn the mapping)
    p = torch.cat([vel_x[:, 0:1], vel_y[:, 0:1], omega_z], dim=-1)  # (B, 22)

    return q, p

def generalized_to_rod_state(
    q: torch.Tensor, p: torch.Tensor, rod_length: float = 0.4, n_elements: int = 20
) -> torch.Tensor:
    """Reconstruct 124-dim rod state from generalized coordinates.

    Reconstructs node positions from head position + yaw angles using
    rod geometry (constant element length, orientation determines next node).
    """
    head_x = q[:, 0:1]
    head_y = q[:, 1:2]
    yaw = q[:, 2:]  # (B, 20)

    head_vx = p[:, 0:1]
    head_vy = p[:, 1:2]
    omega_z = p[:, 2:]  # (B, 20)

    dl = rod_length / n_elements  # element length
    B = q.shape[0]

    # Reconstruct node positions from head + cumulative yaw
    pos_x = [head_x]
    pos_y = [head_y]
    for i in range(n_elements):
        pos_x.append(pos_x[-1] + dl * torch.cos(yaw[:, i:i+1]))
        pos_y.append(pos_y[-1] + dl * torch.sin(yaw[:, i:i+1]))

    pos_x = torch.cat(pos_x, dim=-1)  # (B, 21)
    pos_y = torch.cat(pos_y, dim=-1)  # (B, 21)

    # Reconstruct velocities (approximate: differentiate position w.r.t. yaw changes)
    # Full velocity reconstruction requires M^{-1}(q) p — use network's mass matrix
    vel_x = torch.zeros(B, 21, device=q.device)
    vel_y = torch.zeros(B, 21, device=q.device)
    vel_x[:, 0] = head_vx.squeeze(-1)
    vel_y[:, 0] = head_vy.squeeze(-1)
    # Remaining velocities derived from omega_z and geometry (simplified)
    for i in range(n_elements):
        vel_x[:, i+1] = vel_x[:, i] - dl * torch.sin(yaw[:, i]) * omega_z[:, i]
        vel_y[:, i+1] = vel_y[:, i] + dl * torch.cos(yaw[:, i]) * omega_z[:, i]

    return torch.cat([pos_x, pos_y, vel_x, vel_y, yaw, omega_z], dim=-1)  # (B, 124)
```

### Training Loop Sketch for Port-Hamiltonian ODE

```python
# Source: Adapted from SymODEN training (Zhong et al., ICLR 2020)
from torchdiffeq import odeint

def train_step(model, batch, optimizer, dt=0.01, n_steps=4):
    """One training step for port-Hamiltonian neural ODE.

    Args:
        model: PortHamiltonianODE instance
        batch: dict with 'state' (B,124), 'action' (B,5), 'next_state' (B,124)
        dt: integration timestep
        n_steps: number of ODE integration steps per RL step
    """
    state = batch['state']
    action = batch['action']
    next_state_true = batch['next_state']

    # Convert to generalized coordinates
    q0, p0 = rod_state_to_generalized(state)
    q1_true, p1_true = rod_state_to_generalized(next_state_true)

    # Embed angles for network input
    q0_embedded = embed_angles(q0)  # (cos/sin for yaw dims)

    # Initial condition
    z0 = torch.cat([q0, p0], dim=-1)  # (B, 44)

    # Integrate ODE
    t_span = torch.linspace(0, dt * n_steps, n_steps + 1, device=state.device)

    # Wrap model to include control input
    def ode_func(t, z):
        return model(t, z, u=action)

    z_pred = odeint(ode_func, z0, t_span, method='dopri5',
                    options={'max_num_steps': 500})  # (n_steps+1, B, 44)

    # Loss on final state
    z1_pred = z_pred[-1]  # (B, 44)
    z1_true = torch.cat([q1_true, p1_true], dim=-1)

    loss = torch.nn.functional.mse_loss(z1_pred, z1_true)

    # Optional: energy regularization
    # H_pred = model.compute_hamiltonian(q0, p0)
    # loss += 0.01 * H_pred.var()  # encourage smooth energy landscape

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item()
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Vanilla HNN (conservative only) | Port-Hamiltonian / D-HNN with dissipation | 2020-2022 | Enables modeling of real physical systems with friction |
| Canonical coordinates required | SymODEN with angle embeddings (cos/sin) | ICLR 2020 | Handles rotational DOFs properly |
| LNN with expensive Hessians | HNN/port-Hamiltonian (first-order ODEs) | 2020+ | Avoids Hessian computation; faster training |
| Black-box neural ODE | KNODE hybrid (physics + neural correction) | 2024 | Best accuracy for Cosserat rods specifically |
| Fixed integration scheme | Adaptive ODE solvers (dopri5) with adjoint | 2018+ | Memory-efficient, adaptive accuracy |
| Low-dimensional demos only (2-4 DOF) | Tested up to 40D (sPHNN, 2025) | 2025 | Encouraging for our 22-DOF formulation |

**Deprecated/outdated:**
- Plain LNN without dissipation handling: superseded by DeLaN with friction model
- SympNet for high-dimensional problems: known scalability issues
- Manual symplectic integrators for training: torchdiffeq adaptive solvers are more robust

## Open Questions

1. **Scalability to 22 DOF**
   - What we know: sPHNN has been tested up to 40D state space (thermal processing); D-HNN only tested on 2D. SymODEN tested on 4 DOF (acrobot).
   - What's unclear: Whether the mass matrix parameterization (22x22 PD matrix = 253 Cholesky parameters) will train stably. The 22-DOF snake is in between tested ranges.
   - Recommendation: Start with a reduced model (e.g., 5 elements = 7 DOF) to validate the approach, then scale up.

2. **Correct Momentum Definition**
   - What we know: Angular velocity omega_z is available in the data. True angular momentum L_i = I_i * omega_z_i where I_i is element moment of inertia.
   - What's unclear: Whether to use omega_z directly as "momentum" (letting M(q) learn the inertia mapping) or compute physical momentum from rod properties.
   - Recommendation: Use omega_z and velocities as proxy momenta. The learned M(q) will absorb the inertia mapping. This avoids needing rod physical parameters.

3. **Multi-Step vs Single-Step Training**
   - What we know: HNN/LNN papers typically train on trajectory segments. Our data is single-step (state, action, next_state) transitions.
   - What's unclear: Whether single-step prediction (integrate one RL step) captures enough temporal structure for the ODE to learn meaningful dynamics.
   - Recommendation: Train on single RL steps initially (integrate over the RL step duration with multiple ODE substeps). If results are poor, collect or construct short trajectory sequences (4-8 consecutive steps).

4. **Action Embedding Strategy**
   - What we know: CPG actions (amplitude, frequency, wave_number, phase_offset, direction_bias) are held constant over one RL step. SymODEN uses u directly in dp/dt = ... + g(q)u.
   - What's unclear: Whether treating CPG params as constant external force is physically accurate -- CPG produces time-varying internal actuation patterns.
   - Recommendation: Start with the simple g(q)u formulation. If insufficient, add time-dependent actuation: g(q, t_cpg) * u where t_cpg encodes CPG phase within the step.

5. **Comparison Metrics Against MLP Baseline**
   - What we know: Phase 3 trains MLP surrogates evaluated on single-step MSE and per-component RMSE.
   - What's unclear: Fair comparison requires evaluating both models on the SAME metrics. HNN may win on multi-step rollout stability but lose on single-step MSE.
   - Recommendation: Evaluate on (1) single-step MSE in original 124-dim space, (2) multi-step rollout divergence, (3) energy conservation quality. Report all three.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already configured) |
| Config file | tests/ directory with existing test files |
| Quick run command | `python3 -m pytest tests/test_hnn.py -x -q` |
| Full suite command | `python3 -m pytest tests/ -x -q` |

### Phase Requirements -> Test Map

Phase 12 has no formal requirement IDs (TBD in ROADMAP). The following are derived from the phase description:

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| HNN-01 | Coordinate transform roundtrip (124 -> 44 -> 124) | unit | `pytest tests/test_hnn.py::test_coordinate_roundtrip -x` | No -- Wave 0 |
| HNN-02 | Port-Hamiltonian ODE forward pass produces valid gradients | unit | `pytest tests/test_hnn.py::test_phnn_forward -x` | No -- Wave 0 |
| HNN-03 | Mass matrix is positive definite | unit | `pytest tests/test_hnn.py::test_mass_matrix_pd -x` | No -- Wave 0 |
| HNN-04 | ODE integration does not diverge on random inputs | integration | `pytest tests/test_hnn.py::test_ode_stability -x` | No -- Wave 0 |
| HNN-05 | Training loop runs without error for 5 batches | smoke | `pytest tests/test_hnn.py::test_training_smoke -x` | No -- Wave 0 |
| HNN-06 | Energy decreases over time (dissipation working) | integration | `pytest tests/test_hnn.py::test_energy_dissipation -x` | No -- Wave 0 |

### Sampling Rate
- **Per task commit:** `python3 -m pytest tests/test_hnn.py -x -q`
- **Per wave merge:** `python3 -m pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_hnn.py` -- covers HNN-01 through HNN-06
- [ ] Framework install: `pip install torchdiffeq==0.2.5`

## Sources

### Primary (HIGH confidence)
- [torchdiffeq GitHub](https://github.com/rtqichen/torchdiffeq) -- ODE solver API, version 0.2.5, PyTorch compatibility
- [Dissipative HNN paper](https://ar5iv.labs.arxiv.org/html/2201.10085) -- D-HNN architecture, Rayleigh dissipation parameterization
- [SymODEN paper](https://arxiv.org/html/1909.12077v5) -- Port-Hamiltonian with control, angle embedding, tested systems
- [sPHNN paper + code](https://arxiv.org/html/2502.02480) -- Stable port-Hamiltonian, up to 40D, Lyapunov stability, code at github.com/CPShub/sphnn-publication
- [D-HNN code](https://github.com/greydanus/dissipative_hnns) -- Reference implementation of D-HNN

### Secondary (MEDIUM confidence)
- [KNODE-Cosserat](https://arxiv.org/html/2408.07776v2) -- Hybrid physics+neural ODE for Cosserat rods, 58.7% accuracy improvement, PyTorch
- [Dissipative SymODEN](https://www.researchgate.net/publication/339399030_Dissipative_SymODEN_Encoding_Hamiltonian_Dynamics_with_Dissipation_and_Control_into_Deep_Learning) -- Extension of SymODEN for dissipative controlled systems
- [DeLaN](https://github.com/milutter/deep_lagrangian_networks) -- Deep Lagrangian Networks with friction, JAX/PyTorch, 2-DOF robot tested
- [ICLR 2025 Port-Hamiltonian](https://proceedings.iclr.cc/paper_files/paper/2025/file/abf731c2993f9b1ee417cc3734787d7a-Paper-Conference.pdf) -- Latest port-Hamiltonian architecture with dissipation and control
- [Pseudo-Hamiltonian NN](https://arxiv.org/abs/2206.02660) -- State-dependent external forces, symmetric integration scheme

### Tertiary (LOW confidence)
- [High-dimensional HNN](https://arxiv.org/abs/2008.04214) -- Claims HNNs work for many-dimensional systems but details unclear from abstract
- Scalability of mass matrix parameterization to 22 DOF -- no direct evidence found; extrapolated from sPHNN 40D result

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- torchdiffeq is the established tool, well-tested with PyTorch 2.x
- Architecture: MEDIUM -- port-Hamiltonian formulation is well-established for low-DOF systems; adaptation to 22-DOF Cosserat rod with CPG actuation is novel and untested
- Coordinate reduction: MEDIUM -- the yaw-based generalized coordinates are physically motivated but the roundtrip reconstruction quality is unverified
- Pitfalls: HIGH -- well-documented in literature (mass matrix conditioning, ODE instability, angle wrapping)
- Scalability to 22 DOF: LOW -- largest tested port-Hamiltonian is 40D (sPHNN) but that was a thermal system, not a mechanical system with mass matrix

**Research date:** 2026-03-11
**Valid until:** 2026-04-11 (stable field; 30 days)
