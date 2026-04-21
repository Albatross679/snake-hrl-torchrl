# Phase 13: Implement PINN and DD-PINN Surrogate Models - Context

**Gathered:** 2026-03-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Implement physics-informed neural network approaches as alternative/complementary surrogate models for snake robot Cosserat rod dynamics. Three stages with user checkpoints between each:

1. **Stage 1: Physics regularizer** on existing MLP/Transformer surrogates (kinematic + moment + energy constraints)
2. **Stage 2: DD-PINN prototype** with damped sinusoidal ansatz, full RFT friction in f_SSM from the start
3. **Stage 3: Full DD-PINN** with adaptive collocation, loss balancing, and comprehensive comparison

Each stage pauses for human review before proceeding to the next. All stages use custom PyTorch (no DeepXDE).

Does NOT cover: multi-step trajectory validation (Phase 4), RL training with surrogate (Phase 5), vanilla PINN baseline (DD-PINN strictly dominates for ODE systems).

</domain>

<decisions>
## Implementation Decisions

### Implementation scope (LOCKED)
- All three stages: physics regularizer, DD-PINN prototype, full DD-PINN with friction
- **User checkpoint** gates between stages: after each stage, pause for human review before deciding whether to continue
- No DeepXDE dependency — custom PyTorch for everything
- DD-PINN has no open-source implementation; implement from scratch based on Krauss et al. papers

### DD-PINN ansatz (LOCKED)
- **Damped sinusoidal** basis functions: g_j(t) = sum_i alpha * exp(-delta*t) * [sin(beta*t + gamma) - sin(gamma)]
- 4 parameters per basis per state dimension (alpha, delta, beta, gamma)
- n_g basis functions per state (sweep n_g in {5, 7, 10})
- NN output dimension: 4 * 124 * n_g (2,480 for n_g=5)
- Guarantees g(a, 0) = 0 (exact initial condition satisfaction)
- Closed-form time derivative (no autodiff through network for dx/dt)

### Physics regularizer constraints (LOCKED — Stage 1)
- **All available physics laws** — implement every algebraic constraint that can be formulated without a full differentiable simulator:
  1. **Kinematic consistency**: delta_pos ≈ avg_vel * dt (velocity-position coupling for x, y)
  2. **Angular kinematic consistency**: delta_psi ≈ avg_omega * dt (angular velocity-yaw coupling)
  3. **Curvature-moment consistency**: omega_dot ∝ B*(kappa - kappa_rest) / (rho*I) (constitutive law)
  4. **Energy conservation**: delta(KE + elastic PE) ≈ work done by friction + actuation
- Sweep lambda_phys in {0.001, 0.01, 0.1, 1.0} with curriculum ramp (data-only first 20% of training, then ramp)

### Physics residual depth (LOCKED — Stages 2-3)
- **Full RFT friction from the start** in f_SSM (no simplified isotropic drag intermediate step)
- Differentiable anisotropic RFT: F_t = -c_t * v_tangential, F_n = -c_n * v_normal
- Regularized tangent direction computation (avoid gradient singularity at zero velocity)
- f_SSM must be **validated against PyElastica outputs** before use in training: run test states through both, compare to numerical tolerance
- Internal elastic forces: constitutive law n = S*(epsilon - epsilon_0), m = B*(kappa - kappa_0)
- CPG rest curvature: analytically differentiable (sin function)

### Nondimensionalization (LOCKED)
- **Physics-based nondimensionalization** (not z-score normalizer)
- Scale by physical reference values: length by rod length, time by dt (0.5s), force by characteristic elastic force B/L^2
- Keeps physics residual terms at O(1) magnitude
- Separate from the existing StateNormalizer (which handles model I/O normalization)

### Collocation strategy (LOCKED)
- Start with 100K collocation points (Sobol sampling in [0, 0.5s])
- Scale up to 500K, then 1M if physics loss doesn't converge
- **RAR (Residual-based Adaptive Refinement) from the start**: concentrate collocation points where physics residual is high
- Sobol quasi-random base sampling (consistent with data collection)

### Loss balancing (LOCKED)
- **ReLoBRaLo** (Relative Loss Balancing with Random Lookback)
- Adapts weights based on relative rate of change of each loss term
- Applied to all multi-term losses: data + physics (Stage 1), data + physics + collocation (Stage 2-3)

### Code organization (LOCKED)
- New **src/pinn/** package (separate from papers/aprx_model_elastica/)
- Files: ansatz.py, physics_residual.py, loss_balancing.py, collocation.py, models.py, train_pinn.py, regularizer.py, nondim.py
- Imports dataset, normalizer, and state utilities from papers/aprx_model_elastica/
- **Separate train_pinn.py** entry point (not a flag on existing train_surrogate.py)
- **Separate W&B project**: snake-hrl-pinn

### Model interface (LOCKED)
- DD-PINN models expose the **same forward(state, action, time_encoding) → delta** interface as existing surrogates
- Wraps the ansatz internally: forward() computes g(a, t=dt) to produce single-step delta
- Drop-in compatible with Phase 4 validation and Phase 5 RL training

### Evaluation (LOCKED)
- **Per-component RMSE in physical units**: pos_x/y (mm), vel_x/y (mm/s), yaw (rad), omega_z (rad/s)
- Same evaluation split as Phase 3 (enables direct comparison)
- Success bar for Stage 1: **any measurable improvement** on any component without >5% regression on others
- Baseline: use existing Phase 3 best checkpoint from output/surrogate/best/
- **Generate comparison plots**: per-component RMSE bar charts, predicted-vs-actual scatter, physics residual convergence curves
- Diagnostic figures saved to figures/pinn/

### Claude's Discretion
- Number of training epochs per stage
- Specific optimizer hyperparameters (lr, weight decay) within each stage
- Fourier feature sigma for spectral bias mitigation
- Whether to use causal training weights for DD-PINN
- Exact curriculum schedule for physics loss ramp
- Architecture of the NN that maps (x_0, u) → ansatz parameters in DD-PINN

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### PINN methodology and feasibility
- `knowledge/pinn-ddpinn-snake-locomotion-feasibility.md` — Detailed system-specific PINN/DD-PINN feasibility analysis, implementation paths, anti-patterns, and code examples
- `knowledge/neural-ode-pde-approximation-survey.md` — Broad survey of PINN/Neural ODE/operator methods
- `knowledge/knode-cosserat-hybrid-surrogate-report.md` — KNODE alternative approach (reference only, not in scope)

### Existing surrogate architecture
- `papers/aprx_model_elastica/model.py` — Current SurrogateModel, ResidualSurrogateModel, TransformerSurrogateModel implementations
- `papers/aprx_model_elastica/state.py` — State vector layout (124-dim raw, 130-dim relative), named slices (POS_X, VEL_X, YAW, OMEGA_Z)
- `papers/aprx_model_elastica/train_surrogate.py` — Training loop, auto-batch, early stopping, W&B logging
- `papers/aprx_model_elastica/dataset.py` — FlatStepDataset for Phase 02.2 data
- `papers/aprx_model_elastica/train_config.py` — SurrogateModelConfig dataclass

### Physics implementation reference
- `src/physics/elastica_snake_robot.py` — PyElastica Cosserat rod setup, rod parameters, time integration
- `src/physics/friction.py` — RFT friction implementation (NumPy+Numba, reference for PyTorch reimplementation)
- `src/physics/cpg/oscillators.py` — CPG rest curvature computation

### DD-PINN papers
- arXiv:2408.14951 — DD-PINN original paper (Krauss et al.), sinusoidal ansatz, closed-form gradients
- arXiv:2508.12681 — DD-PINN for Cosserat rod control (Licher et al.), 72-state system, 44,000x speedup
- arXiv:2502.01916 — Generalizable PINN surrogates for soft robot MPC

### Report sections
- `report/report.tex` §2 — Background on Cosserat rod PDEs, RFT friction, CPG actuation
- `report/dd-pinn-explanation.tex` — DD-PINN methodology explanation (standalone LaTeX)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `papers/aprx_model_elastica/state.py`: Named slices (POS_X, VEL_X, YAW, OMEGA_Z) for extracting state components — use in physics regularizer
- `papers/aprx_model_elastica/state.py:StateNormalizer`: z-score normalization with save/load — reuse for model I/O normalization
- `papers/aprx_model_elastica/dataset.py:FlatStepDataset`: Dataset loader for Phase 02.2 data — import for training
- `papers/aprx_model_elastica/model.py:SurrogateModel._init_weights()`: Zero-init output pattern — reuse for DD-PINN output
- `src/physics/friction.py`: RFT friction coefficients and formulas — translate to differentiable PyTorch
- `src/physics/cpg/oscillators.py`: CPG parameters and rest curvature formula — translate to PyTorch for physics residual
- `src/physics/elastica_snake_robot.py`: Rod physical parameters (B, rho, A, I, L, n_elements) — extract for nondimensionalization

### Established Patterns
- Delta prediction: next_state = current_state + model(normalized_input)
- CLI entry: `python -m <package>.<script> --flags`
- Checkpoint format: model.pt + normalizer.pt + config.json in save_dir
- W&B logging: log per-epoch metrics, configs at init
- Auto-batch: binary search for max GPU batch size

### Integration Points
- Reads from `data/surrogate_rl_step/` (Phase 02.2 dataset, via FlatStepDataset import)
- Baseline comparison: `output/surrogate/best/` (Phase 3 best model checkpoint)
- W&B project: snake-hrl-pinn (separate from snake-hrl-surrogate)
- Figures output: `figures/pinn/`
- DD-PINN model output: compatible with Phase 4 validation pipeline via same forward() interface

</code_context>

<specifics>
## Specific Ideas

- The key weakness of the current surrogate is omega_z prediction (R^2=0.23). Physics regularizer should specifically target angular dynamics via curvature-moment consistency.
- DD-PINN's damped sinusoidal ansatz is a natural fit for CPG-driven oscillatory dynamics — the snake's motion is fundamentally sinusoidal.
- f_SSM validation against PyElastica is critical safety check — bugs in the physics residual silently corrupt training.
- RFT friction is the hardest part of f_SSM. The existing `src/physics/friction.py` uses sigmoid regularization at low speeds — translate this approach to PyTorch.
- Physics-based nondimensionalization should use: L_ref = rod_length (~0.3m), t_ref = dt (0.5s), F_ref = B/L^2 (characteristic elastic force).

</specifics>

<deferred>
## Deferred Ideas

- Vanilla PINN baseline (DD-PINN strictly dominates for ODE systems — not worth implementing)
- PIKANs (Kolmogorov-Arnold Networks for PINNs) — promising but immature for 124-state systems
- Separable PINNs — useful for spatial PDEs, not for ODE systems (only 1 independent variable: time)
- Inverse problem: using PINN to estimate rod stiffness/friction parameters from trajectory data
- DeepXDE integration — decided against, using custom PyTorch
- KNODE hybrid approach — documented separately in knowledge/, could be a future phase

</deferred>

---

*Phase: 13-implement-pinn-and-dd-pinn-surrogate-models*
*Context gathered: 2026-03-17*
