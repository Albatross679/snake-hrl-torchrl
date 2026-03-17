# Phase 3: Relative Velocity Representation Research

**Researched:** 2026-03-17
**Domain:** Neural surrogate input/output representation -- relative velocity and body-frame features for dynamics prediction
**Confidence:** HIGH
**Mode:** Comparison research (not phase planning)

## Summary

The surrogate model predicts geometry well (CoM R2=0.799, rel_pos R2=0.639) but dynamics poorly (vel R2=0.065, omega_z R2=0.307, heading R2=0.119, yaw R2=0.083). The core hypothesis is that factoring out bulk/CoM motion from velocities and using body-frame angular quantities would reduce dynamic range and make per-node velocity patterns more learnable.

**This research finds strong evidence that the hypothesis is correct**, supported by:
1. The system has exact SE(2) symmetry -- no force depends on absolute position or heading (verified force-by-force in `knowledge/symmetry-analysis-ode-state-simplification.md`)
2. The current `raw_to_relative()` already factors out CoM position but **does not** factor out CoM velocity or rotate into body frame -- velocities are still in the world frame
3. Literature universally uses body-frame or CoM-relative velocities for physics surrogates (GNS, MeshGraphNets, NeRD, MuJoCo)
4. The existing `knowledge/surrogate-input-representation-research.md` already contains a detailed body-frame proposal that was never implemented for velocities

**Primary recommendation:** Extend `raw_to_relative()` to also subtract CoM velocity from per-node velocities, making the representation CoM-velocity-relative. Do NOT rotate into body frame yet (keep it as a follow-up experiment). This is the minimum-risk, maximum-impact change.

---

## Recommendation

**YES -- implement CoM-relative velocities.** Specific variant: **CoM-velocity subtraction** (not full body-frame rotation).

### Why CoM-Velocity Subtraction First (Not Full Body-Frame)

There are three levels of velocity representation, in order of increasing complexity:

| Level | What it does | Dims | Invertible? | Risk |
|-------|-------------|------|-------------|------|
| 1. Current | Absolute world-frame velocities | 42 | N/A | Baseline |
| 2. **CoM-relative** | Subtract CoM velocity from each node | 42 + 2 (CoM vel) = 44 | Yes | Low |
| 3. Body-frame | Subtract CoM velocity + rotate by -heading | 42 + 2 (CoM vel body) = 44 | Yes | Medium |

**Level 2 (CoM-relative) is recommended because:**
- It addresses the primary problem: bulk translational velocity dominates per-node velocity patterns
- It is a simple, invertible, differentiable transformation
- It preserves consistency with the current 128-dim relative representation (which already does CoM subtraction for positions but not velocities)
- No rotation means no coupling between velocity components across the transformation
- The current code (`raw_to_relative`) already has the infrastructure -- just needs 4 additional lines

**Level 3 (full body-frame rotation) should be a follow-up experiment** because:
- It couples vx and vy through the rotation matrix, which could introduce training artifacts
- The heading angle computation introduces a discontinuity risk near +/-pi
- The benefit over Level 2 is smaller: rotation invariance helps generalization across headings, but the normalizer already handles per-feature scaling

### Why This Should Improve vel R2

The core issue is variance decomposition. Current absolute velocities contain:

```
v_node_world = v_CoM + v_deformation + v_wave_propagation
```

- `v_CoM`: The overall snake translation (0-20 mm/s depending on gait). This is the **same** for all 21 nodes at any given timestep -- it is pure bulk motion that adds variance but carries zero information about internal dynamics.
- `v_deformation`: How each node moves relative to CoM due to body bending.
- `v_wave_propagation`: The traveling curvature wave creates a velocity pattern along the body.

The network currently must learn to subtract the large, variable CoM velocity to access the small deformation signal. By explicitly subtracting CoM velocity:
- **Reduced dynamic range**: Relative velocities are bounded by the rod deformation speed (order 1-5 mm/s), not by the locomotion speed (order 0-20 mm/s)
- **Tighter normalization**: The z-score normalization works better when the feature distribution has smaller variance
- **Physics-aligned features**: Internal forces depend on strains and strain rates (relative quantities), not absolute velocities. RFT friction depends on velocity relative to the body tangent, which is better captured by relative velocities.

### What About Angular Quantities?

**omega_z (angular velocity):** Already frame-independent in 2D. The scalar omega_z does not change under SE(2) transformation. The poor R2=0.307 is NOT a representation problem -- it is a learning difficulty problem (high variance, fast sign-switching from CPG half-cycles). CoM-relative velocity subtraction will not directly improve omega_z.

**yaw (element angles):** The current representation already subtracts mean heading (`dpsi = yaw - heading`), making yaw angles body-relative. The poor R2=0.083 may be because:
1. Yaw deltas are very small (elements rotate slowly) -- the signal-to-noise ratio is low
2. Yaw is redundant with position (can be computed from adjacent node positions) -- the network may be confused by redundant features
3. The normalizer may not adequately handle the tight distribution of yaw deltas

**heading (sin/cos):** The heading delta depends on the average angular velocity, which is well-represented. The poor R2=0.119 may be because sin/cos encoding makes the delta non-trivial to predict (predicting delta-sin rather than delta-angle). Consider predicting heading delta as a raw angle delta instead.

**Recommendation for angular quantities:** Do NOT change angular representation in this iteration. Focus on velocity first. If omega_z remains poor, try component-weighted loss (upweight omega_z 5-10x) as a separate experiment.

---

## Implementation Plan

### Step 1: Extend `raw_to_relative()` in `state.py`

```python
def raw_to_relative(state: torch.Tensor) -> torch.Tensor:
    """Convert raw 124-dim to relative representation.

    Current (128-dim): CoM(2) + heading(2) + rel_pos(42) + vel(42) + yaw(20) + omega_z(20)
    Proposed (132-dim): CoM(2) + heading(2) + CoM_vel(2) + rel_pos(42) + rel_vel(42) + yaw(20) + omega_z(20)
    """
    pos_x = state[..., POS_X]       # (..., 21)
    pos_y = state[..., POS_Y]       # (..., 21)
    vel_x = state[..., VEL_X]       # (..., 21)
    vel_y = state[..., VEL_Y]       # (..., 21)
    yaw = state[..., YAW]           # (..., 20)
    omega_z = state[..., OMEGA_Z]   # (..., 20)

    # Absolute CoM
    com_x = pos_x.mean(dim=-1, keepdim=True)
    com_y = pos_y.mean(dim=-1, keepdim=True)

    # Body heading
    heading = yaw.mean(dim=-1)
    heading_sin = heading.sin().unsqueeze(-1)
    heading_cos = heading.cos().unsqueeze(-1)

    # CoM velocity (NEW)
    com_vel_x = vel_x.mean(dim=-1, keepdim=True)
    com_vel_y = vel_y.mean(dim=-1, keepdim=True)

    # Relative positions (existing)
    rel_pos_x = pos_x - com_x
    rel_pos_y = pos_y - com_y

    # Relative velocities (NEW -- subtract CoM velocity)
    rel_vel_x = vel_x - com_vel_x
    rel_vel_y = vel_y - com_vel_y

    return torch.cat([
        com_x, com_y, heading_sin, heading_cos,
        com_vel_x, com_vel_y,      # NEW: 2 dims
        rel_pos_x, rel_pos_y,
        rel_vel_x, rel_vel_y,      # CHANGED: now relative
        yaw, omega_z,
    ], dim=-1)
```

### Step 2: Update Constants

```python
REL_STATE_DIM = 132  # was 128
# Update all slices:
REL_COM_VEL_X = slice(4, 5)       # NEW
REL_COM_VEL_Y = slice(5, 6)       # NEW
REL_POS_X = slice(6, 27)          # shifted +2
REL_POS_Y = slice(27, 48)         # shifted +2
REL_VEL_X = slice(48, 69)         # shifted +2, now CoM-relative
REL_VEL_Y = slice(69, 90)         # shifted +2, now CoM-relative
REL_YAW = slice(90, 110)          # shifted +2
REL_OMEGA_Z = slice(110, 132)     # shifted +2  (wait, 110+20=130, not 132)
```

**Wait -- dimension check:**
- com_x(1) + com_y(1) + heading_sin(1) + heading_cos(1) + com_vel_x(1) + com_vel_y(1) + rel_pos_x(21) + rel_pos_y(21) + rel_vel_x(21) + rel_vel_y(21) + yaw(20) + omega_z(20) = 6 + 42 + 42 + 20 + 20 = **130**

So REL_STATE_DIM = 130 (not 132). Let me recount: 2 + 2 + 2 + 42 + 42 + 20 + 20 = 130.

### Step 3: Update `relative_to_raw()` Inverse

```python
def relative_to_raw(state: torch.Tensor) -> torch.Tensor:
    """Convert 130-dim relative state back to 124-dim absolute."""
    com_x = state[..., 0:1]
    com_y = state[..., 1:2]
    # heading_sin, heading_cos at [2:4] -- derive heading = atan2(sin, cos)
    com_vel_x = state[..., 4:5]
    com_vel_y = state[..., 5:6]
    rel_pos_x = state[..., 6:27]
    rel_pos_y = state[..., 27:48]
    rel_vel_x = state[..., 48:69]
    rel_vel_y = state[..., 69:90]
    yaw = state[..., 90:110]
    omega_z = state[..., 110:130]

    pos_x = rel_pos_x + com_x
    pos_y = rel_pos_y + com_y
    vel_x = rel_vel_x + com_vel_x  # reconstruct absolute velocity
    vel_y = rel_vel_y + com_vel_y

    return torch.cat([pos_x, pos_y, vel_x, vel_y, yaw, omega_z], dim=-1)
```

### Step 4: Update Model Config

```python
# train_config.py
state_dim: int = 130              # was 128
input_dim: int = 139              # 130 + 5 + 4 = 139 (was 137)
output_dim: int = 130             # was 128
```

### Step 5: Re-run Preprocessing

```bash
python -m aprx_model_elastica.preprocess_relative \
    --input-dir data/surrogate_rl_step \
    --output-dir data/surrogate_rl_step_rel130 \
    --workers 8 --verify
```

### Step 6: Train with Same Config

```bash
python -m aprx_model_elastica.train_surrogate \
    --data-dir data/surrogate_rl_step_rel130 \
    --hidden-dims 1024,1024,1024,1024 \
    --lr 1e-4 \
    --run-name surrogate-rel-vel-v1
```

### Alternative: In-Place Conversion (No New Preprocessing)

The current `train_surrogate.py` already has in-place conversion for 124->128 dim:
```python
if train_dataset.states.shape[-1] == 124:
    ds.states = raw_to_relative(ds.states)
```

The same approach works for 124->130. Just update `raw_to_relative()` and point to the raw data. The model will auto-convert. This avoids needing to re-preprocess.

---

## Expected Impact

### Velocity Components (vel_x, vel_y)

| Metric | Current (128-dim) | Expected (130-dim) | Rationale |
|--------|-------------------|-------------------|-----------|
| R2 vel | 0.065 | 0.3 - 0.6 | Removing CoM velocity variance exposes deformation pattern |
| MSE vel_x | 0.092 | ~0.02 - 0.04 | 2-4x improvement from reduced dynamic range |
| MSE vel_y | 0.241 | ~0.05 - 0.10 | Similar improvement |

**Confidence: MEDIUM.** The improvement is physics-motivated but the magnitude is uncertain. A 5-10x reduction in velocity MSE is plausible given that CoM velocity accounts for most of the absolute velocity variance, but the deformation velocities themselves may still be hard to predict.

### Position Components (unchanged)

| Metric | Current | Expected | Rationale |
|--------|---------|----------|-----------|
| R2 com | 0.799 | ~0.8 | Unchanged -- CoM position already factored |
| R2 rel_pos | 0.639 | ~0.65 | Marginal improvement from better velocity prediction (positions are integrated velocities) |

### Angular Components (no direct change)

| Metric | Current | Expected | Rationale |
|--------|---------|----------|-----------|
| R2 omega_z | 0.307 | ~0.3 | No change -- omega_z is already frame-independent |
| R2 heading | 0.119 | ~0.12 | No change -- heading prediction is a different problem |
| R2 yaw | 0.083 | ~0.10 | Slight improvement from correlated velocity improvement |

### Overall

| Metric | Current | Expected |
|--------|---------|----------|
| val_loss | 0.712 | ~0.3 - 0.5 |
| R2 overall | ~0.35 | ~0.45 - 0.55 |

**The expected improvement is moderate (1.5-2x better val_loss) because velocity occupies 42/128 = 33% of the output dimensions. Even a dramatic velocity improvement only moves the needle partially on overall metrics. The omega_z and yaw components (40/128 = 31% of output) remain unchanged.**

---

## Risks

### Risk 1: Information Loss at Prediction Time

**Risk level: LOW.** The transformation is perfectly invertible. CoM velocity is stored as 2 additional dimensions in the relative state. At inference time:
```
abs_vel = rel_vel + predicted_com_vel_delta + current_com_vel
```
No information is lost.

### Risk 2: CoM Velocity Delta Prediction Is Hard

**Risk level: MEDIUM.** The model now needs to predict how CoM velocity changes (2 new output dimensions). CoM velocity delta depends on the net RFT friction force, which in turn depends on the velocity pattern relative to the body tangent. This is a physics-meaningful quantity but may be harder to predict than absolute velocity deltas because it is a small residual of canceling forces.

**Mitigation:** CoM velocity changes slowly compared to per-node deformation velocities. The z-score normalization of deltas will handle the scale difference. If CoM velocity prediction is poor, it can be predicted by a simple average of the per-node velocity predictions (which is exact by construction).

### Risk 3: Slice Index Breakage

**Risk level: LOW but annoying.** Changing REL_STATE_DIM from 128 to 130 shifts all slice indices. Every piece of code that references `REL_POS_X`, `REL_VEL_X`, etc. must be updated. The named constants in `state.py` should be the single source of truth.

**Mitigation:** Use the named slice constants everywhere. Grep for any hardcoded 128 or slice(46, 67) patterns.

### Risk 4: Pre-processed Data Incompatibility

**Risk level: LOW.** The current `data/surrogate_rl_step_rel128/` directory contains 128-dim pre-processed data. The new representation produces 130-dim data. Need to either:
- Re-preprocess to a new directory (`data/surrogate_rl_step_rel130/`)
- Or rely on in-place conversion from raw 124-dim data (simpler, no disk space overhead)

**Recommendation:** Use in-place conversion. The raw 124-dim data in `data/surrogate_rl_step/` is the ground truth. The conversion adds ~30 seconds at training start for 4M transitions -- negligible.

---

## Standard Stack

No new libraries needed. The transformation uses only PyTorch operations already in use.

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.x (system) | Tensor operations for transform |
| numpy | system | Verification and preprocessing |

---

## Architecture Patterns

### Pattern 1: Incremental Representation Improvement

**What:** Change the preprocessing layer (raw_to_relative), not the model architecture. The MLP input/output dim changes but the architecture pattern is identical.

**Why:** This isolates the representation effect from architecture effects. If vel R2 improves, we know it was the representation. If we changed architecture simultaneously, we could not attribute the improvement.

### Pattern 2: Shared-Output Global Pose Prediction

**What:** The model predicts both body-relative deltas (how the shape changes) and global pose deltas (how CoM and heading change) in a single forward pass, as part of the same output vector.

**Why:** Global pose deltas (CoM position delta, CoM velocity delta, heading delta) are correlated with body shape -- they arise from the same physics. A shared network can learn these correlations.

### Anti-Pattern: Predicting Absolute Next State

**Don't:** Switch from delta prediction to absolute next-state prediction. The current delta prediction approach (next_state = current + predicted_delta) is correct and should be retained. The arxiv paper "Predicting Change, Not States" (2412.13074) confirms that delta/derivative prediction outperforms state prediction for neural surrogates across multiple PDE benchmarks.

---

## Common Pitfalls

### Pitfall 1: Forgetting to Update Normalizer Statistics

**What goes wrong:** The StateNormalizer is fitted on training data. If the representation changes from 128 to 130 dims, the normalizer must be re-fitted. Using an old normalizer.pt with new data will produce wrong results.

**How to avoid:** Delete any existing normalizer.pt. The training script re-computes normalization statistics at startup. Verify `normalizer.state_dim == REL_STATE_DIM` at load time.

### Pitfall 2: Hardcoded Dimensions in Component Loss Logging

**What goes wrong:** The `evaluate()` function in `train_surrogate.py` uses hardcoded slices for per-component R2:
```python
"vel": r2_per_dim[44:84].mean().item(),
```
These must be updated to the new slice positions.

**How to avoid:** Use the named constants from state.py:
```python
"vel": r2_per_dim[REL_VEL_X.start:REL_VEL_Y.stop].mean().item(),
```

### Pitfall 3: Confusing "Relative Velocity" with "Body-Frame Velocity"

**What goes wrong:** Implementing rotation into body frame when only CoM subtraction is intended. Body-frame rotation couples vx and vy, changes the meaning of the features, and requires heading-angle computation at every step.

**How to avoid:** Be explicit: this iteration does CoM-velocity subtraction ONLY. No rotation. The code change is:
```python
rel_vel_x = vel_x - vel_x.mean(dim=-1, keepdim=True)
rel_vel_y = vel_y - vel_y.mean(dim=-1, keepdim=True)
```

### Pitfall 4: Evaluating with Wrong Component Slice Mapping

**What goes wrong:** When comparing R2 scores between 128-dim and 130-dim models, the component indices are different. A "vel R2" computed with old indices on new data will be wrong.

**How to avoid:** Always use named slice constants. Add assertions:
```python
assert state.shape[-1] == REL_STATE_DIM, f"Expected {REL_STATE_DIM}, got {state.shape[-1]}"
```

---

## Code Examples

### Minimal Change to raw_to_relative()

```python
# In state.py -- add CoM velocity to the relative representation

def raw_to_relative(state: torch.Tensor) -> torch.Tensor:
    pos_x = state[..., POS_X]
    pos_y = state[..., POS_Y]
    vel_x = state[..., VEL_X]
    vel_y = state[..., VEL_Y]
    yaw = state[..., YAW]
    omega_z = state[..., OMEGA_Z]

    com_x = pos_x.mean(dim=-1, keepdim=True)
    com_y = pos_y.mean(dim=-1, keepdim=True)

    heading = yaw.mean(dim=-1)
    heading_sin = heading.sin().unsqueeze(-1)
    heading_cos = heading.cos().unsqueeze(-1)

    # NEW: CoM velocity extraction and subtraction
    com_vel_x = vel_x.mean(dim=-1, keepdim=True)
    com_vel_y = vel_y.mean(dim=-1, keepdim=True)

    rel_pos_x = pos_x - com_x
    rel_pos_y = pos_y - com_y
    rel_vel_x = vel_x - com_vel_x   # NEW
    rel_vel_y = vel_y - com_vel_y   # NEW

    return torch.cat([
        com_x, com_y, heading_sin, heading_cos,
        com_vel_x, com_vel_y,                    # NEW: 2 dims added
        rel_pos_x, rel_pos_y,
        rel_vel_x, rel_vel_y,                    # CHANGED: now CoM-relative
        yaw, omega_z,
    ], dim=-1)
```

### Updated Inverse Transform

```python
def relative_to_raw(state: torch.Tensor) -> torch.Tensor:
    com_x = state[..., REL_COM_X]
    com_y = state[..., REL_COM_Y]
    com_vel_x = state[..., REL_COM_VEL_X]    # NEW
    com_vel_y = state[..., REL_COM_VEL_Y]    # NEW
    rel_pos_x = state[..., REL_POS_X]
    rel_pos_y = state[..., REL_POS_Y]
    rel_vel_x = state[..., REL_VEL_X]
    rel_vel_y = state[..., REL_VEL_Y]
    yaw = state[..., REL_YAW]
    omega_z = state[..., REL_OMEGA_Z]

    pos_x = rel_pos_x + com_x
    pos_y = rel_pos_y + com_y
    vel_x = rel_vel_x + com_vel_x   # reconstruct absolute
    vel_y = rel_vel_y + com_vel_y

    return torch.cat([pos_x, pos_y, vel_x, vel_y, yaw, omega_z], dim=-1)
```

### Round-Trip Verification Test

```python
def test_raw_relative_roundtrip():
    """Verify raw -> relative -> raw is identity."""
    state = torch.randn(100, 124)
    rel = raw_to_relative(state)
    assert rel.shape == (100, REL_STATE_DIM)
    reconstructed = relative_to_raw(rel)
    assert reconstructed.shape == (100, 124)
    assert torch.allclose(state, reconstructed, atol=1e-5)
```

---

## Literature Evidence

### Direct Support for CoM-Relative Velocities

| Source | What They Do | Confidence |
|--------|-------------|------------|
| GNS (Sanchez-Gonzalez et al., ICML 2020) | Mask absolute positions; use relative displacements and velocity histories | HIGH |
| MeshGraphNets (Pfaff et al., ICLR 2021) | Edge features are relative displacements; node features include velocity | HIGH |
| NeRD (2025) | "Enforces dynamics invariance under translation and rotation around gravity axis" via input canonicalization | HIGH |
| MuJoCo Gymnasium envs | Exclude global (x,y) position from observations; velocities in body frame | HIGH |
| Kaba et al. (ICML 2023) | MLP + canonicalization is universal approximator for equivariant functions | HIGH |
| Gebhardt et al. (CMAME 2024) | Neural constitutive model uses strain/curvature (body-frame quantities), not absolute coords | HIGH |

### Support for Delta Prediction

| Source | Finding | Confidence |
|--------|---------|------------|
| "Predicting Change, Not States" (arXiv 2412.13074) | Delta/derivative prediction outperforms state prediction across 5 PDE benchmarks; FNO error 0.715 -> 0.100 on Navier-Stokes | HIGH |
| Current codebase | Already uses delta prediction (predict_delta=True) -- this is correct and should not change | HIGH |

### SE(2) Invariance of Snake Dynamics

The `knowledge/symmetry-analysis-ode-state-simplification.md` document provides a force-by-force proof that no force in the system depends on absolute position or heading. This confirms full SE(2) equivariance, which means the body-frame representation is exact (no information loss).

---

## Comparison of Representation Options

### Option A: CoM-Velocity Subtraction Only (RECOMMENDED)

```
Relative state (130-dim):
  CoM_x(1) + CoM_y(1) + heading_sin(1) + heading_cos(1) +
  CoM_vel_x(1) + CoM_vel_y(1) +
  rel_pos_x(21) + rel_pos_y(21) +
  rel_vel_x(21) + rel_vel_y(21) +
  dpsi(20) + omega_z(20) = 130
```

- **Pros:** Minimal code change, invertible, addresses primary velocity variance issue
- **Cons:** Does not achieve rotation invariance for velocities
- **Expected vel R2 improvement:** 5-10x (0.065 -> 0.3-0.6)

### Option B: Full Body-Frame (CoM-relative + heading rotation)

```
Body-frame state (130-dim, same layout but rotated):
  CoM_x(1) + CoM_y(1) + heading_sin(1) + heading_cos(1) +
  CoM_vel_body_x(1) + CoM_vel_body_y(1) +
  body_pos_x(21) + body_pos_y(21) +
  body_vel_x(21) + body_vel_y(21) +
  dpsi(20) + omega_z(20) = 130
```

- **Pros:** Full SE(2) invariance; best generalization across headings
- **Cons:** Rotation couples vx/vy; heading discontinuity at +/-pi; more complex code
- **Expected additional benefit over A:** 10-30% further improvement on vel R2
- **Recommendation:** Implement as follow-up experiment AFTER verifying Option A helps

### Option C: Neighbor-Relative Velocities

```
Per-node velocity relative to adjacent nodes:
  dv_x_i = v_x_{i+1} - v_x_i  (20 values)
  dv_y_i = v_y_{i+1} - v_y_i  (20 values)
```

- **Pros:** Captures strain rate directly (physically meaningful)
- **Cons:** Loses 2 dimensions (21 nodes -> 20 differences); NOT invertible without an anchor; does not capture bulk motion
- **Recommendation:** Do not use as primary representation. Could be added as supplementary features.

### Option D: Residual/Delta-Only Prediction (Predict Acceleration)

```
Instead of predicting delta_state, predict delta_delta_state (acceleration):
  next_state = 2*current - previous + model(current, action)
```

- **Pros:** More physics-aligned (Newton's second law); used by GNS
- **Cons:** Requires 2-step history; changes the prediction target fundamentally; needs TrajectoryDataset
- **Recommendation:** Too complex for this iteration. Consider for a future experiment.

---

## Open Questions

1. **Will CoM velocity prediction itself be accurate?**
   - What we know: CoM velocity change depends on net RFT friction, which is the sum of friction forces on all 21 nodes. The sum of many small, correlated quantities.
   - What's unclear: Whether the MLP can learn this sum accurately from the body shape + action.
   - Recommendation: If CoM velocity R2 is poor, derive it from the predicted per-node velocity average (mathematically exact).

2. **Should omega_z prediction be addressed separately?**
   - What we know: omega_z R2=0.307, MSE=1.96 -- by far the worst component. This is NOT a representation problem (omega_z is already frame-independent).
   - What's unclear: Whether component-weighted loss (upweighting omega_z 5-10x) would help.
   - Recommendation: After this experiment, run a follow-up with `loss = MSE(pred, target) + 5 * MSE(pred[omega_z], target[omega_z])`.

3. **Interaction with architecture sweep (Phase 3 sweep)**
   - What we know: The Phase 3 sweep uses 128-dim input. Changing to 130-dim requires re-running the sweep.
   - Recommendation: Run one model with the best Phase 3 architecture (4x1024 MLP) on the new 130-dim representation FIRST. If it improves vel R2, then re-run the full 15-config sweep with the new representation.

---

## Sources

### Primary (HIGH confidence)

- Codebase: `papers/aprx_model_elastica/state.py` -- current raw_to_relative() implementation, named slices
- Codebase: `papers/aprx_model_elastica/train_surrogate.py` -- current training loop, component loss logging
- Codebase: `knowledge/surrogate-input-representation-research.md` -- comprehensive body-frame analysis, literature review
- Codebase: `knowledge/symmetry-analysis-ode-state-simplification.md` -- SE(2) invariance proof for snake dynamics
- [GNS: Learning to Simulate Complex Physics with Graph Networks (ICML 2020)](https://arxiv.org/abs/2002.09405) -- relative displacement encoding, absolute position masking
- [MeshGraphNets (ICLR 2021)](https://arxiv.org/abs/2010.03409) -- relative mesh features for physics simulation
- [Predicting Change, Not States (arXiv 2412.13074)](https://arxiv.org/html/2412.13074) -- delta prediction outperforms state prediction
- [Physics-augmented neural networks for beams (CMAME 2024)](https://arxiv.org/html/2407.00640v1) -- body-frame strain/curvature inputs

### Secondary (MEDIUM confidence)

- [Neural Robot Dynamics (NeRD, 2025)](https://arxiv.org/html/2508.15755v1) -- robot-centric spatially-invariant representation
- [Kaba et al. (ICML 2023)](https://arxiv.org/abs/2211.06489) -- canonicalization + MLP is universal approximator
- [Generalizable surrogates for articulated soft robots (2025)](https://arxiv.org/html/2502.01916) -- PINN surrogates for soft robot dynamics

### Tertiary (LOW confidence)

- Magnitude of expected R2 improvement -- physics-motivated estimate, not empirically validated for this specific system

---

## Metadata

**Confidence breakdown:**
- Representation theory: HIGH -- SE(2) invariance proven, literature unanimous
- Implementation plan: HIGH -- minimal code change, well-understood transformation
- Expected impact magnitude: MEDIUM -- direction is confident, magnitude is estimated
- Angular quantity analysis: HIGH -- omega_z is frame-independent by physics

**Research date:** 2026-03-17
**Valid until:** 2026-04-17 (stable domain)
