# Issues Log

This document tracks issues encountered during development and simulation.

---

## Issue #1: Floor Friction Shape Mismatch Error

**Date:** 2024-01-26

**Status:** Open

**Component:** `dismech-python/src/dismech/external_forces/ground_contact.py`

### Description

When enabling `floorFriction` force in the environment alongside gravity and floor contact, the simulation crashes with a shape mismatch error.

### Error Message

```
ValueError: operands could not be broadcast together with shapes (21,3) (18,3)
```

### Stack Trace

```
File "dismech-python/src/dismech/time_steppers/time_stepper.py", line 270, in _compute_forces_and_jacobian
    F, J = compute_ground_contact_friction(robot, q, u)
File "dismech-python/src/dismech/external_forces/ground_contact.py", line 67, in compute_ground_contact_friction
    v_dot_n = np.sum(u_vec * n_hat, axis=1, keepdims=True)
ValueError: operands could not be broadcast together with shapes (21,3) (18,3)
```

### Steps to Reproduce

```python
import numpy as np
import dismech

env = dismech.Environment()
env.add_force('gravity', g=np.array([0.0, 0.0, -9.81]))
env.add_force('floorContact', ground_z=0, stiffness=1e3, delta=5e-3, h=1e-3)
env.add_force('floorFriction', mu=0.75, vel_tol=1e-3)  # This causes the error
env.add_force('rft', ct=0.01, cn=0.1)

# Create robot with 21-node geometry
geo = dismech.Geometry.from_txt('dismech-python/tests/resources/rod_cantilever/horizontal_rod_n21.txt')
robot = dismech.SoftRobot(geom, material, geo, sim_params, env)

stepper = dismech.ImplicitEulerTimeStepper(robot)
stepper.simulate(robot)  # Crashes here
```

### Analysis

The error occurs in `compute_ground_contact_friction()` where `u_vec` has shape `(21, 3)` (all 21 nodes) but `n_hat` has shape `(18, 3)`. This suggests that `n_hat` is computed for a subset of nodes (possibly only nodes in contact with the ground) while `u_vec` includes all nodes.

### Workaround

Disable floor friction when using gravity + floor contact:

```python
env = dismech.Environment()
env.add_force('gravity', g=np.array([0.0, 0.0, -9.81]))
env.add_force('floorContact', ground_z=0, stiffness=1e3, delta=5e-3, h=1e-3)
# env.add_force('floorFriction', mu=0.75, vel_tol=1e-3)  # Skip this
env.add_force('rft', ct=0.01, cn=0.1)
```

### Related Files

- `dismech-python/src/dismech/external_forces/ground_contact.py:67`
- `dismech-python/src/dismech/time_steppers/time_stepper.py:270`

---

## Issue #2: Gravity Simulation Without Floor Causes Unbounded Fall

**Date:** 2024-01-26

**Status:** Expected Behavior (Documented)

**Component:** Snake simulation

### Description

When gravity is enabled without floor contact, the snake falls indefinitely in the -Z direction. This is expected physical behavior but should be noted for users.

### Observation

After 5 seconds of simulation with gravity (g = -9.81 m/s²) and no floor contact:
- Initial Z position: 0.0m
- Final Z position: -1.85m

### Solution

Always enable `floorContact` when using gravity:

```python
env = dismech.Environment()
env.add_force('gravity', g=np.array([0.0, 0.0, -9.81]))
env.add_force('floorContact', ground_z=0, stiffness=1e3, delta=5e-3, h=1e-3)
```

---

## Issue #3: Simulation Convergence with Floor Contact

**Date:** 2024-01-26

**Status:** Resolved

**Component:** Time stepper

### Description

Initial simulation steps with floor contact show higher iteration counts and errors during the first few timesteps as the snake settles onto the ground.

### Observation

First timestep iterations with gravity + floor contact:
```
iter: 1, error: 8.871
iter: 2, error: 1.574
iter: 3, error: 0.431
```

Subsequent timesteps converge in 4 iterations:
```
iter: 1, error: 0.147
iter: 2, error: 0.991
iter: 3, error: 0.018
iter: 4, error: 0.020
```

### Resolution

This is normal behavior. The simulation stabilizes after the initial contact transient.

---

## Summary of Gravity Simulation Results

### Test Configurations

| Configuration | Result | Video |
|---------------|--------|-------|
| Gravity only (no floor) | Snake falls to Z=-1.85m | `Media/snake_with_gravity.mp4` |
| Gravity + Floor Contact | Works, snake stays on ground | `Media/snake_gravity_floor_no_friction.mp4` |
| Gravity + Floor + Friction | **CRASHES** (shape mismatch) | N/A |

### Recommended Configuration for Ground-Based Simulation

```python
env = dismech.Environment()
env.add_force('gravity', g=np.array([0.0, 0.0, -9.81]))
env.add_force('floorContact', ground_z=0, stiffness=1e3, delta=5e-3, h=1e-3)
# Do NOT add floorFriction - it has a bug
env.add_force('rft', ct=0.01, cn=0.1)  # RFT provides some lateral resistance
```

---

## Issue #4: DisMech Convergence Warnings Cause Extremely Slow RL Training

**Date:** 2026-01-27

**Status:** Open

**Component:** DisMech implicit solver / RL training with DIRECT curvature control

### Description

When training RL policies with DIRECT curvature control (19-dim action space), the DisMech implicit Euler solver frequently hits its iteration limit, causing extremely slow training. The warning "Iteration limit 25 reached before convergence" appears on nearly every simulation step.

### Impact

| Metric | Expected | Actual |
|--------|----------|--------|
| Training speed | ~100+ steps/second | ~2.7 steps/second |
| Time for 500k frames | ~1-2 hours | ~50+ hours |
| Usability | Practical | Impractical |

### Root Cause Analysis

The DIRECT control method allows the RL policy to output arbitrary curvature values for each of the 19 joints. When the policy outputs:
1. **High curvatures** (near the ±10 limit) - Creates large bending forces
2. **Rapidly changing curvatures** - Large forces at each step
3. **Non-smooth curvature profiles** - Discontinuities stress the solver

The implicit Euler solver uses Newton's method which struggles to converge when:
- Forces are large relative to stiffness
- Configuration changes significantly between steps
- The Jacobian becomes ill-conditioned

### Reproduction

```bash
# This will show many convergence warnings
PYTHONPATH=src python scripts/train_approach_curvature.py \
    --total-frames 10000 \
    --experiment-name test_convergence
```

### Observed Warnings

```
DisMech step warning: Iteration limit 25 reached before convergence
DisMech step warning: Iteration limit 25 reached before convergence
... (repeated on nearly every step)
```

### Potential Solutions

1. **Increase iteration limit** (partial fix):
   ```python
   PhysicsConfig(max_iter=50)  # or higher
   ```
   - May improve convergence but increases step time

2. **Add action smoothing/rate limiting**:
   ```python
   # Limit curvature change per step
   max_delta_curvature = 1.0
   new_action = np.clip(action,
                        prev_action - max_delta_curvature,
                        prev_action + max_delta_curvature)
   ```

3. **Use SERPENOID control instead of DIRECT**:
   - 4-5 dim action space produces smoother curvature profiles
   - Natural wave patterns are more stable

4. **Reduce action scale**:
   ```python
   EnvConfig(action_scale=0.5)  # Limit max curvature to 2.5
   ```
   - Reduces force magnitudes but limits policy capabilities

5. **Increase simulation timestep** (risky):
   ```python
   PhysicsConfig(dt=0.1)  # From 0.05
   ```
   - Fewer steps but may cause instability

6. **Pre-train with SERPENOID, fine-tune with DIRECT**:
   - Use stable control method for initial training
   - Transition to DIRECT for fine-grained control

### Workaround

For now, use SERPENOID or SERPENOID_STEERING control methods which produce smoother curvature profiles:

```python
from snake_hrl.configs.env import CPGConfig, ControlMethod

# Use 5-dim serpenoid steering instead of 19-dim direct
cpg_config = CPGConfig(control_method=ControlMethod.SERPENOID_STEERING)
```

### Related Files

- `src/snake_hrl/envs/base_env.py` - Action scaling and curvature control
- `src/snake_hrl/physics/snake_robot.py` - DisMech integration
- `scripts/train_approach_curvature.py` - Training script affected
- `dismech-python/src/dismech/time_steppers/` - Implicit solver

### Training Session Terminated

Training was stopped after ~20 minutes with only ~3,000 steps completed (of 500,000 target frames). No checkpoints were saved as the first batch (4,096 frames) was not completed.

---

## Issue #5: No Checkpoint Saved When Training Interrupted Before First Batch

**Date:** 2026-01-27

**Status:** Open

**Component:** PPOTrainer checkpoint mechanism

### Description

When training is interrupted (via SIGINT/SIGTERM) before completing the first batch of frames, no checkpoint is saved. The interrupt handler attempts to save but the training loop hasn't yielded to allow the signal to be processed.

### Expected Behavior

- Training interrupted → `interrupted.pt` checkpoint saved
- Allows resuming training from last state

### Actual Behavior

- Training interrupted before first batch → No checkpoint saved
- All progress lost

### Cause

The signal handler in `ppo.py` sets a flag (`self._shutdown_requested = True`) which is checked in the training loop. However, if the interrupt occurs during batch collection (before any loop iteration completes), the flag is never checked.

### Potential Fix

Add periodic checkpoint saving during batch collection, not just after batch completion:

```python
# In collector loop, save every N steps
if steps_collected % emergency_save_interval == 0:
    self._emergency_checkpoint()
```

### Workaround

Use smaller `frames_per_batch` to complete batches faster:

```bash
python scripts/train_approach_curvature.py \
    --frames-per-batch 1024 \  # Smaller batches
    --total-frames 50000
```
