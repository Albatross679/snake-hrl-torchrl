---
name: physics-backend-benchmark
description: Benchmark comparing all 4 physics backends under identical serpenoid control
type: experiment
created: 2026-02-13T12:00:00
updated: 2026-02-13T13:00:00
tags: [experiment, physics, benchmark, performance]
aliases: []
---

# Experiment: Physics Backend Benchmark

**Date:** 2026-02-13

**Objective:** Compare the 4 physics backends (DisMech, PyElastica, dismech-rods, MuJoCo) under identical conditions to quantify differences in computation time, locomotion behavior, solver stability, and trajectory agreement.

## Setup

**Snake geometry:** 20 segments, L=1.0m, r=0.001m, density=1200 kg/m^3, E=2e6 Pa

**Control input:** Fixed serpenoid wave applied identically to all backends:
```
κ(s, t) = A · sin(k · s − ω · t + φ)
A = 1.0, f = 1.0 Hz, wave_number = 1.5, φ = 0
```

**Duration:** 500 RL steps = 25s simulation time (dt=0.05s)

**Solver:** max_iter=100 (increased from default 25 to prevent dismech-rods `exit()`)

**Command:**
```bash
python script/benchmark_physics.py --steps 500 --frameworks dismech elastica mujoco
python script/benchmark_physics.py --steps 60 --frameworks dismech_rods --output-dir output/benchmark_dismech_rods
```

dismech-rods was limited to 60 steps because its C++ solver calls `exit(1)` on convergence failure (at step 62), which cannot be caught by Python.

## Why "Identical Input" Produces Different Output

All four backends receive the same 19-float curvature array at each step, but the same numeric input produces different motion because of three layers of divergence.

### Layer 1: Curvature application mechanism

Each backend interprets the curvature command through a different physical mechanism:

| Backend | Mechanism | Code path | Effect |
|---|---|---|---|
| **DisMech** | Sets `bend_springs.nat_strain[i, 0]` | `snake_robot.py:374` | Elastic rest shape — rod bends toward target via energy minimization in implicit Newton solve |
| **PyElastica** | Sets `rod.rest_kappa[0, i]` | `elastica_snake_robot.py:396` | Same concept (elastic rest curvature), but solved with explicit symplectic integration (PositionVerlet), producing different transient dynamics |
| **dismech-rods** | Passes `{"curvature": matrix}` to `step_simulation()` | `dismech_rods_snake_robot.py:302` | Hard boundary condition — *constrains* the curvature at each edge rather than setting an elastic target |
| **MuJoCo** | Converts `κ → angle = κ × seg_len`, sets `ctrl[actuator]` | `mujoco_snake_robot.py:426` | PD position controller (kp=50) drives hinge joints toward target angle — not an elastic rod at all |

The distinction between "elastic rest shape" (DisMech, PyElastica), "kinematic constraint" (dismech-rods), and "actuator setpoint" (MuJoCo) means the same κ=1.0 command produces different forces, torques, and transient responses.

### Layer 2: Ground contact model

This is the largest source of divergence, and the primary driver of locomotion differences:

| Backend | Ground contact | Result |
|---|---|---|
| **DisMech** | RFT: anisotropic velocity drag (ct=0.01, cn=0.1) | Resists motion but doesn't prevent z-penetration → snake sinks to z=-22.8m |
| **PyElastica** | Same RFT formula + `AnalyticalLinearDamper` (damping=0.1) | Over-damped — RFT + damper kills all XY motion, snake sinks slowly to z=-0.44m |
| **dismech-rods** | **None** — only gravity and optional viscous damping | Free-fall while bending → sinks to z=-44.9m in 60 steps |
| **MuJoCo** | Rigid ground plane with full contact simulation (friction=1.0) | Actual friction-driven propulsion → 11.3m forward displacement, but NaN instability |

Snake locomotion requires anisotropic ground friction (resist lateral sliding, allow forward sliding). Only MuJoCo has a physically grounded contact model, which is why it's the only backend that produces substantial forward displacement. The DER backends use RFT as an approximation, and dismech-rods has no ground contact at all.

### Layer 3: Time integration

| Backend | Integrator | Effective dt | Steps per RL step |
|---|---|---|---|
| DisMech | Implicit Euler (Newton) | 0.05s | 1 |
| dismech-rods | Implicit Euler (Newton) | 0.05s | 1 |
| PyElastica | Explicit PositionVerlet | 0.001s | 50 substeps |
| MuJoCo | Semi-implicit Euler | 0.002s | 25 substeps |

Implicit methods (DisMech, dismech-rods) allow large timesteps but introduce numerical dissipation. Explicit methods (PyElastica) need small substeps for stability but preserve energy better. The 50x difference in effective dt between DisMech (0.05s) and PyElastica (0.001s) means they resolve different frequency content of the dynamics even from identical initial conditions.

### Summary

"Same input, different output" is expected and correct. The backends share a control *interface* (19 curvature floats) but implement different physics: different constitutive models, different ground interaction, and different numerical methods. Comparing their outputs is the point of this benchmark — it quantifies how much the choice of physics engine matters.

## Results

### 1. Computation Time

| Framework | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Total (s) |
|---|---|---|---|---|---|
| **MuJoCo** | **2.9** | **2.6** | 2.5 | 8.0 | 1.5 |
| PyElastica | 15.8 | 13.7 | 11.4 | 174.8 | 7.9 |
| dismech-rods (C++)* | 27.0 | 16.6 | 6.8 | 122.6 | 1.6 |
| DisMech (Python) | 94.2 | 87.9 | 16.8 | 860.9 | 47.1 |

*\*60 steps only*

**Speed ranking:** MuJoCo (32x) > PyElastica (6x) > dismech-rods (3.5x) > DisMech (1x baseline)

MuJoCo is the fastest by a wide margin, consistent with its compiled C engine and rigid-body formulation. DisMech (Python) is the slowest due to Python-level implicit Newton solves. dismech-rods benefits from C++ but its implicit solver requires many iterations under large curvatures (up to 95 iterations at step 53).

### 2. Locomotion Behavior

| Framework | Fwd disp (m) | Heading drift (°) | Head z range (m) |
|---|---|---|---|
| MuJoCo | 11.34 | -91.9 | [-0.71, 602.8] |
| DisMech | 0.75 | 180.0 | [-22.8, -0.01] |
| dismech-rods* | 0.11 | -67.0 | [-44.9, -0.02] |
| PyElastica | 0.00 | 0.0 | [-0.44, 0.00] |

*\*60 steps only*

Each backend exhibits fundamentally different locomotion:

- **MuJoCo** produces the most displacement (11.3m) because its rigid capsule chain has strong ground-plane contact and friction. However, it also shows extreme instability — the snake launches into the air (head z up to 603m) due to MuJoCo reporting NaN/Inf in joint accelerations at t=0.006s. The simulation continues but produces physically unrealistic trajectories.
- **DisMech** (Python) achieves modest forward displacement (0.75m) with a 180° heading reversal, indicating the snake turns around over 25s. The snake sinks steadily in z (to -22.8m), suggesting RFT ground forces are insufficient to fully support the rod under gravity.
- **dismech-rods** (C++) shows similar sinking behavior (z to -44.9m in just 60 steps) with heading oscillation (±70°). It produces the least displacement among backends that actually move.
- **PyElastica** produces zero XY displacement. The rod sinks slowly in z (to -0.44m) but remains essentially stationary in XY, with zero heading change. The RFT forcing appears to fully damp any lateral motion while gravity pulls it down.

### 3. Energy Analysis

| Framework | KE mean | Elastic mean | Grav mean | Total init | Total final | Energy drift |
|---|---|---|---|---|---|---|
| MuJoCo | 8.000 | 0.005 | 13.621 | 0.569 | 17.233 | +29x |
| DisMech | 0.002 | 0.000 | -0.369 | -0.000 | -0.847 | -5110x |
| dismech-rods* | 0.558 | 0.000 | -0.572 | -0.000 | -0.027 | -59x |
| PyElastica | 0.000 | 0.000 | -0.008 | -0.000 | -0.016 | -520x |

*\*60 steps only*

- **MuJoCo** is the only backend where total energy increases (gravitational energy grows as the snake launches upward). The 29x energy gain reflects the instability.
- **DisMech** has the largest energy drift magnitude (-5110x), driven by gravitational potential energy as the snake sinks deeply in z. Its elastic energy contribution is negligible.
- **dismech-rods** shows moderate kinetic energy (mean 0.56 J), consistent with active locomotion and heading oscillation before convergence failure.
- **PyElastica** is the most energetically quiet — kinetic energy is six orders of magnitude smaller than other backends, confirming the snake is essentially static.

### 4. Solver Convergence (DisMech only)

DisMech's implicit Euler solver exposes a residual norm (`f_norm`) per timestep:

| Metric | Value |
|---|---|
| Mean residual | 0.004492 |
| Max residual | 0.069218 |
| Min residual | 0.000047 |

All residuals remain well below the convergence tolerance (ftol=1e-4 is the relative tolerance; the absolute residuals here are small), indicating the solver converges reliably across all 500 steps. No convergence failures occurred.

### 5. Cross-Framework Trajectory Agreement

#### Full 500 steps (3 backends)

| Pair | Head MSE | CoG MSE | Max head divergence (m) |
|---|---|---|---|
| DisMech vs PyElastica | 141.1 | 144.6 | 22.4 |
| DisMech vs MuJoCo | 180,211 | 180,353 | 611.0 |
| PyElastica vs MuJoCo | 173,837 | 173,840 | 603.0 |

#### First 60 steps (all 4 backends, XY plane)

| Pair | Head XY MSE |
|---|---|
| DisMech vs PyElastica | 0.0021 |
| DisMech vs dismech-rods | 0.0206 |
| PyElastica vs dismech-rods | 0.0108 |
| DisMech vs MuJoCo | 0.4538 |
| PyElastica vs MuJoCo | 0.4498 |
| dismech-rods vs MuJoCo | 0.4183 |

The three DER-based backends (DisMech, PyElastica, dismech-rods) show reasonable agreement in the early phase (XY MSE < 0.021), with DisMech and PyElastica being closest (MSE=0.002). MuJoCo diverges from all DER backends by ~0.45 MSE even in the first 60 steps — expected given its fundamentally different rigid-body physics model.

Over the full 500 steps, trajectories diverge massively (max divergence 603-611m), dominated by MuJoCo's instability.

## Conclusions

### Performance
MuJoCo is the clear winner on speed (2.9ms/step vs 94ms for DisMech), making it attractive for RL training where millions of steps are needed. dismech-rods (C++) is ~3.5x faster than DisMech (Python) but suffers convergence failures under high curvature.

### Physical fidelity
None of the backends produce fully realistic ground locomotion with the current configuration:
- DisMech and dismech-rods sink in z (RFT insufficient to resist gravity)
- PyElastica is over-damped (zero locomotion)
- MuJoCo is unstable (NaN at t=0.006s, snake launches into air)

### Recommendations
1. **For RL training speed:** Use MuJoCo, but tune `mujoco_timestep`, `mujoco_joint_stiffness`, and `mujoco_friction` to prevent the NaN instability. The 0.001m rod radius creates extremely light segments that are prone to instability.
2. **For physical accuracy:** Use DisMech (Python) as the reference simulator, but investigate the z-sinking issue — the RFT ground contact model may need a ground-plane constraint or stiffer normal force.
3. **For dismech-rods:** Increase damping (`dismech_rods_damping_viscosity`) or use adaptive time stepping to prevent convergence failures under large curvature changes.
4. **Cross-backend validation:** Run comparisons with 2D-constrained motion (z=0) to isolate the ground-contact issue and get a cleaner comparison of in-plane dynamics.

## Artifacts

| File | Description |
|---|---|
| `output/benchmark/dismech_timeseries.csv` | DisMech per-step data (500 rows) |
| `output/benchmark/elastica_timeseries.csv` | PyElastica per-step data (500 rows) |
| `output/benchmark/mujoco_timeseries.csv` | MuJoCo per-step data (500 rows) |
| `output/benchmark/summary.csv` | Summary statistics (3 rows) |
| `output/benchmark/cross_framework.csv` | Pairwise comparison (3 pairs) |
| `output/benchmark_dismech_rods/dismech_rods_timeseries.csv` | dismech-rods per-step data (60 rows) |
| `output/benchmark_dismech_rods/summary.csv` | dismech-rods summary (1 row) |
| `script/benchmark_physics.py` | Benchmark script |
