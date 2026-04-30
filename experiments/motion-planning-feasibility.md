---
name: motion-planning-feasibility
description: Feasibility study of classical motion planning (RRT, trajectory optimization) for snake locomotion
type: experiment
created: 2026-03-05T14:24:39
updated: 2026-03-05T14:24:39
tags: [motion-planning, rrt, locomotion, feasibility]
aliases: [motion-planning-study]
---

# Motion Planning Feasibility for Snake Locomotion

## Objective

Evaluate whether classical motion planning techniques (RRT, RRT*, PRM, trajectory optimization, CHOMP, TrajOpt) can be applied to the snake robot in the `locomotion/` environment.

## Setup

- **Environment**: `LocomotionEnv` with DisMech rod simulation + RFT anisotropic friction
- **Snake**: 0.5m rod, 20 segments (21 nodes), radius 0.001m
- **Action space**: 5-dim serpenoid (amplitude, frequency, wave_number, phase, turn_bias)
- **Test script**: `locomotion/test_motion_planning.py`

## Key Finding: Bending Plane Issue

**The current `two_d_sim=True` mode constrains DisMech to the XZ plane.** Curvature is applied to `nat_strain[:, 0]` (XZ bending), so the snake oscillates vertically but cannot turn laterally. All motion is strictly along the initial heading axis (dy = 0 always).

For 2D ground locomotion with turning, bending must be applied to `nat_strain[:, 1]` (XY plane) in 3D simulation mode (`two_d_sim=False`).

| Mode | Bending Axis | XY Turning | Forward Motion |
|------|-------------|------------|----------------|
| `two_d_sim=True`, `nat_strain[:,0]` (current) | XZ | No | Yes (1D only) |
| `two_d_sim=False`, `nat_strain[:,1]` (tested) | XY | Yes | Yes (2D) |

## Motion Primitive Characterization

With XY-plane bending (corrected), 30-step primitives:

| Primitive | dx (m) | dy (m) | dtheta (deg) | Sim Time (s) |
|-----------|--------|--------|--------------|---------------|
| forward | -0.258 | +0.039 | -3.9 | 3.66 |
| forward_fast | +0.040 | -0.016 | -64.0 | 24.00 |
| turn_left | -0.043 | -0.004 | -74.2 | 3.61 |
| turn_left_sharp | -0.039 | +0.010 | -37.1 | 3.63 |
| turn_right | -0.015 | -0.032 | -112.9 | 3.64 |
| turn_right_sharp | +0.029 | -0.023 | +103.9 | 4.67 |

Observations:
- The "forward" primitive moves backward (negative dx) due to wave propagation direction
- Turn primitives produce significant heading changes (37-113 deg per primitive)
- Lateral displacement (dy) is small but non-zero
- Simulation cost: ~3.6s per primitive (30 physics steps)

## RRT Planning Test

Tested RRT in reduced (x, y, theta) space using motion primitives as edges.

- **Start**: (0, 0, 0 deg)
- **Goal**: (1.0, 0.5) with radius 0.2m
- **Result**: Path found in 36 iterations (13 nodes)
- **Planning time**: < 0.1s (primitive lookup, no simulation)

Path sequence: `forward_fast -> forward_fast -> turn_left_sharp -> forward x3 -> turn_right -> forward_fast x2 -> forward -> turn_left_sharp -> forward`

**Caveat**: This uses pre-characterized primitives as a lookup table. Open-loop execution in the simulator will accumulate drift since the primitives are state-dependent (the actual motion changes as the snake's shape evolves).

## Feasibility Assessment

### Sampling-Based (RRT, RRT*, PRM)

**Feasible with caveats:**
- Motion primitives provide controllable (dx, dy, dtheta) in reduced space
- RRT planning is fast (~0.1s) when using pre-computed primitives
- PRM could build a reusable roadmap for repeated queries
- **Limitation**: Open-loop execution drifts. Primitives are not truly state-independent because the rod's dynamic shape affects future motion.
- **Limitation**: Simulation cost (~3.6s per edge) makes sim-in-the-loop RRT expensive (500 iters ~ 30 min)

### Optimization-Based (CHOMP, TrajOpt)

**Challenging:**
- DisMech is NOT differentiable (implicit Euler solver with iterative convergence)
- No analytical gradients available for trajectory optimization
- Finite-difference gradients: 5-dim action x 3.6s/eval = 18s per gradient step
- CHOMP requires smooth cost landscapes; contact/friction dynamics are non-smooth
- **Alternative**: CMA-ES or other gradient-free optimizers could work for short segments

### Lattice-Based Planning

**Most practical for offline planning:**
- Pre-compute a motion primitive lattice offline (one-time cost)
- Use A*/Dijkstra over discretized (x, y, theta) grid
- Deterministic, no simulation at planning time
- Limited by primitive granularity

### Hybrid: RL + Planning (Recommended)

**Best fit for this project:**
- Use RL (PPO) for low-level locomotion (already implemented)
- Use planning for high-level waypoint sequencing
- The HRL architecture already supports this: meta-controller (planner) -> sub-policies (RL)
- RL handles physics-dependent motion; planning provides interpretable high-level goals

## Blocking Issues

1. **Bending plane**: Current `_apply_curvature_to_dismech` writes to `nat_strain[:, 0]` (XZ). For XY locomotion, must use `nat_strain[:, 1]`. This needs a config flag or env refactor.
2. **Forward direction**: The "forward" primitive moves backward. The curvature reversal (`curvatures[::-1]`) in `_step` may need tuning or the serpenoid phase needs adjustment.
3. **Simulation speed**: ~3.6s per 30-step primitive limits real-time planning.

## Conclusion

Classical motion planning **can** work for this snake robot, but requires:
1. Fixing the bending plane (XZ -> XY) for 2D ground locomotion
2. Tuning primitives so "forward" actually moves forward
3. Accepting that open-loop plans will drift and need re-planning or closed-loop correction

The **hybrid RL + planning** approach is recommended: use planning for waypoints, RL for execution.
