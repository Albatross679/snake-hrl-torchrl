---
id: 7c7b5285-9780-486d-bcb7-5fc37c322a23
name: elastica-substep-stability-sweep
description: Empirical sweep to find the largest stable PyElastica substep for the current rod configuration
type: experiment
created: 2026-03-09T11:35:02
updated: 2026-03-09T11:35:02
tags: [elastica, stability, substep, cfl, physics, performance]
aliases: []
---

# PyElastica Substep Stability Sweep

## Objective

Determine the largest stable integration timestep (dt_sub) for the PositionVerlet integrator with our current Cosserat rod parameters, to find the maximum possible simulation speedup.

## Rod Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Length | 0.5 m | `locomotion_elastica/config.py:63` |
| Radius | 0.02 m | `locomotion_elastica/config.py:64` |
| Elements | 20 | `locomotion_elastica/config.py:65` |
| Young's modulus | 1e5 Pa | `locomotion_elastica/config.py:73` |
| Density | 1200 kg/m^3 | `locomotion_elastica/config.py:75` |
| Damping | 0.002 | `locomotion_elastica/config.py:81` |
| Element length (dx) | 0.025 m | derived: 0.5/20 |

## Theoretical Stability Limits

PositionVerlet is an explicit symplectic integrator, stable when dt < 2/omega_max.

For bending modes of a Cosserat rod:
```
omega_max = sqrt(EI / rhoA) * (pi / dx)^2
         = sqrt(1e5 * pi*0.02^4/4 / (1200 * pi*0.02^2)) * (pi / 0.025)^2
         = 1441.55 rad/s

dt_verlet = 2 / omega_max = 0.001387 s
```

Other limits (less restrictive):
- CFL longitudinal wave: dt < 0.00274 s
- CFL bending wave (dx^2): dt < 0.00685 s

**Predicted stability boundary: dt_sub < 0.00139 s**

## Empirical Results

Swept `elastica_substeps` (which controls dt_sub = physics.dt / elastica_substeps = 0.05 / N) over 50 RL steps with random actions. Stability judged by max absolute position of rod nodes (should stay < 0.5m for a 0.5m rod).

### Coarse Sweep

| elastica_substeps | dt_sub (s) | Total substeps/RL | max_pos (m) | Status |
|---|---|---|---|---|
| 50 | 0.00100 | 500 | 0.255 | Stable |
| 40 | 0.00125 | 400 | 0.292 | Stable |
| 35 | 0.00143 | 350 | 0.265 | Stable |
| 30 | 0.00167 | 300 | 0.286 | Stable |
| 25 | 0.00200 | 250 | 0.282 | Stable |
| 20 | 0.00250 | 200 | 10.614 | Unphysical |
| 15 | 0.00333 | 150 | 8.228 | Unphysical |
| 10 | 0.00500 | 100 | 6.294 | Unphysical |
| 5 | 0.01000 | 50 | 25.585 | Unphysical |

### Fine Sweep Around Boundary

| elastica_substeps | dt_sub (s) | max_pos (m) | Status |
|---|---|---|---|
| 30 | 0.00167 | 0.318 | Stable |
| 28 | 0.00179 | 0.284 | Stable |
| 26 | 0.00192 | 0.402 | Stable |
| 25 | 0.00200 | 0.297 | Stable |
| 24 | 0.00208 | 0.369 | Stable |
| 23 | 0.00217 | 13.591 | **Unstable** |
| 22 | 0.00227 | 11.900 | Unstable |
| 21 | 0.00238 | 13.439 | Unstable |
| 20 | 0.00250 | 11.215 | Unstable |

## Key Finding

**Empirical stability limit: dt_sub ≈ 0.0021 s (elastica_substeps = 24)**

The instability is silent — no NaN or crash, just unphysical rod positions (rod "explodes" to 10+ meters). This makes it dangerous if not monitored explicitly.

## Comparison to Theory

| Metric | Value |
|--------|-------|
| Theoretical limit (PositionVerlet) | 0.00139 s |
| Empirical limit | 0.0021 s |
| Ratio | 1.5x (theory is conservative) |
| Current setting (elastica_substeps=50) | 0.001 s |
| Safety margin vs empirical | 2.1x |
| Safety margin vs theory | 1.4x |

The theory underestimates the actual limit by ~1.5x because it assumes the highest-frequency bending mode is fully excited, which doesn't happen in practice with smooth serpenoid actuation and damping.

## Speedup Opportunities

| Setting | dt_sub (s) | Substeps/RL | Speedup vs current | Safety margin |
|---------|-----------|-------------|---------------------|---------------|
| Current (50) | 0.001 | 500 | 1.0x | 2.1x |
| Conservative (30) | 0.00167 | 300 | **1.67x** | 1.26x |
| Aggressive (25) | 0.002 | 250 | **2.0x** | 1.05x |
| Unsafe (20) | 0.0025 | 200 | — | <1x (unstable) |

**Recommendation:** `elastica_substeps=30` gives a 1.67x speedup with a comfortable safety margin. Going to 25 doubles speed but leaves almost no margin.

## Related

- [[control-frequency-survey]]
- [[unify-curvature-substep-frequency]]
- [[elastica-curvature-not-updated-substeps]]
