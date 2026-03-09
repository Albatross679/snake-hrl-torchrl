---
id: e5a1733e-8196-4cd1-821e-36d392983874
name: friction-contact-models
description: Friction and contact models in robot simulation across MuJoCo and DisMech
type: knowledge
created: 2026-02-16T00:00:00
updated: 2026-02-16T00:00:00
tags: [physics, contact-models, friction, simulation, mujoco, dismech]
aliases: []
---

# Friction and Contact Models in Robot Simulation

## Hierarchy of Abstractions

Friction/contact models form a hierarchy from most idealized to most general. Each level relaxes assumptions from the level above, and simpler models are special cases of more general ones.

```
Level 0: Frictionless
  │
Level 1: Kinematic constraints (perfect no-slip)
  │
Level 2: Viscous drag (force ∝ velocity)
  │
Level 3: Coulomb friction (constant μ, discontinuous at v=0)
  │
Level 4: Coulomb + Stribeck (smooth static-to-kinetic transition)
  │
Level 5: Soft contact with friction cones (MuJoCo convex optimization)
  │
Level 6: Implicit barrier contact (IPC, unconditionally stable)
```

## Level 1: Nonholonomic Constraints

$$-\dot{x}_i \sin\theta_i + \dot{y}_i \cos\theta_i = 0$$

Perfect no-lateral-slip at each wheel. Algebraic constraint baked into kinematics. No friction forces computed. Valid when wheel contact is ideal. This is the $\mu \to \infty$ limit of Coulomb friction.

## Level 2: Resistive Force Theory

Force balance in low-Reynolds-number fluid: drag is proportional to velocity. Each link experiences tangential drag $c_t$ and normal drag $c_n$ ($c_n > c_t$). The anisotropy $c_n > c_t$ is the fluid analogue of passive wheels.

## Level 3–4: Coulomb + Stribeck Friction

**Stribeck curve** (smooth static-to-kinetic transition):

$$s_i = \mu_c - (\mu_c - \mu_s)\exp\left(-\frac{|\dot{p}|^2}{v_s^2}\right)$$

Where $\mu_s > \mu_c$ (static > kinetic), $v_s$ controls transition width. Normal force via spring-damper: $F_z = -k_1 p_z - k_2 \dot{p}_z$.

Pure Coulomb is the $v_s \to 0$, $\mu_v \to 0$ limit.

## Level 5: MuJoCo Soft Contact

MuJoCo uses **convex optimization** instead of classical LCP (Todorov, 2014):

$$f^* = \arg\min_{\lambda \in \Omega} \frac{1}{2}\lambda^T(A + R)\lambda + \lambda^T(a_u - a_r)$$

Where:
- $A = JM^{-1}J^T$ — inverse inertia in constraint space
- $R$ — diagonal regularizer (contact softness)
- $\Omega$ — friction cone constraints

**Key innovation**: Relaxes complementarity condition. Classical LCP is NP-hard; MuJoCo's convex formulation is polynomial-time.

**Friction cone** (elliptic):

$$K = \{f \in \mathbb{R}^n : f_1 \geq 0,\; f_1^2 \geq \sum_{i>1} f_i^2/\mu_{i-1}^2\}$$

This subsumes Coulomb friction — Coulomb is the hard-contact limit ($R \to 0$).

## Level 6: Incremental Potential Contact (IPC)

Smooth barrier function preventing interpenetration:

$$F = -k \nabla_q \left(\frac{1}{K}\log(1 + e^{K\epsilon})\right)^2, \quad K = \frac{15}{\delta}$$

The softplus $\frac{1}{K}\log(1+e^{K\epsilon})$ is a smooth ($C^1$) approximation of $\max(0, \epsilon)$. Fully implicit solve → **unconditionally stable** → allows large timesteps ($\Delta t = 0.05$s vs. $0.0002$s for explicit penalty methods).

## Comparison

| Concept | Kinematic | Explicit (Coulomb, Penalty) | Implicit (MuJoCo, IPC) |
|---|---|---|---|
| Contact detection | Algebraic | Penetration + Heaviside | Smooth barrier |
| Normal force | Not computed | Spring-damper $kx + c\dot{x}$ | QP (MuJoCo) / barrier (IPC) |
| Friction force | Infinite (no-slip) | $\mu F_N \text{sgn}(\dot{p})$ | Friction cone optimization |
| Differentiability | N/A | Discontinuous | Smooth ($C^1$+) |
| Stability | Unconditional | Timestep-limited | Unconditional |

## Which Is Best for RL?

For RL, the critical properties are:
1. **Smoothness** — discontinuous contact makes policy gradients noisy
2. **Stability at large dt** — faster sim = more training data per hour
3. **Differentiability** — for gradient-based optimization

**MuJoCo** is the most widely used: best balance of speed, fidelity, and ease of use for rigid robots. **DisMech (IPC)** is better for soft robots due to implicit stability (40× speedup over explicit methods).

## References

- Todorov, E. (2014). Convex and analytically-invertible dynamics with contacts and constraints. *ICRA*.
- Khatib, O. (1986). Real-time obstacle avoidance for manipulators and mobile robots. *IJRR*.
- Li, M. et al. (2020). Incremental Potential Contact. *ACM TOG*.
- Choi, A. & Tong, D. (2025). Rapidly Learning Soft Robot Control via Implicit Time-Stepping.
