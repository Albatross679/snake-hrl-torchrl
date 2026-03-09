---
id: c013c6bf-4d45-447c-ba54-9a1ae9634ac3
name: cross-framework-comparison-protocol
description: Detailed comparison of MuJoCo, dismech, and Elastica physics frameworks — architectural differences, parameter matching, and experimental validation protocol
type: knowledge
created: 2026-02-14T00:00:00
updated: 2026-02-14T01:00:00
tags: [knowledge, physics, comparison, dismech, mujoco, elastica, experiment-design]
aliases: []
---

# Cross-Framework Comparison Protocol: MuJoCo vs dismech vs Elastica

This document describes the architectural differences between the three active physics frameworks in this project and defines a protocol for fair comparison by matching physical parameters while keeping conditions identical.

## Architectural Difference: Geometry–Physics Coupling

The most fundamental difference between the frameworks is how they couple body representation (geometry) with force computation (physics).

### MuJoCo: Modular (Geometry and Physics Are Separate Layers)

MuJoCo separates body representation from force computation. You first choose a geometry layer, then attach a physics layer independently:

```
Step 1: Choose body representation (geometry layer)
         → rigid body chain, composite (deprecated), or flex

Step 2: Attach force model (physics layer)
         → actuators, elasticity plugins, or engine-native flex stiffness
```

These are independent, composable choices:

| Geometry | Physics | Result |
|----------|---------|--------|
| Rigid chain | Position actuators (kp) | Current snake implementation |
| Flex (1D) | Cable plugin (bend, twist) | Elastic rod approximation |
| Flex (1D) | Engine edge stiffness | Simple spring model |
| Flex (2D) | Engine elastic2d | Shell/cloth |
| Flex (3D) | Engine FEM (SVK) | Volumetric deformable |

MuJoCo's core engine understands bodies, contacts, and constraints — it does not inherently know what an "elastic rod" is. Rod-specific physics (DER bending/twisting) is bolted on via the cable plugin.

**Consequence:** The physics fidelity depends on which plugin/feature you attach. The same geometry can behave very differently depending on the force model.

### dismech: Monolithic (Geometry IS the Physics)

In dismech, the Discrete Elastic Rod (DER) formulation defines both the geometry and the physics simultaneously. They cannot be separated:

```
State vector = [node positions (x₁...xₙ), edge twist angles (θ₁...θₙ₋₁)]
                         │
         This IS the geometry (defines the rod's shape)
                         │
                         ▼
         Bending energy  = f(discrete curvature between consecutive edges)
         Stretching energy = f(edge length vs rest length)
         Twisting energy  = f(θ vs reference twist)
                         │
         These energies ARE the physics, computed analytically
         from the geometric state — no separate force model needed
```

There is no "geometry layer" distinct from "physics layer." The node positions define the shape, and the elastic energies are analytically derived from those same positions. The rod representation IS the constitutive model.

### Elastica: Also Monolithic (Cosserat Rod = Geometry + Physics)

Elastica uses the same monolithic principle with a richer formulation (Cosserat rod theory):

```
State = [position(s), director frame(s), velocity(s), angular velocity(s)]
                         │
         Rod centerline + director frames ARE the geometry
                         │
                         ▼
         Strain  = computed from directors vs rest configuration
         Stress  = constitutive law applied to strain (material properties)
         Forces  = divergence of internal stress
                         │
         Physics is inseparable from the rod description
```

Cosserat theory is richer than DER — it captures shear and extension naturally in addition to bending and twisting. But the same principle holds: you cannot swap out the geometry and keep the physics, or vice versa.

### Summary Table

| | MuJoCo (rigid chain) | MuJoCo (flex + cable) | dismech | Elastica |
|---|---|---|---|---|
| **Architecture** | Modular | Modular | Monolithic | Monolithic |
| **Geometry** | Rigid capsules + joints | Flex vertices + elements | DER nodes + edges | Cosserat rod centerline + directors |
| **Physics** | PD actuator springs | Cable plugin forces | DER energy functional | Cosserat constitutive law |
| **Separable?** | Yes — can swap physics | Yes — can swap physics | No | No |
| **Elastic forces** | External (actuator kp) | External (plugin) | Intrinsic (from state) | Intrinsic (from state) |
| **Contact** | General-purpose solver | General-purpose solver | Custom (IMC, penalty) | External (must add) |

### What This Means for Comparison

Because MuJoCo's physics is bolted on while dismech/Elastica's is intrinsic, you cannot make the three frameworks compute identical forces. The comparison is between:

- **dismech**: Purpose-built DER solver with analytical elastic energies
- **Elastica**: Purpose-built Cosserat solver with constitutive strain-stress model
- **MuJoCo**: General-purpose rigid-body engine approximating elastic behavior via PD controllers

The goal is not to make them identical — it's to match the controllable parameters so the only remaining differences are the irreducible physics engine differences.

## Three Layers of Comparison

```
┌─────────────────────────────────────────┐
│  Layer 3: Observation / Reward Pipeline │  ← Already unified (wrapper classes)
├─────────────────────────────────────────┤
│  Layer 2: Physical Parameters           │  ← MUST BE MATCHED (controlled variables)
├─────────────────────────────────────────┤
│  Layer 1: Physics Engine                │  ← Fundamentally different (independent variable)
└─────────────────────────────────────────┘
```

- **Layer 3** is already handled by the unified wrapper interface (`get_state()`, `get_observation()`, `get_energy()`).
- **Layer 1** is the independent variable — the thing being compared.
- **Layer 2** is the controlled variable — must be carefully matched.

## Layer 2: Parameters That Must Be Matched

### 1. Geometry (Straightforward — Direct Mapping)

These should be numerically identical across all three frameworks.

| Parameter | Config Field | Units |
|-----------|-------------|-------|
| Snake total length | `config.snake_length` | m |
| Snake radius | `config.snake_radius` | m |
| Number of discrete elements | `config.num_segments` | — |
| Total mass | `config.density × volume` | kg |
| Prey radius | `config.prey_radius` | m |
| Prey length | `config.prey_length` | m |
| Prey position | `config.prey_position` | m |

**Discretization equivalence:**

| dismech | Elastica | MuJoCo |
|---------|----------|--------|
| N nodes, N−1 edges | N elements along rod | N rigid bodies, N−1 joints |
| `num_nodes = num_segments + 1` | `n_elements = num_segments` | `num_segments` bodies |

The `PhysicsConfig` already unifies these. Ensure all three backends interpret `num_segments` consistently.

### 2. Bending Stiffness (Hardest — Different Parameterizations)

Each framework parameterizes bending stiffness differently:

| Framework | Parameter | Units | Meaning |
|-----------|-----------|-------|---------|
| dismech | `EI` | N·m² | Bending stiffness (Young's modulus × second moment of area) |
| Elastica | `E` (Young's modulus) | Pa | Material property; framework computes `EI = E × I` internally |
| MuJoCo (rigid chain) | `kp` | N·m/rad | Position actuator spring stiffness |
| MuJoCo (cable plugin) | `bend` | Pa | Plugin bending stiffness parameter |

**Ground truth:** Define `EI` as the canonical stiffness value. Derive all others from it.

**Conversion formulas:**

```
Second moment of area (circular cross-section):
    I = (π/4) × r⁴

dismech → Elastica:
    E = EI / I = EI / ((π/4) × r⁴)

dismech → MuJoCo (rigid chain):
    For a hinge joint to approximate a segment of elastic rod:
        Rod segment restoring torque:  τ_rod = EI × κ = EI × θ / L_seg
        Hinge joint restoring torque:  τ_joint = kp × θ
        Match: kp = EI / L_seg
    where L_seg = snake_length / num_segments

dismech → MuJoCo (cable plugin):
    bend = E = EI / ((π/4) × r⁴)
    (cable plugin uses material Young's modulus internally)
```

**Conversion chain:**

```
                          ┌──→  MuJoCo kp  = EI / L_seg
                          │
    dismech EI (N·m²)  ──┼──→  Elastica E  = EI / (π r⁴ / 4)
                          │
                          └──→  MuJoCo cable bend = EI / (π r⁴ / 4)
```

### 3. Stretching Stiffness

| Framework | Parameter | Units |
|-----------|-----------|-------|
| dismech | `EA` | N | Axial stiffness (Young's modulus × cross-sectional area) |
| Elastica | `E` | Pa | Same Young's modulus also governs stretching |
| MuJoCo (rigid chain) | N/A | — | Rigid links don't stretch |
| MuJoCo (cable plugin) | N/A | — | Cable is inextensible by design |

**Note:** MuJoCo backends have no stretching DOF. This is an irreducible difference. In practice, DER/Cosserat stretching is small when `EA` is large (stiff rod), so this should not be a major source of discrepancy.

### 4. Twisting Stiffness

| Framework | Parameter | Units |
|-----------|-----------|-------|
| dismech | `GJ` | N·m² | Torsional stiffness (shear modulus × polar moment) |
| Elastica | `G` (shear modulus) | Pa | Framework computes `GJ = G × J` internally |
| MuJoCo (rigid chain) | N/A | — | Hinge joints only allow bending, no twist DOF |
| MuJoCo (cable plugin) | `twist` | Pa | Plugin twisting stiffness parameter |

**Conversion:**

```
Polar moment (circular cross-section):
    J = (π/2) × r⁴

dismech → Elastica:
    G = GJ / J = GJ / ((π/2) × r⁴)

dismech → MuJoCo (cable plugin):
    twist = G = GJ / ((π/2) × r⁴)
```

**Note:** The current MuJoCo rigid-chain backend has **no twist DOF** (hinge joints only rotate about z-axis). This is another irreducible difference for 3D comparison. For 2D planar locomotion, twist is typically negligible.

### 5. Damping (Different Models — Must Match Empirically)

#### What Damping Is

Damping is **internal energy dissipation in the rod itself**. It has nothing to do with ground interaction.

**Physical intuition:** Imagine plucking a guitar string:

- **No damping:** the string vibrates forever
- **Light damping:** vibrations gradually decay over seconds
- **Heavy damping:** vibrations die out almost immediately, no oscillation

For the snake rod: if you suddenly change the target curvature, the rod bends toward the new shape. Without damping, it overshoots and oscillates around the target. Damping controls how quickly those oscillations die out.

#### What Damping Is NOT

Damping is easily confused with other dissipative forces. These are three distinct physical phenomena:

| Term | What It Does | Where It Acts | Example |
|------|-------------|---------------|---------|
| **Stiffness** | Restoring force toward equilibrium shape | Internal — rod wants to return to rest curvature | Guitar string returning to straight after pluck |
| **Damping** | Energy dissipation opposing the rod's own deformation velocity | Internal — rod resists fast bending/unbending | Guitar string vibration decaying over time |
| **Ground friction** | Force from ground interaction opposing sliding | External — ground acts on snake surface | Snake pushing against ground to move forward |

Damping and friction both dissipate energy, but:

- **Damping** opposes the rod's own internal motion (how fast it bends/unbends). A snake in free space (no ground) still has damping.
- **Friction** opposes the rod's motion relative to the ground (how fast it slides). A perfectly rigid snake on the ground has friction but no damping.

#### Framework Implementations

Each framework uses a fundamentally different damping model:

| Framework | Model | Parameter | Acts On |
|-----------|-------|-----------|---------|
| dismech (Python) | Viscous velocity damping | `damping_viscosity` | Node translational velocities |
| dismech (C++) | `DampingForce` class | Damping coefficient | Node translational velocities |
| Elastica | `AnalyticalLinearDamper` | `dissipation_constant` | Node translational + angular velocities |
| MuJoCo | Joint damping | `mujoco_joint_damping` | Joint angular velocities |

#### Why These Can't Be Matched Analytically

The models damp different physical quantities:

- **dismech/Elastica:** `F = -γ × v` applied to each node. Damps **all** velocity components — including forward translation. This means damping slightly resists the snake's overall locomotion, not just its bending oscillation.
- **MuJoCo:** `τ = -d × ω` applied to each joint. Damps **only joint rotation rate** — resists bending velocity but does NOT resist the snake's translational motion at all.

This is a fundamental mismatch: dismech's damping bleeds into locomotion resistance; MuJoCo's does not. There is no analytical conversion between them.

#### Empirical Matching Protocol

1. Set all three to the same initial condition (e.g., rod bent to curvature κ₀, released from rest, no ground contact)
2. Measure the free vibration: oscillation frequency and exponential decay envelope
3. Tune each framework's damping until the decay time constant matches
4. Alternatively: tune until all three dissipate the same fraction of initial energy over a fixed time window

**Target metric:** Half-energy time `t½` — the time for total energy to drop to 50% of its initial value. Measure in free space (no ground) to isolate damping from friction.

### 6. Ground Interaction (Contact Model + Friction)

This is the largest source of behavioral divergence between the frameworks. Ground interaction involves two coupled but distinct physical problems:

1. **Contact model (normal direction):** How do we detect when the snake touches the ground, and what force pushes them apart?
2. **Friction model (tangential direction):** What force resists sliding along the ground surface?

In MuJoCo, these are tightly coupled — friction force is proportional to normal contact force. In dismech/Elastica, they are handled by completely separate, independent force models. This is why the earlier parameter classification listed "contact model" as an irreducible difference while "ground friction" was listed as tunable — the contact normal force model cannot be matched, but the resulting locomotion behavior (driven by friction/drag) can be tuned to match.

#### Contact Model (Normal Forces)

The contact model determines how the snake stays on the ground surface.

| Framework | Contact Detection | Normal Force |
|-----------|-------------------|-------------|
| dismech (Python) | Geometric check: `node_z < ground_z` | Penalty spring: exponential repulsion when penetrating |
| dismech (C++) | Geometric check: `node_z < floor_z` | Penalty spring: `FloorContactForce(floor_delta, floor_z)` |
| Elastica | **None** — no geometric ground | **None** — RFT drag implicitly keeps snake in plane |
| MuJoCo | Collision detection: capsule-plane overlap | Constraint solver: impulse-based non-penetration |

**Key observation:** Elastica has no contact model at all. The snake doesn't "touch" the ground — it just experiences velocity-dependent drag everywhere. There's no concept of "lifting off" the ground.

#### Friction / Locomotion Force Model (Tangential Forces)

This is what actually makes the snake move. The tangential forces convert lateral undulation into forward thrust via anisotropy (more resistance sideways than forward).

| Framework | Model | Formula | Always Active? |
|-----------|-------|---------|---------------|
| dismech (Python) | **RFT** (Resistive Force Theory) | `F = -ct × v_t - cn × v_n` | Yes — applied every timestep regardless of contact |
| dismech (C++) | **Penalty friction** (stick/slip) | Coulomb-like with `floor_mu` | Only when penalty floor is active (node near ground) |
| Elastica | **RFT** (Resistive Force Theory) | `F = -ct × v_t - cn × v_n` | Yes — always applied, no contact condition |
| MuJoCo | **Coulomb friction cones** | `F_tangential ≤ μ × F_normal` | Only during contact (normal force > 0) |

#### Why RFT Is NOT a Contact Model

RFT (Resistive Force Theory) is a **velocity-proportional drag field**, not a contact model. It's important to understand this distinction:

| | RFT (dismech, Elastica) | Contact Friction (MuJoCo) |
|---|---|---|
| **Type** | Continuous drag field | Contact mechanics |
| **Active when** | Always, every node, every timestep | Only when surfaces overlap |
| **Normal force** | Drag on vertical velocity (`-cn × v_n`) | Constraint impulse (rigid non-penetration) |
| **Tangential force** | Drag on horizontal velocity (`-ct × v_t`) | Coulomb: `F_t ≤ μ × F_n` (coupled to normal) |
| **Anisotropy source** | Drag ratio `cn/ct` | Friction coefficient μ |
| **Can the snake lift off?** | No — drag always applies | Yes — loses contact, loses all friction |
| **Physical basis** | Empirical model from granular media / low-Re fluids | Solid mechanics contact theory |

RFT was originally developed for organisms moving through granular media (sand) or low-Reynolds-number fluids, where the medium always surrounds the body. It's a useful simplification for snake locomotion because the snake is always on the ground, but it models a fundamentally different physical process than MuJoCo's rigid-body contact.

#### Coupling Between Contact and Friction

This coupling is the core reason the frameworks can't be analytically matched:

- **MuJoCo:** Friction is **coupled** to contact normal force via `F_t ≤ μ × F_n`. Heavier snake segments get more friction. If a segment lifts off the ground, it loses friction entirely.
- **dismech/Elastica (RFT):** Drag is **independent** of any normal force. It depends only on velocity. Every node gets the same drag regardless of how much weight it carries or whether it's touching the ground.

This means the force profiles are qualitatively different even when tuned to produce the same average locomotion speed.

#### Dismech's Penalty Floor vs RFT

Dismech has **two independent ground forces** that serve different purposes:

```
Penalty floor force:  Prevents snake from falling through ground (normal direction)
                      F_penalty = stiffness × exp(-distance / delta)  when node_z < ground_z

RFT force:            Drives locomotion (tangential direction)
                      F_rft = -ct × v_tangential - cn × v_normal
```

These are separate forces that happen to both relate to the ground. The penalty floor is the "contact model" (keeps the snake on the surface). RFT is the "friction model" (makes the snake move). They operate independently — RFT doesn't know or care about the penalty floor force.

#### Empirical Matching Protocol

Since the models are fundamentally different, match the **locomotion behavior** rather than the parameters:

1. Apply identical sinusoidal curvature wave (serpenoid gait) to all three
2. Measure steady-state forward velocity
3. Tune MuJoCo `friction` parameters until forward speed matches dismech/Elastica
4. Also match:
   - Lateral slip ratio (sideways drift / forward displacement)
   - Turning radius for asymmetric gaits
   - Locomotion efficiency (distance per energy unit)

**Important:** This matching is gait-specific. Parameters tuned for one gait waveform may not transfer to a different gait. Document the gait used for matching.

### 7. Curvature Control Mechanism

All three accept curvature as the control input, but the internal mechanism is different:

| Framework | Mechanism | What Happens Physically |
|-----------|-----------|------------------------|
| dismech | Set natural curvature on bend springs (`nat_strain`) | Energy minimum shifts; rod relaxes toward target shape |
| Elastica | Set `rest_kappa` on rod | Same — rest configuration changes |
| MuJoCo (rigid chain) | Set actuator target: `ctrl = κ × L_seg` | PD controller drives joint angle toward target |

**Key distinction:**

- **dismech/Elastica:** Curvature is the *natural shape* — the rod physically wants to be at that curvature. The approach speed depends on material stiffness and damping (passive dynamics).
- **MuJoCo:** Curvature is a *control target* — the PD controller actively drives toward it. The approach speed depends on `kp` (proportional gain) and damping (derivative term).

**Matching condition:** For a step change in curvature input, all three should:

1. Reach the target curvature in approximately the same time (rise time)
2. Show similar overshoot/undershoot behavior
3. Settle to the same final curvature (steady-state accuracy)

This is coupled to the stiffness and damping matching (sections 2 and 5). If stiffness and damping are correctly matched, curvature response should follow automatically.

### 8. Time Integration

| Framework | Integrator | Typical Internal dt | Substeps per RL Step |
|-----------|-----------|--------------------|--------------------|
| dismech (Python) | Implicit Euler (Newton iterations) | 0.01–0.05 s | 1–5 |
| dismech (C++) | Backward Euler or Implicit Midpoint | 0.01–0.05 s | 1–5 |
| Elastica | PositionVerlet or PEFRL (symplectic) | 0.001 s | ~50 |
| MuJoCo | Semi-implicit Euler | 0.002 s | ~25 |

**Matching condition:** Same total RL timestep, not same internal timestep.

```
dt_rl = internal_dt × num_substeps

Example: dt_rl = 0.05 s for all three
    dismech:  0.05 s × 1 substep  = 0.05 s  ✓
    Elastica: 0.001 s × 50 substeps = 0.05 s  ✓
    MuJoCo:   0.002 s × 25 substeps = 0.05 s  ✓
```

The internal timestep is dictated by each engine's stability requirements — do not try to unify it.

## Parameter Classification Summary

```
FIXED (numerically identical across all three):
├── Snake geometry: length, radius, num_segments, total mass
├── Prey geometry: radius, length, position
├── RL timestep: dt_rl (e.g., 0.05 s)
├── Curvature control inputs: same command sequence to all three
├── Observation extraction: unified wrapper interface
└── Reward function: identical for all three

DERIVED (analytically converted from a single ground truth):
├── Bending stiffness: define EI → derive kp = EI/L_seg, E = EI/I
├── Stretching stiffness: define EA → derive E (Elastica only)
├── Twisting stiffness: define GJ → derive twist, G
└── Mass distribution: define density → compute per-node/segment masses

TUNED (empirically matched via controlled experiments):
├── Damping → match free vibration decay time t½ (test in free space, no ground)
├── Ground friction / locomotion → match steady-state forward speed for reference gait
└── Curvature response → verify similar transient behavior (rise time, overshoot)

IRREDUCIBLE DIFFERENCES (document, do not try to unify):
├── Internal timestep (engine stability requirement)
├── Integration scheme (implicit vs symplectic vs semi-implicit)
├── Contact normal force model (penalty spring vs constraint solver vs none)
├── Friction–contact coupling (RFT: independent of normal force; MuJoCo: F_t ≤ μ×F_n)
├── Damping target (dismech: all velocities; MuJoCo: joint angular velocity only)
├── Stretching DOF (DER/Cosserat have it, MuJoCo rigid chain doesn't)
├── Twist DOF (DER/Cosserat have it, MuJoCo hinge joints don't)
├── Elastic energy fidelity (analytical vs actuator-spring approximation)
└── Geometry–physics coupling (monolithic vs modular)
```

## Validation Experiments

Before running RL or comparing training results, run these controlled tests to verify parameter matching. All tests use the same initial conditions and curvature inputs.

### Test 1: Static Bending (Validates Stiffness Matching)

- **Setup:** Straight rod, apply constant curvature κ = 5 rad/m, wait for equilibrium
- **Measure:** Final node positions
- **Expected:** All three should converge to the same arc shape
- **Diagnostic:** If shapes differ, stiffness conversion is wrong

### Test 2: Free Vibration (Validates Stiffness + Damping Matching)

- **Setup:** Bend rod to κ₀ = 10 rad/m, release from rest (set curvature to 0)
- **Measure:** Curvature at midpoint over time → oscillation frequency and decay envelope
- **Expected:** Same natural frequency (validates stiffness); same decay rate (validates damping)
- **Diagnostic:** Frequency mismatch → fix stiffness. Decay mismatch → tune damping.

### Test 3: Straight-Line Locomotion (Validates Friction Matching)

- **Setup:** Apply serpenoid gait: `κ(s,t) = A × sin(ωt − ks)` with fixed A, ω, k
- **Measure:** Forward velocity at steady state, lateral slip ratio, energy consumed
- **Expected:** Same forward speed (validates friction matching)
- **Diagnostic:** Speed mismatch → tune MuJoCo friction parameters

### Test 4: Curvature Step Response (Validates Control Dynamics Matching)

- **Setup:** Straight rod at rest, step curvature from 0 to κ_target at t=0
- **Measure:** Curvature at midpoint over time → rise time, overshoot, settling time
- **Expected:** Similar transient response across all three
- **Diagnostic:** If MuJoCo responds much faster/slower, adjust kp and damping together

### Test 5: Energy Comparison (Quantifies Irreducible Difference)

- **Setup:** Same gait as Test 3, run for 10 seconds
- **Measure:** Total energy (kinetic + elastic + gravitational) at each RL step
- **Expected:** Energy profiles will differ — this quantifies the irreducible physics gap
- **Purpose:** Not for tuning. This is the measurement of how different the engines truly are after matching everything you can control.

## Recommended Workflow

```
1. Set geometry and mass      (Section 1 — direct copy)
         │
2. Derive stiffness params    (Section 2–4 — analytical conversion)
         │
3. Run Test 1 (static bend)   → verify stiffness
         │
4. Tune damping               (Section 5 — empirical)
         │
5. Run Test 2 (free vibration) → verify stiffness + damping
         │
6. Tune friction              (Section 6 — empirical)
         │
7. Run Test 3 (locomotion)    → verify friction
         │
8. Run Test 4 (step response) → verify control dynamics
         │
9. Run Test 5 (energy)        → quantify irreducible gap
         │
10. Run RL comparison          → compare training with matched parameters
```

## Related

- `doc/knowledge/physics-framework-comparison.md` — Full comparison of all backends (API, contact, energy, etc.)
- `doc/knowledge/mujoco-body-representations-and-physics.md` — MuJoCo body representations and elasticity plugin history
- `src/snake_hrl/physics/__init__.py` — Factory function routing by `SolverFramework`
- `src/snake_hrl/configs/env.py` — `PhysicsConfig` with backend-specific fields
- `src/snake_hrl/physics/mujoco_snake_robot.py` — Current MuJoCo rigid-chain implementation
- `src/snake_hrl/physics/snake_robot.py` — dismech (Python) backend
