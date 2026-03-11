---
name: mujoco-body-representations-and-physics
description: MuJoCo body representation systems (rigid chain, composite, flex) and elasticity physics frameworks
type: knowledge
created: 2026-02-14T00:00:00
updated: 2026-02-16T00:00:00
tags: [knowledge, mujoco, physics, flex, elasticity, deformable, body-representation, cosserat, kirchhoff, der]
aliases: []
---

# MuJoCo Body Representations and Physics Frameworks

MuJoCo provides three distinct systems for representing deformable or articulated bodies, and a set of elasticity plugins/engine features for computing physics forces on them. This document covers both.

## Body Representation Systems

### 1. Rigid Body Chain

The oldest and most common approach. Individual rigid bodies connected by explicit joints.

```xml
<body name="seg_0">
  <freejoint/>
  <geom type="capsule" size="0.01" fromto="0 0 0  0.05 0 0"/>
  <body name="seg_1" pos="0.05 0 0">
    <joint type="hinge" axis="0 0 1"/>
    <geom type="capsule" size="0.01" fromto="0 0 0  0.05 0 0"/>
  </body>
</body>
```

**Characteristics:**

- Each body has its own mass, inertia, and geometry
- Bodies connected by explicit joints (hinge, ball, slide, free, etc.)
- Forces come from actuators (position, velocity, torque) and joint springs/damping
- No inherent elasticity — elastic behavior must be faked with PD controllers
- Curvature control: `angle = curvature × segment_length`, driven by position actuators

**Use cases:** Articulated robots, kinematic chains, rigid-body approximations of flexible systems.

**This project:** The current `MujocoSnakeRobot` uses this approach — rigid capsule segments + hinge joints + position actuators (`src/snake_hrl/physics/mujoco_snake_robot.py`).

### 2. Composite (deprecated)

A macro system that auto-generates rigid body chains with specific topologies.

```xml
<composite type="rope" count="20" spacing="0.04">
  <joint kind="ball" damping="0.1"/>
  <geom type="capsule" size="0.01"/>
</composite>
```

**Available types (before deprecation):**

| Type | Topology | Description |
|------|----------|-------------|
| `rope` | 1D chain | Ball-joint chain of capsules |
| `loop` | 1D closed | Rope with ends connected |
| `cloth` | 2D grid | Grid of bodies with cross-joints |
| `grid` | 2D grid | Similar to cloth, different connectivity |
| `box` | 3D volume | Volumetric body grid |
| `cylinder` | 3D volume | Cylindrical body grid |
| `sphere` | 3D volume | Spherical body grid |
| `particle` | point cloud | Unconnected mass points |

**Deprecation timeline:**

| Version | Date | Change |
|---------|------|--------|
| 3.2.3 | Sep 2024 | Experimental elasticity plugins with composite discontinued |
| 3.2.5 | Nov 2024 | Box, cylinder, sphere types removed |
| 3.2.6 | Dec 2024 | Rope and loop types removed |

All composite types have been replaced by **flexcomp** equivalents.

**Relationship to rigid body chain:** Composite is fundamentally still rigid bodies with joints under the hood — just auto-generated. It is a convenience macro, not a new physics model.

### 3. Flex (current, MuJoCo 3.0+)

A fundamentally different deformable body primitive introduced in MuJoCo 3.0.0 (October 2023). Not rigid bodies with joints — instead, **collections of bodies connected by massless stretchable elements** forming simplicial complexes.

```xml
<flexcomp name="rod" type="grid" count="20 1 1"
          spacing="0.04" radius="0.01" mass="0.5"
          dim="1">
  <edge damping="0.01" stiffness="5.0"/>
  <plugin plugin="mujoco.elasticity.cable">
    <config key="twist" value="1e4"/>
    <config key="bend" value="1e4"/>
  </plugin>
</flexcomp>
```

**Dimensionality:**

| Dimension | Element Type | Example |
|-----------|-------------|---------|
| 1D | Segments (capsules) | Ropes, cables, elastic rods |
| 2D | Triangles | Cloth, membranes, shells |
| 3D | Tetrahedra | Volumetric soft bodies, tissue |

**Deformation models (built into engine):**

1. **Edge-based soft equality constraints** — simple, permits large timesteps
2. **Continuum FEM** — piecewise linear finite elements with separate shear and volumetric stiffness, based on Saint Venant-Kirchhoff hyperelastic theory

**Key differences from rigid body chains:**

| | Rigid Body Chain | Flex |
|---|---|---|
| **Connectivity** | Explicit joints (hinge, ball) | Massless stretchable elements |
| **Deformation** | Joint angles | Vertex displacements |
| **Elasticity** | Faked via actuators/springs | Native (edge stiffness, FEM, plugins) |
| **Collision** | Per-geom, pairwise | Distributed across elements (up to 8 bodies) |
| **Dimensionality** | 1D only (chains) | 1D, 2D, 3D |
| **Physics model** | Rigid body dynamics | Saint Venant-Kirchhoff / elastic rods |

**`flex` vs `flexcomp`:**

- `flex` — low-level MJCF element inside `<deformable>`, requires manual vertex/element specification
- `flexcomp` — high-level convenience macro (analogous to composite), auto-generates flex from specs. Supports mesh files, GMSH tetrahedral format, and grid topologies.

## Elasticity Physics Frameworks

Separate from the body representation, MuJoCo provides physics models that compute passive elastic forces on flex (or formerly composite) bodies. These started as external plugins and have been progressively absorbed into the engine.

The 1D/2D/3D labels refer to the **dimensionality of the object**, not the space it lives in — all objects deform in 3D space:

| Dimension | Object geometry | Example | Rod theory |
|-----------|----------------|---------|------------|
| **1D** | Curve (length only) | Ropes, rods, snake centerline | DER (Kirchhoff), Cosserat rod |
| **2D** | Surface (length + width) | Cloth, shells, membranes | Kirchhoff-Love plate, Reissner-Mindlin |
| **3D** | Volume (length + width + depth) | Rubber blocks, tissue | Continuum FEM |

### Cable Plugin (`mujoco.elasticity.cable`)

Models an **inextensible 1D elastic rod** — the same physics as Discrete Elastic Rods (DER), which is what dismech-python and dismech-rods implement. Both DER and MuJoCo's cable plugin implement **Kirchhoff rod** theory: the rod is inextensible and unshearable, with only bending and twisting degrees of freedom. This is a special case of the more general **Cosserat rod** theory (used by PyElastica), which additionally allows the centerline to stretch/compress and cross-sections to shear relative to the centerline.

**Parameters:**

| Parameter | Unit | Description |
|-----------|------|-------------|
| `twist` | Pa | Twisting stiffness |
| `bend` | Pa | Bending stiffness |
| `flat` | bool | If true, stress-free config is straight; if false, uses XML shape |
| `vmax` | N/m² | Optional stress visualization color scale |

**Status:** Still an active external plugin (the only elasticity plugin remaining).

**Timeline:**

| Version | Date | Event |
|---------|------|-------|
| 2.3.0 | Oct 2022 | **Created** as cable passive force plugin |
| 2.3.1 | Dec 2022 | Plugin library renamed from "cable" to "elasticity" |
| 3.0.0 | Oct 2023 | Migrated to work with flex system |
| 3.1.0 | Dec 2023 | Bug fix for use with pinned flex vertices |
| 3.2.6 | Dec 2024 | Recommended as replacement for deprecated composite rope/loop |

### Shell Plugin (`mujoco.elasticity.shell`)

Computed **bending forces for thin 2D surfaces** using a constant precomputed Hessian.

**Timeline:**

| Version | Date | Event |
|---------|------|-------|
| 3.0.0 | Oct 2023 | **Created** |
| 3.3.3 | Jun 2025 | **Removed** — replaced by `flexcomp` with `elastic2d` attribute |
| 3.3.6 | Sep 2025 | Curved reference configurations supported natively in engine |

### Solid Plugin (`mujoco.elasticity.solid`)

Computed **volumetric elasticity** for 3D deformable bodies using piecewise-constant strain (equivalent to linear FEM).

**Timeline:**

| Version | Date | Event |
|---------|------|-------|
| 2.3.1 | Dec 2022 | **Created** — tetrahedral mesh with piecewise-constant strain |
| 3.0.0 | Oct 2023 | Migrated to work with flex system |
| 3.2.4 | Oct 2024 | **Removed as plugin** — moved into engine |

### Membrane Plugin (`mujoco.elasticity.membrane`)

Computed **2D surface forces** for thin deformable surfaces (no bending, just in-plane stretch).

**Timeline:**

| Version | Date | Event |
|---------|------|-------|
| 3.0.0 | Oct 2023 | **Created** |
| 3.2.4 | Oct 2024 | **Removed as plugin** — moved into engine |

### Summary Table

| Plugin | Introduced | Removed/Absorbed | Current Status |
|--------|-----------|-------------------|---------------|
| Cable | v2.3.0 (Oct 2022) | — | **Active plugin** |
| Shell | v3.0.0 (Oct 2023) | v3.3.3 (Jun 2025) | Absorbed into engine (`elastic2d`) |
| Solid | v2.3.1 (Dec 2022) | v3.2.4 (Oct 2024) | Absorbed into engine (flex stiffness) |
| Membrane | v3.0.0 (Oct 2023) | v3.2.4 (Oct 2024) | Absorbed into engine (flex stiffness) |

**Trend:** MuJoCo is moving all elasticity computations into the engine natively via the flex system. Cable is the only remaining external plugin.

## Key MuJoCo Versions for Deformable Bodies

| Version | Date | Milestone |
|---------|------|-----------|
| 2.3.0 | Oct 2022 | Cable plugin and composite cable type introduced |
| 2.3.1 | Dec 2022 | Solid plugin added, plugin library renamed to "elasticity" |
| **3.0.0** | **Oct 2023** | **Flex system introduced** — flex, flexcomp, deformable section, shell/membrane plugins |
| 3.1.0 | Dec 2023 | Bug fixes for flex + elasticity plugin interaction |
| 3.2.3 | Sep 2024 | Elasticity plugins with composite deprecated |
| 3.2.4 | Oct 2024 | Solid and membrane plugins absorbed into engine |
| 3.2.5 | Nov 2024 | Composite box/cylinder/sphere removed, model editing stabilized |
| 3.2.6 | Dec 2024 | Composite rope/loop removed |
| 3.3.0 | Feb 2025 | Fast deformable bodies via trilinear flexcomp; separate collision/deformation meshes |
| 3.3.3 | Jun 2025 | Shell plugin removed, replaced by `elastic2d` |
| 3.3.6 | Sep 2025 | Curved shell reference configurations |
| 3.4.0 | Dec 2025 | Quadratic flexcomp/dof option for fast deformables |
| **3.5.0** | **Feb 2026** | **Latest** — implicit integration for deformable flex, flexvert equality constraints |

## Relevance to This Project

The current `MujocoSnakeRobot` uses **rigid body chain** (approach #1) with position actuators. This has no true elastic physics — curvature is driven by PD controllers that approximate bending.

A **flex + cable plugin** approach (approach #3) would:

- Model the snake as an actual elastic rod with material stiffness (twist, bend in Pascals)
- Match the DER physics used by dismech-python and dismech-rods backends
- Use native MuJoCo elastic energy instead of the `0.5 * kp * (q - q_target)²` approximation
- Maintain MuJoCo's advantages: fast C engine, native contact, easy install

**Open questions for migration:**

- How to apply curvature control to a flex body (natural curvature vs actuator control)
- Whether MuJoCo's cable plugin provides enough fidelity compared to dismech's DER implementation
- Contact handling between flex cable and rigid prey geometry
- How to extract node positions/velocities from flex state for the existing observation pipeline

## Related

- `doc/knowledge/physics-framework-comparison.md` — Comparison of all physics backends in this project
- `src/snake_hrl/physics/mujoco_snake_robot.py` — Current rigid-body MuJoCo implementation
- [MuJoCo Modeling Documentation](https://mujoco.readthedocs.io/en/latest/modeling.html)
- [MuJoCo Elasticity Plugin README](https://github.com/google-deepmind/mujoco/blob/main/plugin/elasticity/README.md)
- [MuJoCo Changelog](https://mujoco.readthedocs.io/en/stable/changelog.html)
