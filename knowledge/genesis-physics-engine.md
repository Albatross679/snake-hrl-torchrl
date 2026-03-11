---
name: genesis-physics-engine
description: Research on Genesis generative physics engine for robotics and embodied AI
type: knowledge
created: 2026-03-09T00:00:00
updated: 2026-03-09T00:00:00
tags: [genesis, physics-engine, gpu-simulation, robotics, rl]
aliases: []
---

# Genesis Physics Engine

## What is Genesis?

Genesis is a **generative and universal physics engine** for robotics and embodied AI, developed through a 24-month collaboration across 20+ research labs (MIT, Stanford, UC Berkeley, CMU, ETH Zurich, and others). It was publicly released in December 2024 and is actively developed under the **Genesis AI** organization.

- **GitHub**: https://github.com/Genesis-Embodied-AI/Genesis
- **Docs**: https://genesis-world.readthedocs.io/
- **License**: Apache 2.0 (fully open source)
- **PyPI**: `pip install genesis-world`
- **Current version**: 0.4.1 (as of March 2026)
- **Stars**: ~28k GitHub stars, 87+ contributors

Genesis is written in **100% Python** (using Taichi for GPU kernels) and positions itself as a lightweight, Pythonic alternative to heavier frameworks like Isaac Sim.

## Physics Backends / Solvers

Genesis integrates **six physics solvers** into a unified, coupled framework:

| Solver | Type | Materials / Use Cases |
|--------|------|----------------------|
| **Rigid Body** | Articulated body dynamics | Robots, rigid objects, joints, URDF/MJCF |
| **MPM** (Material Point Method) | Continuum | Snow, granular materials, soft muscles, elastoplastic bodies |
| **SPH** (Smoothed Particle Hydrodynamics) | Particle-based fluid | Water, liquids |
| **FEM** (Finite Element Method) | Continuum | Deformable tissue, soft bodies |
| **PBD** (Position-Based Dynamics) | Constraint-based | Cloth, rope, thin-shell objects |
| **Stable Fluid** | Eulerian fluid | Smoke, gas |

These solvers can run simultaneously and are **coupled** through a `gs.engine.Coupler` system that handles cross-solver interactions (e.g., rigid-soft contact) using impulse-based methods with configurable restitution and Coulomb friction.

## Soft Body / Elastic Rod Support

### What Genesis supports:
- **Volumetric soft bodies** via MPM and FEM (e.g., deformable tissue, soft robot actuators)
- **Muscle simulation** via `gs.materials.MPM.Muscle` with configurable Young's modulus, Poisson's ratio, density, and constitutive models (neo-Hookean, stable neo-Hookean)
- **Hybrid robots** (rigid skeleton + soft MPM skin)
- **Thin-shell objects** via PBD (cloth, fabrics)
- **Granular materials** via MPM

### What Genesis does NOT support:
- **No explicit Cosserat rod / elastic rod (1D) solver**. Genesis handles deformable bodies through volumetric methods (MPM, FEM) and thin shells (PBD), but does not have a dedicated 1D rod solver like PyElastica's Cosserat rod model or DisMech's DER formulation.
- A snake body modeled as a Cosserat rod would need to be approximated as either a volumetric FEM/MPM mesh or an articulated rigid body with joints (URDF/MJCF).

## GPU-Parallelized RL Environments

Genesis has **first-class support** for massively parallel RL training:

- Gym-style environment API with `__init__`, `reset_idx`, `step` methods
- Supports **thousands of parallel environments** on a single GPU (examples show 8192 envs)
- Claims **43 million FPS** simulating a Franka arm on RTX 4090 (430,000x real-time)
- Claims **10-80x faster** than Isaac Gym/Sim/Lab and MuJoCo MJX
- Built-in reward function registration system
- Integration with **rsl-rl** (PPO) for locomotion policy training
- Observations, actions, and rewards are batched tensors (PyTorch)

### Speed Claims Caveat
The MuJoCo community has challenged Genesis's benchmark methodology:
- Speed comparisons were primarily against Isaac Sim, not MuJoCo MJX directly
- Benchmarks reportedly used "fastest physics settings" with reduced accuracy
- Self-collisions were disabled by default in benchmarks
- Mostly static scenes (1 action + 999 no-op steps)
- MJX lacks mesh-based collision, making fair comparison difficult

Real-world performance is likely excellent but the "10-80x" figure should be taken with context.

## Python API

Genesis is entirely Pythonic. Example workflow:

```python
import genesis as gs

gs.init(backend=gs.cuda)  # or gs.cpu, gs.metal

# Create scene
scene = gs.Scene(dt=0.01, gravity=(0, 0, -9.81))

# Add entities
plane = scene.add_entity(gs.morphs.Plane())
robot = scene.add_entity(
    gs.morphs.URDF(file='robot.urdf', pos=(0, 0, 0.5)),
)

# Soft body
soft = scene.add_entity(
    morph=gs.morphs.Sphere(pos=(0.5, 0.2, 0.3), radius=0.1),
    material=gs.materials.MPM.Muscle(E=3e4, nu=0.45, rho=1000.)
)

# Build scene (compiles GPU kernels)
scene.build(n_envs=4096)  # parallel environments

# Control
robot.set_dofs_kp(...)
robot.set_dofs_kv(...)
robot.control_dofs_position(target_pos, dofs_idx_local=dof_ids)
robot.control_dofs_force(torques, dofs_idx_local=dof_ids)

# Step
scene.step()

# State queries
pos = robot.get_dofs_position()
vel = robot.get_dofs_velocity()
force = robot.get_dofs_force()
```

### File Format Support
MJCF (.xml), URDF, .obj, .glb, .ply, .stl

### Control Modes
- `control_dofs_position()` - PD position control (persistent)
- `control_dofs_velocity()` - velocity control
- `control_dofs_force()` - direct torque/force control
- `set_dofs_position()` - hard state setting (bypasses physics)

### External Forces
- `apply_links_external_force()` and `apply_links_external_torque()` on rigid bodies (added Dec 2024, PR #135)
- `CustomForceField` API with Taichi function: `f(pos, vel, t) -> acceleration`

## Comparison with Current Stack

| Feature | Genesis | PyElastica | MuJoCo | Isaac Gym |
|---------|---------|------------|--------|-----------|
| **Cosserat rods** | No | Yes (core feature) | No | No |
| **Elastic rod (1D)** | No | Yes | No | No |
| **Soft body (3D)** | Yes (MPM/FEM) | No | Limited | Yes |
| **GPU parallel envs** | Yes (thousands) | No (CPU only) | MJX (JAX) | Yes |
| **Speed** | Very fast | Slow (~57 FPS) | Fast (MJX) | Fast |
| **Differentiable** | Partial (MPM only) | Yes | MJX: Yes | No |
| **Custom force models** | CustomForceField | Yes (RFT, etc.) | Custom | Limited |
| **Python API** | Pure Python | Pure Python | C + Python | Python/YAML |
| **Open source** | Apache 2.0 | MIT | Apache 2.0 | Proprietary |
| **Maturity** | Young (v0.4) | Mature | Very mature | Mature |
| **Articulated bodies** | Yes (URDF/MJCF) | No | Yes | Yes |
| **Multi-material coupling** | Yes (6 solvers) | No | Limited | Limited |

### Key Trade-offs for This Project (Snake Locomotion)

**Advantages of switching to Genesis:**
- Massively parallel GPU environments would dramatically speed up RL training (currently ~57 FPS with 16 CPU envs in PyElastica)
- Native RL training pipeline with gym-style API
- Could model snake as articulated rigid body (URDF) with joint-based actuation
- Multi-material coupling could enable interesting terrain interactions

**Disadvantages / Risks:**
- **No Cosserat rod model** -- the snake body cannot be modeled as a continuous elastic rod with bending/torsion dynamics; would need to be discretized as an articulated rigid body chain
- **No built-in RFT** (Resistive Force Theory) for ground friction; would need to implement via CustomForceField (limited to `f(pos, vel, t)` signature -- may not have access to surface normals or body geometry)
- **Young project** (v0.4) -- API may change, documentation gaps, fewer community resources
- **Differentiable simulation only for MPM** currently, not for rigid body solver
- **Speed claims contested** -- real speedup over MJX likely less than advertised

## Custom Force Models (RFT Feasibility)

Genesis provides `gs.force_fields.Custom` which accepts a Taichi function:
```python
@ti.func
def rft_force(pos: ti.types.vector(3), vel: ti.types.vector(3), t: ti.f32) -> ti.types.vector(3):
    # Only has access to position, velocity, and time
    # Does NOT have access to body orientation, surface normals, or geometry
    return acceleration
```

This is **insufficient for proper RFT implementation**, which requires:
- Local body frame orientation (to decompose velocity into tangential/normal components)
- Surface normal direction at each contact point
- Body geometry (cross-section, curvature)

A workaround might involve using the external force API (`apply_links_external_force`) computed manually each step, but this would run on CPU and negate GPU parallelization benefits.

## Installation Requirements

- Python >= 3.10, < 3.14
- PyTorch (installed separately)
- Linux, macOS, or Windows
- GPU: NVIDIA (CUDA), AMD, or Apple Metal
- CPU-only mode also supported
- `pip install genesis-world`
- Optional extras: `[dev]`, `[docs]`, `[render]`, `[usd]`

## Maturity Assessment

- **Strengths**: Extremely fast rigid body simulation, excellent GPU parallelization, clean Python API, strong institutional backing, active development, Apache 2.0 license
- **Weaknesses**: Young project (first public release Dec 2024), partial differentiability, no elastic rod solver, contested benchmarks, generative features still rolling out, documentation gaps in advanced features
- **Production readiness**: Suitable for rigid-body RL training (locomotion, manipulation). Not yet suitable for specialized soft body simulations requiring Cosserat rod models.

## Recommendation for This Project

Genesis is not a drop-in replacement for PyElastica for snake locomotion because it lacks Cosserat rod physics. Two paths forward:

1. **Articulated rigid body approximation**: Model the snake as a chain of rigid links connected by revolute joints (URDF). This loses continuous elastic dynamics but gains massive GPU parallelization. Could work if the RL policy can learn to compensate for the discretization.

2. **Hybrid approach**: Use PyElastica for physics-accurate Cosserat rod simulation during development/validation, then transfer to a Genesis rigid-body approximation for large-scale RL training.

The MuJoCo path (already in the project's physics backends) may be a more practical middle ground -- MuJoCo has mature articulated body support, MJX provides GPU parallelization via JAX, and it has a much larger community and ecosystem.
