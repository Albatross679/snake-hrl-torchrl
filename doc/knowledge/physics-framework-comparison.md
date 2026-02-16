---
name: physics-framework-comparison
description: Structured comparison of all physics frameworks used or considered for snake robot simulation
type: knowledge
created: 2026-02-09T23:00:00
updated: 2026-02-10T00:00:00
tags: [knowledge, physics, comparison, dismech, mujoco, pybullet, elastica]
aliases: []
---

# Physics Framework Comparison

Five physics frameworks are used or considered in this project for snake robot simulation. Three are discrete elastic rod (DER) simulators that model the snake as a continuous flexible rod. Two are rigid-body simulators that model the snake as a chain of rigid links connected by joints.

## Quick Reference

| | dismech-python | dismech-rods (py_dismech) | PyElastica | MuJoCo | PyBullet |
|---|---|---|---|---|---|
| **Status** | Default backend | Backend | Backend | Backend | Legacy (`snakebot-gym/`) |
| **Import** | `import dismech` | `import py_dismech` | `import elastica` | `import mujoco` | `import pybullet` |
| **Language** | Pure Python | C++ + pybind11 | Pure Python | C (Python bindings) | C++ (Python bindings) |
| **Physics model** | Discrete elastic rod | Discrete elastic rod | Cosserat rod | Rigid-body multibody | Rigid-body multibody |
| **Snake representation** | 1D elastic rod (nodes + edges) | 1D elastic rod (limb) | Cosserat rod (continuous) | Chain of capsule bodies + hinge joints | Chain of box bodies + revolute joints |
| **Integrator** | Implicit Euler | Backward Euler / Implicit Midpoint | Symplectic (PositionVerlet, PEFRL) | Semi-implicit Euler | Semi-implicit Euler |
| **Typical dt** | 0.05 s | 0.05 s | 0.001 s (×50 substeps) | 0.002 s (×25 substeps) | 1/240 s |
| **Config enum** | `DISMECH` | `DISMECH_RODS` | `ELASTICA` | `MUJOCO` | N/A |
| **Wrapper class** | `SnakeRobot` | `DismechRodsSnakeRobot` | `ElasticaSnakeRobot` | `MujocoSnakeRobot` | `SnakeRobot` (snakebot-gym) |

## Physics Model

### Elastic Rod Backends (dismech-python, dismech-rods, PyElastica)

All three model the snake as a continuous deformable rod discretized into nodes and edges. Deformation produces real elastic strain energy (bending, stretching, twisting). The rod has intrinsic material properties (Young's modulus, density, Poisson ratio) that govern stiffness.

- **dismech-python / dismech-rods**: Discrete Elastic Rod (DER) model. The rod state is a vector of node positions + edge twist angles. Bending energy is computed from discrete curvature (turning angle between consecutive edges). Both use the same underlying mathematical model but have completely different implementations and APIs.
- **PyElastica**: Cosserat rod theory. Tracks position, director frame, velocity, and angular velocity along the rod centerline. Richer model than DER — captures shear and extension naturally. Uses explicit symplectic integrators so requires much smaller timesteps (compensated by substeps).

### Rigid-Body Backends (MuJoCo, PyBullet)

Both model the snake as a kinematic chain of rigid bodies connected by joints. No elastic deformation — curvature is achieved by setting joint angles via position actuators/servos. "Elastic energy" in MuJoCo is approximated as actuator spring energy (`0.5 * kp * (q - q_target)^2`).

- **MuJoCo**: MJCF-defined scene. Capsule geometry for each segment (avoids contact instabilities at endpoints). Position actuators with configurable stiffness (`kp`) and damping. Ground contact via MuJoCo's native contact solver.
- **PyBullet**: `createMultiBody` API. Box geometry for each segment. Three motor modes (position, velocity, torque). Anisotropic friction for locomotion. Legacy implementation in `snakebot-gym/`.

## Curvature Control

| Backend | Mechanism | Internal Representation |
|---|---|---|
| dismech-python | Mutate `bend_springs.nat_strain[i, 0]` directly | Natural curvature of DER bend springs |
| dismech-rods | Pass `{"curvature": matrix}` to `step_simulation()` | `(n, 4)` matrix: `[limb_idx, edge_idx, cx, cy]`, edge_idx 1-based |
| PyElastica | Set `rod.rest_kappa` array | Rest curvature of Cosserat rod |
| MuJoCo | Set `data.ctrl[i] = curvature * segment_length` | Joint angle targets for position actuators |
| PyBullet | `setJointMotorControl2(targetPosition=angle)` | Joint angle targets for position servos |

**Key distinction**: In DER/Cosserat backends, curvature is the native control variable (the rod "wants" to bend to the specified curvature). In rigid-body backends, curvature is converted to joint angles (`angle = curvature × arc_length`), and position controllers drive joints to those angles.

## Time Integration

| Backend | Method | Stability | Step Size |
|---|---|---|---|
| dismech-python | Implicit Euler (Newton iterations) | Unconditionally stable | Large (0.05 s) |
| dismech-rods | Backward Euler or Implicit Midpoint | Unconditionally stable | Large (0.05 s) |
| PyElastica | PositionVerlet or PEFRL (symplectic) | Conditionally stable | Small (0.001 s, 50 substeps per RL step) |
| MuJoCo | Semi-implicit Euler | Conditionally stable | Small (0.002 s, 25 substeps per RL step) |
| PyBullet | Semi-implicit Euler | Conditionally stable | Small (1/240 s) |

Implicit integrators (dismech backends) allow large timesteps without instability but require solving a nonlinear system (Newton iterations) each step. Explicit/semi-implicit integrators (PyElastica, MuJoCo, PyBullet) are cheaper per step but need small dt for stability, compensated by running multiple substeps per RL step.

## Contact

Three types of contact are relevant to the snake simulation: snake-ground (locomotion), snake-prey (coiling), and self-contact (snake body crossing itself). Each backend handles these very differently.

### Contact Capabilities Summary

| Capability | dismech-python | dismech-rods | PyElastica | MuJoCo | PyBullet |
|---|---|---|---|---|---|
| **Snake-ground** | RFT + penalty floor | Penalty floor + friction | RFT or damping | Native contact solver | Native contact solver |
| **Snake-prey (physics)** | No | No | No | Supported (currently off) | Supported |
| **Self-contact** | Yes (IMC energy) | Yes (FCL + Lumelsky) | No | Yes (automatic) | Yes (automatic) |
| **Non-planar ground** | No (plane only) | No (plane only) | No (implicit plane) | Yes (arbitrary mesh) | Yes (arbitrary mesh) |
| **Contact friction** | Stick/slip model | Stick/slip model | No | Friction cones | Coulomb friction |

### Snake-Ground Contact (Locomotion)

This is what makes the snake move. Anisotropic ground interaction (more resistance to sideways sliding than forward sliding) converts lateral undulation into forward thrust.

| Backend | Method | Details |
|---|---|---|
| dismech-python | **RFT** (velocity drag) + **penalty floor** | RFT: `F = -ct*v_t - cn*v_n` (anisotropic drag). Floor: exponential penalty when `node_z < ground_z`. Config: `rft_ct`, `rft_cn`, `floorContact: {ground_z, stiffness, delta}` |
| dismech-rods | **Penalty floor** + optional **DampingForce** | `FloorContactForce(floor_delta, floor_slipTol, floor_z, floor_mu)`. Same exponential penalty as dismech-python. Optional friction with stick/slip. RFT not available. |
| PyElastica | **RFT** (velocity drag) | Custom `RFTForcing` class: `F = -ct*v_t - cn*v_n`. No geometric floor — drag is applied unconditionally based on velocity. Alternative: `AnalyticalLinearDamper`. |
| MuJoCo | **Native contact solver** | Ground plane with configurable friction `(slide, spin, roll)`. Rigid-body contact with friction cones. Config: `mujoco_friction: (1.0, 0.005, 0.0001)` |
| PyBullet | **Native contact solver** | Ground plane with anisotropic friction `[1, 0.01, 0.01]` + lateral friction. |

**Note on RFT**: Resistive Force Theory is a velocity-proportional drag model, not a geometric contact model. It cannot model contact normals, non-penetration, or forces between two bodies. It only models the anisotropic ground drag that enables serpentine locomotion.

### Snake-Prey Contact (Coiling)

In the current design, snake-prey contact is **purely analytical** across all backends — no physics forces are exchanged. The prey is not part of any physics simulation. Contact is detected geometrically in `geometry.py` and used only for reward computation and observations.

- `compute_contact_points(snake, prey)` — per-node distance to prey cylinder surface, returns boolean contact mask
- `compute_wrap_angle(snake, prey)` — total signed angle wrapped around prey axis

This means the snake does not physically "feel" the prey. It can pass through the prey geometry — coiling behavior must be learned entirely through reward shaping.

**To enable physical snake-prey contact:**

| Backend | Difficulty | How |
|---|---|---|
| dismech-python | Hard | Would need a new force class computing penalty forces against the prey cylinder geometry. No rod-rigid-body contact exists. |
| dismech-rods | Hard | Same — `ContactForce` handles rod-rod only (capsule geometry via FCL). Extending to rod-cylinder would require modifying C++ collision detection. |
| PyElastica | Hard | No contact infrastructure in the current wrapper. |
| MuJoCo | **Trivial** | Change prey from `contype="0"` to `contype="1"` in MJCF XML. MuJoCo's contact solver handles capsule-cylinder collisions automatically. |
| PyBullet | Easy | Add collision shape to prey body. Built-in rigid-body contact handles the rest. |

### Self-Contact

Self-contact prevents the snake body from passing through itself. Important for coiling (the snake must wrap around prey without self-intersection).

| Backend | Support | Implementation |
|---|---|---|
| dismech-python | **Yes** | IMC (Implicit Minimal Coordinate) contact energy. Three contact types: Point-to-Point, Point-to-Edge, Edge-to-Edge. Smooth penalty potential with energy-based compliance. Includes stick/slip friction. Config: `selfContact: {delta, h, kc}`, `selfFriction: {mu, vel_tol}` in `Environment`. Source: `dismech/contact/imc_energy.py` |
| dismech-rods | **Yes** | FCL (Flexible Collision Library) for broad-phase AABB tree, Lumelsky distance computation for narrow-phase. Same P2P/P2E/E2E constraint types. Capsule-based geometry. Configurable friction with ZERO_VEL/SLIDING/STICKING states. API: `py_dismech.ContactForce(soft_robots, col_limit, delta, k_scaler, friction, nu, self_contact=True)`. Source: `contact_force.cpp`, `collision_detector.cpp` |
| PyElastica | **No** | Not wired up in this project (PyElastica library may support it). |
| MuJoCo | **Yes** | Automatic — all capsule geoms collide with each other via MuJoCo's built-in contact solver. No additional configuration needed. |
| PyBullet | **Yes** | Automatic for multi-body links with collision shapes. |

**Current status**: Self-contact is **not enabled** in any backend's wrapper class. The dismech backends have the capability but it is not configured in `SnakeRobot` or `DismechRodsSnakeRobot`. MuJoCo handles it automatically but the effect may be minimal at the current snake radius.

### Non-Planar Ground

| Backend | Support | Details |
|---|---|---|
| dismech-python | **No** | Floor is a single `ground_z` height. Penalty force checks `node_z < ground_z`. |
| dismech-rods | **No** | `FloorContactForce` takes a single `floor_z` parameter. Flat plane only. |
| PyElastica | **No** | RFT has no surface geometry — drag is applied based on velocity alone. |
| MuJoCo | **Yes** | Ground can be any MJCF geometry: heightfield, mesh, composed primitives. Replace `<geom type="plane">` with arbitrary terrain. |
| PyBullet | **Yes** | Supports heightfield terrain and arbitrary mesh collision shapes via URDF/SDF. |

## Energy Computation

| Backend | Source | Fidelity |
|---|---|---|
| dismech-python | `stepper.compute_total_elastic_energy(state)` | Exact (from internal force model) |
| dismech-rods | Fallback: `0.5 * EA * sum(stretch^2)` from positions | Approximate (stretch only, no bending energy) |
| PyElastica | Computed from rod internal strains | Exact (Cosserat strain energy) |
| MuJoCo | `0.5 * kp * (q - q_target)^2` per actuator | Approximate (rigid-body analog) |
| PyBullet | Not computed | N/A |

## State Reading

| Backend | Positions | Velocities |
|---|---|---|
| dismech-python | `robot.state.q[:3N].reshape(N, 3)` (flat DOF vector) | `robot.state.u[:3N].reshape(N, 3)` |
| dismech-rods | `limb.getVertices()` → `(N, 3)` | `limb.getVelocities()` → `(N, 3)` |
| PyElastica | `rod.position_collection` → `(3, N)` (transposed) | `rod.velocity_collection` → `(3, N)` |
| MuJoCo | `data.xpos[body_ids]` + tail extrapolation | `mj_objectVelocity()` per body |
| PyBullet | `getBasePositionAndOrientation()` + `getJointStates()` | From joint state tuples |

## Installation

| Backend | Dependencies | Build Complexity |
|---|---|---|
| dismech-python | `pip install -e dismech-python/` | Trivial (pure Python, git submodule) |
| dismech-rods | eigen3, libccd, libfcl, freeglut, symengine, MKL, pybind11 | High (C++ build: cmake + make, MKL threading workaround) |
| PyElastica | `pip install pyelastica` | Trivial (pure Python, pip) |
| MuJoCo | `pip install mujoco>=3.0.0` | Trivial (pre-built wheels) |
| PyBullet | `pip install pybullet` | Low (pre-built wheels) |

### Known Build Issues

- **dismech-rods**: PARDISO solver requires `MKL_NUM_THREADS=1` when torch is also loaded, otherwise symbolic factorization crashes with error -3. Set `os.environ.setdefault("MKL_NUM_THREADS", "1")` before importing `py_dismech`.
- **dismech-rods**: CMake needs `-Dpybind11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')` to find pybind11.

## API Design Pattern

| Backend | Pattern |
|---|---|
| dismech-python | **Decomposed objects**: 5 separate objects (Geometry, GeomParams, Material, SimParams, Environment) → `SoftRobot`. Mutable state on robot, separate `TimeStepper`. |
| dismech-rods | **Centralized manager**: `SimulationManager` owns everything. Configure via properties (`sim_params`, `render_params`, `soft_robots`, `forces`), then `initialize([])`. Control passed at step time via dict. |
| PyElastica | **Simulator collection**: `BaseSystemCollection` with mixin classes (Constraints, Forcing, Damping). Add rod + forces, finalize, then step with `extend_stepper_interface`. |
| MuJoCo | **XML + C state**: Generate MJCF XML → `MjModel.from_xml_string()` → `MjData`. Set `data.ctrl`, call `mj_step()`. All state in `MjData` struct. |
| PyBullet | **Client/server**: `BulletClient` connection. `createMultiBody()` returns integer ID. Control via `setJointMotorControl2()`. State via `getJointStates()`. |

## Strengths and Limitations

### dismech-python
- **+** Default backend, most tested
- **+** Pure Python — easy to debug and modify
- **+** Exact elastic energy computation
- **+** RFT ground interaction (physically motivated for snakes)
- **-** Slow (Python-level Newton solver)
- **-** Only implicit Euler integrator

### dismech-rods (py_dismech)
- **+** Same DER physics as dismech-python but in C++ (faster)
- **+** Multiple integrators (Backward Euler, Implicit Midpoint)
- **+** Rich force model (gravity, damping, floor contact, self-contact)
- **-** Complex build process (C++ deps, MKL, pybind11)
- **-** MKL threading conflict with torch
- **-** No direct elastic energy API (must approximate from positions)
- **-** API uses mutable global state (SimulationManager)

### PyElastica
- **+** Full Cosserat rod physics (richer than DER)
- **+** Pure Python — easy to install
- **+** Symplectic integrators (energy-preserving)
- **+** Active open-source community
- **-** Requires many substeps (small dt for stability)
- **-** Different array conventions (3×N instead of N×3)

### MuJoCo
- **+** Fast (optimized C engine, pre-built wheels)
- **+** Excellent contact simulation (native solver)
- **+** Easy install (`pip install mujoco`)
- **+** GPU rendering, visualization tools
- **+** Widely used in RL research (policy transfer, sim-to-real)
- **-** Rigid-body only — no true elastic deformation
- **-** Elastic energy is an approximation (actuator springs)
- **-** Curvature control is indirect (angle conversion)

### PyBullet
- **+** Mature, well-documented rigid-body engine
- **+** Easy install (`pip install pybullet`)
- **+** Multiple motor modes (position, velocity, torque)
- **+** Built-in GUI renderer
- **-** Legacy in this project (not integrated into factory pattern)
- **-** Box geometry (less realistic for snake)
- **-** No elastic energy computation
- **-** Older API design (integer handles, global state)

## When to Use Which

| Use Case | Recommended Backend |
|---|---|
| Physically accurate soft-body snake | dismech-python or PyElastica |
| Fast training with RL | MuJoCo |
| C++ performance with DER physics | dismech-rods |
| Quick prototyping / baseline | MuJoCo or PyBullet |
| Sim-to-real transfer research | MuJoCo |
| Studying elastic energy / strain | dismech-python or PyElastica |
| Physical snake-prey contact (constriction forces) | MuJoCo |
| Self-contact with elastic rod physics | dismech-python or dismech-rods |
| Non-planar terrain | MuJoCo or PyBullet |

## Related

- `src/snake_hrl/physics/__init__.py` — Factory function routing by `SolverFramework`
- `src/snake_hrl/configs/env.py` — `SolverFramework` enum and backend-specific config fields
- `doc/knowledge/dismech-api-comparison.md` — Detailed API comparison between the two dismech backends
- `snakebot-gym/` — Legacy PyBullet environment (not part of current factory pattern)
