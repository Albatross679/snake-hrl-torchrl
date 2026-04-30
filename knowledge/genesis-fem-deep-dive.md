---
name: genesis-fem-deep-dive
description: "Genesis FEM solver capabilities, snake robot feasibility, and FEM vs Cosserat comparison"
type: knowledge
created: 2026-03-09T00:00:00
updated: 2026-03-09T00:00:00
tags: [genesis, fem, soft-robot, snake-robot, cosserat, physics-simulation, gpu, reinforcement-learning]
aliases: []
---

# Genesis FEM Solver Deep Dive

## 1. Genesis FEM Capabilities in Detail

### 1.1 Element Types

Genesis FEM uses **tetrahedral elements** exclusively. The `FEMEntity` class stores:
- `elems`: array of shape `(n_elements, 4)` indexing into vertices (4 vertices per tetrahedron)
- `init_positions`: array of shape `(n_vertices, 3)`
- Surface triangles extracted from tetrahedral faces

Tetrahedralization is performed automatically when loading surface meshes (OBJ, STL, etc.) or creating primitive shapes (Sphere, Box, Cylinder). The tetrahedralization parameters (from `gs.morphs.Mesh`) suggest **TetGen** is used internally:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mindihedral` | 10 | Minimum dihedral angle (degrees) |
| `minratio` | 1.1 | Minimum tetrahedron quality ratio |
| `quality` | True | Enable quality improvement |
| `nobisect` | True | Disable bisection |
| `maxvolume` | -1.0 | Maximum tetrahedron volume (-1 = no limit) |
| `order` | 1 | FEM mesh order (1 = linear) |
| `force_retet` | False | Force re-tetrahedralization |

**No hexahedral or beam elements are supported.**

### 1.2 Constitutive Models

**FEM.Elastic** supports three constitutive models (via `model` parameter):

| Model | Description | Large Deformation |
|-------|-------------|-------------------|
| `'linear'` (default) | Linear elasticity | No -- small strain only |
| `'stable_neohookean'` | Stable Neo-Hookean hyperelastic | Yes -- designed for large deformations |
| `'linear_corotated'` | Linear corotated elasticity | Moderate -- handles rotation but linearized strain |

**FEM.Muscle** supports:
- `'linear'` (default)
- `'stable_neohookean'`

(Note: `'linear_corotated'` is not listed for Muscle, only Elastic.)

### 1.3 Material Parameters

**FEM.Elastic:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `E` | float | 1e6 | Young's modulus (stiffness) |
| `nu` | float | 0.2 | Poisson's ratio |
| `rho` | float | 1000.0 | Density (kg/m^3) |
| `hydroelastic_modulus` | float | 1e7 | Hydroelastic contact modulus |
| `friction_mu` | float | 0.1 | Surface friction coefficient |
| `model` | str | 'linear' | Constitutive model |

**FEM.Muscle** (inherits from Elastic, adds):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_groups` | int | 1 | Number of muscle groups |

### 1.4 Time Integration

Genesis FEM supports **two integration schemes**:

1. **Explicit (default)**: `use_implicit_solver = False` -- symplectic Euler-like integration with Rayleigh damping (`damping` parameter)
2. **Implicit**: `use_implicit_solver = True` -- Newton's method with Preconditioned Conjugate Gradient (PCG) linear solver

**Implicit solver parameters (FEMOptions):**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_implicit_solver` | False | Enable implicit integration |
| `n_newton_iterations` | 1 | Max Newton iterations |
| `n_pcg_iterations` | 500 | Max PCG iterations per Newton step |
| `n_linesearch_iterations` | 0 | Max line search iterations |
| `newton_dx_threshold` | 1e-6 | Newton convergence threshold |
| `pcg_threshold` | 1e-6 | PCG convergence threshold |
| `linesearch_c` | 1e-4 | Sufficient decrease parameter |
| `linesearch_tau` | 0.5 | Step size reduction factor |
| `damping_alpha` | 0.5 | Rayleigh mass damping (implicit) |
| `damping_beta` | 5e-4 | Rayleigh stiffness damping (implicit) |
| `enable_vertex_constraints` | False | Enable vertex constraints |

### 1.5 Rest Shapes and Rest Curvatures

Genesis FEM uses the **initial mesh configuration as the rest shape**. The `init_positions` array defines the stress-free reference configuration. To set a rest curvature, you would need to provide a pre-curved mesh as input (e.g., a curved OBJ file). There is no explicit API to set rest curvatures or pre-stress on individual elements.

Vertex constraints (`set_vertex_constraints`) can enforce positions through hard or soft (spring-based) constraints, which could approximate pre-stress but are not the same as setting rest curvatures.

### 1.6 Actuation for FEM Bodies

Genesis FEM supports **volumetric muscle actuation**:

```python
# Create FEM muscle entity
robot = scene.add_entity(
    morph=gs.morphs.Mesh(file='snake.obj', scale=0.1),
    material=gs.materials.FEM.Muscle(
        E=5e5, nu=0.45, rho=10000,
        model='stable_neohookean', n_groups=4
    ),
)

# Define muscle groups and directions per element
robot.set_muscle(
    muscle_group=group_array,      # shape: (n_elements,), int 0..n_groups-1
    muscle_direction=dir_array,    # shape: (n_elements, 3), normalized vectors
)

# Apply actuation each step
robot.set_actuation(actu)  # shape: (n_groups,) or (n_envs, n_groups)
```

Actuation applies a **contractile force along the muscle direction** for each element, scaled by the actuation signal. This is analogous to biological muscle contraction.

### 1.7 Performance

**No published FEM-specific benchmarks exist.** The claimed 43M FPS figure is for rigid body (Franka arm) only. Independent analysis (Stone Tao, 2025) found that under realistic conditions (self-collisions enabled, active control), rigid body performance dropped to ~290K FPS, and Genesis was actually 3-10x slower than ManiSkill/SAPIEN for manipulation tasks.

For FEM soft bodies, realistic performance expectations:
- **Much slower than rigid body** due to tetrahedral element computation
- The implicit solver with Newton/PCG iterations adds significant cost
- No published FEM-specific FPS numbers from Genesis team or independent benchmarks
- GPU parallelization should still provide advantage over CPU-based FEM (SOFA, etc.)

## 2. Genesis FEM for Snake/Worm Robots

### 2.1 Existing Examples

Genesis includes a **worm example** in its documentation that demonstrates:
- MPM-based worm with 4 muscle groups (upper-fore, upper-hind, lower-fore, lower-hind)
- FEM-based equivalent using `gs.materials.FEM.Muscle`
- Sinusoidal actuation signals to produce peristaltic crawling

**No snake-specific examples exist.** The worm example uses peristaltic (earthworm-like) locomotion, not serpentine (lateral undulation). Adapting to serpentine locomotion would require:
- Different muscle group layout (left/right alternating, not upper/lower)
- Muscle directions aligned along the body axis
- Wave-like actuation pattern (traveling wave of contraction)

### 2.2 MPM vs FEM for Soft Robots in Genesis

| Aspect | MPM | FEM |
|--------|-----|-----|
| Discretization | Particles | Tetrahedral elements |
| Unit access | `n_particles` | `n_elements` via `get_el2v()` |
| Constitutive models | `'neohooken'` | `'linear'`, `'stable_neohookean'`, `'linear_corotated'` |
| Differentiable | Yes | Not currently |
| Contact | Particle-grid coupling | Surface vertex collision only |
| Topology | Can handle fracture/splitting | Fixed mesh topology |

### 2.3 Known Issues (from GitHub)

- **Issue #1013**: Multiple FEM entities had broken surface property initialization and incorrect vertex/element counts
- **Issue #1072**: FEM-rigid entity collision bugs
- **Issue #1139**: Changing friction vectors does not affect contact behavior
- **Issue #2090**: Coupling issues between FEM and MPM entities
- **FEM differentiability not yet implemented** (only MPM and Tool solvers are differentiable)

## 3. FEM vs Cosserat Rod for Snake Robots: Academic Literature

### 3.1 When Cosserat Rods Are Sufficient

Rod models are appropriate when (from arxiv:2407.05886):
- The body is **slender** (length >> cross-section radius)
- Deformations are primarily **bending, torsion, stretching, and shear**
- **Real-time computation** is required
- Cross-sectional shape does not change significantly

For a snake robot with L=1m, r=0.02m (aspect ratio 50:1), Cosserat rod theory is well-suited.

**Key advantages of Cosserat rods:**
- 1D reduction: O(N) DOFs along the arc length vs O(N^3) for 3D FEM
- Real-time capable: demonstrated at faster-than-real-time with implicit integration (Till et al., 2019)
- Position errors <1% of robot length in validation studies
- Quantitative agreement within 5% vs experimental force-displacement curves
- Natural parameterization by arc length matches snake body kinematics

### 3.2 When FEM Is Necessary

FEM is preferred when (from the literature):
- **Complex 3D geometry** matters (non-circular cross-sections, internal structure)
- **Cross-sectional deformation** occurs (squishing, inflation)
- **Contact mechanics** require 3D surface resolution
- **Multi-material bodies** with spatial material property variation
- **Initial design validation** before reduced-order model deployment

### 3.3 What FEM Captures That Cosserat Misses

1. **Cross-sectional deformation**: Cosserat assumes rigid cross-sections. FEM captures squishing when pressed against ground.
2. **3D contact geometry**: Cosserat treats contact as point/line forces. FEM resolves distributed contact pressure over the belly surface.
3. **Internal stress distribution**: FEM provides full 3D stress/strain fields, important for failure analysis.
4. **Non-uniform material**: FEM can model spatially varying stiffness (e.g., stiffer belly scales).

### 3.4 Does 3D Geometry Matter for Snake Ground Contact?

**For locomotion RL training: probably not.** Key arguments:

- Snake scales create **anisotropic friction** which is a surface property, modelable with either approach
- Ground contact in serpentine locomotion is primarily a **friction problem**, not a contact geometry problem
- Cosserat rod + RFT (Resistive Force Theory) captures the essential physics for locomotion
- FEM's distributed contact pressure provides marginal accuracy improvement at enormous computational cost
- No published work demonstrates that FEM-resolved belly contact significantly improves locomotion policy quality

**Where 3D matters**: coiling around prey (the HRL coil task), where the snake wraps around a cylindrical object. Here, 3D contact geometry determines grip force distribution. However, this could still be approximated with Cosserat rods + contact models.

### 3.5 Computational Cost Comparison

| Metric | Cosserat Rod (PyElastica) | FEM (SOFA) | FEM (Genesis GPU) |
|--------|--------------------------|------------|-------------------|
| DOFs for 1m snake | ~100-300 (20-100 nodes x 3) | ~5000-15000 (1000-5000 tets x 3) | Same as SOFA |
| Real-time capable | Yes (with implicit) | Yes (coarse mesh, ~2200 verts) | Likely yes (GPU) |
| RL training speed | ~57 FPS (16 CPU envs) | Not practical for RL | Unknown (no benchmarks) |
| GPU parallelization | No (PyElastica is CPU-only) | No (SOFA is CPU-only) | Yes (Taichi backend) |

### 3.6 Key References

- **Till et al. (2019)**: "Real-time dynamics of soft and continuum robots based on Cosserat rod models" -- demonstrates real-time Cosserat simulation, *Int J Robotics Research*
- **arxiv:2407.05886 (2024)**: "Rod models in continuum and soft robot control: a review" -- comprehensive comparison of rod vs FEM approaches
- **Duriez et al. (2023)**: "Modeling and Simulation of Dynamics in Soft Robotics: a Review of Numerical Approaches" -- *Current Robotics Reports*, Springer
- **Wan et al. (2023)**: "Design, Analysis, and Real-Time Simulation of a 3D Soft Robotic Snake" -- *Soft Robotics*, using NVidia Flex particle simulation
- **Coevoet et al. (2019)**: "Soft robots locomotion and manipulation control using FEM simulation and quadratic programming" -- SOFA-based FEM locomotion
- **Grazioso et al. (2019)**: "A Geometrically Exact Model for Soft Continuum Robots: The Finite Element Deformation Space Formulation" -- FEM formulation for continuum robots

## 4. Practical Considerations for This Project

### 4.1 Mesh Size Estimation for 1m Snake (r=0.02m)

Volume of cylinder: pi * r^2 * L = pi * 0.0004 * 1.0 = 0.00126 m^3

| Element Size | Approx. Elements | Approx. Vertices | DOFs |
|-------------|-------------------|-------------------|------|
| 0.01m (coarse) | 500-1,000 | 200-400 | 600-1,200 |
| 0.005m (medium) | 2,000-5,000 | 800-2,000 | 2,400-6,000 |
| 0.002m (fine) | 30,000-80,000 | 10,000-25,000 | 30,000-75,000 |

For RL training, a **coarse mesh (500-1000 elements)** is likely sufficient. SOFA literature recommends limiting to ~2200 vertices for real-time performance on CPU.

### 4.2 Simulation Speed Estimates

No Genesis FEM benchmarks exist, but reasonable estimates:
- **Explicit solver, 1000 elements, 1 env**: Likely 100s-1000s FPS (GPU overhead may not help at small scale)
- **Implicit solver, 1000 elements, 1 env**: Likely 10s-100s FPS (Newton/PCG iterations expensive)
- **GPU parallelization benefit**: Becomes significant with 100+ parallel environments
- **Comparison**: PyElastica Cosserat rod currently achieves ~57 FPS with 16 CPU parallel envs (~3.6 FPS per env)

### 4.3 Ground Contact in Genesis FEM

FEM-rigid coupling uses:
- **Surface vertex collision only** (not volumetric)
- Impulse-based response with Coulomb friction (`coup_friction` parameter)
- `friction_mu` on the FEM material (default 0.1)
- **No built-in anisotropic friction** -- Genesis uses isotropic Coulomb friction for FEM-rigid coupling
- No way to specify direction-dependent friction coefficients on FEM surfaces

**This is a significant limitation for snake locomotion**, which fundamentally relies on anisotropic friction (low friction forward, high friction lateral/backward).

### 4.4 Can You Implement Anisotropic Friction?

**Difficult.** Options:
1. **CustomForceField**: Only has access to `(pos, vel, t)`, not surface normals or body orientation. Insufficient.
2. **Manual per-step forces**: Use `apply_links_external_force()` but this is for rigid bodies, not FEM entities.
3. **Modify Genesis source**: Add anisotropic friction to the FEM-rigid coupler. This requires modifying Taichi kernel code.
4. **Workaround with surface geometry**: Add anisotropic surface features (scales/ridges) to the mesh to create directional friction mechanically. This increases element count.

### 4.5 Summary Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| FEM solver maturity | Moderate | Basic features work, implicit solver available, but bugs in multi-entity and coupling |
| Snake locomotion feasibility | Low-Moderate | Muscle actuation works, but no anisotropic friction |
| Performance for RL | Unknown | No benchmarks; likely slower than rigid body by 10-100x |
| Advantage over Cosserat rod | Minimal | Snake is slender (aspect ratio 50:1), Cosserat is more appropriate |
| GPU parallelization value | High | Main advantage over PyElastica |
| Differentiability | Not available | FEM solver is not yet differentiable |

### 4.6 Recommendation

**Genesis FEM is not recommended for this snake locomotion project** at this time, because:

1. The snake body is highly slender (L/r = 50), making Cosserat rod theory the natural choice
2. Genesis FEM lacks anisotropic friction, which is essential for serpentine locomotion
3. No FEM performance benchmarks exist -- the speed advantage is uncertain
4. FEM is not differentiable in Genesis (only MPM is)
5. The FEM solver has known bugs in multi-entity and coupling scenarios

**Better alternatives:**
- **PyElastica (current)**: Cosserat rod + RFT friction, physically accurate, but CPU-only
- **Genesis rigid body**: Model snake as URDF articulated chain for GPU-parallel RL (loses continuous dynamics)
- **MuJoCo MJX**: Articulated body with GPU parallelization via JAX, mature ecosystem
- **Genesis MPM**: If volumetric soft body is desired, MPM is more mature in Genesis (differentiable, better tested)
