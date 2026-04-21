---
name: dismech-contact-pipe-tunnels
description: Using DisMech rod-to-rod contact for snake robot pipe tunnel navigation
type: knowledge
created: 2026-02-17T00:00:00
updated: 2026-02-17T00:00:00
tags: [knowledge, physics, dismech, contact, pipe, tunnel, imc, collision, rod-to-rod]
aliases: []
---

# DisMech Contact for Pipe Tunnel Navigation

How to use DisMech's existing rod-to-rod contact infrastructure to simulate a snake robot navigating through pipe tunnels, without modifying the core contact code.

## DisMech Contact Capabilities

### What Exists

DisMech supports exactly two types of contact:

**1. Rod-to-ground (flat plane only)**

Both backends provide a penalty-based floor contact that checks `node_z < ground_z + delta` and applies an exponential repulsion force. Hardcoded to a single horizontal plane.

- dismech-python: `compute_ground_contact()` in `dismech/external_forces/ground_contact.py`
- dismech-rods: `FloorContactForce` class

**2. Rod-to-rod (self-contact and multi-limb contact)**

Both backends detect proximity between rod edge pairs using the **Lumelsky minimum distance algorithm**, then classify each contact into one of three proximity primitives:

| Primitive | Geometry | Lumelsky Parameters |
|-----------|----------|-------------------|
| Point-to-Point (P2P) | Node vs. Node | Both t, u at endpoints (0 or 1) |
| Point-to-Edge (P2E) | Node vs. Edge interior | One at endpoint, one interior |
| Edge-to-Edge (E2E) | Edge interior vs. Edge interior | Both interior |

**dismech-python** uses IMC (Implicit Minimal Coordinate) penalty energy with SymPy-derived analytical gradients and Hessians. Supports stick/slip friction.

**dismech-rods** uses FCL (Flexible Collision Library) for broad-phase AABB tree pruning, then the same Lumelsky narrow-phase in C++. Supports friction with `ZERO_VEL`, `SLIDING`, `STICKING` states.

### What Does NOT Exist

- Rod-to-rigid-body contact (no rigid cylinders, meshes, or arbitrary surfaces)
- Rod-to-curved-surface contact
- Any geometric primitive other than rod edges and a flat floor

## Rod-as-Pipe-Wall Approach

### Key Insight

The pipe's inner surface can be discretized as a set of **fixed rod segments** (additional limbs with all nodes pinned). The existing multi-limb contact machinery handles snake-to-wall forces automatically.

### Why It Works

1. **Multi-limb contact already exists.** The `CollisionDetector` in dismech-rods iterates over all limb pairs (`collision_detector.cpp:171-184`). The snake is limb 0; pipe wall segments become limbs 1..N.

2. **FCL broad-phase efficiently culls distant pairs.** Only nearby wall segments trigger the expensive Lumelsky narrow-phase computation.

3. **Same P2P/P2E/E2E primitives apply.** Snake edge ↔ wall edge contact uses identical penalty forces, gradients, and friction. No new physics code needed.

4. **Collision groups** (`col_group` bitmask on each limb) control which limbs interact. Snake collides with wall limbs; wall limbs don't need to collide with each other.

5. **Boundary conditions pin wall nodes.** `lockEdge(limb_idx, edge_idx)` fixes both endpoint nodes and the twist angle of an edge, making the wall static.

### Pipe Discretization: Circumferential Rings

Represent the pipe as a series of **polygonal rings** along the pipe centerline. Each ring is a separate limb forming an open polygon:

```
Side view:                    Cross-section (12-sided):

ring  ring  ring  ring              * --- *
 |     |     |     |             /           \
 |  ~~~snake~~~~   |           *               *
 |     |     |     |           |               |
ring  ring  ring  ring         *               *
                                 \           /
                                    * --- *
```

Each ring has `n_sides` nodes arranged at equal angles around the pipe circumference. Ring edges are **circumferential** (perpendicular to the pipe axis), while snake edges are **axial** — this gives well-conditioned Edge-to-Edge contact geometry (near-perpendicular edges have a well-defined closest approach).

### Contact Direction Analysis

The critical question: does the repulsion force point the right way when the snake is *inside* the pipe?

The IMC/penalty contact force is proportional to the **negative gradient of the distance** between two edges. When the snake edge approaches a wall edge from inside the pipe:

- The closest-point vector points from the snake edge toward the wall edge (outward radially)
- The repulsion force pushes the snake **away** from the wall, **toward the pipe center**

This is correct. The repulsion direction is determined by the Lumelsky closest-point geometry, not by any notion of "inside" vs "outside." As long as the snake starts inside the pipe (closer to the center than to any wall edge), the forces naturally confine it.

### API Usage (dismech-rods)

```python
import py_dismech

soft_robots = sim_manager.soft_robots

# Limb 0: the snake (col_group = 0x0001)
soft_robots.addLimb(snake_start, snake_end, num_snake_nodes,
                    density, snake_radius, E, nu, mu,
                    col_group=0x0001)

# Limbs 1..N: pipe wall rings (col_group = 0x0001 so they collide with snake)
for ring_nodes in pipe_ring_node_lists:
    soft_robots.addLimb(ring_nodes, wall_density, wall_radius, wall_E, wall_nu,
                        mu=0.0, col_group=0x0001)

# Lock all edges of all wall limbs
for limb_idx in range(1, len(pipe_ring_node_lists) + 1):
    n_edges = len(pipe_ring_node_lists[limb_idx - 1]) - 1
    for edge_idx in range(n_edges):
        soft_robots.lockEdge(limb_idx, edge_idx)

# Enable contact between limbs (self_contact=False: no intra-limb self-contact)
contact = py_dismech.ContactForce(
    soft_robots,
    col_limit=1e-3,    # broad-phase distance buffer
    delta=5e-4,         # penalty activation distance
    k_scaler=1e5,       # contact stiffness
    friction=True,
    nu=1e-3,            # friction smoothing
    self_contact=False   # only inter-limb contact
)
sim_manager.forces.addForce(contact)
```

### Design Parameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `n_sides` | Polygon sides per ring | 12-16 (more = smoother, more limbs) |
| `ring_spacing` | Axial distance between rings | 1-2× snake segment length |
| `wall_radius` | Capsule radius of wall edges | ~snake_radius (thin enough for contact geometry) |
| `wall_E` | Wall Young's modulus | High (1e8+, effectively rigid) |
| `wall_density` | Wall density | Irrelevant (nodes are locked) |
| `col_limit` | FCL broad-phase buffer | Pipe radius (ensures snake-wall pairs aren't culled) |
| `delta` | Penalty activation distance | ~snake_radius (snake feels wall before geometric contact) |
| `k_scaler` | Contact stiffness | Tune: too low = penetration, too high = instability |

### Coverage Gaps

Each ring is an **open** polygon (dismech-rods doesn't support closed-loop rods). An N-sided ring has N-1 edges, leaving one angular gap of `360°/N`. Mitigation:

- Use higher `n_sides` (16 sides = 22.5° gap = 6.25% uncovered)
- Stagger rings: alternate rings rotated by half the angular spacing, so no two adjacent rings share the same gap position
- The snake's finite radius partially bridges small gaps

### Computational Cost

For a pipe with `n_rings` rings of `n_sides` sides each:

- Total wall limbs: `n_rings`
- Total wall edges: `n_rings × (n_sides - 1)`
- FCL broad-phase: efficient (AABB tree prunes most distant pairs)
- Narrow-phase: only wall edges near the snake are checked
- Solver: wall DOFs are eliminated by boundary conditions (no added unknowns)

A 1m pipe with 0.04m ring spacing and 12 sides = 25 rings × 11 edges = 275 wall edges. FCL handles this efficiently.

## Alternatives Considered

| Approach | Pros | Cons |
|----------|------|------|
| **Rod-as-wall (this approach)** | Zero core code changes, existing contact | Discretization gaps, many limbs |
| **Custom cylinder distance function** | Smooth contact, exact geometry | Requires new contact energy class |
| **MuJoCo rigid-body** | Native contact, trivial setup | No elastic rod physics |
| **Hybrid: DER snake + MuJoCo contact** | Best of both worlds | Two simulators, complex coupling |

## Implementation

The pipe geometry generator is implemented at `src/physics/pipe_geometry.py`. It produces ring node lists compatible with `soft_robots.addLimb(nodes, ...)`.

## Related

- `doc/knowledge/physics-framework-comparison.md` — Full backend comparison
- `src/physics/dismech_rods_snake_robot.py` — Current dismech-rods snake wrapper
- `dismech-rods/src/rod_mechanics/external_forces/collision_detector.cpp` — FCL broad-phase + Lumelsky narrow-phase
- `dismech-rods/src/rod_mechanics/external_forces/contact_force.cpp` — Penalty contact forces
- `dismech-rods/cpp_examples/active_entanglement_case/` — Multi-limb contact example
