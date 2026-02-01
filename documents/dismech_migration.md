# DisMech Integration Migration Guide

## Overview

This document describes the migration from the custom semi-implicit Euler physics implementation to **dismech-python** (pure Python) from [StructuresComp/dismech-python](https://github.com/StructuresComp/dismech-python).

## Why DisMech?

DisMech provides:
- **Implicit time integration** for better stability with soft materials
- **Discrete elastic rod model** specifically designed for soft robotics
- **Resistive Force Theory (RFT)** for ground interaction
- **Vectorized NumPy operations** for good performance
- **Well-tested physics** from published research

## Parameter Changes

### PhysicsConfig Updates

| Parameter | Old Value | New Value (DisMech) | Notes |
|-----------|-----------|---------------------|-------|
| `snake_radius` | 0.02 | 0.001 | DisMech uses smaller rod radius |
| `density` | 1000.0 | 1200.0 | Updated material density |
| `youngs_modulus` | 1e6 | 2e6 | Stiffer rod for stability |
| `dt` | 1e-3 | 5e-2 | Larger timestep (implicit integration) |
| `substeps` | 10 | *Removed* | DisMech handles internally |
| `damping` | 0.1 | *Removed* | RFT provides damping |
| `ground_friction` | 0.5 | *Removed* | RFT replaces friction model |

### New Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `poisson_ratio` | 0.5 | Poisson's ratio for rod |
| `max_iter` | 25 | Max Newton iterations per timestep |
| `tol` | 1e-4 | Force tolerance for convergence |
| `ftol` | 1e-4 | Relative force tolerance |
| `dtol` | 1e-2 | Displacement tolerance |
| `use_rft` | True | Enable Resistive Force Theory |
| `rft_ct` | 0.01 | Tangential drag coefficient |
| `rft_cn` | 0.1 | Normal drag coefficient |

## API Compatibility

The `SnakeRobot` class maintains the same public API:

```python
# Same initialization
robot = SnakeRobot(config, initial_snake_position, initial_prey_position)

# Same methods
robot.reset(snake_position, prey_position)
robot.set_curvature_control(curvatures)
state = robot.step(dt)
obs = robot.get_observation(include_curvatures=True)
energy = robot.get_energy()
state = robot.get_state()
```

### State Dictionary

The state dictionary returned by `get_state()` contains the same keys:
- `positions`: Snake node positions (n_nodes, 3)
- `velocities`: Snake node velocities (n_nodes, 3)
- `curvatures`: Current curvatures at joints
- `prey_position`: Prey center position
- `prey_orientation`: Prey orientation
- `prey_distance`: Distance from snake head to prey
- `contact_mask`: Which nodes are in contact with prey
- `contact_fraction`: Fraction of nodes in contact
- `wrap_angle`: Total angle wrapped around prey
- `wrap_count`: Number of complete wraps
- `time`: Current simulation time

## Deprecated Functions

The following functions are deprecated but kept for backward compatibility:

- `create_snake_geometry()` - Snake geometry is now managed internally by DisMech

## Installation

1. Clone dismech-python:
   ```bash
   git clone https://github.com/StructuresComp/dismech-python.git
   ```

2. Install in development mode:
   ```bash
   cd dismech-python
   pip install -e .
   ```

## Testing

Run the physics tests to verify the migration:

```bash
pytest tests/test_physics.py -v
```

## Manual Verification

```python
from snake_hrl.configs.env import PhysicsConfig
from snake_hrl.physics.snake_robot import SnakeRobot
import numpy as np

config = PhysicsConfig()
robot = SnakeRobot(config)
robot.set_curvature_control(np.zeros(config.num_segments - 1))

for _ in range(100):
    state = robot.step()

print("Positions:", state["positions"][:3])
print("Energy:", robot.get_energy())
```

## Technical Details

### DisMech Integration

The `SnakeRobot` class now wraps DisMech's `SoftRobot` and `ImplicitEulerTimeStepper`:

1. **Geometry**: Rod geometry is created using DisMech's `Geometry` class
2. **Material**: Material properties are set via `GeomParams` and `Material`
3. **Environment**: Forces (gravity, RFT) are added via `Environment.add_force()`
4. **Time Stepping**: Uses implicit Euler with Newton-Raphson iteration
5. **Curvature Control**: Mapped to DisMech's natural curvature in bend springs

### SnakeGeometryAdapter

The `SnakeGeometryAdapter` class provides the same interface as the old `SnakeGeometry` dataclass but reads positions from DisMech's state vector.

### Prey Contact

Prey contact and wrapping computations remain custom (not part of DisMech) since prey is not simulated as a soft body.

## DisMech Control Methods

DisMech provides control through the **natural strain** mechanism:

### Natural Strain Control
Each spring type has a `nat_strain` property representing its rest configuration:
- `robot.bend_springs.nat_strain` - curvature at rest, shape `(Nb, 2)` for κ₁ and κ₂
- `robot.stretch_springs.nat_strain` - stretch at rest
- `robot.twist_springs.nat_strain` - twist at rest

The elastic energy penalizes deviations from natural strain, driving the rod toward the target configuration.

### Built-in Controllers
DisMech provides PI controllers in `dismech.controllers`:
- **`CurvaturePI`** - PI controller for bending curvature
- **`CurvaturePITracker`** - Trajectory tracking for curvature
- **`LongitudinalPI`** - PI controller for stretch

### How Curvature Control Works in SnakeRobot
The `set_curvature_control()` method maps target curvatures to `bend_springs.nat_strain`:
```python
# Target curvature is set on the first component (κ₁) for planar bending
bend_springs.nat_strain[i, 0] = target_curvature
bend_springs.nat_strain[i, 1] = 0.0  # κ₂ = 0 for planar motion
```

## Known Limitations

1. **2D Mode**: Currently runs in 2D mode (`two_d_sim=True`) for snake locomotion
2. **Prey Contact Forces**: Not integrated into DisMech solver (computed separately)
3. **Rod Radius**: Uses small rod radius (0.001m) for DisMech compatibility

## References

- [DisMech Paper](https://doi.org/10.1177/02783649231191091)
- [dismech-python Repository](https://github.com/StructuresComp/dismech-python)
