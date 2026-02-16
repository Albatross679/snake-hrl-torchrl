"""Physics configuration hierarchy for all simulation backends.

Hierarchy:
    PhysicsConfig (base: geometry, dt, density, gravity, solver, prey)
    └── RodConfig (material: youngs_modulus, poisson_ratio, convergence, RFT)
        ├── DERConfig (Discrete Elastic Rods)
        │   ├── DismechConfig (DisMech Python — no extra fields)
        │   └── DismechRodsConfig (dismech-rods C++ — integrator, adaptive, damping)
        └── CosseratConfig (Cosserat Rod Theory)
            └── ElasticaConfig (PyElastica — damping, time_stepper, substeps, ground_contact)
    MujocoPhysicsConfig(PhysicsConfig) (timestep, substeps, joint_damping, joint_stiffness, friction)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

from configs.geometry import GeometryConfig


class SolverFramework(str, Enum):
    """Physics solver framework selection.

    DISMECH: DisMech library with implicit Euler integration
        - Uses discrete elastic rod model
        - Implicit time stepping for stability at larger dt
        - Control via bend spring natural strain

    ELASTICA: PyElastica (Cosserat rod theory)
        - Full Cosserat rod dynamics
        - Explicit symplectic integrators (PositionVerlet, PEFRL)
        - Control via rest curvature (rest_kappa)

    DISMECH_RODS: dismech-rods (C++ with pybind11 bindings)
        - C++ discrete elastic rod simulation via py_dismech
        - SimulationManager-based API with addLimb / step_simulation
        - Control via curvature dict passed to step_simulation

    MUJOCO: MuJoCo rigid-body simulation
        - Chain of rigid capsule links connected by hinge joints
        - Position actuators for curvature control
        - Fast, stable contact simulation with ground plane
    """
    DISMECH = "dismech"
    ELASTICA = "elastica"
    DISMECH_RODS = "dismech_rods"
    MUJOCO = "mujoco"


class ElasticaGroundContact(str, Enum):
    """Ground contact method for PyElastica.

    RFT: Resistive Force Theory (custom implementation matching DisMech)
    DAMPING: Use Elastica's built-in damping mechanisms
    NONE: No ground contact forces
    """
    RFT = "rft"
    DAMPING = "damping"
    NONE = "none"


# ---------------------------------------------------------------------------
# Base physics config
# ---------------------------------------------------------------------------


@dataclass
class PhysicsConfig:
    """Base physics config shared by all backends.

    Contains only parameters that every backend needs: geometry, time stepping,
    density, gravity, solver selection, and prey scene parameters.
    """

    # Snake body geometry (grouped)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)

    # Time stepping
    dt: float = 5e-2  # Simulation timestep (seconds)

    # Material
    density: float = 1200.0  # Material density (kg/m^3)

    # Gravity
    enable_gravity: bool = True
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)

    # Solver framework selection
    solver_framework: SolverFramework = SolverFramework.DISMECH

    # Prey geometry (scene setup, not body representation)
    prey_radius: float = 0.1  # Radius of cylindrical prey (meters)
    prey_length: float = 0.3  # Length of prey (meters)

    # --- Backward-compat property aliases → geometry fields ---

    @property
    def snake_length(self) -> float:
        return self.geometry.snake_length

    @property
    def snake_radius(self) -> float:
        return self.geometry.snake_radius

    @property
    def num_segments(self) -> int:
        return self.geometry.num_segments


# ---------------------------------------------------------------------------
# Rod-based physics (shared by DER and Cosserat theories)
# ---------------------------------------------------------------------------


@dataclass
class RodConfig(PhysicsConfig):
    """Shared config for rod-based physics (both DER and Cosserat).

    Common material and solver properties shared by all rod theories.
    """

    # Newton solver convergence
    max_iter: int = 25  # Maximum Newton iterations per timestep
    tol: float = 1e-4  # Force tolerance for convergence
    ftol: float = 1e-4  # Relative force tolerance
    dtol: float = 1e-2  # Displacement tolerance

    # Material properties
    youngs_modulus: float = 2e6  # Young's modulus for rod (Pa)
    poisson_ratio: float = 0.5  # Poisson's ratio for rod

    # Resistive Force Theory (RFT) for ground interaction
    use_rft: bool = True  # Enable RFT for ground forces
    rft_ct: float = 0.01  # Tangential drag coefficient
    rft_cn: float = 0.1  # Normal drag coefficient


# ---------------------------------------------------------------------------
# Discrete Elastic Rods (DER) backends
# ---------------------------------------------------------------------------


@dataclass
class DERConfig(RodConfig):
    """Discrete Elastic Rods — used by DisMech and dismech-rods backends."""
    pass


@dataclass
class DismechConfig(DERConfig):
    """DisMech (Python) backend config.

    No extra fields beyond DERConfig — DisMech uses all the rod defaults.
    """
    solver_framework: SolverFramework = SolverFramework.DISMECH


@dataclass
class DismechRodsConfig(DERConfig):
    """dismech-rods (C++) backend config.

    Adds C++-specific integrator and adaptive time stepping options.
    """
    solver_framework: SolverFramework = SolverFramework.DISMECH_RODS
    dismech_rods_integrator: str = "BACKWARD_EULER"  # Time integration scheme
    dismech_rods_adaptive_time_stepping: int = 0  # 0 = disabled, 1 = enabled
    dismech_rods_damping_viscosity: float = 0.0  # Viscous damping coefficient


# ---------------------------------------------------------------------------
# Cosserat rod theory backends
# ---------------------------------------------------------------------------


@dataclass
class CosseratConfig(RodConfig):
    """Cosserat rod theory — used by PyElastica backend."""
    pass


@dataclass
class ElasticaConfig(CosseratConfig):
    """PyElastica backend config (Cosserat rod dynamics).

    Adds Elastica-specific parameters: damping, time stepper, substeps,
    and ground contact method.
    """
    solver_framework: SolverFramework = SolverFramework.ELASTICA
    elastica_damping: float = 0.1  # Numerical damping coefficient
    elastica_time_stepper: str = "PositionVerlet"  # "PositionVerlet" or "PEFRL"
    elastica_substeps: int = 50  # Internal substeps per RL step
    elastica_ground_contact: ElasticaGroundContact = ElasticaGroundContact.RFT


# ---------------------------------------------------------------------------
# MuJoCo rigid-body backend
# ---------------------------------------------------------------------------


@dataclass
class MujocoPhysicsConfig(PhysicsConfig):
    """MuJoCo rigid-body backend config.

    Not a rod simulation — models the snake as a chain of rigid capsule
    links connected by hinge joints.
    """
    solver_framework: SolverFramework = SolverFramework.MUJOCO
    mujoco_timestep: float = 0.002  # Internal sim timestep (seconds)
    mujoco_substeps: int = 25  # Steps per RL step (0.002*25=0.05s=dt)
    mujoco_joint_damping: float = 0.1  # Joint damping coefficient
    mujoco_joint_stiffness: float = 50.0  # Position actuator kp
    mujoco_friction: Tuple[float, float, float] = (1.0, 0.005, 0.0001)
