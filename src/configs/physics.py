"""Physics configuration hierarchy for all simulation backends.

Hierarchy:
    PhysicsConfig (base: geometry, dt, density, gravity, solver, prey)
    └── RodConfig (material: youngs_modulus, poisson_ratio, convergence, friction)
        ├── DERConfig (Discrete Elastic Rods)
        │   ├── DismechConfig (DisMech Python — no extra fields)
        │   └── DismechRodsConfig (dismech-rods C++ — integrator, adaptive, damping)
        └── CosseratConfig (Cosserat Rod Theory)
            └── ElasticaConfig (PyElastica — damping, time_stepper, dt_substep)
    MujocoPhysicsConfig(PhysicsConfig) (timestep, substeps, joint_damping, joint_stiffness, friction)

Time stepping convention (two levels):
    substep:  The smallest integration timestep (dt for DisMech, dt_substep for Elastica)
    RL step:  substeps_per_action × substep duration (defined in control configs)

Friction models (FrictionModel enum):
    NONE: No ground forces
    RFT: Resistive Force Theory (viscous drag)
    COULOMB: Coulomb friction with smooth barrier normal force
    STRIBECK: Coulomb + Stribeck static-to-kinetic transition
    NATIVE: Backend's built-in model (MuJoCo soft contact, dismech-rods DampingForce)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

from .geometry import GeometryConfig


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

    Deprecated: Use FrictionModel instead.

    RFT: Resistive Force Theory (custom implementation matching DisMech)
    DAMPING: Use Elastica's built-in damping mechanisms
    NONE: No ground contact forces
    """
    RFT = "rft"
    DAMPING = "damping"
    NONE = "none"


class FrictionModel(str, Enum):
    """Friction/contact model for ground interaction.

    NONE: No ground forces
    RFT: Resistive Force Theory (viscous drag, anisotropic ct/cn)
    COULOMB: Coulomb friction with smooth barrier normal force
    STRIBECK: Coulomb + Stribeck static-to-kinetic transition
    NATIVE: Backend's built-in model (MuJoCo soft contact, dismech-rods DampingForce)
    """
    NONE = "none"
    RFT = "rft"
    COULOMB = "coulomb"
    STRIBECK = "stribeck"
    NATIVE = "native"


@dataclass
class FrictionConfig:
    """Configurable friction/contact parameters for ground interaction.

    Parameters are used selectively depending on which FrictionModel is active:
    - RFT: uses rft_ct, rft_cn
    - COULOMB: uses mu_kinetic, ground_stiffness, ground_delta
    - STRIBECK: uses mu_kinetic, mu_static, stribeck_velocity, ground_stiffness, ground_delta
    - NATIVE: uses backend-specific defaults
    - NONE: no parameters used
    """
    model: FrictionModel = FrictionModel.RFT

    # RFT parameters
    rft_ct: float = 0.01   # Tangential drag coefficient
    rft_cn: float = 0.1    # Normal drag coefficient

    # Coulomb parameters
    mu_kinetic: float = 0.3       # Kinetic friction coefficient
    mu_static: float = 0.5        # Static friction coefficient (Stribeck only)
    stribeck_velocity: float = 0.01  # Stribeck transition velocity v_s

    # Ground contact (barrier-based normal force)
    ground_stiffness: float = 50000.0  # Normal force stiffness
    ground_delta: float = 0.01         # Barrier activation distance


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

    # Friction / ground interaction
    friction: FrictionConfig = field(default_factory=FrictionConfig)

    # --- Deprecated fields (kept for backward compat) ---
    use_rft: bool = True  # Deprecated: use friction.model instead
    rft_ct: float = 0.01  # Deprecated: use friction.rft_ct instead
    rft_cn: float = 0.1   # Deprecated: use friction.rft_cn instead

    def __post_init__(self):
        """Migrate deprecated fields to FrictionConfig."""
        # If friction is still at defaults but deprecated fields were customized,
        # migrate them into the FrictionConfig
        default_friction = FrictionConfig()
        if self.friction == default_friction:
            if not self.use_rft:
                self.friction = FrictionConfig(model=FrictionModel.NONE)
            elif self.rft_ct != 0.01 or self.rft_cn != 0.1:
                self.friction = FrictionConfig(
                    model=FrictionModel.RFT,
                    rft_ct=self.rft_ct,
                    rft_cn=self.rft_cn,
                )


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
    Default friction is NATIVE (DampingForce) since dismech-rods C++ API
    does not expose RFT or Coulomb forces.
    """
    solver_framework: SolverFramework = SolverFramework.DISMECH_RODS
    dismech_rods_integrator: str = "BACKWARD_EULER"  # Time integration scheme
    dismech_rods_adaptive_time_stepping: int = 0  # 0 = disabled, 1 = enabled
    dismech_rods_damping_viscosity: float = 0.0  # Viscous damping coefficient

    # Override default: NATIVE uses DampingForce
    friction: FrictionConfig = field(
        default_factory=lambda: FrictionConfig(model=FrictionModel.NATIVE)
    )


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

    Adds Elastica-specific parameters: damping, time stepper, and ground
    contact method. Provides dt_substep property for the actual integration
    timestep (= dt / elastica_substeps).
    """
    solver_framework: SolverFramework = SolverFramework.ELASTICA
    elastica_damping: float = 0.1  # Numerical damping coefficient
    elastica_time_stepper: str = "PositionVerlet"  # "PositionVerlet" or "PEFRL"
    elastica_substeps: int = 50  # Substeps per dt interval (dt_substep = dt / elastica_substeps)
    elastica_ground_contact: ElasticaGroundContact = ElasticaGroundContact.RFT  # Deprecated

    @property
    def dt_substep(self) -> float:
        """Elastica integration timestep (seconds) = dt / elastica_substeps."""
        return self.dt / self.elastica_substeps

    def __post_init__(self):
        """Migrate deprecated elastica_ground_contact to friction config."""
        super().__post_init__()
        # If friction is at RFT default but elastica_ground_contact was changed,
        # migrate the old field
        if self.friction.model == FrictionModel.RFT:
            if self.elastica_ground_contact == ElasticaGroundContact.NONE:
                self.friction = FrictionConfig(model=FrictionModel.NONE)
            elif self.elastica_ground_contact == ElasticaGroundContact.DAMPING:
                # DAMPING was Elastica-specific, map to NONE (damping is added separately)
                self.friction = FrictionConfig(model=FrictionModel.NONE)


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

    # Friction / ground interaction (default NATIVE for MuJoCo)
    friction: FrictionConfig = field(
        default_factory=lambda: FrictionConfig(model=FrictionModel.NATIVE)
    )
