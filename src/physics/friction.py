"""Friction force implementations for physics backends.

Provides standalone Coulomb and Stribeck friction force computations,
plus PyElastica-compatible forcing classes (CoulombForcing, StribeckForcing).

Physics:
    Normal force uses a smooth barrier (softplus-based) that activates when
    a node is within `delta` of the ground plane (z=0):

        F_n = k * softplus(-z, K)^2,   K = 15/delta

    Coulomb tangential friction:
        F_t = -mu * |F_n| * v_t / |v_t|   (regularized with sigmoid at low speed)

    Stribeck extension adds static-to-kinetic transition:
        mu_eff = mu_k + (mu_s - mu_k) * exp(-|v_t|^2 / v_s^2)
"""

import numpy as np

from configs.physics import FrictionConfig


# ---------------------------------------------------------------------------
# Core force computation (used by all backends)
# ---------------------------------------------------------------------------


def compute_barrier_normal_force(
    z: np.ndarray,
    stiffness: float,
    delta: float,
) -> np.ndarray:
    """Smooth barrier normal force for ground contact at z=0.

    Uses softplus-based barrier: F = k * softplus(-z, K)^2 where K = 15/delta.
    Force is positive (upward) when z < delta.

    Args:
        z: Vertical positions of nodes (n,)
        stiffness: Normal force stiffness k
        delta: Barrier activation distance

    Returns:
        Normal force magnitudes (n,), positive = upward
    """
    K = 15.0 / delta
    # softplus(-z, K) = (1/K) * log(1 + exp(K * (-z)))
    # For numerical stability, use log-sum-exp trick
    arg = -K * z
    softplus = np.where(
        arg > 20.0,
        -z,  # For large arg, softplus ≈ -z
        np.log1p(np.exp(np.clip(arg, -50, 20))) / K,
    )
    return stiffness * softplus ** 2


def compute_coulomb_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    config: FrictionConfig,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute Coulomb friction forces with smooth barrier normal force.

    Args:
        positions: Node positions (n, 3)
        velocities: Node velocities (n, 3)
        config: Friction configuration
        eps: Regularization for velocity normalization

    Returns:
        Force vectors (n, 3) to apply to each node
    """
    n = len(positions)
    forces = np.zeros((n, 3))

    # Normal force from barrier
    z = positions[:, 2]
    f_normal = compute_barrier_normal_force(z, config.ground_stiffness, config.ground_delta)

    # Apply upward normal force
    forces[:, 2] += f_normal

    # Tangential friction: F_t = -mu * F_n * v_t / |v_t|
    # Only for nodes with nonzero normal force
    active = f_normal > eps
    if not np.any(active):
        return forces

    # Tangential velocity (xy plane for ground contact)
    v_xy = velocities[active, :2]
    v_speed = np.linalg.norm(v_xy, axis=1)

    # Sigmoid regularization for smooth transition at v=0
    # sigma(v) = v / sqrt(v^2 + eps^2) — smooth approximation of sign(v)
    reg_factor = v_speed / np.sqrt(v_speed ** 2 + config.stribeck_velocity ** 2)

    # Normalized tangential direction (safe division)
    v_dir = np.zeros_like(v_xy)
    nonzero = v_speed > eps
    v_dir[nonzero] = v_xy[nonzero] / v_speed[nonzero, np.newaxis]

    # Friction force magnitude
    f_friction = config.mu_kinetic * f_normal[active] * reg_factor

    # Apply tangential friction (opposing velocity)
    forces[active, 0] -= f_friction * v_dir[:, 0]
    forces[active, 1] -= f_friction * v_dir[:, 1]

    return forces


def compute_stribeck_force(
    positions: np.ndarray,
    velocities: np.ndarray,
    config: FrictionConfig,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute Stribeck friction forces (Coulomb + static-to-kinetic transition).

    The effective friction coefficient varies with speed:
        mu_eff = mu_k + (mu_s - mu_k) * exp(-|v_t|^2 / v_s^2)

    At zero velocity, mu_eff = mu_s (static friction).
    At high velocity, mu_eff → mu_k (kinetic friction).

    Args:
        positions: Node positions (n, 3)
        velocities: Node velocities (n, 3)
        config: Friction configuration
        eps: Regularization for velocity normalization

    Returns:
        Force vectors (n, 3) to apply to each node
    """
    n = len(positions)
    forces = np.zeros((n, 3))

    # Normal force from barrier
    z = positions[:, 2]
    f_normal = compute_barrier_normal_force(z, config.ground_stiffness, config.ground_delta)

    # Apply upward normal force
    forces[:, 2] += f_normal

    # Tangential friction with Stribeck curve
    active = f_normal > eps
    if not np.any(active):
        return forces

    v_xy = velocities[active, :2]
    v_speed = np.linalg.norm(v_xy, axis=1)

    # Stribeck curve: mu_eff = mu_k + (mu_s - mu_k) * exp(-v^2 / v_s^2)
    v_s = config.stribeck_velocity
    mu_eff = config.mu_kinetic + (config.mu_static - config.mu_kinetic) * np.exp(
        -v_speed ** 2 / (v_s ** 2 + 1e-12)
    )

    # Sigmoid regularization for smooth transition at v=0
    reg_factor = v_speed / np.sqrt(v_speed ** 2 + v_s ** 2)

    # Normalized tangential direction
    v_dir = np.zeros_like(v_xy)
    nonzero = v_speed > eps
    v_dir[nonzero] = v_xy[nonzero] / v_speed[nonzero, np.newaxis]

    # Friction force magnitude
    f_friction = mu_eff * f_normal[active] * reg_factor

    # Apply tangential friction (opposing velocity)
    forces[active, 0] -= f_friction * v_dir[:, 0]
    forces[active, 1] -= f_friction * v_dir[:, 1]

    return forces


# ---------------------------------------------------------------------------
# PyElastica forcing classes
# ---------------------------------------------------------------------------


class CoulombForcing:
    """Coulomb friction forcing for PyElastica rods.

    Applies smooth-barrier normal force and Coulomb tangential friction
    to a CosseratRod system during each substep.
    """

    def __init__(self, config: FrictionConfig):
        self.config = config

    def apply_forces(self, system, time: float = 0.0):
        """Apply Coulomb friction forces to the rod.

        Args:
            system: CosseratRod system (PyElastica)
            time: Current simulation time
        """
        # PyElastica stores positions as (3, n_nodes) and velocities as (3, n_nodes)
        positions = system.position_collection.T  # → (n_nodes, 3)
        velocities = system.velocity_collection.T  # → (n_nodes, 3)

        forces = compute_coulomb_force(positions, velocities, self.config)

        # Add to external forces (3, n_nodes format)
        system.external_forces += forces.T


class StribeckForcing:
    """Stribeck friction forcing for PyElastica rods.

    Applies smooth-barrier normal force and Stribeck (static-to-kinetic)
    tangential friction to a CosseratRod system.
    """

    def __init__(self, config: FrictionConfig):
        self.config = config

    def apply_forces(self, system, time: float = 0.0):
        """Apply Stribeck friction forces to the rod.

        Args:
            system: CosseratRod system (PyElastica)
            time: Current simulation time
        """
        positions = system.position_collection.T  # → (n_nodes, 3)
        velocities = system.velocity_collection.T  # → (n_nodes, 3)

        forces = compute_stribeck_force(positions, velocities, self.config)

        system.external_forces += forces.T
