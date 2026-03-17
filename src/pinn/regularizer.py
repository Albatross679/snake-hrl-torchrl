"""Physics regularizer for surrogate model training.

Computes soft physics constraint losses from predicted state deltas:
1. Kinematic consistency (position-velocity coupling, x and y)
2. Angular kinematic consistency (yaw-omega coupling)
3. Curvature-moment consistency (approximate constitutive law)
4. Energy conservation (bounded kinetic energy change)

All constraints are algebraic (no differentiable simulator needed) and
operate on raw 124-dim state vectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from papers.aprx_model_elastica.state import (
    POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z,
    NUM_ELEMENTS,
)


class PhysicsRegularizer(nn.Module):
    """Physics constraint regularizer for Cosserat rod surrogate models.

    Computes a scalar loss penalizing violations of known physics laws.
    All constraints use trapezoidal integration (average of current and
    next velocities) for improved accuracy.

    Args:
        dt: Control timestep in seconds.
        snake_length: Rod total length in meters.
        energy_threshold: Maximum allowed dimensionless KE change before penalty.
    """

    def __init__(
        self,
        dt: float = 0.5,
        snake_length: float = 1.0,
        energy_threshold: float = 10.0,
    ):
        super().__init__()
        self.register_buffer("dt", torch.tensor(dt))
        self.dl = snake_length / NUM_ELEMENTS  # element length
        self.energy_threshold = energy_threshold

    def forward(self, state: torch.Tensor, delta_pred: torch.Tensor) -> torch.Tensor:
        """Compute physics regularization loss.

        Args:
            state: (..., 124) current raw state tensor.
            delta_pred: (..., 124) predicted state delta (next - current).

        Returns:
            Scalar loss tensor with gradients.
        """
        dt = self.dt

        # Next state
        next_state = state + delta_pred

        # ------------------------------------------------------------------
        # Constraint 1: Kinematic consistency (x, y positions)
        # delta_pos ≈ (v_current + 0.5 * delta_v) * dt  (trapezoidal)
        # ------------------------------------------------------------------
        avg_vel_x = state[..., VEL_X] + 0.5 * delta_pred[..., VEL_X]
        expected_dx = avg_vel_x * dt
        loss_kin_x = F.mse_loss(delta_pred[..., POS_X], expected_dx)

        avg_vel_y = state[..., VEL_Y] + 0.5 * delta_pred[..., VEL_Y]
        expected_dy = avg_vel_y * dt
        loss_kin_y = F.mse_loss(delta_pred[..., POS_Y], expected_dy)

        loss_kin = loss_kin_x + loss_kin_y

        # ------------------------------------------------------------------
        # Constraint 2: Angular kinematic consistency
        # delta_yaw ≈ (omega + 0.5 * delta_omega) * dt
        # ------------------------------------------------------------------
        avg_omega = state[..., OMEGA_Z] + 0.5 * delta_pred[..., OMEGA_Z]
        expected_dyaw = avg_omega * dt
        loss_ang = F.mse_loss(delta_pred[..., YAW], expected_dyaw)

        # ------------------------------------------------------------------
        # Constraint 3: Curvature-moment consistency (approximate)
        # kappa ≈ diff(yaw) / dl, curvature change should be smooth
        # Low weight (0.1x) since this is approximate
        # ------------------------------------------------------------------
        yaw_curr = state[..., YAW]
        yaw_next = next_state[..., YAW]

        kappa_curr = torch.diff(yaw_curr, dim=-1) / self.dl  # (..., 19)
        kappa_next = torch.diff(yaw_next, dim=-1) / self.dl  # (..., 19)
        delta_kappa = kappa_next - kappa_curr

        # Curvature change should be consistent with angular velocity difference
        omega_curr = state[..., OMEGA_Z]
        delta_omega_diff = torch.diff(delta_pred[..., OMEGA_Z], dim=-1)  # (..., 19)

        loss_curv = 0.1 * F.mse_loss(delta_kappa, delta_omega_diff * dt / self.dl)

        # ------------------------------------------------------------------
        # Constraint 4: Energy conservation (bounded KE change)
        # Penalize unreasonably large kinetic energy changes
        # ------------------------------------------------------------------
        vel_sq = state[..., VEL_X] ** 2 + state[..., VEL_Y] ** 2
        next_vel_sq = next_state[..., VEL_X] ** 2 + next_state[..., VEL_Y] ** 2

        KE = 0.5 * vel_sq.sum(dim=-1)          # (...,)
        KE_next = 0.5 * next_vel_sq.sum(dim=-1)  # (...,)
        delta_KE = KE_next - KE

        loss_energy = torch.mean(
            F.relu(delta_KE.abs() - self.energy_threshold) ** 2
        )

        return loss_kin + loss_ang + loss_curv + loss_energy
