"""Differentiable Cosserat rod right-hand side (f_SSM) for DD-PINN training.

Computes dx/dt = f(x, u) entirely in PyTorch so gradients flow to network
parameters. Implements:
- Kinematic equations (position-velocity, yaw-omega coupling)
- Internal elastic moments (bending via curvature)
- Anisotropic RFT friction forces (tangential + normal drag)

This is an approximate f_SSM for a 2D planar snake rod: bending + friction +
kinematics capture the dominant dynamics. Stretching is omitted (inextensible
rod approximation).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.pinn._state_slices import (
    POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z,
    NUM_NODES, NUM_ELEMENTS,
)


class CosseratRHS(nn.Module):
    """Differentiable Cosserat rod right-hand side for 2D planar snake.

    Computes the time derivative dx/dt = f(x) of the 124-dim state vector,
    including elastic bending moments and anisotropic RFT friction.

    Args:
        E: Young's modulus (Pa).
        rho: Material density (kg/m^3).
        r: Rod cross-section radius (m).
        L: Rod total length (m).
        n_elem: Number of rod elements.
        ct: RFT tangential drag coefficient.
        cn: RFT normal drag coefficient.
        eps: Regularization constant for tangent normalization.
    """

    def __init__(
        self,
        E: float = 2e6,
        rho: float = 1200.0,
        r: float = 0.001,
        L: float = 1.0,
        n_elem: int = 20,
        ct: float = 0.01,
        cn: float = 0.1,
        eps: float = 1e-6,
    ):
        super().__init__()

        # Geometric properties
        I = math.pi * r ** 4 / 4.0  # second moment of area
        A = math.pi * r ** 2  # cross-section area
        B = E * I  # bending stiffness
        S = E * A  # stretching stiffness (unused — inextensible approx)
        dl = L / n_elem  # element length
        mass = rho * A * dl  # mass per element

        # Register as buffers so they move with .to(device)
        self.register_buffer("B", torch.tensor(B, dtype=torch.float32))
        self.register_buffer("S", torch.tensor(S, dtype=torch.float32))
        self.register_buffer("dl", torch.tensor(dl, dtype=torch.float32))
        self.register_buffer("mass", torch.tensor(mass, dtype=torch.float32))
        self.register_buffer("I_area", torch.tensor(I, dtype=torch.float32))
        self.register_buffer("rho_val", torch.tensor(rho, dtype=torch.float32))

        self.ct = ct
        self.cn = cn
        self.eps = eps
        self.n_elem = n_elem
        self.n_nodes = n_elem + 1

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute time derivative of 124-dim state vector.

        Args:
            state: (B, 124) raw state tensor.

        Returns:
            (B, 124) time derivative tensor
            [d_pos_x, d_pos_y, d_vel_x, d_vel_y, d_yaw, d_omega_z].
        """
        # Extract state components
        pos_x = state[..., POS_X]  # (B, 21)
        pos_y = state[..., POS_Y]  # (B, 21)
        vel_x = state[..., VEL_X]  # (B, 21)
        vel_y = state[..., VEL_Y]  # (B, 21)
        yaw = state[..., YAW]  # (B, 20)
        omega_z = state[..., OMEGA_Z]  # (B, 20)

        # ------- Kinematic equations -------
        d_pos_x = vel_x  # (B, 21)
        d_pos_y = vel_y  # (B, 21)
        d_yaw = omega_z  # (B, 20)

        # ------- Elastic bending moments -------
        # Curvature from yaw differences: kappa_j = (yaw_{j+1} - yaw_j) / dl
        kappa = torch.diff(yaw, dim=-1) / self.dl  # (B, 19)

        # Bending moment at internal joints
        moment = self.B * kappa  # (B, 19)

        # Torque on elements from bending moment gradient
        # tau_i = (M_{i} - M_{i-1}) / dl for interior elements
        # Boundary elements get single-sided contribution
        tau_elastic = torch.zeros_like(yaw)  # (B, 20)

        # Interior elements (1 to n_elem-2): two-sided moment difference
        tau_elastic[..., 1:-1] = (moment[..., 1:] - moment[..., :-1]) / self.dl

        # Boundary elements: single-sided
        tau_elastic[..., 0] = moment[..., 0] / self.dl
        tau_elastic[..., -1] = -moment[..., -1] / self.dl

        # ------- RFT friction forces -------
        F_friction_x, F_friction_y = self._compute_rft_forces(
            pos_x, pos_y, vel_x, vel_y
        )

        # ------- Internal elastic forces (bending) -------
        # Transverse force from moment gradient at nodes
        # F_elastic_node = -d(moment)/ds projected onto x,y
        F_elastic_x, F_elastic_y = self._compute_elastic_forces(
            pos_x, pos_y, moment
        )

        # ------- Force balance (Newton's 2nd law) -------
        # Mass at nodes: interior nodes have full element mass,
        # boundary nodes have half
        mass_node = self.mass * torch.ones_like(vel_x)  # (B, 21)
        mass_node[..., 0] = self.mass / 2.0
        mass_node[..., -1] = self.mass / 2.0

        F_total_x = F_friction_x + F_elastic_x
        F_total_y = F_friction_y + F_elastic_y

        d_vel_x = F_total_x / mass_node  # (B, 21)
        d_vel_y = F_total_y / mass_node  # (B, 21)

        # ------- Angular momentum balance -------
        # d(omega_z)/dt = tau / (rho * I * dl)
        moment_of_inertia = self.rho_val * self.I_area * self.dl
        d_omega_z = tau_elastic / moment_of_inertia  # (B, 20)

        # Assemble output
        return torch.cat([d_pos_x, d_pos_y, d_vel_x, d_vel_y, d_yaw, d_omega_z], dim=-1)

    def _compute_rft_forces(
        self,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        vel_x: torch.Tensor,
        vel_y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute anisotropic RFT friction forces at nodes.

        Uses regularized tangent computation to avoid gradient singularity
        at zero element length.

        Args:
            pos_x, pos_y: Node positions (B, 21).
            vel_x, vel_y: Node velocities (B, 21).

        Returns:
            (F_x, F_y): Friction force components at each node, both (B, 21).
        """
        eps = self.eps

        # Element tangent vectors from position differences
        dx = torch.diff(pos_x, dim=-1)  # (B, 20)
        dy = torch.diff(pos_y, dim=-1)  # (B, 20)

        # Regularized normalization: tangent / sqrt(norm^2 + eps^2)
        norm_sq = dx ** 2 + dy ** 2
        norm_reg = torch.sqrt(norm_sq + eps ** 2)
        tang_x = dx / norm_reg  # (B, 20)
        tang_y = dy / norm_reg  # (B, 20)

        # Interpolate tangent to nodes (average neighbors)
        node_tang_x = torch.zeros_like(vel_x)  # (B, 21)
        node_tang_y = torch.zeros_like(vel_y)  # (B, 21)

        # First and last nodes use nearest element tangent
        node_tang_x[..., 0] = tang_x[..., 0]
        node_tang_y[..., 0] = tang_y[..., 0]
        node_tang_x[..., -1] = tang_x[..., -1]
        node_tang_y[..., -1] = tang_y[..., -1]

        # Interior nodes: average of neighboring element tangents
        node_tang_x[..., 1:-1] = 0.5 * (tang_x[..., :-1] + tang_x[..., 1:])
        node_tang_y[..., 1:-1] = 0.5 * (tang_y[..., :-1] + tang_y[..., 1:])

        # Re-normalize interpolated tangents
        nt_norm_sq = node_tang_x ** 2 + node_tang_y ** 2
        nt_norm_reg = torch.sqrt(nt_norm_sq + eps ** 2)
        node_tang_x = node_tang_x / nt_norm_reg
        node_tang_y = node_tang_y / nt_norm_reg

        # Tangential velocity: v_tan = (v . tangent) * tangent
        v_dot_t = vel_x * node_tang_x + vel_y * node_tang_y  # (B, 21)
        v_tan_x = v_dot_t * node_tang_x
        v_tan_y = v_dot_t * node_tang_y

        # Normal velocity: v_norm = v - v_tan
        v_norm_x = vel_x - v_tan_x
        v_norm_y = vel_y - v_tan_y

        # RFT: F = -ct * v_tan - cn * v_norm
        F_x = -self.ct * v_tan_x - self.cn * v_norm_x
        F_y = -self.ct * v_tan_y - self.cn * v_norm_y

        return F_x, F_y

    def _compute_elastic_forces(
        self,
        pos_x: torch.Tensor,
        pos_y: torch.Tensor,
        moment: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute elastic forces at nodes from bending moment gradient.

        Converts the moment gradient along the rod into transverse forces
        at nodes, projected onto x and y directions.

        Args:
            pos_x, pos_y: Node positions (B, 21).
            moment: Bending moment at internal joints (B, 19).

        Returns:
            (F_x, F_y): Elastic force components at each node, both (B, 21).
        """
        eps = self.eps

        # Normal direction at internal joints (perpendicular to tangent)
        # Tangent at joint j is average of element j and element j+1
        dx = torch.diff(pos_x, dim=-1)  # (B, 20)
        dy = torch.diff(pos_y, dim=-1)  # (B, 20)

        # Tangent at each internal joint (average of neighboring elements)
        # Joint j sits between element j and element j+1
        tang_x_joint = 0.5 * (dx[..., :-1] + dx[..., 1:])  # (B, 19)
        tang_y_joint = 0.5 * (dy[..., :-1] + dy[..., 1:])  # (B, 19)

        # Normalize
        jt_norm_sq = tang_x_joint ** 2 + tang_y_joint ** 2
        jt_norm_reg = torch.sqrt(jt_norm_sq + eps ** 2)
        tang_x_joint = tang_x_joint / jt_norm_reg
        tang_y_joint = tang_y_joint / jt_norm_reg

        # Normal (perpendicular) direction: rotate tangent by 90 degrees
        norm_x_joint = -tang_y_joint  # (B, 19)
        norm_y_joint = tang_x_joint  # (B, 19)

        # Shear force at joints: V = dM/ds
        # V_j = (M_{j+1} - M_{j-1}) / (2*dl) for interior joints
        # But we only have 19 joints, so use finite differences
        # Force = moment * curvature_gradient, applied as transverse force
        # F_transverse_j = M_j * normal_j / dl (simplified)

        # Distribute moment forces to nodes
        # 21 nodes, 20 elements, 19 internal joints.
        # Joint j connects elements j and j+1, located at node j+1.
        F_x = torch.zeros_like(pos_x)  # (B, 21)
        F_y = torch.zeros_like(pos_y)  # (B, 21)

        # Moment gradient force at joints
        moment_force_x = moment * norm_x_joint / (self.dl ** 2)  # (B, 19)
        moment_force_y = moment * norm_y_joint / (self.dl ** 2)  # (B, 19)

        # Joint j is at node j+1 (j=0..18 → nodes 1..19)
        F_x[..., 1:20] = moment_force_x
        F_y[..., 1:20] = moment_force_y

        return F_x, F_y
