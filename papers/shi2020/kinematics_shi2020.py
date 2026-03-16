"""Geometric mechanics kinematics for 3-link snake robots (Shi et al., 2020).

Implements the kinematic reconstruction equation:
    ξ = -A(α) · α̇

where:
    ξ = (ξ_x, ξ_y, ξ_θ) are body velocities in SE(2)
    α = (α₁, α₂) are joint angles (shape variables)
    A(α) is the local connection form (3×2 matrix)

Two variants:
    1. Wheeled robot: nonholonomic constraints from wheel contacts
    2. Swimming robot: low Reynolds number drag forces
"""

import numpy as np


class ConnectionForm:
    """Computes the local connection form A(α) for kinematic snake robots.

    The connection form maps shape velocities to body velocities:
        ξ = -A(α) · α̇

    For the wheeled robot, A is derived from nonholonomic (Pfaffian)
    constraints. For the swimming robot, A comes from force balance
    in low Reynolds number fluid.
    """

    def __init__(self, robot_type: str = "wheeled", link_length: float = 1.0,
                 drag_coefficient: float = 1.0):
        self.robot_type = robot_type
        self.R = link_length
        self.k = drag_coefficient

    def evaluate(self, alpha: np.ndarray) -> np.ndarray:
        """Compute A(α) at the given joint configuration.

        Args:
            alpha: Joint angles [α₁, α₂], shape (2,).

        Returns:
            Connection form A, shape (3, 2).
        """
        if self.robot_type == "wheeled":
            return self._wheeled_connection(alpha)
        elif self.robot_type == "swimming":
            return self._swimming_connection(alpha)
        else:
            raise ValueError(f"Unknown robot type: {self.robot_type}")

    def _wheeled_connection(self, alpha: np.ndarray) -> np.ndarray:
        """Connection form for nonholonomic wheeled 3-link robot.

        From Eq. (4) in the paper:
            A(α) = -(1/D) * [[R/2*(cos α₁ + cos(α₁-α₂)), R/2*(1+cos α₁)],
                              [0,                           0              ],
                              [sin α₁ + sin(α₁-α₂),        sin α₁        ]]

        where D = sin α₁ + sin(α₁ - α₂) - sin α₂
        """
        a1, a2 = alpha
        R = self.R

        D = np.sin(a1) + np.sin(a1 - a2) - np.sin(a2)
        if abs(D) < 1e-10:
            return np.zeros((3, 2))

        A = np.zeros((3, 2))
        A[0, 0] = R / 2 * (np.cos(a1) + np.cos(a1 - a2))
        A[0, 1] = R / 2 * (1 + np.cos(a1))
        A[1, 0] = 0.0
        A[1, 1] = 0.0
        A[2, 0] = np.sin(a1) + np.sin(a1 - a2)
        A[2, 1] = np.sin(a1)

        return -A / D

    def _swimming_connection(self, alpha: np.ndarray) -> np.ndarray:
        """Connection form for 3-link low Reynolds number swimmer.

        Derived from force/torque balance with viscous drag, following
        Hatton and Choset (2013). The connection has the same structure
        as the wheeled robot but with nonzero y-component.
        """
        a1, a2 = alpha
        R = self.R

        # Drag-based reconstruction (simplified from full derivation)
        c1, c2 = np.cos(a1), np.cos(a2)
        s1, s2 = np.sin(a1), np.sin(a2)
        c12 = np.cos(a1 - a2)
        s12 = np.sin(a1 - a2)

        D = 3 + 2 * c1 + 2 * c2 + 2 * c12
        if abs(D) < 1e-10:
            return np.zeros((3, 2))

        A = np.zeros((3, 2))
        # x-body velocity
        A[0, 0] = R / 2 * (c1 + c12) / D * 3
        A[0, 1] = R / 2 * (1 + c1) / D * 3
        # y-body velocity (nonzero for swimmer)
        A[1, 0] = R / 2 * (s1 + s12) / D * 3
        A[1, 1] = R / 2 * s1 / D * 3
        # θ-body velocity
        A[2, 0] = (s1 + s12) / D * 3
        A[2, 1] = s1 / D * 3

        return -A

    def body_velocity(self, alpha: np.ndarray, alpha_dot: np.ndarray) -> np.ndarray:
        """Compute body velocity from shape velocity.

        Args:
            alpha: Joint angles [α₁, α₂], shape (2,).
            alpha_dot: Joint velocities [α̇₁, α̇₂], shape (2,).

        Returns:
            Body velocity [ξ_x, ξ_y, ξ_θ], shape (3,).
        """
        A = self.evaluate(alpha)
        return -A @ alpha_dot

    def world_velocity(self, alpha: np.ndarray, alpha_dot: np.ndarray,
                       theta: float) -> np.ndarray:
        """Compute world-frame velocity from body velocity.

        Uses the left translation T_e L_g to map body frame → world frame:
            ẏ = T_e L_g · ξ

        Args:
            alpha: Joint angles, shape (2,).
            alpha_dot: Joint velocities, shape (2,).
            theta: Current orientation angle.

        Returns:
            World velocity [ẋ, ẏ, θ̇], shape (3,).
        """
        xi = self.body_velocity(alpha, alpha_dot)

        # SE(2) left translation
        c, s = np.cos(theta), np.sin(theta)
        T = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1],
        ])

        return T @ xi

    def exterior_derivative(self, alpha: np.ndarray, component: int = 0,
                            h: float = 1e-5) -> float:
        """Compute exterior derivative dA_i at a joint configuration.

        dA_i(α) = ∂A_{i,2}/∂α₁ - ∂A_{i,1}/∂α₂

        This measures the geometric phase (displacement per area in
        joint space) and is visualized as surfaces in Fig. 3 and 5.

        Args:
            alpha: Joint angles, shape (2,).
            component: 0=x, 1=y, 2=θ.
            h: Finite difference step.

        Returns:
            Scalar exterior derivative value.
        """
        a1, a2 = alpha

        # ∂A_{i,2}/∂α₁ (derivative of second column w.r.t. first shape var)
        A_plus = self.evaluate(np.array([a1 + h, a2]))
        A_minus = self.evaluate(np.array([a1 - h, a2]))
        dA_col2_da1 = (A_plus[component, 1] - A_minus[component, 1]) / (2 * h)

        # ∂A_{i,1}/∂α₂ (derivative of first column w.r.t. second shape var)
        A_plus = self.evaluate(np.array([a1, a2 + h]))
        A_minus = self.evaluate(np.array([a1, a2 - h]))
        dA_col1_da2 = (A_plus[component, 0] - A_minus[component, 0]) / (2 * h)

        return dA_col2_da1 - dA_col1_da2
