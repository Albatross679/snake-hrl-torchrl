"""Dual CPG controller for COBRA snake robot (Jiang et al., 2024).

Implements the Bing (2017) coupled phase oscillator (Eq. 9-11 from the paper).
Two CPGs control pitch (5 oscillators) and yaw (6 oscillators) joints separately,
with the RL agent modulating 7 CPG parameters.

This is *not* reusing src/physics/cpg/ — the formulation is fundamentally different
(tridiagonal coupling matrix, phase shift matrix, second-order amplitude dynamics).
"""

import numpy as np


class BingCPG:
    """Single CPG based on Bing (2017) coupled phase oscillator model.

    State variables per oscillator i:
        phi_i: phase (rad)
        r_i: amplitude
        rdot_i: amplitude velocity

    Dynamics (Eq. 9-11):
        dphi_i/dt = omega + sum_j A_ij * r_j * sin(phi_j - phi_i - theta_ij)
        drdot_i/dt = a * (a/4 * (R_i - r_i) - rdot_i)
        dr_i/dt = rdot_i

    Output:
        x_i = r_i * cos(phi_i) + delta_i

    Args:
        n: Number of oscillators.
        a: Amplitude convergence rate (controls how fast r_i -> R_i).
        coupling_weight: Weight for nearest-neighbor coupling in matrix A.
    """

    def __init__(self, n: int, a: float = 20.0, coupling_weight: float = 1.0):
        self.n = n
        self.a = a

        # Tridiagonal coupling matrix A (nearest-neighbor)
        self.A = np.zeros((n, n))
        for i in range(n - 1):
            self.A[i, i + 1] = coupling_weight
            self.A[i + 1, i] = coupling_weight

        # State
        self.phi = np.zeros(n)       # Phase
        self.r = np.zeros(n)         # Amplitude
        self.rdot = np.zeros(n)      # Amplitude velocity

    def reset(self):
        """Reset CPG state to zero."""
        self.phi = np.zeros(self.n)
        self.r = np.zeros(self.n)
        self.rdot = np.zeros(self.n)

    def step(
        self,
        dt: float,
        R: float,
        omega: float,
        theta: float,
        delta: float,
    ) -> np.ndarray:
        """Advance CPG by one timestep.

        Args:
            dt: Time step (seconds).
            R: Target amplitude (uniform across oscillators).
            omega: Angular frequency (rad/s).
            theta: Phase shift between adjacent oscillators (rad).
            delta: Output offset.

        Returns:
            Output array of shape (n,).
        """
        # Build phase shift matrix B (theta between adjacent oscillators)
        B = np.zeros((self.n, self.n))
        for i in range(self.n - 1):
            B[i, i + 1] = theta
            B[i + 1, i] = -theta

        # Phase dynamics (Eq. 9)
        dphi = np.full(self.n, omega)
        for i in range(self.n):
            for j in range(self.n):
                if self.A[i, j] != 0:
                    dphi[i] += self.A[i, j] * self.r[j] * np.sin(
                        self.phi[j] - self.phi[i] - B[i, j]
                    )

        # Amplitude dynamics (Eq. 10-11)
        a = self.a
        drdot = a * (a / 4.0 * (R - self.r) - self.rdot)
        dr = self.rdot

        # Euler integration
        self.phi += dphi * dt
        self.rdot += drdot * dt
        self.r += dr * dt

        # Output
        output = self.r * np.cos(self.phi) + delta
        return output


class DualCPGController:
    """Two CPGs controlling pitch and yaw joints of the 11-joint COBRA robot.

    Joint layout (0-indexed):
        Even indices (0, 2, 4, 6, 8, 10) -> yaw joints (6 total)
        Odd indices (1, 3, 5, 7, 9) -> pitch joints (5 total)

    RL action (7-dim):
        [R1, R2, omega, theta1, theta2, delta1, delta2]
        R1: pitch amplitude, R2: yaw amplitude
        omega: shared angular frequency
        theta1: pitch phase shift, theta2: yaw phase shift
        delta1: pitch offset, delta2: yaw offset

    Args:
        num_joints: Total number of joints (11 for COBRA).
        cpg_dt: Time step for CPG integration (seconds).
        num_cpg_steps: Number of CPG steps per RL step.
    """

    def __init__(
        self,
        num_joints: int = 11,
        cpg_dt: float = 0.01,
        num_cpg_steps: int = 100,
    ):
        self.num_joints = num_joints
        self.cpg_dt = cpg_dt
        self.num_cpg_steps = num_cpg_steps

        # 5 pitch oscillators, 6 yaw oscillators
        self.n_pitch = 5
        self.n_yaw = 6

        self.cpg_pitch = BingCPG(n=self.n_pitch)
        self.cpg_yaw = BingCPG(n=self.n_yaw)

        # Index mapping: pitch joints are odd indices, yaw are even
        self.pitch_indices = [1, 3, 5, 7, 9]
        self.yaw_indices = [0, 2, 4, 6, 8, 10]

    def reset(self):
        """Reset both CPGs."""
        self.cpg_pitch.reset()
        self.cpg_yaw.reset()

    def step(self, action: np.ndarray) -> np.ndarray:
        """Generate joint targets from a 7-dim RL action.

        Runs num_cpg_steps of CPG integration and returns the final joint targets.

        Args:
            action: 7-dim array [R1, R2, omega, theta1, theta2, delta1, delta2].

        Returns:
            Joint targets of shape (num_joints,).
        """
        R1, R2, omega, theta1, theta2, delta1, delta2 = action

        # Run CPG steps
        for _ in range(self.num_cpg_steps):
            pitch_out = self.cpg_pitch.step(self.cpg_dt, R1, omega, theta1, delta1)
            yaw_out = self.cpg_yaw.step(self.cpg_dt, R2, omega, theta2, delta2)

        # Interleave into joint targets
        targets = np.zeros(self.num_joints)
        for i, idx in enumerate(self.pitch_indices):
            targets[idx] = pitch_out[i]
        for i, idx in enumerate(self.yaw_indices):
            targets[idx] = yaw_out[i]

        return targets
