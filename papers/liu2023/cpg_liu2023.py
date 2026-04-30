"""Matsuoka CPG oscillator network for Liu et al. (2023).

Implements equations (1)-(5) from:
    Liu et al., "Reinforcement Learning of CPG-regulated Locomotion Controller
    for a Soft Snake Robot", IEEE Transactions on Industrial Electronics, 2023.

Key differences from the generic MatsuokaOscillator in src/physics/cpg/:
    - Separate excitatory/flexor tonic inputs u^e, u^f with sigmoid mapping
    - FOC constant c = 0.75 added to dynamics
    - Frequency scaling via K_f multiplying all time constants
    - Explicit asymmetric cross-oscillator coupling (w_ij != w_ji)
    - Output scaled by a_psi: y = a_psi * (z^e - z^f)
"""

import numpy as np


class LiuMatsuokaOscillator:
    """Single Matsuoka oscillator with Liu et al. 2023 FOC extensions.

    State per neuron (e=excitatory, f=flexor):
        x^e, x^f: membrane potentials
        v^e, v^f: adaptation variables
        z^e = max(0, x^e), z^f = max(0, x^f): firing rates

    Dynamics (eq. 1-3):
        tau_r * dx^e/dt = c - x^e - b*v^e - w_ji*z^f + u^e
        tau_a * dv^e/dt = -v^e + z^e
        (and symmetrically for flexor neuron)

    Tonic input mapping (eq. 4):
        u^e = u_max * sigma(alpha)
        u^f = u_max * sigma(-alpha)
        where sigma(x) = 1/(1+exp(-x))
    """

    def __init__(
        self,
        a_psi: float = 2.0935,
        b: float = 10.0355,
        tau_r: float = 0.7696,
        tau_a: float = 1.7728,
        a_i: float = 4.6062,
        w_ij: float = 8.8669,
        w_ji: float = 0.7844,
        c: float = 0.75,
        u_max: float = 5.0,
    ):
        self.a_psi = a_psi
        self.b = b
        self.tau_r = tau_r
        self.tau_a = tau_a
        self.a_i = a_i
        self.w_ij = w_ij  # excitatory → flexor inhibition weight
        self.w_ji = w_ji  # flexor → excitatory inhibition weight
        self.c = c
        self.u_max = u_max

        # State: [x_e, v_e, x_f, v_f]
        self.state = np.array([0.5, 0.0, -0.5, 0.0])
        self.z_e = 0.0
        self.z_f = 0.0

    def reset(self, perturb: float = 0.0) -> None:
        """Reset oscillator state with optional perturbation for symmetry breaking."""
        self.state = np.array([0.5 + perturb, 0.0, -0.5 - perturb, 0.0])
        self.z_e = max(0.0, self.state[0])
        self.z_f = max(0.0, self.state[2])

    def step(self, dt: float, alpha: float, kf: float = 1.0) -> float:
        """Advance oscillator by one timestep.

        Args:
            dt: Timestep (seconds).
            alpha: Tonic input from RL policy (raw, before sigmoid).
            kf: Frequency scaling factor (K_f). Divides all time constants.

        Returns:
            Oscillator output in range approximately [-a_psi, a_psi].
        """
        # Sigmoid tonic mapping (eq. 4)
        sig = 1.0 / (1.0 + np.exp(-alpha))
        u_e = self.u_max * sig
        u_f = self.u_max * (1.0 - sig)

        x_e, v_e, x_f, v_f = self.state

        # Firing rates (ReLU)
        self.z_e = max(0.0, x_e)
        self.z_f = max(0.0, x_f)

        # Scaled time constants
        tau_r = self.tau_r / kf
        tau_a = self.tau_a / kf

        # Dynamics (eq. 1-3)
        dx_e = (self.c - x_e - self.b * v_e - self.w_ji * self.z_f + u_e) / tau_r
        dv_e = (-v_e + self.z_e) / tau_a
        dx_f = (self.c - x_f - self.b * v_f - self.w_ij * self.z_e + u_f) / tau_r
        dv_f = (-v_f + self.z_f) / tau_a

        # Euler integration
        self.state += dt * np.array([dx_e, dv_e, dx_f, dv_f])

        return self.output

    @property
    def output(self) -> float:
        """Scaled output: a_psi * (z_e - z_f)."""
        return self.a_psi * (self.z_e - self.z_f)


class LiuCPGNetwork:
    """Network of coupled Matsuoka oscillators for 4-link snake.

    Each of the 4 oscillators receives an independent tonic input alpha_i
    from the RL policy. Inter-oscillator coupling uses weights from Table II.

    The coupling adds a_i * sum_j(z^e_j - z^f_j) to each neuron's input,
    creating phase coordination between adjacent oscillators.
    """

    def __init__(
        self,
        num_oscillators: int = 4,
        a_psi: float = 2.0935,
        b: float = 10.0355,
        tau_r: float = 0.7696,
        tau_a: float = 1.7728,
        a_i: float = 4.6062,
        w_ij: float = 8.8669,
        w_ji: float = 0.7844,
        c: float = 0.75,
        u_max: float = 5.0,
        max_pressure: float = 1.0,
    ):
        self.num_oscillators = num_oscillators
        self.max_pressure = max_pressure

        self.oscillators = [
            LiuMatsuokaOscillator(
                a_psi=a_psi, b=b, tau_r=tau_r, tau_a=tau_a,
                a_i=a_i, w_ij=w_ij, w_ji=w_ji, c=c, u_max=u_max,
            )
            for _ in range(num_oscillators)
        ]

        # Store coupling weight
        self._a_i = a_i

    def reset(self) -> None:
        """Reset all oscillators with small perturbations for symmetry breaking."""
        for i, osc in enumerate(self.oscillators):
            osc.reset(perturb=0.01 * i)

    def step(self, dt: float, alphas: np.ndarray, kf: float = 1.0) -> np.ndarray:
        """Advance the CPG network by one physics timestep.

        Args:
            dt: Physics timestep (seconds).
            alphas: Tonic inputs from RL, shape (num_oscillators,).
            kf: Frequency scaling factor K_f.

        Returns:
            Actuator commands in [-1, 1], shape (num_oscillators,).
        """
        # Compute inter-oscillator coupling input
        # Each oscillator gets contribution from all other oscillators
        coupling = np.zeros(self.num_oscillators)
        for i in range(self.num_oscillators):
            for j in range(self.num_oscillators):
                if i != j:
                    osc_j = self.oscillators[j]
                    coupling[i] += self._a_i * (osc_j.z_e - osc_j.z_f)

        # Step each oscillator with its tonic input + coupling
        raw_outputs = np.zeros(self.num_oscillators)
        for i, osc in enumerate(self.oscillators):
            # Add coupling to the tonic input
            alpha_coupled = alphas[i] + coupling[i]
            raw_outputs[i] = osc.step(dt, alpha_coupled, kf)

        # Scale to [-1, 1] and apply max_pressure
        outputs = np.clip(raw_outputs / osc.a_psi, -1.0, 1.0) * self.max_pressure
        return outputs

    @property
    def outputs(self) -> np.ndarray:
        """Current outputs from all oscillators."""
        return np.array([osc.output for osc in self.oscillators])
