"""Matsuoka CPG network for soft snake actuation (Liu, Onal & Fu, 2021).

Primitive Matsuoka CPG oscillators consist of mutually inhibited
extensor/flexor neuron pairs. The network of coupled oscillators
produces coordinated rhythmic patterns for soft pneumatic actuation.

Key property: steering control via imbalanced tonic inputs.
The bias of a Matsuoka oscillator is proportional to the amplitude
of u^e when u^f = 1 - u^e.

Tonic input composition from C1 and R2:
    u = w₁·u₁ + w₂·u₂
"""

import numpy as np

from liu2021.configs_liu2021 import MatsuokaCPGConfig


class MatsuokaOscillator:
    """Single Matsuoka oscillator (extensor + flexor neuron pair).

    Dynamics (Eq. 2 in paper):
        K_f·τ_r·ẋᵢᵉ = -xᵢᵉ - a·zᵢᶠ - b·yᵢᵉ - Σⱼ wⱼᵢ·yⱼᵉ + uᵢᵉ
        K_f·τ_a·ẏᵢᵉ = zᵢᵉ - yᵢᵉ
        (symmetric for flexor with superscript f)

    where zᵢᵍ = max(0, xᵢᵍ) is the output, g ∈ {e, f}.
    """

    def __init__(self, config: MatsuokaCPGConfig):
        self.config = config

        # Extensor neuron state: (x, y)
        self.xe = 0.0
        self.ye = 0.0
        # Flexor neuron state: (x, y)
        self.xf = 0.0
        self.yf = 0.0

    @property
    def output_e(self) -> float:
        """Extensor output z^e = max(0, x^e)."""
        return max(0.0, self.xe)

    @property
    def output_f(self) -> float:
        """Flexor output z^f = max(0, x^f)."""
        return max(0.0, self.xf)

    @property
    def output(self) -> float:
        """Net oscillator output (extensor - flexor)."""
        return self.output_e - self.output_f

    def step(self, ue: float, uf: float, K_f: float, dt: float,
             coupling_input_e: float = 0.0, coupling_input_f: float = 0.0) -> float:
        """Advance one timestep.

        Args:
            ue: Extensor tonic input.
            uf: Flexor tonic input.
            K_f: Frequency ratio.
            dt: Timestep.
            coupling_input_e: Inhibitory coupling from neighbors (extensor).
            coupling_input_f: Inhibitory coupling from neighbors (flexor).

        Returns:
            Net oscillator output.
        """
        cfg = self.config
        tau_r = cfg.tau_r
        tau_a = cfg.tau_a
        a = cfg.a
        b = cfg.b

        # Extensor dynamics
        dxe = (-self.xe - a * self.output_f - b * self.ye
               - coupling_input_e + ue) / (K_f * tau_r)
        dye = (self.output_e - self.ye) / (K_f * tau_a)

        # Flexor dynamics
        dxf = (-self.xf - a * self.output_e - b * self.yf
               - coupling_input_f + uf) / (K_f * tau_r)
        dyf = (self.output_f - self.yf) / (K_f * tau_a)

        # Euler integration
        self.xe += dxe * dt
        self.ye += dye * dt
        self.xf += dxf * dt
        self.yf += dyf * dt

        return self.output

    def reset(self, rng: np.random.Generator = None):
        """Reset oscillator state with small random perturbation."""
        if rng is not None:
            self.xe = rng.uniform(-0.1, 0.1)
            self.ye = rng.uniform(0, 0.1)
            self.xf = rng.uniform(-0.1, 0.1)
            self.yf = rng.uniform(0, 0.1)
        else:
            self.xe = self.ye = self.xf = self.yf = 0.0


class MatsuokaCPG:
    """Coupled Matsuoka CPG network for soft snake locomotion.

    N coupled oscillators produce coordinated rhythmic signals.
    Tonic inputs control steering bias; frequency ratio controls speed.

    Tonic input mapping (Eq. 3):
        uᵢᵉ = sigmoid(aᵢ) = 1 / (1 + exp(-aᵢ))
        uᵢᶠ = 1 - uᵢᵉ

    This bounds tonic inputs to [0, 1] and ensures complementary
    extensor/flexor drive.
    """

    def __init__(self, config: MatsuokaCPGConfig = None):
        self.config = config or MatsuokaCPGConfig()
        n = self.config.num_oscillators
        self.oscillators = [MatsuokaOscillator(self.config) for _ in range(n)]
        self._rng = np.random.default_rng(42)

    @property
    def num_oscillators(self) -> int:
        return self.config.num_oscillators

    def reset(self, rng: np.random.Generator = None):
        """Reset all oscillators."""
        for osc in self.oscillators:
            osc.reset(rng or self._rng)

    def step(self, action_vector: np.ndarray, K_f: float, dt: float) -> np.ndarray:
        """Advance CPG one timestep.

        Args:
            action_vector: Raw action from RL [a₁, a₂, a₃, a₄], shape (4,).
                Mapped to tonic inputs via sigmoid.
            K_f: Frequency ratio (controls oscillation speed).
            dt: Timestep.

        Returns:
            Actuation outputs for each link, shape (num_oscillators,).
        """
        n = self.num_oscillators
        w = self.config.w_coupling

        # Map actions to tonic inputs (Eq. 3)
        ue = 1.0 / (1.0 + np.exp(-action_vector))
        uf = 1.0 - ue

        # Compute coupling inputs (nearest-neighbor inhibition)
        outputs = np.zeros(n)
        for i in range(n):
            coupling_e = 0.0
            coupling_f = 0.0
            # Coupling from neighbors
            if i > 0:
                coupling_e += w * self.oscillators[i - 1].output_e
                coupling_f += w * self.oscillators[i - 1].output_f
            if i < n - 1:
                coupling_e += w * self.oscillators[i + 1].output_e
                coupling_f += w * self.oscillators[i + 1].output_f

            outputs[i] = self.oscillators[i].step(
                ue[i], uf[i], K_f, dt, coupling_e, coupling_f
            )

        return outputs

    def compose_tonic_inputs(
        self,
        a1: np.ndarray,
        a2: np.ndarray,
        w1: float = 0.5,
        w2: float = 0.5,
    ) -> np.ndarray:
        """Compose tonic inputs from C1 and R2 controllers (Eq. 4).

        u = w₁·u₁ + w₂·u₂

        The linear composition property of Matsuoka oscillators ensures
        the bias is a linear combination of the individual biases.

        Args:
            a1: C1 action vector, shape (4,).
            a2: R2 action vector, shape (4,).
            w1: C1 weight.
            w2: R2 weight.

        Returns:
            Combined tonic input vector, shape (4,).
        """
        u1 = 1.0 / (1.0 + np.exp(-a1))
        u2 = 1.0 / (1.0 + np.exp(-a2))

        # Combined as 8-dim (extensor + flexor) per Eq. (4)
        ue = w1 * u1 + w2 * u2
        return ue
