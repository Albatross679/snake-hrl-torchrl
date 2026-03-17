"""Physics-based nondimensionalization scales for Cosserat rod dynamics.

Uses physical reference quantities (rod length, control timestep, elastic force)
rather than statistical z-score normalization. This keeps physics residual terms
at O(1) magnitude, which is critical for PINN loss balancing.
"""

from __future__ import annotations

import math

import torch

from papers.aprx_model_elastica.state import (
    POS_X, POS_Y, VEL_X, VEL_Y, YAW, OMEGA_Z,
    RAW_STATE_DIM,
)


class NondimScales:
    """Physics-based nondimensionalization for 124-dim rod state vectors.

    Reference scales:
        L_ref: rod length (characteristic length)
        t_ref: control timestep (characteristic time)
        V_ref: L_ref / t_ref (characteristic velocity)
        omega_ref: 1 / t_ref (characteristic angular velocity)
        F_ref: E * I / L_ref^2 (characteristic elastic force)

    Args:
        snake_length: Rod total length in meters.
        snake_radius: Rod cross-section radius in meters.
        dt_ctrl: RL control timestep in seconds.
        youngs_modulus: Young's modulus in Pa.
    """

    def __init__(
        self,
        snake_length: float = 1.0,
        snake_radius: float = 0.001,
        dt_ctrl: float = 0.5,
        youngs_modulus: float = 2e6,
    ):
        self.L_ref = snake_length
        self.t_ref = dt_ctrl
        self.V_ref = snake_length / dt_ctrl
        self.omega_ref = 1.0 / dt_ctrl

        # Second moment of area for circular cross-section
        I = math.pi * snake_radius ** 4 / 4.0
        self.I = I
        self.F_ref = youngs_modulus * I / snake_length ** 2

    def nondim_state(self, state: torch.Tensor) -> torch.Tensor:
        """Nondimensionalize a raw 124-dim state tensor.

        Args:
            state: (..., 124) raw state tensor in physical units.

        Returns:
            (..., 124) nondimensionalized state tensor.
        """
        out = state.clone()
        out[..., POS_X] = state[..., POS_X] / self.L_ref
        out[..., POS_Y] = state[..., POS_Y] / self.L_ref
        out[..., VEL_X] = state[..., VEL_X] / self.V_ref
        out[..., VEL_Y] = state[..., VEL_Y] / self.V_ref
        # YAW is dimensionless (radians) — no scaling
        out[..., OMEGA_Z] = state[..., OMEGA_Z] / self.omega_ref
        return out

    def redim_state(self, nondim_state: torch.Tensor) -> torch.Tensor:
        """Re-dimensionalize a nondimensional state back to physical units.

        Args:
            nondim_state: (..., 124) nondimensionalized state tensor.

        Returns:
            (..., 124) state tensor in physical units.
        """
        out = nondim_state.clone()
        out[..., POS_X] = nondim_state[..., POS_X] * self.L_ref
        out[..., POS_Y] = nondim_state[..., POS_Y] * self.L_ref
        out[..., VEL_X] = nondim_state[..., VEL_X] * self.V_ref
        out[..., VEL_Y] = nondim_state[..., VEL_Y] * self.V_ref
        out[..., OMEGA_Z] = nondim_state[..., OMEGA_Z] * self.omega_ref
        return out

    def nondim_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Nondimensionalize a state delta tensor (same scales as state)."""
        return self.nondim_state(delta)

    def redim_delta(self, nondim_delta: torch.Tensor) -> torch.Tensor:
        """Re-dimensionalize a nondimensional delta back to physical units."""
        return self.redim_state(nondim_delta)
