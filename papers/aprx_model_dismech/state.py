"""Rod state utilities: pack/unpack between DisMech and flat 2D vectors.

DisMech stores full 3D state (positions q, velocities u) for all nodes.
This module extracts the 2D (x, y) projection and computes yaw and angular
velocity from finite differences of segment tangents, producing the same
124-dim flat representation used by the Elastica surrogate.

Raw state (124-dim, stored on disk):
    - positions (x, y) for 21 nodes:    42 floats  (absolute)
    - velocities (x, y) for 21 nodes:   42 floats
    - yaw angles for 20 elements:        20 floats
    - angular velocities (z) for 20 el:  20 floats
    Total:                               124 floats

Relative state (128-dim, used as model input):
    - CoM (x, y):                         2 floats  (absolute)
    - heading (sin, cos):                  2 floats  (from mean element yaw)
    - relative positions (x, y) for 21:   42 floats  (node pos - CoM)
    - velocities (x, y) for 21 nodes:     42 floats  (unchanged)
    - yaw angles for 20 elements:         20 floats  (unchanged)
    - angular velocities (z) for 20 el:   20 floats  (unchanged)
    Total:                                128 floats
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_NODES = 21
NUM_ELEMENTS = 20
NUM_JOINTS = 19
ACTION_DIM = 5
TIME_ENC_DIM = 4  # sin/cos phase (2) + sin/cos n_cycles (2)
PER_ELEMENT_PHASE_DIM = 60  # 20 elements x (sin + cos + kappa) = 60

# Raw state (124-dim, as stored in .pt files)
RAW_STATE_DIM = 124
STATE_DIM = RAW_STATE_DIM  # Backward-compat alias

# Named slices into the raw 124-dim state vector
POS_X = slice(0, 21)
POS_Y = slice(21, 42)
VEL_X = slice(42, 63)
VEL_Y = slice(63, 84)
YAW = slice(84, 104)
OMEGA_Z = slice(104, 124)

# Relative state (128-dim, used as model input/output)
REL_STATE_DIM = 128
REL_INPUT_DIM = REL_STATE_DIM + ACTION_DIM + PER_ELEMENT_PHASE_DIM  # 193

# Named slices into the relative 128-dim state vector
REL_COM_X = slice(0, 1)
REL_COM_Y = slice(1, 2)
REL_HEADING_SIN = slice(2, 3)
REL_HEADING_COS = slice(3, 4)
REL_POS_X = slice(4, 25)
REL_POS_Y = slice(25, 46)
REL_VEL_X = slice(46, 67)
REL_VEL_Y = slice(67, 88)
REL_YAW = slice(88, 108)
REL_OMEGA_Z = slice(108, 128)

# Legacy alias (code that imported INPUT_DIM)
INPUT_DIM = REL_INPUT_DIM


# ---------------------------------------------------------------------------
# RodState2D: pack / unpack
# ---------------------------------------------------------------------------


class RodState2D:
    """Static methods for converting between DisMech rod and flat 2D state."""

    @staticmethod
    def pack_from_dismech(snake_robot) -> np.ndarray:
        """Extract 124-dim 2D state from a DisMech SnakeRobot.

        DisMech stores full 3D state vectors. We extract (x, y) components
        for positions and velocities, then compute yaw and angular velocity
        from segment tangent vectors and their time derivatives.

        Args:
            snake_robot: SnakeRobot instance (from src.physics.snake_robot).

        Returns:
            Flat numpy array of shape (124,).
        """
        # Extract 3D positions: q contains [x0,y0,z0, x1,y1,z1, ...] for all nodes
        q = snake_robot._dismech_robot.state.q
        positions = q[:3 * NUM_NODES].reshape(NUM_NODES, 3)

        pos_x = positions[:, 0].copy()  # (21,)
        pos_y = positions[:, 1].copy()  # (21,)

        # Extract 3D velocities
        u = snake_robot._dismech_robot.state.u
        velocities = u[:3 * NUM_NODES].reshape(NUM_NODES, 3)

        vel_x = velocities[:, 0].copy()  # (21,)
        vel_y = velocities[:, 1].copy()  # (21,)

        # Compute yaw (20 elements) from segment tangent vectors
        # tangent_e[i] = positions[i+1] - positions[i]
        dx = positions[1:, 0] - positions[:-1, 0]  # (20,)
        dy = positions[1:, 1] - positions[:-1, 1]  # (20,)
        yaw = np.arctan2(dy, dx).copy()  # (20,)

        # Compute omega_z (20 elements) from velocity cross product
        # omega_z_e = (dx * dv_y - dy * dv_x) / (dx^2 + dy^2 + eps)
        # This is d(yaw)/dt computed via the cross product of tangent
        # and velocity difference vectors.
        dv_x = velocities[1:, 0] - velocities[:-1, 0]  # (20,)
        dv_y = velocities[1:, 1] - velocities[:-1, 1]  # (20,)
        seg_len_sq = dx**2 + dy**2 + 1e-10
        omega_z = (dx * dv_y - dy * dv_x) / seg_len_sq  # (20,)

        return np.concatenate([pos_x, pos_y, vel_x, vel_y, yaw, omega_z]).astype(
            np.float32
        )

    @staticmethod
    def pack_forces(snake_robot) -> Dict[str, np.ndarray]:
        """Capture force / torque snapshots from the DisMech rod (full 3D).

        DisMech does not expose force arrays in the same way as PyElastica.
        Returns zeros as placeholders for API compatibility.

        Args:
            snake_robot: SnakeRobot instance (from src.physics.snake_robot).

        Returns:
            Dictionary with keys:
                external_forces  (3, 21)
                internal_forces  (3, 21)
                external_torques (3, 20)
                internal_torques (3, 20)
        """
        # TODO: Extract forces from DisMech if the API provides them.
        # Currently DisMech does not expose per-node force arrays directly.
        return {
            "external_forces": np.zeros((3, NUM_NODES), dtype=np.float32),
            "internal_forces": np.zeros((3, NUM_NODES), dtype=np.float32),
            "external_torques": np.zeros((3, NUM_ELEMENTS), dtype=np.float32),
            "internal_torques": np.zeros((3, NUM_ELEMENTS), dtype=np.float32),
        }

    @staticmethod
    def unpack_positions_xy(state: torch.Tensor) -> torch.Tensor:
        """Extract (x, y) node positions from state tensor.

        Args:
            state: (..., 124) state tensor.

        Returns:
            (..., 2, 21) positions tensor.
        """
        px = state[..., POS_X]  # (..., 21)
        py = state[..., POS_Y]  # (..., 21)
        return torch.stack([px, py], dim=-2)  # (..., 2, 21)

    @staticmethod
    def unpack_velocities_xy(state: torch.Tensor) -> torch.Tensor:
        """Extract (x, y) node velocities from state tensor.

        Args:
            state: (..., 124) state tensor.

        Returns:
            (..., 2, 21) velocities tensor.
        """
        vx = state[..., VEL_X]
        vy = state[..., VEL_Y]
        return torch.stack([vx, vy], dim=-2)


# ---------------------------------------------------------------------------
# Raw <-> Relative state conversion
# ---------------------------------------------------------------------------


def raw_to_relative(state: torch.Tensor) -> torch.Tensor:
    """Convert raw 124-dim absolute state to 128-dim relative representation.

    Factorises absolute node positions into:
        - CoM (2): absolute center-of-mass (x, y)
        - Heading (2): sin/cos of mean element yaw angle
        - Relative positions (42): node positions minus CoM

    Velocities, yaw, and omega_z are passed through unchanged.

    Args:
        state: (..., 124) raw state tensor.

    Returns:
        (..., 128) relative state tensor.
    """
    pos_x = state[..., POS_X]       # (..., 21)
    pos_y = state[..., POS_Y]       # (..., 21)
    vel_x = state[..., VEL_X]       # (..., 21)
    vel_y = state[..., VEL_Y]       # (..., 21)
    yaw = state[..., YAW]           # (..., 20)
    omega_z = state[..., OMEGA_Z]   # (..., 20)

    # Absolute CoM
    com_x = pos_x.mean(dim=-1, keepdim=True)   # (..., 1)
    com_y = pos_y.mean(dim=-1, keepdim=True)   # (..., 1)

    # Body heading from mean element yaw (sin/cos avoids wraparound issues)
    heading = yaw.mean(dim=-1)                  # (...,)
    heading_sin = heading.sin().unsqueeze(-1)   # (..., 1)
    heading_cos = heading.cos().unsqueeze(-1)   # (..., 1)

    # Relative positions (node - CoM)
    rel_pos_x = pos_x - com_x                  # (..., 21)
    rel_pos_y = pos_y - com_y                  # (..., 21)

    return torch.cat([
        com_x, com_y, heading_sin, heading_cos,
        rel_pos_x, rel_pos_y,
        vel_x, vel_y,
        yaw, omega_z,
    ], dim=-1)


def relative_to_raw(state: torch.Tensor) -> torch.Tensor:
    """Convert 128-dim relative state back to 124-dim raw absolute state.

    Reconstructs absolute node positions from CoM + relative positions.
    Heading (sin/cos) is discarded (redundant with yaw array).

    Args:
        state: (..., 128) relative state tensor.

    Returns:
        (..., 124) raw state tensor.
    """
    com_x = state[..., REL_COM_X]          # (..., 1)
    com_y = state[..., REL_COM_Y]          # (..., 1)
    rel_pos_x = state[..., REL_POS_X]     # (..., 21)
    rel_pos_y = state[..., REL_POS_Y]     # (..., 21)
    vel_x = state[..., REL_VEL_X]         # (..., 21)
    vel_y = state[..., REL_VEL_Y]         # (..., 21)
    yaw = state[..., REL_YAW]             # (..., 20)
    omega_z = state[..., REL_OMEGA_Z]     # (..., 20)

    # Reconstruct absolute positions
    pos_x = rel_pos_x + com_x              # (..., 21)
    pos_y = rel_pos_y + com_y              # (..., 21)

    return torch.cat([pos_x, pos_y, vel_x, vel_y, yaw, omega_z], dim=-1)


# ---------------------------------------------------------------------------
# Phase encoding (omega * t)
# ---------------------------------------------------------------------------

# The serpenoid curvature is kappa(s, t) = A * sin(k*s + omega*t + phi).
# The surrogate needs the oscillation phase omega*t, not raw time t, because
# omega = 2*pi*frequency varies per RL step (frequency is an action output).
# An MLP cannot easily compute sin(omega*t) from sin(t), cos(t), and omega.

DEFAULT_FREQUENCY_RANGE = (0.5, 3.0)


def action_to_omega(action: np.ndarray, freq_range: tuple = DEFAULT_FREQUENCY_RANGE) -> float:
    """Extract angular frequency omega from a single action vector.

    Args:
        action: Action array where index 1 is normalized frequency in [-1, 1].
        freq_range: (min_freq, max_freq) in Hz.

    Returns:
        omega = 2*pi*frequency in rad/s.
    """
    freq_norm = (float(np.clip(action[1], -1.0, 1.0)) + 1) / 2
    frequency = freq_range[0] + freq_norm * (freq_range[1] - freq_range[0])
    return 2 * np.pi * frequency


def action_to_omega_batch(action: torch.Tensor, freq_range: tuple = DEFAULT_FREQUENCY_RANGE) -> torch.Tensor:
    """Extract angular frequency omega from a batch of action tensors.

    Args:
        action: (..., 5) action tensor where index 1 is normalized frequency.
        freq_range: (min_freq, max_freq) in Hz.

    Returns:
        (...,) tensor of omega = 2*pi*frequency in rad/s.
    """
    import math
    freq_norm = (action[..., 1].clamp(-1.0, 1.0) + 1) / 2
    frequency = freq_range[0] + freq_norm * (freq_range[1] - freq_range[0])
    return 2 * math.pi * frequency


def encode_phase(phase: float) -> np.ndarray:
    """Encode oscillation phase as [sin(phase), cos(phase)].

    Args:
        phase: Oscillation phase omega*t (radians).

    Returns:
        Array of shape (2,).
    """
    return np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)


def encode_phase_batch(phase: torch.Tensor) -> torch.Tensor:
    """Encode oscillation phase for a batch.

    Args:
        phase: (...,) tensor of phases omega*t (radians).

    Returns:
        (..., 2) tensor of [sin(phase), cos(phase)].
    """
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)


# ---------------------------------------------------------------------------
# n_cycles encoding
# ---------------------------------------------------------------------------

DT_CTRL = 0.05  # seconds per RL macro-step (1 implicit step x 0.05s)


def encode_n_cycles(action: np.ndarray, freq_range: tuple = DEFAULT_FREQUENCY_RANGE) -> np.ndarray:
    """Encode number of CPG oscillation cycles per RL step as [sin, cos].

    n_cycles = frequency * dt_ctrl. When n_cycles is integer (full cycles),
    sin=0, cos=1; when half-integer (reversal), sin=0, cos=-1.

    Args:
        action: (5,) action array, index 1 is normalized frequency in [-1, 1].
        freq_range: (min_freq, max_freq) in Hz.

    Returns:
        (2,) array: [sin(2*pi*n_cycles), cos(2*pi*n_cycles)].
    """
    freq_norm = (float(np.clip(action[1], -1.0, 1.0)) + 1) / 2
    frequency = freq_range[0] + freq_norm * (freq_range[1] - freq_range[0])
    n_cycles = frequency * DT_CTRL
    phase = 2 * np.pi * n_cycles
    return np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)


def encode_n_cycles_batch(action: torch.Tensor, freq_range: tuple = DEFAULT_FREQUENCY_RANGE) -> torch.Tensor:
    """Batch version of encode_n_cycles.

    Args:
        action: (..., 5) action tensor, index 1 is normalized frequency.
        freq_range: (min_freq, max_freq) in Hz.

    Returns:
        (..., 2) tensor: [sin(2*pi*n_cycles), cos(2*pi*n_cycles)].
    """
    import math
    freq_norm = (action[..., 1].clamp(-1.0, 1.0) + 1) / 2
    frequency = freq_range[0] + freq_norm * (freq_range[1] - freq_range[0])
    n_cycles = frequency * DT_CTRL
    phase = 2 * math.pi * n_cycles
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1)


# Backward-compatible aliases (deprecated -- use encode_phase / encode_phase_batch)
def encode_serpenoid_time(t: float) -> np.ndarray:
    """Deprecated: use encode_phase(omega * t) instead."""
    return encode_phase(t)


def encode_serpenoid_time_batch(t: torch.Tensor) -> torch.Tensor:
    """Deprecated: use encode_phase_batch(omega * t) instead."""
    return encode_phase_batch(t)


# ---------------------------------------------------------------------------
# Per-element phase encoding
# ---------------------------------------------------------------------------

# Element arc-positions: 20 elements uniformly spaced in [0, 1].
# Matches the convention used in the Elastica surrogate.
# Do NOT import from locomotion_elastica -- hardcoded to avoid circular imports.
_ELEMENT_ARC_POSITIONS = np.linspace(0.0, 1.0, NUM_ELEMENTS, dtype=np.float32)  # (20,)
_ELEMENT_ARC_POSITIONS_TORCH = torch.from_numpy(_ELEMENT_ARC_POSITIONS)
_FREQUENCY_RANGE = (0.5, 3.0)  # Hz


def encode_per_element_phase(
    action: np.ndarray,
    t: float,
    freq_range: tuple = _FREQUENCY_RANGE,
) -> np.ndarray:
    """Encode per-element CPG phase for 20 rod elements.

    Replaces the 2-dim global [sin(wt), cos(wt)] with 60 per-element features:
        For element j (j=0..19, s_j uniformly spaced in [0, 1]):
            sin(k*s_j + w*t + phi)    -- 20 features
            cos(k*s_j + w*t + phi)    -- 20 features
            A*sin(k*s_j + w*t + phi)  -- 20 features (commanded curvature kappa_j)

    Args:
        action: (5,) array [amplitude, frequency, wave_number, phase_offset, turn_bias]
                normalized in [-1, 1].
        t: Accumulated serpenoid time (seconds).
        freq_range: (min_freq, max_freq) in Hz for denormalizing action[1].

    Returns:
        (60,) float32 array: [sin_0..sin_19, cos_0..cos_19, kappa_0..kappa_19].
    """
    # Denormalize action components
    amp_norm = float(np.clip(action[0], -1.0, 1.0))
    A = (amp_norm + 1.0) / 2.0 * 5.0  # amplitude in [0, 5] rad/m

    freq_norm = float(np.clip(action[1], -1.0, 1.0))
    frequency = freq_range[0] + (freq_norm + 1.0) / 2.0 * (freq_range[1] - freq_range[0])
    omega = 2.0 * np.pi * frequency  # rad/s

    wave_norm = float(np.clip(action[2], -1.0, 1.0))
    # wave_number range [0.5, 3.5] -- matches perturb_rod_state() in collect_data.py
    wave_number = 0.5 + (wave_norm + 1.0) / 2.0 * 3.0
    k = 2.0 * np.pi * wave_number  # spatial frequency rad/m

    phi = float(np.clip(action[3], -1.0, 1.0)) * np.pi  # phase_offset in [-pi, pi]

    # Per-element phase angles: shape (20,)
    phase_angles = k * _ELEMENT_ARC_POSITIONS + omega * t + phi

    sin_features = np.sin(phase_angles).astype(np.float32)   # (20,)
    cos_features = np.cos(phase_angles).astype(np.float32)   # (20,)
    kappa_features = (A * sin_features).astype(np.float32)   # (20,)

    return np.concatenate([sin_features, cos_features, kappa_features])  # (60,)


def encode_per_element_phase_batch(
    actions: torch.Tensor,
    t: torch.Tensor,
    freq_range: tuple = _FREQUENCY_RANGE,
) -> torch.Tensor:
    """Batch version of encode_per_element_phase for training.

    Args:
        actions: (N, 5) action tensor, normalized [-1, 1].
        t: (N,) tensor of accumulated serpenoid times (seconds).
        freq_range: (min_freq, max_freq) in Hz.

    Returns:
        (N, 60) float32 tensor.
    """
    import math

    s = _ELEMENT_ARC_POSITIONS_TORCH.to(actions.device)  # (20,)

    # Denormalize
    A = (actions[:, 0].clamp(-1.0, 1.0) + 1.0) / 2.0 * 5.0        # (N,)
    freq_norm = (actions[:, 1].clamp(-1.0, 1.0) + 1.0) / 2.0
    frequency = freq_range[0] + freq_norm * (freq_range[1] - freq_range[0])
    omega = 2.0 * math.pi * frequency                                # (N,)
    wave_norm = (actions[:, 2].clamp(-1.0, 1.0) + 1.0) / 2.0
    wave_number = 0.5 + wave_norm * 3.0
    k = 2.0 * math.pi * wave_number                                  # (N,)
    phi = actions[:, 3].clamp(-1.0, 1.0) * math.pi                  # (N,)

    # Phase angles: (N, 20)
    phase_angles = (
        k[:, None] * s[None, :]
        + (omega * t + phi)[:, None]
    )

    sin_f = torch.sin(phase_angles)    # (N, 20)
    cos_f = torch.cos(phase_angles)    # (N, 20)
    kappa_f = A[:, None] * sin_f       # (N, 20)

    return torch.cat([sin_f, cos_f, kappa_f], dim=-1).float()  # (N, 60)


# ---------------------------------------------------------------------------
# StateNormalizer
# ---------------------------------------------------------------------------


class StateNormalizer:
    """Per-feature z-score normalization for rod state vectors.

    Computes and stores running mean/std from training data. Applies
    (x - mean) / (std + eps) normalization.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        device: str = "cpu",
        eps: float = 1e-8,
    ):
        self.state_dim = state_dim
        self.eps = eps
        self.device = device

        self.state_mean = torch.zeros(state_dim, device=device)
        self.state_std = torch.ones(state_dim, device=device)
        self.delta_mean = torch.zeros(state_dim, device=device)
        self.delta_std = torch.ones(state_dim, device=device)

    def fit(
        self,
        states: torch.Tensor,
        deltas: torch.Tensor,
    ) -> None:
        """Compute normalization statistics from training data.

        Args:
            states: (N, state_dim) tensor of rod states.
            deltas: (N, state_dim) tensor of state deltas (next - current).
        """
        self.state_mean = states.mean(dim=0).to(self.device)
        self.state_std = states.std(dim=0).to(self.device)
        self.delta_mean = deltas.mean(dim=0).to(self.device)
        self.delta_std = deltas.std(dim=0).to(self.device)

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize a state tensor."""
        return (state - self.state_mean) / (self.state_std + self.eps)

    def normalize_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Normalize a state delta tensor."""
        return (delta - self.delta_mean) / (self.delta_std + self.eps)

    def denormalize_delta(self, delta_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize a predicted state delta."""
        return delta_norm * (self.delta_std + self.eps) + self.delta_mean

    def to(self, device: str) -> "StateNormalizer":
        """Move normalizer to a device."""
        self.device = device
        self.state_mean = self.state_mean.to(device)
        self.state_std = self.state_std.to(device)
        self.delta_mean = self.delta_mean.to(device)
        self.delta_std = self.delta_std.to(device)
        return self

    def save(self, path: str) -> None:
        """Save normalizer statistics to disk."""
        torch.save(
            {
                "state_mean": self.state_mean.cpu(),
                "state_std": self.state_std.cpu(),
                "delta_mean": self.delta_mean.cpu(),
                "delta_std": self.delta_std.cpu(),
                "eps": self.eps,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "StateNormalizer":
        """Load normalizer statistics from disk."""
        data = torch.load(path, map_location=device, weights_only=True)
        norm = cls(state_dim=data["state_mean"].shape[0], device=device, eps=data["eps"])
        norm.state_mean = data["state_mean"].to(device)
        norm.state_std = data["state_std"].to(device)
        norm.delta_mean = data["delta_mean"].to(device)
        norm.delta_std = data["delta_std"].to(device)
        return norm
