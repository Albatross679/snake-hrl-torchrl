"""Tests for per-element CPG phase encoding in state.py.

Verifies:
  - encode_per_element_phase() returns (60,) float32 array
  - encode_per_element_phase_batch() returns (N, 60) float32 tensor
  - INPUT_DIM == 189, PER_ELEMENT_PHASE_DIM == 60
  - Backward compatibility: encode_phase still returns (2,)
  - Formula correctness for element j=0 (s_j=0)
"""

import math

import numpy as np
import pytest
import torch

from aprx_model_elastica.state import (
    ACTION_DIM,
    INPUT_DIM,
    PER_ELEMENT_PHASE_DIM,
    STATE_DIM,
    encode_per_element_phase,
    encode_per_element_phase_batch,
)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_encode_per_element_phase_shape():
    """encode_per_element_phase returns (60,) for any valid action/time."""
    action = np.zeros(5, dtype=np.float32)
    out = encode_per_element_phase(action, t=0.0)
    assert out.shape == (60,), f"Expected (60,), got {out.shape}"
    assert out.dtype == np.float32, f"Expected float32, got {out.dtype}"


def test_encode_per_element_phase_batch_shape():
    """encode_per_element_phase_batch with (4, 5) actions and (4,) t returns (4, 60)."""
    actions = torch.zeros(4, 5)
    t = torch.zeros(4)
    out = encode_per_element_phase_batch(actions, t)
    assert out.shape == (4, 60), f"Expected (4, 60), got {out.shape}"
    assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


def test_input_dim():
    """INPUT_DIM == 189, PER_ELEMENT_PHASE_DIM == 60."""
    assert PER_ELEMENT_PHASE_DIM == 60, f"PER_ELEMENT_PHASE_DIM={PER_ELEMENT_PHASE_DIM}, expected 60"
    assert INPUT_DIM == 189, f"INPUT_DIM={INPUT_DIM}, expected 189"
    # Verify the arithmetic identity
    assert INPUT_DIM == STATE_DIM + ACTION_DIM + PER_ELEMENT_PHASE_DIM, (
        f"INPUT_DIM {INPUT_DIM} != {STATE_DIM} + {ACTION_DIM} + {PER_ELEMENT_PHASE_DIM}"
    )


# ---------------------------------------------------------------------------
# Formula correctness tests
# ---------------------------------------------------------------------------


def test_encode_per_element_phase_values():
    """For element j=0 (s_j=0.0), verify sin/cos/kappa features.

    Action: amplitude_norm=0 → A=2.5 (midpoint of [0,5])
            frequency_norm=0 → freq=1.75 Hz → omega=2*pi*1.75
            wave_number_norm=0 → wave_number=2.0 → k=2*pi*2.0
            phase_offset_norm=0 → phi=0
            turn_bias_norm=0 → not used in encoding

    At t=1.0:
        phase_angle_j0 = k*0.0 + omega*t + phi
                       = 0 + 2*pi*1.75*1.0 + 0
                       = 2*pi*1.75

        sin_j0 = sin(2*pi*1.75)
        cos_j0 = cos(2*pi*1.75)
        kappa_j0 = A * sin_j0 = 2.5 * sin(2*pi*1.75)
    """
    action = np.zeros(5, dtype=np.float32)  # all normalized to 0
    t = 1.0

    # Expected values
    A = 2.5  # (0+1)/2 * 5 = 2.5
    freq = 1.75  # 0.5 + (0+1)/2 * 2.5 = 0.5 + 1.25 = 1.75 Hz
    omega = 2 * math.pi * freq
    k = 2 * math.pi * 2.0  # wave_norm=0 → wave_number = 0.5 + (0+1)/2*3.0 = 2.0
    phi = 0.0

    phase_j0 = k * 0.0 + omega * t + phi
    expected_sin_j0 = math.sin(phase_j0)
    expected_cos_j0 = math.cos(phase_j0)
    expected_kappa_j0 = A * expected_sin_j0

    out = encode_per_element_phase(action, t)
    # Layout: [sin_0..sin_19, cos_0..cos_19, kappa_0..kappa_19]
    sin_j0 = float(out[0])
    cos_j0 = float(out[20])
    kappa_j0 = float(out[40])

    assert abs(sin_j0 - expected_sin_j0) < 1e-5, f"sin_j0: {sin_j0} != {expected_sin_j0}"
    assert abs(cos_j0 - expected_cos_j0) < 1e-5, f"cos_j0: {cos_j0} != {expected_cos_j0}"
    assert abs(kappa_j0 - expected_kappa_j0) < 1e-5, f"kappa_j0: {kappa_j0} != {expected_kappa_j0}"


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_backward_compat():
    """encode_phase still works and returns (2,); STATE_DIM and ACTION_DIM unchanged."""
    from aprx_model_elastica.state import encode_phase

    result = encode_phase(0.0)
    assert result.shape == (2,), f"encode_phase shape: {result.shape}"

    assert STATE_DIM == 124, f"STATE_DIM changed: {STATE_DIM}"
    assert ACTION_DIM == 5, f"ACTION_DIM changed: {ACTION_DIM}"
