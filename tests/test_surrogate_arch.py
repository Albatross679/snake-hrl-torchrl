"""Tests for Phase 3.1 architectural experiments: residual MLP, history-window MLP,
trajectory dataset windows, and CLI argument wiring.

Covers ARCH-01 through ARCH-04 as defined in Phase 3.1 plan 01.
"""

import sys
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# ARCH-01: ResidualSurrogateModel forward pass
# ---------------------------------------------------------------------------


def test_residual_model_forward():
    """ARCH-01: ResidualSurrogateModel produces (B, 124) output from (B,124)/(B,5)/(B,2)."""
    from aprx_model_elastica.model import ResidualSurrogateModel
    from aprx_model_elastica.train_config import SurrogateModelConfig

    config = SurrogateModelConfig(hidden_dims=[512, 512, 512])
    model = ResidualSurrogateModel(config)

    state = torch.randn(2, 124)
    action = torch.randn(2, 5)
    time_enc = torch.randn(2, 2)

    output = model(state, action, time_enc)

    assert output.shape == (2, 124), f"Expected (2, 124), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"


# ---------------------------------------------------------------------------
# ARCH-02: HistorySurrogateModel forward pass
# ---------------------------------------------------------------------------


def test_history_model_forward():
    """ARCH-02: HistorySurrogateModel produces (B, 124) output with K=2 history."""
    from aprx_model_elastica.model import HistorySurrogateModel
    from aprx_model_elastica.train_config import SurrogateModelConfig

    config = SurrogateModelConfig(hidden_dims=[512, 512, 512])
    model = HistorySurrogateModel(config, history_k=2)

    state = torch.randn(2, 124)
    action = torch.randn(2, 5)
    time_enc = torch.randn(2, 2)
    history_states = torch.randn(2, 2, 124)   # (B, K, 124)
    history_actions = torch.randn(2, 2, 5)    # (B, K, 5)

    output = model(state, action, time_enc, history_states, history_actions)

    assert output.shape == (2, 124), f"Expected (2, 124), got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"


# ---------------------------------------------------------------------------
# ARCH-03: TrajectoryDataset produces windows for rollout_length=16
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not Path("data/surrogate").exists(),
    reason="data/surrogate not present (CI environment without data)",
)
def test_trajectory_dataset_windows():
    """ARCH-03: TrajectoryDataset returns >0 windows for rollout_length=16."""
    from aprx_model_elastica.dataset import TrajectoryDataset

    dataset = TrajectoryDataset("data/surrogate", rollout_length=16)
    assert len(dataset) > 0, (
        f"TrajectoryDataset has 0 windows for rollout_length=16 — "
        "episodes may be too short or data_dir is empty"
    )


# ---------------------------------------------------------------------------
# ARCH-04: CLI args --rollout-weight and --rollout-steps wire to config
# ---------------------------------------------------------------------------


def test_train_cli_args(monkeypatch):
    """ARCH-04: parse_args() honours --rollout-weight, --rollout-steps, --use-residual, --history-k."""
    from aprx_model_elastica.train_surrogate import parse_args
    from aprx_model_elastica.train_config import SurrogateTrainConfig

    monkeypatch.setattr(
        sys, "argv",
        ["train_surrogate", "--rollout-weight", "0.3", "--rollout-steps", "16",
         "--use-residual", "--history-k", "4",
         "--epochs", "0"],
    )

    args = parse_args()

    # Verify namespace values
    assert args.rollout_weight == pytest.approx(0.3), (
        f"Expected rollout_weight=0.3, got {args.rollout_weight}"
    )
    assert args.rollout_steps == 16, (
        f"Expected rollout_steps=16, got {args.rollout_steps}"
    )
    assert args.use_residual is True, "Expected use_residual=True"
    assert args.history_k == 4, f"Expected history_k=4, got {args.history_k}"

    # Verify values propagate to SurrogateTrainConfig
    config = SurrogateTrainConfig()
    if args.rollout_weight is not None:
        config.rollout_loss_weight = args.rollout_weight
    if args.rollout_steps is not None:
        config.rollout_steps = args.rollout_steps
    config.use_residual = args.use_residual
    if args.history_k is not None:
        config.history_k = args.history_k

    assert config.rollout_loss_weight == pytest.approx(0.3)
    assert config.rollout_steps == 16
    assert config.use_residual is True
    assert config.history_k == 4
