"""Unit tests for Phase 3 surrogate model changes.

Tests TransformerSurrogateModel, updated configs, FlatStepDataset wiring,
and 15-config sweep setup.
"""

import torch
import pytest

from aprx_model_elastica.train_config import SurrogateModelConfig
from aprx_model_elastica.model import (
    TransformerSurrogateModel,
    RMSNorm,
    SurrogateModel,
    ResidualSurrogateModel,
)


class TestSurrogateModelConfig:
    def test_has_arch_field(self):
        cfg = SurrogateModelConfig()
        assert hasattr(cfg, "arch")
        assert cfg.arch == "mlp"

    def test_has_transformer_fields(self):
        cfg = SurrogateModelConfig()
        assert cfg.n_layers == 6
        assert cfg.n_heads == 8
        assert cfg.d_model == 256


class TestTransformerSurrogateModel:
    @pytest.fixture
    def config(self):
        return SurrogateModelConfig(
            arch="transformer", d_model=128, n_layers=4, n_heads=4
        )

    @pytest.fixture
    def model(self, config):
        return TransformerSurrogateModel(config)

    def test_forward_shape(self, model):
        B = 4
        state = torch.randn(B, 124)
        action = torch.randn(B, 5)
        time_enc = torch.randn(B, 2)
        out = model(state, action, time_enc)
        assert out.shape == (B, 124)

    def test_output_near_zero_on_init(self, model):
        """Zero-initialized output head should produce near-zero deltas."""
        state = torch.randn(2, 124)
        action = torch.randn(2, 5)
        time_enc = torch.randn(2, 2)
        out = model(state, action, time_enc)
        assert out.abs().max().item() < 0.1

    def test_predict_next_state(self, model):
        state = torch.randn(2, 124)
        action = torch.randn(2, 5)
        time_enc = torch.randn(2, 2)
        next_state = model.predict_next_state(state, action, time_enc)
        assert next_state.shape == (2, 124)
        # With zero-init output head, next_state ≈ state
        assert torch.allclose(next_state, state, atol=0.1)

    def test_uses_rmsnorm(self, model):
        """TransformerSurrogateModel must use RMSNorm, not LayerNorm."""
        has_rmsnorm = any(isinstance(m, RMSNorm) for m in model.modules())
        has_layernorm = any(
            isinstance(m, torch.nn.LayerNorm) for m in model.modules()
        )
        assert has_rmsnorm, "TransformerSurrogateModel should use RMSNorm"
        assert not has_layernorm, "TransformerSurrogateModel should not use LayerNorm"


class TestRMSNorm:
    def test_output_shape(self):
        norm = RMSNorm(64)
        x = torch.randn(4, 64)
        assert norm(x).shape == (4, 64)


class TestSweepConfigs:
    def test_has_15_configs(self):
        from aprx_model_elastica.sweep import SWEEP_CONFIGS
        assert len(SWEEP_CONFIGS) == 15

    def test_all_configs_have_required_keys(self):
        from aprx_model_elastica.sweep import SWEEP_CONFIGS
        required = {"name", "lr", "hidden_dims", "arch", "rollout_weight"}
        for cfg in SWEEP_CONFIGS:
            missing = required - set(cfg.keys())
            assert not missing, f"Config {cfg['name']} missing keys: {missing}"

    def test_arch_distribution(self):
        from aprx_model_elastica.sweep import SWEEP_CONFIGS
        archs = [c["arch"] for c in SWEEP_CONFIGS]
        assert archs.count("mlp") == 8  # 5 MLP + 3 Wide/Deep
        assert archs.count("residual") == 3
        assert archs.count("transformer") == 4

    def test_transformer_configs_have_extra_keys(self):
        from aprx_model_elastica.sweep import SWEEP_CONFIGS
        for cfg in SWEEP_CONFIGS:
            if cfg["arch"] == "transformer":
                assert "n_layers" in cfg, f"{cfg['name']} missing n_layers"
                assert "n_heads" in cfg, f"{cfg['name']} missing n_heads"
                assert "d_model" in cfg, f"{cfg['name']} missing d_model"


class TestCLIArgs:
    def test_arch_arg_accepted(self):
        """--arch arg should be accepted by parse_args."""
        import sys
        from unittest.mock import patch
        with patch.object(sys, "argv", ["prog", "--arch", "transformer"]):
            from aprx_model_elastica.train_surrogate import parse_args
            args = parse_args()
            assert args.arch == "transformer"
