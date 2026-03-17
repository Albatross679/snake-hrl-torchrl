"""Unit tests for Phase 3 surrogate model infrastructure.

Tests for:
- TransformerSurrogateModel (FT-Transformer architecture)
- SurrogateModelConfig new fields (arch, n_layers, n_heads, d_model)
- RMSNorm usage inside TransformerSurrogateModel
"""

import torch
import pytest

from aprx_model_elastica.train_config import SurrogateModelConfig


class TestSurrogateModelConfig:
    """Test new config fields for architecture selection."""

    def test_config_has_arch_field(self):
        config = SurrogateModelConfig()
        assert hasattr(config, "arch")
        assert config.arch == "mlp"

    def test_config_has_n_layers_field(self):
        config = SurrogateModelConfig()
        assert hasattr(config, "n_layers")
        assert config.n_layers == 6

    def test_config_has_n_heads_field(self):
        config = SurrogateModelConfig()
        assert hasattr(config, "n_heads")
        assert config.n_heads == 8

    def test_config_has_d_model_field(self):
        config = SurrogateModelConfig()
        assert hasattr(config, "d_model")
        assert config.d_model == 256


class TestTransformerSurrogateModel:
    """Test TransformerSurrogateModel with FT-Transformer architecture."""

    @pytest.fixture
    def config(self):
        return SurrogateModelConfig(d_model=128, n_layers=4, n_heads=4, arch="transformer")

    @pytest.fixture
    def model(self, config):
        from aprx_model_elastica.model import TransformerSurrogateModel
        return TransformerSurrogateModel(config)

    def test_forward_output_shape(self, model, config):
        """TransformerSurrogateModel.forward returns shape (B, output_dim)."""
        B = 8
        state = torch.randn(B, config.state_dim)
        action = torch.randn(B, config.action_dim)
        time_enc = torch.randn(B, config.time_encoding_dim)
        out = model(state, action, time_enc)
        assert out.shape == (B, config.output_dim)

    def test_output_near_zero_on_init(self, model, config):
        """Zero-initialized output head means near-zero output on init."""
        B = 4
        state = torch.randn(B, config.state_dim)
        action = torch.randn(B, config.action_dim)
        time_enc = torch.randn(B, config.time_encoding_dim)
        out = model(state, action, time_enc)
        assert out.abs().max().item() < 0.01, (
            f"Expected near-zero output on init, got max={out.abs().max().item():.4f}"
        )

    def test_predict_next_state_returns_state_plus_delta(self, model, config):
        """predict_next_state returns state + delta (same interface as MLP models)."""
        B = 4
        state = torch.randn(B, config.state_dim)
        action = torch.randn(B, config.action_dim)
        time_enc = torch.randn(B, config.time_encoding_dim)
        next_state = model.predict_next_state(state, action, time_enc)
        # On init, delta ~0 so next_state ~state
        assert next_state.shape == (B, config.output_dim)
        assert torch.allclose(next_state, state, atol=0.01)

    def test_rmsnorm_used_not_layernorm(self, model):
        """RMSNorm should be used inside TransformerSurrogateModel, not LayerNorm."""
        from aprx_model_elastica.model import RMSNorm
        has_rmsnorm = False
        has_layernorm = False
        for name, module in model.named_modules():
            if isinstance(module, RMSNorm):
                has_rmsnorm = True
            if isinstance(module, torch.nn.LayerNorm):
                has_layernorm = True
        assert has_rmsnorm, "TransformerSurrogateModel should use RMSNorm"
        assert not has_layernorm, "TransformerSurrogateModel should NOT use LayerNorm"

    def test_cls_token_exists(self, model):
        """[CLS] token should be a learnable parameter."""
        assert hasattr(model, "cls_token")
        assert isinstance(model.cls_token, torch.nn.Parameter)

    def test_single_sample_batch(self, model, config):
        """Forward works with batch size 1."""
        state = torch.randn(1, config.state_dim)
        action = torch.randn(1, config.action_dim)
        time_enc = torch.randn(1, config.time_encoding_dim)
        out = model(state, action, time_enc)
        assert out.shape == (1, config.output_dim)
