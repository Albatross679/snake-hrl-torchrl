"""Unit tests for Phase 3 surrogate model infrastructure.

Tests for:
- TransformerSurrogateModel (FT-Transformer architecture)
- SurrogateModelConfig new fields (arch, n_layers, n_heads, d_model)
- RMSNorm usage inside TransformerSurrogateModel
- sweep.py 15-config table
- --arch CLI arg acceptance
"""

import sys
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


class TestSweepConfigs:
    """Integration tests for sweep.py configuration table."""

    def test_sweep_has_15_configs(self):
        from aprx_model_elastica.sweep import SWEEP_CONFIGS
        assert len(SWEEP_CONFIGS) == 15, f"Expected 15 configs, got {len(SWEEP_CONFIGS)}"

    def test_all_configs_have_required_keys(self):
        from aprx_model_elastica.sweep import SWEEP_CONFIGS
        required_keys = {"name", "lr", "hidden_dims", "arch", "rollout_weight"}
        for cfg in SWEEP_CONFIGS:
            missing = required_keys - set(cfg.keys())
            assert not missing, f"Config {cfg.get('name', '?')} missing keys: {missing}"

    def test_transformer_configs_have_extra_keys(self):
        from aprx_model_elastica.sweep import SWEEP_CONFIGS
        transformer_configs = [c for c in SWEEP_CONFIGS if c["arch"] == "transformer"]
        assert len(transformer_configs) == 4, f"Expected 4 transformer configs, got {len(transformer_configs)}"
        for cfg in transformer_configs:
            for key in ("n_layers", "n_heads", "d_model"):
                assert key in cfg, f"Transformer config {cfg['name']} missing {key}"

    def test_all_rollout_weights_zero(self):
        from aprx_model_elastica.sweep import SWEEP_CONFIGS
        for cfg in SWEEP_CONFIGS:
            assert cfg["rollout_weight"] == 0.0, (
                f"Config {cfg['name']} has rollout_weight={cfg['rollout_weight']}, expected 0.0"
            )

    def test_config_names_unique(self):
        from aprx_model_elastica.sweep import SWEEP_CONFIGS
        names = [c["name"] for c in SWEEP_CONFIGS]
        assert len(names) == len(set(names)), f"Duplicate config names: {names}"


class TestArchCLIArg:
    """Test --arch CLI arg acceptance by train_surrogate.py parse_args()."""

    def test_arch_arg_accepted(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "train_surrogate", "--arch", "transformer", "--no-wandb",
        ])
        from aprx_model_elastica.train_surrogate import parse_args
        args = parse_args()
        assert args.arch == "transformer"

    def test_arch_mlp_accepted(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "train_surrogate", "--arch", "mlp", "--no-wandb",
        ])
        from aprx_model_elastica.train_surrogate import parse_args
        args = parse_args()
        assert args.arch == "mlp"

    def test_arch_residual_accepted(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "train_surrogate", "--arch", "residual", "--no-wandb",
        ])
        from aprx_model_elastica.train_surrogate import parse_args
        args = parse_args()
        assert args.arch == "residual"

    def test_transformer_args_accepted(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", [
            "train_surrogate", "--arch", "transformer",
            "--n-layers", "4", "--n-heads", "4", "--d-model", "128",
            "--no-wandb",
        ])
        from aprx_model_elastica.train_surrogate import parse_args
        args = parse_args()
        assert args.n_layers == 4
        assert args.n_heads == 4
        assert args.d_model == 128
