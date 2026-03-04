"""Tests for DDPG trainer and DDPGConfig."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest
import torch

from src.configs.base import MetricGroups, TensorBoard, save_config, load_config
from src.configs.training import DDPGConfig, PPOConfig
from src.trainers.ddpg import DDPGTrainer, DeterministicActor, OUNoise
from src.trainers.logging_utils import compute_grad_norm

try:
    from jiang2024.configs_jiang2024 import CobraEnvConfig
    from jiang2024.env_jiang2024 import CobraNavigationEnv
    has_jiang2024 = True
except ImportError:
    has_jiang2024 = False


class TestDDPGConfig:
    """Test DDPGConfig defaults and inheritance."""

    def test_defaults(self):
        cfg = DDPGConfig()
        assert cfg.tau == 0.001
        assert cfg.buffer_size == 1_000_000
        assert cfg.batch_size == 256
        assert cfg.warmup_steps == 10000
        assert cfg.noise_type == "ou"
        assert cfg.noise_sigma == 0.2
        assert cfg.noise_theta == 0.15
        assert cfg.critic_lr == 1e-3
        assert cfg.actor_lr == 1e-4

    def test_inherits_rl_config(self):
        cfg = DDPGConfig()
        assert hasattr(cfg, "total_frames")
        assert hasattr(cfg, "gamma")
        assert cfg.gamma == 0.99

    def test_custom_values(self):
        cfg = DDPGConfig(tau=0.01, batch_size=128, noise_type="gaussian")
        assert cfg.tau == 0.01
        assert cfg.batch_size == 128
        assert cfg.noise_type == "gaussian"


class TestOUNoise:
    """Test Ornstein-Uhlenbeck noise process."""

    def test_output_shape(self):
        noise = OUNoise(action_dim=7)
        sample = noise.sample()
        assert sample.shape == (7,)

    def test_reset(self):
        noise = OUNoise(action_dim=4, mu=0.0)
        noise.sample()
        noise.sample()
        noise.reset()
        assert np.allclose(noise.state, 0.0)

    def test_mean_reversion(self):
        """OU noise should revert toward mu over many steps."""
        noise = OUNoise(action_dim=1, mu=0.0, theta=0.5, sigma=0.01)
        noise.state = np.array([5.0])
        for _ in range(1000):
            noise.sample()
        # Should be much closer to mu=0
        assert abs(noise.state[0]) < 1.0


class TestDeterministicActor:
    """Test deterministic actor network."""

    def test_output_shape(self):
        actor = DeterministicActor(obs_dim=21, action_dim=7, hidden_dims=[64, 32])
        obs = torch.randn(1, 21)
        action = actor(obs)
        assert action.shape == (1, 7)

    def test_output_bounded(self):
        actor = DeterministicActor(
            obs_dim=10, action_dim=3, hidden_dims=[32],
            action_low=-1.5, action_high=1.5,
        )
        obs = torch.randn(100, 10)
        actions = actor(obs)
        assert torch.all(actions >= -1.5)
        assert torch.all(actions <= 1.5)


@pytest.mark.skipif(not has_jiang2024, reason="jiang2024 not available")
class TestDDPGTrainer:
    """Test DDPG trainer with CobraNavigationEnv."""

    @pytest.fixture
    def trainer(self):
        config = DDPGConfig(
            total_frames=100,
            warmup_steps=10,
            batch_size=8,
            buffer_size=100,
        )
        config.tensorboard.enabled = False
        env = CobraNavigationEnv(config=CobraEnvConfig())
        t = DDPGTrainer(env=env, config=config)
        yield t
        env.close()

    def test_init(self, trainer):
        assert trainer.actor is not None
        assert trainer.critic is not None
        assert trainer.actor_target is not None
        assert trainer.critic_target is not None

    def test_select_action(self, trainer):
        obs = torch.randn(21)
        action = trainer._select_action(obs, add_noise=True)
        assert action.shape == (7,)

    def test_select_action_no_noise(self, trainer):
        obs = torch.randn(21)
        a1 = trainer._select_action(obs, add_noise=False)
        a2 = trainer._select_action(obs, add_noise=False)
        assert torch.allclose(a1, a2)

    def test_soft_update(self, trainer):
        """Soft update should move target params toward source."""
        # Set source and target params to different values
        for p in trainer.actor.parameters():
            p.data.fill_(1.0)
        for p in trainer.actor_target.parameters():
            p.data.fill_(0.0)

        trainer._soft_update(trainer.actor, trainer.actor_target)

        # Target should move toward source by tau
        for p in trainer.actor_target.parameters():
            assert torch.allclose(p.data, torch.full_like(p.data, trainer.config.tau), atol=1e-5)


class TestMetricGroups:
    """Test MetricGroups config and integration."""

    def test_defaults(self):
        mg = MetricGroups()
        assert mg.episode is True
        assert mg.train is True
        assert mg.q_values is True
        assert mg.gradients is True
        assert mg.system is True
        assert mg.timing is True
        assert mg.system_interval == 10

    def test_ddpg_config_has_metric_groups(self):
        cfg = DDPGConfig()
        assert hasattr(cfg.tensorboard, "metrics")
        assert cfg.tensorboard.metrics.q_values is True

    def test_ppo_config_has_metric_groups(self):
        cfg = PPOConfig()
        assert cfg.tensorboard.metrics.gradients is True

    def test_custom_metric_groups(self):
        mg = MetricGroups(system=False, q_values=False, system_interval=5)
        tb = TensorBoard(metrics=mg)
        cfg = DDPGConfig(tensorboard=tb)
        assert cfg.tensorboard.metrics.system is False
        assert cfg.tensorboard.metrics.q_values is False
        assert cfg.tensorboard.metrics.system_interval == 5
        # Other defaults preserved
        assert cfg.tensorboard.metrics.train is True

    def test_json_round_trip(self, tmp_path):
        mg = MetricGroups(system=False, gradients=False, system_interval=20)
        tb = TensorBoard(metrics=mg)
        cfg = DDPGConfig(tensorboard=tb)

        path = tmp_path / "cfg.json"
        save_config(cfg, path)
        loaded = load_config(DDPGConfig, path)

        assert loaded.tensorboard.metrics.system is False
        assert loaded.tensorboard.metrics.gradients is False
        assert loaded.tensorboard.metrics.system_interval == 20
        assert loaded.tensorboard.metrics.train is True


class TestComputeGradNorm:
    """Test gradient norm computation."""

    def test_grad_norm(self):
        model = torch.nn.Linear(4, 2)
        x = torch.randn(1, 4)
        loss = model(x).sum()
        loss.backward()
        norm = compute_grad_norm(model)
        assert norm > 0.0

    def test_zero_grad(self):
        model = torch.nn.Linear(4, 2)
        model.zero_grad()
        norm = compute_grad_norm(model)
        assert norm == 0.0
