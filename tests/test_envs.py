"""Tests for TorchRL environment implementations."""

import pytest
import torch
import numpy as np

from envs import BaseSnakeEnv, ApproachEnv, CoilEnv, HRLEnv
from configs.env import (
    EnvConfig,
    ApproachEnvConfig,
    CoilEnvConfig,
    HRLEnvConfig,
)


class TestBaseSnakeEnv:
    """Tests for BaseSnakeEnv."""

    @pytest.fixture
    def env(self):
        """Create base environment for testing."""
        config = EnvConfig(max_episode_steps=100)
        return BaseSnakeEnv(config=config)

    def test_env_initialization(self, env):
        """Test environment initializes correctly."""
        assert env.observation_spec is not None
        assert env.action_spec is not None
        assert env.reward_spec is not None

    def test_env_reset(self, env):
        """Test environment reset."""
        td = env.reset()

        assert "observation" in td.keys()
        assert "done" in td.keys()
        assert td["done"].item() == False

    def test_env_step(self, env):
        """Test environment step."""
        env.reset()

        # Create random action
        action = torch.rand(env.action_spec.shape) * 2 - 1
        td = env._make_tensordict()
        td["action"] = action

        result = env.step(td)

        assert "observation" in result.keys()
        assert "reward" in result.keys()
        assert "done" in result.keys()
        assert "terminated" in result.keys()
        assert "truncated" in result.keys()

    def test_env_truncation(self, env):
        """Test episode truncation at max steps."""
        env.reset()

        for _ in range(env.config.max_episode_steps):
            action = torch.rand(env.action_spec.shape) * 2 - 1
            td = env._make_tensordict()
            td["action"] = action
            result = env.step(td)

        # TorchRL puts next state in "next" key
        next_td = result.get("next", result)
        assert next_td["truncated"].item() == True
        assert next_td["done"].item() == True

    def test_env_observation_spec(self, env):
        """Test observation spec produces valid observations."""
        td = env.reset()
        obs = td["observation"]

        # Check observation is a valid tensor with expected properties
        assert obs.ndim == 1
        assert obs.dtype == torch.float32
        assert torch.all(torch.isfinite(obs))

    def test_env_action_spec(self, env):
        """Test action spec bounds."""
        action_spec = env.action_spec

        # Check bounds are scalars or all elements have same bounds
        assert float(action_spec.space.low.min()) == -1.0
        assert float(action_spec.space.high.max()) == 1.0


class TestApproachEnv:
    """Tests for ApproachEnv."""

    @pytest.fixture
    def env(self):
        """Create approach environment for testing."""
        config = ApproachEnvConfig(
            max_episode_steps=200,
            approach_distance_threshold=0.15,
        )
        return ApproachEnv(config=config)

    def test_approach_initialization(self, env):
        """Test approach environment initializes correctly."""
        assert env.approach_config is not None
        assert env.approach_config.approach_distance_threshold == 0.15

    def test_approach_reset(self, env):
        """Test approach environment reset."""
        td = env.reset()

        assert "observation" in td.keys()
        assert "prey_distance" in td.keys()

    def test_approach_reward(self, env):
        """Test approach reward computation.

        Clean architecture: Only energy penalty + success bonus + PBRS shaping.
        No direct distance/velocity rewards (handled by PBRS).
        """
        env.reset()

        # Take a step towards prey
        action = torch.zeros(env.action_spec.shape)
        td = env._make_tensordict()
        td["action"] = action

        result = env.step(td)

        # Reward should be a scalar
        assert result["reward"].shape == (1,)

    def test_approach_reward_structure(self, env):
        """Test that approach reward uses clean architecture."""
        env.reset()

        # With zero action, energy penalty should be zero
        action = torch.zeros(env.action_spec.shape)
        td = env._make_tensordict()
        td["action"] = action

        result = env.step(td)

        # Reward should be finite (PBRS shaping may contribute)
        reward = result["reward"].item()
        assert np.isfinite(reward)

    def test_approach_success_detection(self, env):
        """Test success detection when close to prey."""
        env.reset()

        # Manually set close to prey
        env.sim.snake.positions[:] = env.sim.prey.position - np.array([0.1, 0, 0])
        env._current_state = env.sim.get_state()

        # Check if success is detected
        assert env._current_state["prey_distance"] < env.approach_config.approach_distance_threshold

    def test_approach_metrics(self, env):
        """Test approach metrics."""
        env.reset()

        metrics = env.get_metrics()

        assert "prey_distance" in metrics
        assert "is_success" in metrics
        assert "episode_reward" in metrics


class TestCoilEnv:
    """Tests for CoilEnv."""

    @pytest.fixture
    def env(self):
        """Create coil environment for testing."""
        config = CoilEnvConfig(
            max_episode_steps=200,
            min_coil_wraps=1.0,
            contact_fraction_threshold=0.5,
        )
        return CoilEnv(config=config)

    def test_coil_initialization(self, env):
        """Test coil environment initializes correctly."""
        assert env.coil_config is not None
        assert env.coil_config.min_coil_wraps == 1.0
        assert env.coil_config.contact_fraction_threshold == 0.5

    def test_coil_reset(self, env):
        """Test coil environment reset."""
        td = env.reset()

        assert "observation" in td.keys()

    def test_coil_reward(self, env):
        """Test coil reward computation.

        Clean architecture: Only stability + energy penalty + success bonus + PBRS shaping.
        No direct contact/wrap/constriction rewards (handled by PBRS).
        """
        env.reset()

        action = torch.zeros(env.action_spec.shape)
        td = env._make_tensordict()
        td["action"] = action

        result = env.step(td)

        assert result["reward"].shape == (1,)

    def test_coil_reward_structure(self, env):
        """Test that coil reward uses clean architecture."""
        env.reset()

        # With zero action, energy penalty should be zero
        action = torch.zeros(env.action_spec.shape)
        td = env._make_tensordict()
        td["action"] = action

        result = env.step(td)

        # Reward should be finite (PBRS shaping may contribute)
        reward = result["reward"].item()
        assert np.isfinite(reward)

    def test_coil_metrics(self, env):
        """Test coil metrics."""
        env.reset()

        metrics = env.get_metrics()

        assert "contact_fraction" in metrics
        assert "wrap_count" in metrics
        assert "is_success" in metrics


class TestHRLEnv:
    """Tests for HRLEnv."""

    @pytest.fixture
    def env(self):
        """Create HRL environment for testing."""
        config = HRLEnvConfig(max_episode_steps=500)
        return HRLEnv(config=config)

    def test_hrl_initialization(self, env):
        """Test HRL environment initializes correctly."""
        assert env.hrl_config is not None
        assert env.hrl_config.num_skills == 2

    def test_hrl_reset(self, env):
        """Test HRL environment reset."""
        td = env.reset()

        assert "observation" in td.keys()
        assert "current_skill" in td.keys()
        assert "task_progress" in td.keys()

    def test_hrl_step(self, env):
        """Test HRL environment step."""
        env.reset()

        action = torch.zeros(env.action_spec.shape)
        td = env._make_tensordict()
        td["action"] = action

        result = env.step(td)

        # TorchRL puts next state in "next" key
        next_td = result.get("next", result)
        assert "current_skill" in next_td.keys()
        assert "task_progress" in next_td.keys()

    def test_hrl_skill_setting(self, env):
        """Test skill setting."""
        env.reset()

        env.set_skill(0)
        assert env.current_skill == 0

        env.set_skill(1)
        assert env.current_skill == 1

    def test_hrl_metrics(self, env):
        """Test HRL metrics."""
        env.reset()

        metrics = env.get_metrics()

        assert "approach_complete" in metrics
        assert "coil_complete" in metrics
        assert "task_success" in metrics
        assert "approach_progress" in metrics
        assert "coil_progress" in metrics

    def test_hrl_reward_structure(self, env):
        """Test that HRL reward uses clean architecture.

        Clean architecture: Phase-specific sparse bonuses + stability + energy penalty
        + skill switch penalty + PBRS shaping. No direct distance/contact improvements.
        """
        env.reset()

        # With zero action, energy penalty should be minimal
        action = torch.zeros(env.action_spec.shape)
        td = env._make_tensordict()
        td["action"] = action

        result = env.step(td)

        # Reward should be finite (PBRS shaping may contribute)
        reward = result["reward"].item()
        assert np.isfinite(reward)

    def test_hrl_has_reward_shaper(self, env):
        """Test that HRL env initializes reward shaper when enabled."""
        # Default config has use_reward_shaping=True
        assert env.reward_shaper is not None
        assert hasattr(env.reward_shaper, "compute_shaping_reward")


class TestEnvSeeding:
    """Tests for environment reproducibility."""

    def test_deterministic_reset(self):
        """Test reset is deterministic with same seed when randomization disabled."""
        # Disable randomization for deterministic test
        config = EnvConfig(randomize_initial_state=False, randomize_prey_position=False)

        env1 = BaseSnakeEnv(config=config)
        env2 = BaseSnakeEnv(config=config)

        td1 = env1.reset()
        td2 = env2.reset()

        torch.testing.assert_close(td1["observation"], td2["observation"])

    def test_different_seeds_different_states(self):
        """Test different seeds produce different states."""
        config = EnvConfig(randomize_initial_state=True)

        env1 = BaseSnakeEnv(config=config)
        env2 = BaseSnakeEnv(config=config)

        env1._set_seed(42)
        env2._set_seed(123)

        td1 = env1.reset()
        td2 = env2.reset()

        # Observations should differ
        assert not torch.allclose(td1["observation"], td2["observation"])


class TestEnvBatching:
    """Tests for batched environment operations."""

    def test_unbatched_env(self):
        """Test unbatched environment has empty batch size."""
        env = BaseSnakeEnv()
        assert env.batch_size == torch.Size([])

    def test_tensordict_batch_size(self):
        """Test TensorDict has correct batch size."""
        env = BaseSnakeEnv()
        td = env.reset()

        assert td.batch_size == env.batch_size


class TestEnvDevices:
    """Tests for device handling."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_env(self):
        """Test environment on CUDA device."""
        env = BaseSnakeEnv(device="cuda")
        td = env.reset()

        assert td["observation"].device.type == "cuda"

    def test_cpu_env(self):
        """Test environment on CPU device."""
        env = BaseSnakeEnv(device="cpu")
        td = env.reset()

        assert td["observation"].device.type == "cpu"
