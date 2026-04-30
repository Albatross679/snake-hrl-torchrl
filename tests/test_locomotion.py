"""Tests for the planar snake locomotion environment.

NOTE: The locomotion package has been moved to 'Bing et al. (IJCAI 2019)/'
as a non-importable reference directory. These tests are skipped.
"""

import math
import os

# Ensure headless rendering works
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest
import torch

try:
    from bing2019.configs_bing2019 import LocomotionEnvConfig, LocomotionPhysicsConfig
    from bing2019 import PlanarSnakeEnv, TracksGenerator
    from bing2019.rewards_bing2019 import (
        compute_energy_normalized,
        compute_power_velocity_reward,
        compute_target_tracking_reward,
    )
    has_locomotion = True
except ImportError:
    has_locomotion = False

pytestmark = pytest.mark.skipif(not has_locomotion, reason="locomotion package not importable")


class TestXMLLoading:
    """Test that MuJoCo XML models load correctly."""

    def test_power_velocity_xml_loads(self):
        config = LocomotionEnvConfig(task="power_velocity")
        env = PlanarSnakeEnv(config=config)
        assert env.model.nq == 34
        assert env.model.nv == 34
        assert env.model.nu == 8
        env.close()

    def test_tracking_xml_loads(self):
        config = LocomotionEnvConfig(task="target_tracking")
        env = PlanarSnakeEnv(config=config)
        assert env.model.nq == 34
        assert env.model.nu == 8
        env.close()


class TestPlanarSnakeEnvPowerVelocity:
    """Tests for the power-velocity task."""

    @pytest.fixture
    def env(self):
        config = LocomotionEnvConfig(task="power_velocity")
        e = PlanarSnakeEnv(config=config)
        yield e
        e.close()

    def test_reset_shape(self, env):
        td = env.reset()
        assert td["observation"].shape == (27,)
        assert td["done"].shape == (1,)

    def test_step_shape(self, env):
        td = env.reset()
        action = torch.zeros(8, dtype=torch.float32)
        td["action"] = action
        td_out = env.step(td)
        # TorchRL wraps _step output in "next" key
        td_next = td_out["next"]
        assert td_next["observation"].shape == (27,)
        assert td_next["reward"].shape == (1,)
        assert td_next["done"].shape == (1,)

    def test_action_bounds(self, env):
        low = env.action_spec.space.low
        high = env.action_spec.space.high
        assert torch.all(low == -1.5)
        assert torch.all(high == 1.5)
        assert low.shape == (8,)

    def test_observation_content(self, env):
        """Verify observation structure: joint_pos(8) + joint_vel(8) + angle(1) + vel(1) + forces(8) + target_v(1)."""
        td = env.reset()
        obs = td["observation"].numpy()

        # Last element should be target_v
        assert obs[-1] == pytest.approx(0.1, abs=0.01)

        # Joint positions (first 8) should be near zero at reset
        joint_pos = obs[:8]
        assert np.allclose(joint_pos, 0.0, atol=0.1)

    def test_reward_finite(self, env):
        td = env.reset()
        action = torch.randn(8, dtype=torch.float32) * 0.5
        td["action"] = action
        td_out = env.step(td)
        reward = td_out["next", "reward"].item()
        assert np.isfinite(reward)

    def test_episode_truncation(self, env):
        td = env.reset()
        for i in range(env.config.max_episode_steps):
            action = torch.zeros(8, dtype=torch.float32)
            td["action"] = action
            td = env.step(td)
            td = td["next"]
        assert td["truncated"].item() is True

    def test_multiple_steps(self, env):
        """Run 10 steps and verify state changes."""
        td = env.reset()
        obs_0 = td["observation"].clone()

        for _ in range(10):
            action = torch.ones(8, dtype=torch.float32) * 0.5
            td["action"] = action
            td = env.step(td)
            td = td["next"]

        obs_10 = td["observation"]
        # Observation should change after applying non-zero actions
        assert not torch.allclose(obs_0, obs_10)


class TestPlanarSnakeEnvTargetTracking:
    """Tests for the target-tracking task."""

    @pytest.fixture
    def env(self):
        config = LocomotionEnvConfig(task="target_tracking")
        e = PlanarSnakeEnv(config=config)
        yield e
        e.close()

    def test_reset_shape(self, env):
        td = env.reset()
        assert td["observation"].shape == (49,)

    def test_step_shape(self, env):
        td = env.reset()
        action = torch.zeros(8, dtype=torch.float32)
        td["action"] = action
        td_out = env.step(td)
        td_next = td_out["next"]
        assert td_next["observation"].shape == (49,)
        assert td_next["reward"].shape == (1,)


class TestRewardFunctions:
    """Test reward computation functions."""

    def test_power_velocity_perfect_match(self):
        """Perfect velocity match + zero power -> high reward."""
        reward = compute_power_velocity_reward(
            target_v=0.1, velocity=0.1, power_normalized=0.0
        )
        assert reward > 0.9

    def test_power_velocity_zero_velocity(self):
        """Zero velocity when target_v=0.1 -> low reward."""
        reward = compute_power_velocity_reward(
            target_v=0.1, velocity=0.0, power_normalized=0.5
        )
        assert reward < 0.5

    def test_target_tracking_closer(self):
        """Moving closer to target distance -> positive reward."""
        reward = compute_target_tracking_reward(
            target_distance=4.0, dist_before=5.0, dist_after=4.5
        )
        assert reward > 0

    def test_target_tracking_farther(self):
        """Moving farther from target distance -> negative reward."""
        reward = compute_target_tracking_reward(
            target_distance=4.0, dist_before=4.0, dist_after=5.0
        )
        assert reward < 0

    def test_energy_normalized(self):
        forces = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        velocities = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        pn, _ = compute_energy_normalized(forces, velocities)
        assert 0.0 <= pn <= 1.0


class TestTracksGenerator:
    """Test all track trajectory generators."""

    @pytest.fixture
    def gen(self):
        return TracksGenerator(target_v=0.3, head_target_dist=4.0)

    def test_line(self, gen):
        x, y = gen.gen_line_step(0, 0, 4, 0, 0.05)
        assert y == 0.0
        assert x > 0

    def test_wave(self, gen):
        x, y = gen.gen_wave_step(0, 0, 4, 0, 0.05)
        assert x > 0

    def test_zigzag(self, gen):
        x, y = gen.gen_zigzag_step(0, 0, 4, 0, 0.05)
        assert x > 0

    def test_circle(self, gen):
        x, y = gen.gen_circle_step(0, 0, 4, 0, 0.05)
        assert x > 0

    def test_random(self, gen):
        x, y = gen.gen_random_step(0, 0, 4, 0, 0.05, seed=42)
        assert np.isfinite(x)
        assert np.isfinite(y)

    def test_dispatch(self, gen):
        """Test the unified step() dispatcher."""
        for track_type in ["line", "wave", "zigzag", "circle", "random"]:
            kwargs = {"seed": 42} if track_type == "random" else {}
            x, y = gen.step(track_type, 0, 0, 4, 0, 0.05, **kwargs)
            assert np.isfinite(x) and np.isfinite(y), f"Failed for {track_type}"

    def test_unknown_track_type(self, gen):
        with pytest.raises(ValueError, match="Unknown track type"):
            gen.step("unknown", 0, 0, 4, 0, 0.05)

    def test_distance_maintenance_line(self, gen):
        """Target stays within [min, max] distance bounds."""
        x, y = 4.0, 0.0
        for _ in range(100):
            x, y = gen.gen_line_step(0, 0, x, y, 0.05)
        # Should not go infinitely far
        assert x < 100
