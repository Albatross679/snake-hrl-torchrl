"""Smoke tests for the underwater snake environment."""

import sys
from pathlib import Path

import pytest
import numpy as np
import torch

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from configs import ZhengConfig
from env import UnderwaterSnakeEnv, build_mjcf_xml
from reward import CurriculumReward

mujoco = pytest.importorskip("mujoco")


class TestMJCFModel:
    """Test MJCF XML generation and loading."""

    def test_xml_generation(self):
        config = ZhengConfig()
        xml = build_mjcf_xml(config)
        assert isinstance(xml, str)
        assert "underwater_snake_zheng2022" in xml
        assert f'density="{config.fluid_density}"' in xml
        assert f'viscosity="{config.fluid_viscosity}"' in xml

    def test_model_loads(self):
        config = ZhengConfig()
        xml = build_mjcf_xml(config)
        model = mujoco.MjModel.from_xml_string(xml)
        assert model is not None

    def test_model_structure(self):
        config = ZhengConfig()
        xml = build_mjcf_xml(config)
        model = mujoco.MjModel.from_xml_string(xml)

        # 7 bodies (link_0 through link_6) + worldbody
        assert model.nbody == config.num_links + 1

        # 6 hinge joints + 1 freejoint (7 total in joint array)
        # Freejoint expands to 7 qpos (3 pos + 4 quat) and 6 qvel
        assert model.njnt == config.num_joints + 1  # 6 hinge + 1 free

        # 6 motor actuators
        assert model.nu == config.num_joints

    def test_custom_stiffness(self):
        config = ZhengConfig(joint_stiffness=2.5)
        xml = build_mjcf_xml(config)
        assert 'stiffness="2.5"' in xml


class TestEnvironment:
    """Test the TorchRL environment."""

    @pytest.fixture
    def env(self):
        config = ZhengConfig()
        return UnderwaterSnakeEnv(config=config)

    def test_reset(self, env):
        td = env.reset()
        assert "observation" in td.keys()
        assert td["observation"].shape == (16,)
        assert td["done"].item() is False

    def test_observation_shape(self, env):
        td = env.reset()
        obs = td["observation"]
        assert obs.shape == (16,)
        assert obs.dtype == torch.float32
        assert not torch.any(torch.isnan(obs))

    def test_step_with_zero_action(self, env):
        td = env.reset()
        td["action"] = torch.zeros(6)
        td = env.step(td)
        assert "observation" in td.keys()
        assert "reward" in td.keys()
        assert "head_velocity_x" in td.keys()
        assert "power" in td.keys()

    def test_step_with_random_action(self, env):
        td = env.reset()
        td["action"] = torch.rand(6) * 2 - 1  # [-1, 1]
        td = env.step(td)
        obs = td["observation"]
        assert obs.shape == (16,)
        assert not torch.any(torch.isnan(obs))

    def test_action_spec(self, env):
        assert env.action_spec.shape == (6,)
        assert (env.action_spec.space.low == -1.0).all()
        assert (env.action_spec.space.high == 1.0).all()

    def test_stability_100_steps(self, env):
        """Run 100 steps with random actions and check for NaN/Inf."""
        td = env.reset()
        for _ in range(100):
            td["action"] = torch.rand(6) * 2 - 1
            td = env.step(td)
            obs = td["observation"]
            assert not torch.any(torch.isnan(obs)), "NaN in observation"
            assert not torch.any(torch.isinf(obs)), "Inf in observation"
            if td["done"].item():
                td = env.reset()

    def test_power_nonnegative(self, env):
        """Power should always be non-negative."""
        td = env.reset()
        for _ in range(50):
            td["action"] = torch.rand(6) * 2 - 1
            td = env.step(td)
            assert td["power"].item() >= 0.0

    def test_neutral_buoyancy(self, env):
        """Snake should not sink or float significantly with zero action."""
        td = env.reset()
        initial_z = env.data.qpos[2]

        for _ in range(200):
            td["action"] = torch.zeros(6)
            td = env.step(td)

        final_z = env.data.qpos[2]
        # Allow small drift but no large sinking/floating
        assert abs(final_z - initial_z) < 0.05, (
            f"Z drift = {final_z - initial_z:.4f} (initial={initial_z:.4f}, final={final_z:.4f})"
        )

    def test_truncation_at_max_steps(self, env):
        """Episode should truncate at max_episode_steps."""
        env.config.max_episode_steps = 10
        td = env.reset()
        for i in range(10):
            td["action"] = torch.zeros(6)
            td = env.step(td)
        # TorchRL puts step results under "next" key
        assert td["next", "truncated"].item() is True
        assert td["next", "done"].item() is True


class TestCurriculumReward:
    """Test the curriculum reward function."""

    @pytest.fixture
    def curriculum(self):
        return CurriculumReward(ZhengConfig())

    def test_phase1_initial(self, curriculum):
        assert curriculum.phase == 1
        reward = curriculum.compute_reward(head_velocity_x=0.1, power=0.01)
        assert isinstance(reward, float)

    def test_phase_transition(self, curriculum):
        curriculum.set_epoch(0)
        assert curriculum.phase == 1
        curriculum.set_epoch(1999)
        assert curriculum.phase == 1
        curriculum.set_epoch(2000)
        assert curriculum.phase == 2

    def test_phase1_rewards_velocity(self, curriculum):
        """Higher velocity should give higher reward in Phase 1."""
        curriculum.set_epoch(0)
        r_slow = curriculum.compute_reward(head_velocity_x=0.01, power=0.01)
        r_fast = curriculum.compute_reward(head_velocity_x=0.1, power=0.01)
        assert r_fast > r_slow

    def test_target_velocity_decreases(self, curriculum):
        curriculum.set_epoch(2000)
        v1 = curriculum.target_velocity
        curriculum.set_epoch(3000)
        v2 = curriculum.target_velocity
        curriculum.set_epoch(4000)
        v3 = curriculum.target_velocity
        assert v2 < v1
        assert v3 < v2

    def test_target_velocity_minimum(self, curriculum):
        """Target velocity should not go below minimum."""
        curriculum.set_epoch(100000)
        assert curriculum.target_velocity >= curriculum.min_target_velocity
