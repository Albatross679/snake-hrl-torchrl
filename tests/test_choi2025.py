"""Tests for choi2025 package (Choi & Tong, 2025)."""

import numpy as np
import pytest
import torch

from choi2025.config import (
    Choi2025Config,
    Choi2025EnvConfig,
    Choi2025PhysicsConfig,
    DeltaCurvatureControlConfig,
    TaskType,
)
from choi2025.control import DeltaCurvatureController
from choi2025.env import SoftManipulatorEnv
from choi2025.rewards import (
    compute_follow_target_reward,
    compute_ik_reward,
    compute_obstacle_reward,
)
from choi2025.tasks import ObstacleManager, TargetGenerator


class TestConfigs:
    def test_default_config(self):
        config = Choi2025Config()
        assert config.name == "choi2025"
        assert config.actor_lr == 0.001
        assert config.batch_size == 2048
        assert config.buffer_size == 2_000_000
        assert config.num_updates == 4

    def test_physics_config(self):
        pc = Choi2025PhysicsConfig()
        assert pc.clamp_first_node is True
        assert pc.two_d_sim is False
        assert pc.dt == 0.01
        assert pc.youngs_modulus == 2e6
        assert pc.geometry.num_segments == 20
        assert pc.geometry.num_nodes == 21

    def test_task_types(self):
        assert len(TaskType) == 4
        assert TaskType("follow_target") == TaskType.FOLLOW_TARGET


class TestDeltaCurvatureController:
    def test_voronoi_weights_shape(self):
        ctrl = DeltaCurvatureController(num_bend_springs=19)
        assert ctrl.W.shape == (19, 5)
        # Each row should sum to ~1 (partition of unity)
        row_sums = ctrl.W.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_apply_delta_2d(self):
        ctrl = DeltaCurvatureController(num_bend_springs=19)
        action = np.ones(5) * 0.5
        state = ctrl.apply_delta(action, two_d_sim=True)
        assert state.shape == (19, 2)
        assert np.all(state[:, 0] != 0)  # kappa1 changed
        assert np.all(state[:, 1] == 0)  # kappa2 unchanged

    def test_apply_delta_3d(self):
        ctrl = DeltaCurvatureController(num_bend_springs=19)
        action = np.ones(10) * 0.5
        state = ctrl.apply_delta(action, two_d_sim=False)
        assert state.shape == (19, 2)
        assert np.all(state[:, 0] != 0)
        assert np.all(state[:, 1] != 0)

    def test_reset(self):
        ctrl = DeltaCurvatureController(num_bend_springs=19)
        ctrl.apply_delta(np.ones(10) * 0.5, two_d_sim=False)
        ctrl.reset()
        np.testing.assert_array_equal(ctrl.curvature_state, 0.0)

    def test_accumulation(self):
        ctrl = DeltaCurvatureController(num_bend_springs=19)
        action = np.ones(5) * 0.5
        state1 = ctrl.apply_delta(action, two_d_sim=True)
        state2 = ctrl.apply_delta(action, two_d_sim=True)
        # Second application should roughly double the curvature
        np.testing.assert_allclose(state2[:, 0], 2 * state1[:, 0], atol=1e-10)


class TestRewards:
    def test_follow_target_reward(self):
        tip = np.array([0.5, 0.0, 0.0])
        target = np.array([0.8, 0.0, 0.0])
        prev_tip = np.array([0.4, 0.0, 0.0])
        reward = compute_follow_target_reward(tip, target, prev_tip)
        assert isinstance(reward, float)
        assert reward > 0  # Moving closer

    def test_ik_reward(self):
        tip = np.array([0.5, 0.0, 0.0])
        tangent = np.array([1.0, 0.0, 0.0])
        target = np.array([0.5, 0.0, 0.0])
        orient = np.array([1.0, 0.0, 0.0])
        reward = compute_ik_reward(tip, tangent, target, orient)
        assert reward > 0.9  # Perfect match

    def test_obstacle_reward_with_penetration(self):
        tip = np.array([0.5, 0.0, 0.0])
        target = np.array([0.8, 0.0, 0.0])
        prev_tip = np.array([0.4, 0.0, 0.0])
        reward_no_pen = compute_obstacle_reward(tip, target, prev_tip, 0.0)
        reward_pen = compute_obstacle_reward(tip, target, prev_tip, 0.1)
        assert reward_no_pen > reward_pen  # Penalty reduces reward


class TestTargetAndObstacles:
    def test_target_generator(self):
        rng = np.random.default_rng(42)
        tg = TargetGenerator(config=Choi2025EnvConfig().target, rng=rng)
        tg.sample(TaskType.FOLLOW_TARGET)
        r = np.linalg.norm(tg.position)
        assert 0.3 <= r <= 0.9

    def test_obstacle_manager_tight(self):
        rng = np.random.default_rng(42)
        om = ObstacleManager(config=Choi2025EnvConfig().obstacles, rng=rng)
        om.setup(TaskType.TIGHT_OBSTACLES)
        assert len(om.positions) == 2
        assert len(om.radii) == 2

    def test_obstacle_manager_random(self):
        rng = np.random.default_rng(42)
        om = ObstacleManager(config=Choi2025EnvConfig().obstacles, rng=rng)
        om.setup(TaskType.RANDOM_OBSTACLES)
        assert len(om.positions) == 3

    def test_penetrations(self):
        rng = np.random.default_rng(42)
        om = ObstacleManager(config=Choi2025EnvConfig().obstacles, rng=rng)
        om.positions = np.array([[0.5, 0.0, 0.0]])
        om.radii = np.array([0.1])

        # Point inside obstacle
        rod_positions = np.array([[0.5, 0.0, 0.0]])
        pen = om.compute_penetrations(rod_positions)
        assert pen[0] == pytest.approx(0.1)

        # Point far away
        rod_positions = np.array([[10.0, 0.0, 0.0]])
        pen = om.compute_penetrations(rod_positions)
        assert pen[0] == 0.0


class TestSoftManipulatorEnv:
    @pytest.fixture
    def env(self):
        config = Choi2025EnvConfig(task=TaskType.FOLLOW_TARGET)
        e = SoftManipulatorEnv(config)
        yield e
        e.close()

    def test_reset(self, env):
        td = env.reset()
        assert "observation" in td.keys()
        assert td["observation"].shape == (env._obs_dim,)

    def test_step(self, env):
        td = env.reset()
        action = torch.zeros(env.action_spec.shape)
        td["action"] = action
        next_td = env.step(td)
        assert "next" in next_td.keys()
        assert "reward" in next_td["next"].keys()
        assert "done" in next_td["next"].keys()

    def test_obs_dims_per_task(self):
        for task in TaskType:
            config = Choi2025EnvConfig(task=task)
            env = SoftManipulatorEnv(config)
            td = env.reset()
            assert td["observation"].shape == (env._obs_dim,), f"Failed for {task}"
            env.close()

    def test_episode_truncation(self):
        config = Choi2025EnvConfig(task=TaskType.FOLLOW_TARGET, max_episode_steps=5)
        env = SoftManipulatorEnv(config)
        td = env.reset()

        for _ in range(5):
            td["action"] = torch.zeros(env.action_spec.shape)
            td = env.step(td)

        assert td["next"]["done"].item() is True
        assert td["next"]["truncated"].item() is True
        env.close()
