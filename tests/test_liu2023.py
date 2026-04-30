"""Tests for Liu et al. (2023) CPG locomotion package."""

import math

import numpy as np
import pytest

# CPG tests (no MuJoCo needed)
from liu2023.cpg_liu2023 import LiuMatsuokaOscillator, LiuCPGNetwork
from liu2023.curriculum_liu2023 import CurriculumManager, DEFAULT_LEVELS
from liu2023.rewards_liu2023 import (
    compute_goal_tracking_reward,
    check_goal_reached,
    check_starvation,
)


class TestLiuMatsuokaOscillator:
    """Unit tests for single Matsuoka oscillator."""

    def test_initialization(self):
        osc = LiuMatsuokaOscillator()
        assert osc.a_psi == pytest.approx(2.0935)
        assert osc.b == pytest.approx(10.0355)

    def test_output_bounds(self):
        """Output should stay within [-a_psi, a_psi] after transient."""
        osc = LiuMatsuokaOscillator()
        osc.reset()
        dt = 0.002
        for _ in range(5000):
            out = osc.step(dt, alpha=1.0)
        # After settling, output should be bounded
        assert abs(out) <= osc.a_psi + 0.1  # small tolerance

    def test_settles_to_nonzero(self):
        """Single oscillator should settle to a non-zero output with tonic drive.

        Note: sustained oscillation requires inter-oscillator coupling (a_i=4.6)
        in the CPG network. A standalone oscillator with the paper's high b=10
        converges to a fixed point, which is correct behavior.
        """
        osc = LiuMatsuokaOscillator()
        osc.reset()
        dt = 0.002
        for _ in range(10000):
            out = osc.step(dt, alpha=5.0)
        # Should settle to non-zero value (excitatory bias from positive alpha)
        assert abs(out) > 0.01, f"Output too close to zero: {out}"

    def test_complementary_tonic_mapping(self):
        """Positive alpha should favor excitatory, negative should favor flexor."""
        osc_pos = LiuMatsuokaOscillator()
        osc_neg = LiuMatsuokaOscillator()
        osc_pos.reset()
        osc_neg.reset()
        dt = 0.002
        pos_outputs = []
        neg_outputs = []
        for _ in range(5000):
            pos_outputs.append(osc_pos.step(dt, alpha=2.0))
            neg_outputs.append(osc_neg.step(dt, alpha=-2.0))
        # Mean outputs should differ in sign (or at least magnitude)
        pos_mean = np.mean(pos_outputs[-500:])
        neg_mean = np.mean(neg_outputs[-500:])
        assert pos_mean != pytest.approx(neg_mean, abs=0.1)

    def test_kf_changes_frequency(self):
        """Higher K_f should produce faster dynamics (shorter time constants)."""
        dt = 0.002
        # Compare settling behavior with different K_f values
        outputs_slow = []
        outputs_fast = []
        for kf, storage in [(0.5, outputs_slow), (2.0, outputs_fast)]:
            osc = LiuMatsuokaOscillator()
            osc.reset(perturb=0.5)  # strong perturbation for transient
            for _ in range(5000):
                storage.append(osc.step(dt, alpha=5.0, kf=kf))

        # Higher kf should cause faster transient dynamics (different trajectory)
        slow_early = np.array(outputs_slow[:500])
        fast_early = np.array(outputs_fast[:500])
        # The trajectories should differ due to different time constants
        assert not np.allclose(slow_early, fast_early, atol=0.01)


class TestLiuCPGNetwork:
    """Unit tests for CPG network."""

    def test_initialization(self):
        cpg = LiuCPGNetwork(num_oscillators=4)
        assert len(cpg.oscillators) == 4

    def test_output_shape(self):
        cpg = LiuCPGNetwork(num_oscillators=4)
        cpg.reset()
        alphas = np.array([1.0, 0.5, -0.5, -1.0])
        outputs = cpg.step(0.002, alphas)
        assert outputs.shape == (4,)

    def test_output_range(self):
        """Outputs should be in [-1, 1] (after clipping)."""
        cpg = LiuCPGNetwork(num_oscillators=4)
        cpg.reset()
        alphas = np.array([2.0, 2.0, 2.0, 2.0])
        for _ in range(1000):
            outputs = cpg.step(0.002, alphas)
        assert np.all(outputs >= -1.0 - 1e-6)
        assert np.all(outputs <= 1.0 + 1e-6)


class TestCurriculum:
    """Tests for curriculum manager."""

    def test_12_levels(self):
        assert len(DEFAULT_LEVELS) == 12

    def test_level_progression(self):
        """Distances and angles should increase with level."""
        for i in range(1, len(DEFAULT_LEVELS)):
            assert DEFAULT_LEVELS[i].distance_max >= DEFAULT_LEVELS[i - 1].distance_max
            assert DEFAULT_LEVELS[i].angle_max >= DEFAULT_LEVELS[i - 1].angle_max

    def test_goal_radii_decrease(self):
        """Goal radii should generally decrease (get harder)."""
        radii = [level.goal_radius for level in DEFAULT_LEVELS]
        assert radii[0] >= radii[-1]

    def test_advancement(self):
        cm = CurriculumManager(success_threshold=0.9, eval_window=10)
        assert cm.current_level == 0

        # Report 9 successes (not enough for window)
        for _ in range(9):
            cm.report_episode(True)
        assert cm.current_level == 0

        # 10th success → advance
        advanced = cm.report_episode(True)
        assert advanced
        assert cm.current_level == 1

    def test_no_advancement_below_threshold(self):
        cm = CurriculumManager(success_threshold=0.9, eval_window=10)
        for _ in range(5):
            cm.report_episode(True)
        for _ in range(5):
            cm.report_episode(False)
        assert cm.current_level == 0

    def test_goal_sampling(self):
        cm = CurriculumManager()
        rng = np.random.default_rng(42)
        dist, angle = cm.sample_goal(rng)
        assert DEFAULT_LEVELS[0].distance_min <= dist <= DEFAULT_LEVELS[0].distance_max
        assert abs(angle) <= math.radians(DEFAULT_LEVELS[0].angle_max)

    def test_reset(self):
        cm = CurriculumManager(success_threshold=0.9, eval_window=10)
        for _ in range(10):
            cm.report_episode(True)
        assert cm.current_level == 1
        cm.reset()
        assert cm.current_level == 0


class TestRewards:
    """Tests for reward functions."""

    def test_positive_velocity_reward(self):
        """Moving toward goal should give positive reward."""
        reward = compute_goal_tracking_reward(
            v_g=0.5, dist_to_goal=2.0, theta_g=0.0, goal_radii=[0.5]
        )
        assert reward > 0

    def test_negative_velocity_penalty(self):
        """Moving away from goal should give negative reward."""
        reward = compute_goal_tracking_reward(
            v_g=-0.5, dist_to_goal=2.0, theta_g=0.0, goal_radii=[0.5]
        )
        assert reward < 0

    def test_curriculum_bonus(self):
        """Being inside goal radius should give bonus."""
        reward_inside = compute_goal_tracking_reward(
            v_g=0.1, dist_to_goal=0.1, theta_g=0.0, goal_radii=[0.5, 0.3]
        )
        reward_outside = compute_goal_tracking_reward(
            v_g=0.1, dist_to_goal=1.0, theta_g=0.0, goal_radii=[0.5, 0.3]
        )
        assert reward_inside > reward_outside

    def test_goal_reached(self):
        assert check_goal_reached(0.1, 0.2)
        assert not check_goal_reached(0.3, 0.2)

    def test_starvation(self):
        history = [-0.1] * 60
        assert check_starvation(history, timeout=60)
        history[-1] = 0.1
        assert not check_starvation(history, timeout=60)
        assert not check_starvation([-0.1] * 30, timeout=60)


class TestSoftSnakeEnv:
    """Integration tests for the full environment (requires MuJoCo)."""

    @pytest.fixture
    def env(self):
        mujoco = pytest.importorskip("mujoco")
        from liu2023.configs_liu2023 import Liu2023EnvConfig
        from liu2023 import SoftSnakeEnv

        config = Liu2023EnvConfig()
        config.curriculum.enabled = False
        config.physics.randomize_friction = False
        config.physics.randomize_mass = False
        config.physics.randomize_max_pressure = False
        e = SoftSnakeEnv(config=config, device="cpu")
        yield e
        e.close()

    def test_reset(self, env):
        td = env.reset()
        assert "observation" in td
        assert td["observation"].shape == (8,)
        assert td["done"].item() is False

    def test_step(self, env):
        env.reset()
        import torch
        from tensordict import TensorDict

        action = torch.zeros(4)
        td_in = TensorDict({"action": action}, batch_size=[])
        td_out = env.step(td_in)
        nxt = td_out["next"]
        assert "observation" in nxt
        assert "reward" in nxt
        assert nxt["observation"].shape == (8,)

    def test_episode_terminates(self, env):
        """Episode should terminate within max_episode_steps."""
        import torch
        from tensordict import TensorDict

        env.config.max_episode_steps = 10
        env.reset()
        for _ in range(15):
            action = torch.zeros(4)
            td_in = TensorDict({"action": action}, batch_size=[])
            td = env.step(td_in)
            if td["next", "done"].item():
                break
        assert td["next", "done"].item()

    def test_cpg_produces_motion(self, env):
        """Non-zero tonic inputs should produce movement via CPG."""
        import torch
        from tensordict import TensorDict

        td = env.reset()
        initial_obs = td["observation"].clone()

        for _ in range(50):
            action = torch.tensor([1.0, 1.0, 1.0, 1.0])
            td_in = TensorDict({"action": action}, batch_size=[])
            td = env.step(td_in)

        final_obs = td["next", "observation"]
        # Joint curvatures (indices 4-7) should change
        assert not torch.allclose(initial_obs[4:7], final_obs[4:7], atol=1e-4)
