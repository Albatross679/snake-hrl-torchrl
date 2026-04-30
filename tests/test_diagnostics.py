"""Tests for RL training diagnostics and probe environments."""

import torch
import pytest
from unittest.mock import MagicMock, patch

from src.trainers.diagnostics import (
    compute_explained_variance,
    compute_action_stats,
    compute_advantage_stats,
    compute_ratio_stats,
    compute_q_value_stats,
    compute_log_prob_stats,
    check_alerts,
)
from src.trainers.probe_envs import (
    ProbeEnv1,
    ProbeEnv2,
    ProbeEnv3,
    ProbeEnv4,
    ProbeEnv5,
    ALL_PROBES,
)


# ============================================================
# Diagnostic metric computation tests
# ============================================================


class TestExplainedVariance:
    def test_perfect_prediction(self):
        """Explained variance = 1.0 when predictions match targets exactly."""
        targets = torch.randn(100)
        ev = compute_explained_variance(targets, targets)
        assert abs(ev - 1.0) < 1e-5

    def test_mean_prediction(self):
        """Explained variance ≈ 0 when predicting constant mean."""
        targets = torch.randn(100)
        preds = torch.full_like(targets, targets.mean())
        ev = compute_explained_variance(preds, targets)
        assert abs(ev) < 0.05

    def test_anti_correlated(self):
        """Explained variance is negative when predictions are anti-correlated."""
        targets = torch.randn(100)
        preds = -targets  # Opposite sign
        ev = compute_explained_variance(preds, targets)
        assert ev < -0.5

    def test_constant_targets(self):
        """Returns 0.0 when target variance is near zero."""
        targets = torch.ones(100)
        preds = torch.randn(100)
        ev = compute_explained_variance(preds, targets)
        assert ev == 0.0


class TestActionStats:
    def test_basic_output_keys(self):
        actions = torch.randn(64, 5)
        stats = compute_action_stats(actions)
        assert "action_mean" in stats
        assert "action_std_mean" in stats
        assert "action_std_min" in stats
        # Per-dimension keys for 5 dims
        for i in range(5):
            assert f"action_dim{i}_std" in stats

    def test_collapsed_dimension(self):
        """Detects when one action dimension has near-zero variance."""
        actions = torch.randn(100, 3)
        actions[:, 1] = 0.5  # Collapse dimension 1
        stats = compute_action_stats(actions)
        assert stats["action_dim1_std"] < 0.01
        assert stats["action_std_min"] < 0.01

    def test_max_10_dims(self):
        """Only logs first 10 dimensions even with more."""
        actions = torch.randn(32, 15)
        stats = compute_action_stats(actions)
        assert "action_dim9_std" in stats
        assert "action_dim10_std" not in stats


class TestAdvantageStats:
    def test_normalized_advantages(self):
        """Normalized advantages should have mean ≈ 0."""
        adv = torch.randn(256)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        stats = compute_advantage_stats(adv)
        assert abs(stats["advantage_mean"]) < 0.1

    def test_large_advantages_detected(self):
        """Large advantage_abs_max indicates problem."""
        adv = torch.randn(100)
        adv[0] = 500.0
        stats = compute_advantage_stats(adv)
        assert stats["advantage_abs_max"] >= 500.0


class TestRatioStats:
    def test_identity_ratio(self):
        """Ratio = 1.0 when old and new log probs are equal."""
        log_prob = torch.randn(64)
        stats = compute_ratio_stats(log_prob, log_prob)
        assert abs(stats["ratio_mean"] - 1.0) < 1e-5

    def test_different_policies(self):
        """Ratio deviates from 1.0 when policies differ."""
        old = torch.zeros(64)
        new = torch.ones(64)
        stats = compute_ratio_stats(new, old)
        assert stats["ratio_mean"] > 1.0


class TestQValueStats:
    def test_basic_output(self):
        q1 = torch.randn(64, 1)
        q2 = torch.randn(64, 1)
        stats = compute_q_value_stats(q1, q2)
        assert "q_value_spread" in stats
        assert "q_max" in stats
        assert "q_min" in stats

    def test_with_target_value(self):
        q1 = torch.randn(64, 1)
        q2 = torch.randn(64, 1)
        target = torch.randn(64, 1)
        stats = compute_q_value_stats(q1, q2, target)
        assert "target_value_mean" in stats
        assert "target_value_std" in stats

    def test_diverged_q_values(self):
        """Detects when twin Q-networks have diverged."""
        q1 = torch.full((64, 1), 100.0)
        q2 = torch.full((64, 1), -100.0)
        stats = compute_q_value_stats(q1, q2)
        assert stats["q_value_spread"] > 100


class TestLogProbStats:
    def test_basic_output(self):
        log_prob = torch.randn(64)
        stats = compute_log_prob_stats(log_prob)
        assert "log_prob_mean" in stats
        assert "log_prob_std" in stats
        assert "entropy_proxy" in stats
        # Entropy proxy is negative of mean log prob
        assert abs(stats["entropy_proxy"] + stats["log_prob_mean"]) < 1e-6


# ============================================================
# Alert system tests
# ============================================================


class TestAlerts:
    def test_no_alerts_healthy_metrics(self):
        """No alerts fire for healthy training metrics."""
        with patch("wandb.alert") as mock_alert:
            check_alerts(
                MagicMock(),
                {"grad_norm": 1.0, "kl_divergence": 0.01, "clip_fraction": 0.1},
                step=1000,
                algorithm="ppo",
            )
            mock_alert.assert_not_called()

    def test_gradient_explosion_alert(self):
        """Alert fires when gradient norm exceeds threshold."""
        with patch("wandb.alert") as mock_alert:
            check_alerts(
                MagicMock(),
                {"grad_norm": 1e5},
                step=1000,
                algorithm="ppo",
            )
            mock_alert.assert_called_once()
            assert "explosion" in mock_alert.call_args[1]["title"].lower()

    def test_entropy_collapse_alert(self):
        """Alert fires when entropy drops near zero."""
        with patch("wandb.alert") as mock_alert:
            check_alerts(
                MagicMock(),
                {"entropy_proxy": 0.001, "grad_norm": 1.0},
                step=1000,
                algorithm="ppo",
            )
            assert any(
                "entropy" in call[1]["title"].lower()
                for call in mock_alert.call_args_list
            )

    def test_q_value_divergence_alert_sac(self):
        """Alert fires for SAC when Q-values are divergent."""
        with patch("wandb.alert") as mock_alert:
            check_alerts(
                MagicMock(),
                {"actor_grad_norm": 1.0, "q1_mean": 5000.0, "q2_mean": 1.0},
                step=1000,
                algorithm="sac",
            )
            assert any(
                "q-value" in call[1]["title"].lower()
                for call in mock_alert.call_args_list
            )

    def test_explained_variance_alert(self):
        """Alert fires when explained variance is strongly negative."""
        with patch("wandb.alert") as mock_alert:
            check_alerts(
                MagicMock(),
                {"grad_norm": 1.0, "explained_variance": -1.0},
                step=1000,
                algorithm="ppo",
            )
            assert any(
                "anti-correlated" in call[1]["title"].lower()
                for call in mock_alert.call_args_list
            )

    def test_nan_alert(self):
        """Alert fires when NaN appears in metrics."""
        with patch("wandb.alert") as mock_alert:
            check_alerts(
                MagicMock(),
                {"grad_norm": float("nan")},
                step=1000,
                algorithm="ppo",
            )
            titles = [call[1]["title"] for call in mock_alert.call_args_list]
            assert any("nan" in t.lower() for t in titles)

    def test_no_alert_when_wandb_disabled(self):
        """No crash when wandb_run is None."""
        # Should not raise
        check_alerts(None, {"grad_norm": 1e10}, step=1000, algorithm="ppo")

    def test_action_std_collapse_alert(self):
        """Alert fires when action_std_min drops below threshold."""
        with patch("wandb.alert") as mock_alert:
            check_alerts(
                MagicMock(),
                {"grad_norm": 1.0, "action_std_min": 0.001},
                step=1000,
                algorithm="ppo",
            )
            assert any(
                "collapsed" in call[1]["title"].lower()
                for call in mock_alert.call_args_list
            )

    def test_excessive_clipping_alert(self):
        """Alert fires when clip_fraction is too high."""
        with patch("wandb.alert") as mock_alert:
            check_alerts(
                MagicMock(),
                {"grad_norm": 1.0, "clip_fraction": 0.7},
                step=1000,
                algorithm="ppo",
            )
            assert any(
                "clipping" in call[1]["title"].lower()
                for call in mock_alert.call_args_list
            )


# ============================================================
# Probe environment tests
# ============================================================


class TestProbeEnvs:
    """Test that probe environments conform to TorchRL EnvBase contract."""

    @pytest.mark.parametrize("name,env_cls", ALL_PROBES)
    def test_reset_returns_valid_tensordict(self, name, env_cls):
        env = env_cls(device="cpu")
        td = env.reset()
        assert "observation" in td.keys()
        assert "done" in td.keys()
        assert td["done"].item() is False

    @pytest.mark.parametrize("name,env_cls", ALL_PROBES)
    def test_step_returns_valid_tensordict(self, name, env_cls):
        env = env_cls(device="cpu")
        td = env.reset()
        td["action"] = env.action_spec.rand()
        next_td = env.step(td)
        # Step output should have "next" key (TorchRL convention)
        if "next" in next_td.keys():
            result = next_td["next"]
        else:
            result = next_td
        assert "observation" in result.keys()
        assert "reward" in result.keys()
        assert "done" in result.keys()

    @pytest.mark.parametrize("name,env_cls", ALL_PROBES)
    def test_specs_match_output(self, name, env_cls):
        """Observation and action specs match actual tensor shapes."""
        env = env_cls(device="cpu")
        td = env.reset()
        obs = td["observation"]
        assert obs.shape == env.observation_spec["observation"].shape

    def test_probe1_constant_reward(self):
        """Probe 1 always gives +1 reward."""
        env = ProbeEnv1(device="cpu")
        td = env.reset()
        td["action"] = env.action_spec.rand()
        next_td = env.step(td)
        result = next_td["next"] if "next" in next_td.keys() else next_td
        assert result["reward"].item() == 1.0
        assert result["done"].item() is True

    def test_probe2_obs_dependent_reward(self):
        """Probe 2 reward matches observation."""
        env = ProbeEnv2(device="cpu")
        for _ in range(10):
            td = env.reset()
            obs = td["observation"].item()
            td["action"] = env.action_spec.rand()
            next_td = env.step(td)
            result = next_td["next"] if "next" in next_td.keys() else next_td
            assert result["reward"].item() == obs

    def test_probe3_delayed_reward(self):
        """Probe 3: step 0 gives 0 reward, step 1 gives +1."""
        env = ProbeEnv3(device="cpu")
        td = env.reset()

        # Step 0: reward = 0, not done
        td["action"] = env.action_spec.rand()
        next_td = env.step(td)
        result = next_td["next"] if "next" in next_td.keys() else next_td
        assert result["reward"].item() == 0.0
        assert result["done"].item() is False

        # Step 1: reward = 1, done
        td2 = result.clone()
        td2["action"] = env.action_spec.rand()
        next_td2 = env.step(td2)
        result2 = next_td2["next"] if "next" in next_td2.keys() else next_td2
        assert result2["reward"].item() == 1.0
        assert result2["done"].item() is True

    def test_probe4_action_dependent_reward(self):
        """Probe 4: positive action → +1, negative action → -1."""
        env = ProbeEnv4(device="cpu")

        # Positive action
        td = env.reset()
        td["action"] = torch.tensor([0.5])
        next_td = env.step(td)
        result = next_td["next"] if "next" in next_td.keys() else next_td
        assert result["reward"].item() == 1.0

        # Negative action
        td = env.reset()
        td["action"] = torch.tensor([-0.5])
        next_td = env.step(td)
        result = next_td["next"] if "next" in next_td.keys() else next_td
        assert result["reward"].item() == -1.0

    def test_probe5_joint_obs_action_reward(self):
        """Probe 5: reward = obs * sign(action)."""
        env = ProbeEnv5(device="cpu")
        torch.manual_seed(42)

        for _ in range(20):
            td = env.reset()
            obs = td["observation"].item()
            # Choose action that matches obs sign → reward should be +1
            td["action"] = torch.tensor([obs])
            next_td = env.step(td)
            result = next_td["next"] if "next" in next_td.keys() else next_td
            assert result["reward"].item() == 1.0
