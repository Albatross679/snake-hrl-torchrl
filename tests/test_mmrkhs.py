"""Unit tests for MM-RKHS (Gupta & Mahajan) trainer.

Tests MMRKHSConfig, MMRKHSTrainer, and MMD computation using a
SimplePendulum TorchRL-native environment (no gymnasium dependency).
"""

import math
import pytest
import torch
from pathlib import Path

from tensordict import TensorDict
from torchrl.envs import EnvBase, check_env_specs

try:
    from torchrl.data import BoundedTensorSpec, UnboundedContinuousTensorSpec, CompositeSpec
except ImportError:
    from torchrl.data import Bounded as BoundedTensorSpec
    from torchrl.data import Unbounded as UnboundedContinuousTensorSpec
    from torchrl.data import Composite as CompositeSpec

from src.configs.training import MMRKHSConfig, RLConfig
from src.configs.network import NetworkConfig, ActorConfig, CriticConfig
from src.configs.base import WandB
from src.trainers.mmrkhs import MMRKHSTrainer


# ---------------------------------------------------------------------------
# SimplePendulum: TorchRL-native environment (no gymnasium)
# ---------------------------------------------------------------------------


class SimplePendulum(EnvBase):
    """TorchRL-native pendulum environment for testing.

    Observation: [cos(theta), sin(theta), angular_velocity] -- 3-dim
    Action: torque in [-2, 2] -- 1-dim
    Reward: -(theta^2 + 0.1 * vel^2 + 0.001 * torque^2)
    Episode: 200 steps (truncation only, no termination)
    """

    def __init__(self, device="cpu"):
        super().__init__(device=torch.device(device), batch_size=torch.Size([]))

        self.observation_spec = CompositeSpec(
            observation=UnboundedContinuousTensorSpec(
                shape=(3,), dtype=torch.float32, device=self.device,
            ),
            shape=(),
        )
        self.action_spec = BoundedTensorSpec(
            low=-2.0, high=2.0, shape=(1,),
            dtype=torch.float32, device=self.device,
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,), dtype=torch.float32, device=self.device,
        )
        self.done_spec = CompositeSpec(
            done=UnboundedContinuousTensorSpec(
                shape=(1,), dtype=torch.bool, device=self.device,
            ),
            terminated=UnboundedContinuousTensorSpec(
                shape=(1,), dtype=torch.bool, device=self.device,
            ),
            truncated=UnboundedContinuousTensorSpec(
                shape=(1,), dtype=torch.bool, device=self.device,
            ),
            shape=(),
        )

        # Physics parameters
        self._max_speed = 8.0
        self._dt = 0.05
        self._g = 10.0
        self._m = 1.0
        self._l = 1.0
        self._max_steps = 200
        self._step_count = 0
        self._th = 0.0
        self._thdot = 0.0

    def _set_seed(self, seed):
        self._rng = torch.manual_seed(seed)

    def _reset(self, tensordict=None, **kwargs):
        self._th = torch.empty(()).uniform_(-math.pi, math.pi).item()
        self._thdot = torch.empty(()).uniform_(-1.0, 1.0).item()
        self._step_count = 0
        obs = torch.tensor(
            [math.cos(self._th), math.sin(self._th), self._thdot],
            dtype=torch.float32, device=self.device,
        )
        return TensorDict(
            {
                "observation": obs,
                "done": torch.tensor([False], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([False], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([False], dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict):
        u = tensordict["action"].squeeze(-1).clamp(-2.0, 2.0).item()
        th_norm = ((self._th + math.pi) % (2 * math.pi)) - math.pi
        cost = th_norm ** 2 + 0.1 * self._thdot ** 2 + 0.001 * u ** 2

        new_thdot = self._thdot + (
            3 * self._g / (2 * self._l) * math.sin(self._th)
            + 3.0 / (self._m * self._l ** 2) * u
        ) * self._dt
        new_thdot = max(-self._max_speed, min(self._max_speed, new_thdot))
        self._th = self._th + new_thdot * self._dt
        self._thdot = new_thdot
        self._step_count += 1

        truncated = self._step_count >= self._max_steps
        obs = torch.tensor(
            [math.cos(self._th), math.sin(self._th), self._thdot],
            dtype=torch.float32, device=self.device,
        )
        return TensorDict(
            {
                "observation": obs,
                "reward": torch.tensor([-cost], dtype=torch.float32, device=self.device),
                "done": torch.tensor([truncated], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([False], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_test_env():
    """Create a SimplePendulum environment for testing."""
    return SimplePendulum(device="cpu")


def _make_small_config():
    """Create a minimal MMRKHSConfig for fast tests."""
    return MMRKHSConfig(
        total_frames=512,
        frames_per_batch=128,
        num_epochs=2,
        mini_batch_size=64,
        use_amp=False,
        patience_batches=0,
        lr_schedule="constant",
        wandb=WandB(enabled=False),
    )


def _make_small_network():
    """Create a small NetworkConfig for fast tests."""
    return NetworkConfig(
        actor=ActorConfig(hidden_dims=[32, 32]),
        critic=CriticConfig(hidden_dims=[32, 32]),
    )


# ---------------------------------------------------------------------------
# Tests: SimplePendulum environment
# ---------------------------------------------------------------------------


class TestSimplePendulum:
    """Verify SimplePendulum environment passes TorchRL spec checks."""

    def test_simple_pendulum_specs(self):
        """SimplePendulum passes check_env_specs (TorchRL validation)."""
        env = SimplePendulum()
        check_env_specs(env)

    def test_reset_shape(self):
        """Reset returns correctly shaped observation."""
        env = SimplePendulum()
        td = env.reset()
        assert td["observation"].shape == (3,)
        assert td["done"].shape == (1,)

    def test_step_shape(self):
        """Step returns correctly shaped tensors."""
        env = SimplePendulum()
        td = env.reset()
        td["action"] = torch.tensor([0.5])
        td_next = env.step(td)
        # env.step() wraps next-state data under "next" key
        assert td_next["next", "observation"].shape == (3,)
        assert td_next["next", "reward"].shape == (1,)
        assert td_next["next", "done"].shape == (1,)


# ---------------------------------------------------------------------------
# Tests: MMRKHSConfig
# ---------------------------------------------------------------------------


class TestMMRKHSConfig:
    """Test MMRKHSConfig dataclass defaults and inheritance."""

    def test_config(self):
        """MMRKHSConfig has expected default values."""
        c = MMRKHSConfig()
        assert c.beta == 1.0
        assert c.eta == 1.0
        assert c.mmd_kernel == "rbf"
        assert c.mmd_bandwidth == 1.0
        assert c.mmd_num_samples == 16
        assert c.gae_lambda == 0.95
        assert c.normalize_advantage is True
        assert c.value_coef == 0.5
        assert c.lr_schedule == "linear"
        assert c.lr_end == 1e-5
        assert c.patience_batches == 200

    def test_config_inherits_rl(self):
        """MMRKHSConfig inherits RLConfig fields."""
        c = MMRKHSConfig()
        assert isinstance(c, RLConfig)
        assert c.gamma == 0.99
        assert c.learning_rate == 3e-4
        assert c.total_frames == 1_000_000
        assert c.max_grad_norm == 0.5
        assert c.frames_per_batch == 4096

    def test_no_clip_or_entropy(self):
        """MMRKHSConfig does NOT have clip_epsilon or entropy_coef."""
        c = MMRKHSConfig()
        assert not hasattr(c, "clip_epsilon")
        assert not hasattr(c, "entropy_coef")


# ---------------------------------------------------------------------------
# Tests: MMRKHSTrainer
# ---------------------------------------------------------------------------


class TestMMRKHSTrainer:
    """Test MMRKHSTrainer initialization, MMD, update, training, and checkpoints."""

    @pytest.fixture
    def trainer(self, tmp_path):
        """Create an MMRKHSTrainer with SimplePendulum and small config."""
        env = _make_test_env()
        config = _make_small_config()
        network_config = _make_small_network()
        t = MMRKHSTrainer(
            env=env,
            config=config,
            network_config=network_config,
            device="cpu",
            run_dir=tmp_path,
        )
        yield t
        # Cleanup: restore signal handlers
        t._restore_signal_handlers()

    def test_trainer_init(self, trainer):
        """Trainer creates actor, critic, optimizer, advantage_module, collector."""
        assert hasattr(trainer, "actor")
        assert hasattr(trainer, "critic")
        assert hasattr(trainer, "optimizer")
        assert hasattr(trainer, "advantage_module")
        assert hasattr(trainer, "collector")
        assert hasattr(trainer, "action_spec")
        assert trainer.total_frames == 0
        assert trainer.best_reward == float("-inf")

    def test_mmd_penalty(self, trainer):
        """_compute_mmd_penalty returns finite non-negative scalar."""
        batch_size = 32
        action_dim = 1
        obs_dim = 3

        # Create synthetic distribution params
        old_loc = torch.randn(batch_size, action_dim)
        old_scale = torch.ones(batch_size, action_dim) * 0.5
        new_loc = torch.randn(batch_size, action_dim)
        new_scale = torch.ones(batch_size, action_dim) * 0.5
        obs = torch.randn(batch_size, obs_dim)

        mmd = trainer._compute_mmd_penalty(
            obs=obs,
            old_loc=old_loc,
            old_scale=old_scale,
            new_loc=new_loc,
            new_scale=new_scale,
        )

        assert mmd.ndim == 0, "MMD should be a scalar"
        assert torch.isfinite(mmd), f"MMD should be finite, got {mmd}"
        assert mmd.item() >= 0.0, f"MMD should be non-negative, got {mmd.item()}"

    def test_mmd_penalty_identical_dists(self, trainer):
        """MMD between identical distributions should be near zero."""
        batch_size = 64
        action_dim = 1

        loc = torch.zeros(batch_size, action_dim)
        scale = torch.ones(batch_size, action_dim) * 0.3
        obs = torch.randn(batch_size, 3)

        mmd = trainer._compute_mmd_penalty(
            obs=obs,
            old_loc=loc,
            old_scale=scale,
            new_loc=loc.clone(),
            new_scale=scale.clone(),
        )

        assert torch.isfinite(mmd)
        assert mmd.item() >= 0.0
        # With identical params, MMD should be small (stochastic, so use generous threshold)
        assert mmd.item() < 1.0, f"MMD for identical dists should be small, got {mmd.item()}"

    def test_update_step(self, trainer):
        """_update(batch) returns dict with all expected metric keys, all finite."""
        # Collect one batch via the collector
        batch = None
        for b in trainer.collector:
            batch = b.to("cpu")
            break

        assert batch is not None, "Collector should produce at least one batch"

        # Flatten if needed
        if batch.ndim > 1:
            batch = batch.reshape(-1)

        # Compute advantages
        with torch.no_grad():
            trainer.advantage_module(batch)

        # Run update
        metrics = trainer._update(batch)

        # Check all expected keys
        expected_keys = [
            "loss_policy", "loss_critic", "mmd_penalty",
            "kl_divergence", "grad_norm", "policy_entropy",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric key: {key}"
            assert math.isfinite(metrics[key]), f"Metric {key} is not finite: {metrics[key]}"

    def test_short_training(self, trainer):
        """512-frame training on SimplePendulum completes without crash."""
        results = trainer.train()

        assert "total_frames" in results
        assert "best_reward" in results
        assert "stop_reason" in results
        assert results["total_frames"] >= 256, (
            f"Expected >= 256 frames, got {results['total_frames']}"
        )

    def test_checkpoint(self, trainer, tmp_path):
        """save_checkpoint/load_checkpoint roundtrip restores model state."""
        # Modify training state
        trainer.total_frames = 42
        trainer.total_episodes = 7
        trainer.best_reward = -100.0

        # Save
        trainer.save_checkpoint("test")
        ckpt_path = trainer.save_dir / "test.pt"
        assert ckpt_path.exists(), "Checkpoint file should exist"

        # Record original actor params
        original_params = {
            name: p.clone() for name, p in trainer.actor.named_parameters()
        }

        # Zero all actor params (to verify load restores them)
        with torch.no_grad():
            for p in trainer.actor.parameters():
                p.zero_()

        # Verify params are zeroed
        for name, p in trainer.actor.named_parameters():
            assert (p == 0).all(), f"Param {name} should be zeroed"

        # Load checkpoint
        trainer.load_checkpoint(str(ckpt_path))

        # Verify state restored
        assert trainer.total_frames == 42
        assert trainer.total_episodes == 7
        assert trainer.best_reward == -100.0

        # Verify actor params restored (not all zeros)
        for name, p in trainer.actor.named_parameters():
            original = original_params[name]
            if original.abs().sum() > 0:  # Only check params that weren't originally zero
                assert not (p == 0).all(), (
                    f"Param {name} should be restored from checkpoint, but is still zero"
                )
                assert torch.allclose(p, original, atol=1e-6), (
                    f"Param {name} not properly restored"
                )
