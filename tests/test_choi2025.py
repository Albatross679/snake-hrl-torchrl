"""Tests for choi2025 package (Choi & Tong, 2025)."""

import inspect

import numpy as np
import pytest
import torch

from choi2025.config import (
    Choi2025Config,
    Choi2025EnvConfig,
    Choi2025NetworkConfig,
    Choi2025PPOConfig,
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
        # __post_init__ sets name to fixed_{task}_sac_lr1e3_{num_envs}envs
        assert config.name.startswith("fixed_")
        assert "sac" in config.name
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


# ---------------------------------------------------------------------------
# CHOI-01: Choi2025PPOConfig defaults and naming
# ---------------------------------------------------------------------------


class TestChoi2025PPOConfig:
    """Verify PPO config has correct hyperparameters, naming, and network dims."""

    def test_ppo_config_clip_epsilon(self):
        config = Choi2025PPOConfig()
        assert config.clip_epsilon == 0.2

    def test_ppo_config_num_epochs(self):
        config = Choi2025PPOConfig()
        assert config.num_epochs == 10

    def test_ppo_config_mini_batch_size(self):
        config = Choi2025PPOConfig()
        assert config.mini_batch_size == 64

    def test_ppo_config_learning_rate(self):
        config = Choi2025PPOConfig()
        assert config.learning_rate == 3e-4

    def test_ppo_config_frames_per_batch(self):
        config = Choi2025PPOConfig()
        assert config.frames_per_batch == 4096

    def test_ppo_config_gae_lambda(self):
        config = Choi2025PPOConfig()
        assert config.gae_lambda == 0.95

    def test_ppo_config_patience_disabled(self):
        config = Choi2025PPOConfig()
        assert config.patience_batches == 0

    def test_ppo_post_init_naming_follows_convention(self):
        """__post_init__ produces fixed_{task}_ppo_{lr}_{envs} naming."""
        config = Choi2025PPOConfig()
        assert config.name.startswith("fixed_")
        assert "_ppo_" in config.name
        assert config.name == config.experiment_name

    def test_ppo_post_init_reflects_task(self):
        """Name contains the task value from the env config."""
        for task in TaskType:
            env_cfg = Choi2025EnvConfig(task=task)
            config = Choi2025PPOConfig(env=env_cfg)
            assert task.value in config.name, (
                f"Expected '{task.value}' in name '{config.name}'"
            )

    def test_ppo_post_init_reflects_num_envs(self):
        """Name contains the num_envs value."""
        config = Choi2025PPOConfig()
        config.num_envs = 32
        config.__post_init__()
        assert "32envs" in config.name

    def test_ppo_network_is_3x1024(self):
        """Network config uses 3x1024 hidden dims for both actor and critic."""
        config = Choi2025PPOConfig()
        assert config.network.actor.hidden_dims == [1024, 1024, 1024]
        assert config.network.critic.hidden_dims == [1024, 1024, 1024]

    def test_ppo_inherits_from_ppoconfig(self):
        from src.configs.training import PPOConfig
        assert isinstance(Choi2025PPOConfig(), PPOConfig)

    def test_ppo_wandb_project(self):
        config = Choi2025PPOConfig()
        assert config.wandb.project == "choi2025-replication"

    def test_sac_post_init_naming_follows_convention(self):
        """SAC __post_init__ also produces fixed_{task}_sac naming."""
        config = Choi2025Config()
        assert config.name.startswith("fixed_")
        assert "_sac_" in config.name
        assert config.name == config.experiment_name

    def test_sac_wandb_project(self):
        config = Choi2025Config()
        assert config.wandb.project == "choi2025-replication"


# ---------------------------------------------------------------------------
# CHOI-02: train_ppo.py wiring and structure
# ---------------------------------------------------------------------------


class TestTrainPPOWiring:
    """Verify train_ppo.py imports resolve and has required structure."""

    def test_train_ppo_imports_resolve(self):
        """train_ppo.py can be imported without errors."""
        import choi2025.train_ppo as mod
        assert hasattr(mod, "main")
        assert hasattr(mod, "parse_args")

    def test_train_ppo_uses_ppo_trainer(self):
        """train_ppo.py imports PPOTrainer from the correct location."""
        import choi2025.train_ppo as mod
        source = inspect.getsource(mod)
        assert "from src.trainers.ppo import PPOTrainer" in source

    def test_train_ppo_uses_choi2025ppoconfig(self):
        """train_ppo.py imports Choi2025PPOConfig."""
        import choi2025.train_ppo as mod
        source = inspect.getsource(mod)
        assert "Choi2025PPOConfig" in source

    def test_train_ppo_uses_parallel_env(self):
        """train_ppo.py uses ParallelEnv (not SerialEnv) for multi-env."""
        import choi2025.train_ppo as mod
        source = inspect.getsource(mod)
        assert "ParallelEnv" in source
        assert "SerialEnv" not in source

    def test_train_ppo_has_env_close(self):
        """train_ppo.py calls env.close() in cleanup."""
        import choi2025.train_ppo as mod
        source = inspect.getsource(mod)
        assert "env.close()" in source

    def test_train_ppo_has_gpu_lock_in_main(self):
        """train_ppo.py wraps main() with GpuLock in __main__ block."""
        import importlib
        source_path = importlib.util.find_spec("choi2025.train_ppo").origin
        with open(source_path, "r") as f:
            source = f.read()
        assert "GpuLock" in source
        assert 'if __name__ == "__main__"' in source


# ---------------------------------------------------------------------------
# CHOI-03: run_experiment.py and evaluate.py structure
# ---------------------------------------------------------------------------


class TestRunExperimentWiring:
    """Verify run_experiment.py has correct matrix, flags, and watchdog."""

    def test_experiment_matrix_has_8_entries(self):
        """EXPERIMENT_MATRIX contains 4 tasks x 2 algos = 8 entries."""
        from choi2025.run_experiment import EXPERIMENT_MATRIX
        assert len(EXPERIMENT_MATRIX) == 8

    def test_experiment_matrix_covers_all_tasks(self):
        """Every TaskType appears in the matrix."""
        from choi2025.run_experiment import EXPERIMENT_MATRIX
        tasks_in_matrix = {e["task"] for e in EXPERIMENT_MATRIX}
        for task in TaskType:
            assert task.value in tasks_in_matrix

    def test_experiment_matrix_covers_both_algos(self):
        """Both SAC and PPO appear in the matrix."""
        from choi2025.run_experiment import EXPERIMENT_MATRIX
        algos = {e["algo"] for e in EXPERIMENT_MATRIX}
        assert algos == {"sac", "ppo"}

    def test_quick_flag_exists(self):
        """run_experiment.py accepts --quick flag."""
        import choi2025.run_experiment as mod
        source = inspect.getsource(mod)
        assert "--quick" in source

    def test_watchdog_timeout_defined(self):
        """run_experiment.py defines WATCHDOG_TIMEOUT for hung process detection."""
        from choi2025.run_experiment import WATCHDOG_TIMEOUT
        assert isinstance(WATCHDOG_TIMEOUT, (int, float))
        assert WATCHDOG_TIMEOUT > 0

    def test_uses_subprocess_popen(self):
        """run_experiment.py uses subprocess.Popen (not .run) for watchdog."""
        import choi2025.run_experiment as mod
        source = inspect.getsource(mod)
        assert "subprocess.Popen" in source

    def test_handles_hung_exit_codes(self):
        """run_experiment.py classifies exit codes 137/143 as hung."""
        import choi2025.run_experiment as mod
        source = inspect.getsource(mod)
        assert "137" in source
        assert "143" in source

    def test_gpu_cleanup_between_runs(self):
        """run_experiment.py calls torch.cuda.empty_cache between runs."""
        import choi2025.run_experiment as mod
        source = inspect.getsource(mod)
        assert "torch.cuda.empty_cache" in source


class TestEvaluateWiring:
    """Verify evaluate.py supports dual-algorithm evaluation."""

    def test_evaluate_imports_resolve(self):
        """evaluate.py can be imported without errors."""
        import choi2025.evaluate as mod
        assert hasattr(mod, "main")

    def test_evaluate_has_algo_flag(self):
        """evaluate.py accepts --algo flag with sac/ppo choices."""
        import choi2025.evaluate as mod
        source = inspect.getsource(mod)
        assert "--algo" in source
        assert '"sac"' in source
        assert '"ppo"' in source

    def test_evaluate_imports_both_configs(self):
        """evaluate.py imports both Choi2025Config and Choi2025PPOConfig."""
        import choi2025.evaluate as mod
        source = inspect.getsource(mod)
        assert "Choi2025Config" in source
        assert "Choi2025PPOConfig" in source


# ---------------------------------------------------------------------------
# CHOI-04: Lightweight env + trainer instantiation smoke test
# ---------------------------------------------------------------------------


class TestConfigToTrainerWiring:
    """Verify SAC and PPO configs can wire up to env + trainer without crashing."""

    def test_sac_config_can_instantiate_env(self):
        """Choi2025Config produces an env that resets successfully."""
        config = Choi2025Config()
        env = SoftManipulatorEnv(config.env)
        td = env.reset()
        assert "observation" in td.keys()
        env.close()

    def test_ppo_config_can_instantiate_env(self):
        """Choi2025PPOConfig produces an env that resets successfully."""
        config = Choi2025PPOConfig()
        env = SoftManipulatorEnv(config.env)
        td = env.reset()
        assert "observation" in td.keys()
        env.close()

    def test_sac_config_can_create_trainer(self):
        """SACTrainer can be instantiated with Choi2025Config without crashing."""
        from src.trainers.sac import SACTrainer
        import tempfile

        config = Choi2025Config()
        config.wandb.enabled = False  # No W&B for test
        env = SoftManipulatorEnv(config.env)
        run_dir = tempfile.mkdtemp(prefix="test_sac_")
        try:
            trainer = SACTrainer(
                env=env,
                config=config,
                network_config=config.network,
                device="cpu",
                run_dir=run_dir,
            )
            assert trainer.actor is not None
            assert trainer.critic is not None
        finally:
            env.close()

    def test_ppo_config_can_create_trainer(self):
        """PPOTrainer can be instantiated with Choi2025PPOConfig without crashing."""
        from src.trainers.ppo import PPOTrainer
        import tempfile

        config = Choi2025PPOConfig()
        config.wandb.enabled = False  # No W&B for test
        env = SoftManipulatorEnv(config.env)
        run_dir = tempfile.mkdtemp(prefix="test_ppo_")
        try:
            trainer = PPOTrainer(
                env=env,
                config=config,
                network_config=config.network,
                device="cpu",
                run_dir=run_dir,
            )
            assert trainer.actor is not None
            assert trainer.critic is not None
        finally:
            env.close()


# ---------------------------------------------------------------------------
# CHOI-06: record.py dual-algorithm support
# ---------------------------------------------------------------------------


class TestRecordDualAlgo:
    """Verify record.py supports --algo flag with checkpoint auto-detection."""

    def test_record_imports_resolve(self):
        """record.py can be imported without errors."""
        import choi2025.record as mod
        assert hasattr(mod, "main")
        assert hasattr(mod, "load_actor")

    def test_record_has_algo_argument(self):
        """record.py accepts --algo flag with sac/ppo choices."""
        import choi2025.record as mod
        source = inspect.getsource(mod)
        assert "--algo" in source
        assert '"sac"' in source
        assert '"ppo"' in source

    def test_record_has_output_dir_argument(self):
        """record.py accepts --output-dir argument."""
        import choi2025.record as mod
        source = inspect.getsource(mod)
        assert "--output-dir" in source

    def test_record_auto_output_naming(self):
        """Output filename auto-generated from task and algo when --output not given."""
        import choi2025.record as mod
        source = inspect.getsource(mod)
        # Should construct path like: fixed_{task}_{algo}.mp4
        assert "fixed_" in source
        assert "args.task" in source
        assert "args.algo" in source

    def test_record_checkpoint_format_auto_detection(self):
        """load_actor detects actor_state_dict key in checkpoint dict."""
        import choi2025.record as mod
        source = inspect.getsource(mod.load_actor)
        assert "actor_state_dict" in source
