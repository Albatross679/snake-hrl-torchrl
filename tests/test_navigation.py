"""Tests for COBRA snake robot navigation (Jiang et al., 2024)."""

import math
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest
import torch

try:
    from jiang2024.configs_jiang2024 import (
        CobraPhysicsConfig,
        CobraCPGConfig,
        CobraEnvConfig,
        CobraMazeEnvConfig,
        CobraNetworkConfig,
        CobraNavigationConfig,
    )
    from jiang2024.cpg_jiang2024 import BingCPG, DualCPGController
    from jiang2024.env_jiang2024 import CobraNavigationEnv, CobraMazeEnv
    from jiang2024.rewards_jiang2024 import (
        compute_proximity_reward,
        compute_velocity_reward,
        compute_smoothness_penalty,
        compute_navigation_reward,
    )
    from jiang2024.planner_jiang2024 import AStarPlanner, OccupancyGrid
    from jiang2024.maze_jiang2024 import KruskalMazeGenerator
    has_jiang2024 = True
except ImportError:
    has_jiang2024 = False

pytestmark = pytest.mark.skipif(not has_jiang2024, reason="jiang2024 package not importable")


# === COBRA XML Model ===

class TestCobraXML:
    """Test that COBRA MuJoCo model loads correctly."""

    def test_xml_loads(self):
        config = CobraEnvConfig()
        env = CobraNavigationEnv(config=config)
        assert env.model is not None
        env.close()

    def test_num_actuators(self):
        env = CobraNavigationEnv()
        assert env.model.nu == 11  # 11 position actuators
        env.close()

    def test_joint_axes_alternate(self):
        """Verify odd joints are yaw (z-axis), even joints are pitch (y-axis)."""
        import mujoco
        env = CobraNavigationEnv()
        m = env.model

        for i in range(1, 12):
            joint_name = f"joint{i:02d}"
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            axis = m.jnt_axis[jid]

            if i % 2 == 1:  # odd joints: yaw (z-axis)
                assert axis[2] != 0, f"Joint {joint_name} should be yaw (z-axis)"
            else:  # even joints: pitch (y-axis)
                assert axis[1] != 0, f"Joint {joint_name} should be pitch (y-axis)"

        env.close()


# === CPG ===

class TestBingCPG:
    """Test single Bing CPG oscillator."""

    def test_output_shape(self):
        cpg = BingCPG(n=5)
        output = cpg.step(dt=0.01, R=1.0, omega=2.0, theta=0.5, delta=0.0)
        assert output.shape == (5,)

    def test_amplitude_convergence(self):
        """Amplitude should converge to target R."""
        cpg = BingCPG(n=3, a=20.0)
        R_target = 0.8
        for _ in range(5000):
            output = cpg.step(dt=0.01, R=R_target, omega=2.0, theta=0.5, delta=0.0)
        # After convergence, max amplitude should be close to R
        assert np.allclose(cpg.r, R_target, atol=0.1)

    def test_offset(self):
        """Output should include offset delta."""
        cpg = BingCPG(n=2)
        # Run enough steps for amplitude to build up
        for _ in range(1000):
            output = cpg.step(dt=0.01, R=0.0, omega=0.0, theta=0.0, delta=0.5)
        # With R=0, output should be just the offset
        assert np.allclose(output, 0.5, atol=0.1)

    def test_reset(self):
        cpg = BingCPG(n=3)
        cpg.step(dt=0.01, R=1.0, omega=2.0, theta=0.5, delta=0.0)
        cpg.reset()
        assert np.allclose(cpg.phi, 0.0)
        assert np.allclose(cpg.r, 0.0)
        assert np.allclose(cpg.rdot, 0.0)


class TestDualCPGController:
    """Test dual CPG controller for COBRA."""

    def test_output_shape(self):
        cpg = DualCPGController(num_joints=11)
        action = np.array([0.5, 0.5, 0.05, 1.0, 1.0, 0.0, 0.0])
        targets = cpg.step(action)
        assert targets.shape == (11,)

    def test_interleaving(self):
        """Pitch and yaw outputs should be assigned to correct joint indices."""
        cpg = DualCPGController(num_joints=11)
        # Set different amplitudes for pitch vs yaw
        action = np.array([1.0, 0.0, 0.05, 1.0, 1.0, 0.0, 0.0])
        targets = cpg.step(action)

        # Yaw joints (even indices 0,2,4,6,8,10) should have ~0 output (R2=0)
        for idx in [0, 2, 4, 6, 8, 10]:
            assert abs(targets[idx]) < 0.1, f"Yaw joint {idx} should be near zero"

    def test_reset(self):
        cpg = DualCPGController()
        action = np.array([1.0, 1.0, 0.05, 1.0, 1.0, 0.0, 0.0])
        cpg.step(action)
        cpg.reset()
        assert np.allclose(cpg.cpg_pitch.phi, 0.0)
        assert np.allclose(cpg.cpg_yaw.phi, 0.0)


# === Environment ===

class TestCobraNavigationEnv:
    """Tests for CobraNavigationEnv."""

    @pytest.fixture
    def env(self):
        config = CobraEnvConfig()
        e = CobraNavigationEnv(config=config)
        yield e
        e.close()

    def test_reset_shape(self, env):
        td = env.reset()
        assert td["observation"].shape == (21,)
        assert td["done"].shape == (1,)

    def test_step_shape(self, env):
        td = env.reset()
        action = torch.zeros(7, dtype=torch.float32)
        td["action"] = action
        td_out = env.step(td)
        td_next = td_out["next"]
        assert td_next["observation"].shape == (21,)
        assert td_next["reward"].shape == (1,)
        assert td_next["done"].shape == (1,)

    def test_action_spec(self, env):
        spec = env.action_spec
        assert spec.shape == (7,)

    def test_reward_finite(self, env):
        td = env.reset()
        action = torch.zeros(7, dtype=torch.float32)
        td["action"] = action
        td_out = env.step(td)
        reward = td_out["next", "reward"].item()
        assert np.isfinite(reward)

    def test_episode_truncation(self, env):
        td = env.reset()
        for i in range(env.config.max_episode_steps):
            action = torch.zeros(7, dtype=torch.float32)
            td["action"] = action
            td = env.step(td)
            td = td["next"]
        assert td["truncated"].item() is True

    def test_multiple_steps(self, env):
        """Observation should change after non-zero actions."""
        td = env.reset()
        obs_0 = td["observation"].clone()

        for _ in range(3):
            action = torch.tensor([0.5, 0.5, 0.05, 1.0, 1.0, 0.0, 0.0], dtype=torch.float32)
            td["action"] = action
            td = env.step(td)
            td = td["next"]

        obs_n = td["observation"]
        assert not torch.allclose(obs_0, obs_n)


# === Rewards ===

class TestRewardFunctions:
    """Test navigation reward functions."""

    def test_proximity_close(self):
        """Close distance -> high proximity reward."""
        r = compute_proximity_reward(0.1)
        assert r > 4.0

    def test_proximity_far(self):
        """Far distance -> low proximity reward."""
        r = compute_proximity_reward(10.0)
        assert r < 0.15

    def test_velocity_approaching(self):
        """Getting closer -> positive velocity reward."""
        r = compute_velocity_reward(5.0, 4.0)
        assert r > 0

    def test_velocity_retreating(self):
        """Getting farther -> negative velocity reward."""
        r = compute_velocity_reward(4.0, 5.0)
        assert r < 0

    def test_smoothness_zero_change(self):
        """Same action -> zero penalty."""
        a = np.array([1.0, 2.0, 3.0])
        r = compute_smoothness_penalty(a, a)
        assert r == 0.0

    def test_smoothness_large_change(self):
        """Large action change -> large penalty."""
        a1 = np.zeros(7)
        a2 = np.ones(7)
        r = compute_smoothness_penalty(a2, a1)
        assert r < -1.0

    def test_combined_reward_finite(self):
        r = compute_navigation_reward(
            distance=3.0,
            dist_prev=4.0,
            action=np.zeros(7),
            prev_action=np.zeros(7),
        )
        assert np.isfinite(r)


# === A* Planner ===

class TestAStarPlanner:
    """Test A* pathfinding."""

    def test_empty_grid(self):
        """Path in empty grid should be found."""
        grid = OccupancyGrid(width=20, height=20, resolution=0.5)
        planner = AStarPlanner(grid)
        path = planner.plan((0.5, 0.5), (9.0, 9.0))
        assert path is not None
        assert len(path) >= 2

    def test_path_start_end(self):
        """Path should start near start and end near goal."""
        grid = OccupancyGrid(width=20, height=20, resolution=0.5)
        planner = AStarPlanner(grid)
        path = planner.plan((0.5, 0.5), (9.0, 9.0))
        assert path is not None
        # First point should be near start
        assert abs(path[0][0] - 0.5) < 1.0
        assert abs(path[0][1] - 0.5) < 1.0
        # Last point should be near goal
        assert abs(path[-1][0] - 9.0) < 1.0
        assert abs(path[-1][1] - 9.0) < 1.0

    def test_with_obstacles(self):
        """Path should navigate around obstacles."""
        grid = OccupancyGrid(width=20, height=20, resolution=0.5)
        # Place wall across the middle
        for gx in range(0, 15):
            grid.set_obstacle(gx, 10)
        planner = AStarPlanner(grid)
        path = planner.plan((0.5, 0.5), (9.0, 9.0))
        assert path is not None

    def test_unreachable(self):
        """Fully blocked goal should return None."""
        grid = OccupancyGrid(width=10, height=10, resolution=1.0)
        # Surround goal cell
        for gx in range(10):
            grid.set_obstacle(gx, 5)
        planner = AStarPlanner(grid)
        path = planner.plan((0.5, 0.5), (5.0, 9.0))
        assert path is None


# === Kruskal Maze ===

class TestKruskalMaze:
    """Test maze generation."""

    def test_generate_walls(self):
        maze = KruskalMazeGenerator(rows=3, cols=3, cell_size=2.0)
        walls = maze.generate(seed=42)
        assert len(walls) > 0

    def test_deterministic_seeding(self):
        """Same seed -> same walls."""
        maze1 = KruskalMazeGenerator(rows=4, cols=4)
        maze2 = KruskalMazeGenerator(rows=4, cols=4)
        w1 = maze1.generate(seed=123)
        w2 = maze2.generate(seed=123)
        assert len(w1) == len(w2)
        for a, b in zip(w1, w2):
            assert a == pytest.approx(b)

    def test_different_seeds(self):
        """Different seeds -> different mazes (with high probability)."""
        maze1 = KruskalMazeGenerator(rows=5, cols=5)
        maze2 = KruskalMazeGenerator(rows=5, cols=5)
        w1 = maze1.generate(seed=1)
        w2 = maze2.generate(seed=2)
        # At least some walls should differ
        differ = any(
            a != pytest.approx(b) for a, b in zip(w1, w2)
        )
        assert differ

    def test_occupancy_grid(self):
        maze = KruskalMazeGenerator(rows=3, cols=3, cell_size=2.0)
        maze.generate(seed=42)
        grid = maze.to_occupancy_grid(resolution=0.5, inflation=0.1)
        assert grid.width > 0
        assert grid.height > 0
        # Some cells should be occupied (walls)
        assert grid.grid.sum() > 0

    def test_mjcf_bodies(self):
        maze = KruskalMazeGenerator(rows=3, cols=3)
        maze.generate(seed=42)
        xml = maze.to_mjcf_bodies()
        assert "wall_" in xml
        assert "geom" in xml

    def test_cell_center(self):
        maze = KruskalMazeGenerator(rows=3, cols=3, cell_size=2.0, origin=(0.0, 0.0))
        cx, cy = maze.get_cell_center(0, 0)
        assert cx == pytest.approx(1.0)
        assert cy == pytest.approx(1.0)

    def test_connectivity(self):
        """A perfect maze should connect all cells via A*."""
        maze = KruskalMazeGenerator(rows=4, cols=4, cell_size=2.0)
        maze.generate(seed=42)
        grid = maze.to_occupancy_grid(resolution=0.25, inflation=0.1)
        planner = AStarPlanner(grid)

        start = maze.get_cell_center(0, 0)
        goal = maze.get_cell_center(3, 3)
        path = planner.plan(start, goal)
        assert path is not None, "Maze should be fully connected"


# === Configs ===

class TestConfigs:
    """Test configuration dataclasses."""

    def test_cobra_physics_defaults(self):
        cfg = CobraPhysicsConfig()
        assert cfg.num_joints == 11
        assert cfg.mujoco_timestep == 0.001
        assert cfg.mujoco_substeps == 20

    def test_cpg_config_defaults(self):
        cfg = CobraCPGConfig()
        assert cfg.num_cpg_steps == 100
        assert cfg.R_max == 1.5

    def test_env_config_defaults(self):
        cfg = CobraEnvConfig()
        assert cfg.max_episode_steps == 80
        assert cfg.waypoint_threshold == 0.5

    def test_maze_config_extends_env(self):
        cfg = CobraMazeEnvConfig()
        assert cfg.maze_rows == 5
        assert cfg.max_episode_steps == 200  # Overridden for maze

    def test_navigation_config_inherits_ddpg(self):
        cfg = CobraNavigationConfig()
        assert cfg.total_frames == 3_000_000
        assert cfg.tau == 0.001
        assert cfg.noise_type == "ou"

    def test_network_config(self):
        cfg = CobraNetworkConfig()
        assert cfg.actor.hidden_dims == [512, 256, 128]
        assert cfg.critic.hidden_dims == [512, 256, 256]
        assert cfg.actor.activation == "relu"
