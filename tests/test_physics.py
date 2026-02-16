"""Tests for physics simulation module with DisMech integration."""

import pytest
import numpy as np
import warnings

from physics.geometry import (
    SnakeGeometry,
    PreyGeometry,
    create_snake_geometry,
    create_prey_geometry,
    compute_contact_points,
    compute_wrap_angle,
)
from physics.snake_robot import SnakeRobot
from configs.physics import (
    PhysicsConfig,
    SolverFramework,
    DismechConfig,
    DismechRodsConfig,
    MujocoPhysicsConfig,
)
from configs.geometry import GeometryConfig


class TestGeometry:
    """Tests for geometry creation and manipulation."""

    @pytest.fixture
    def physics_config(self):
        """Create a physics configuration for testing."""
        return DismechConfig(
            geometry=GeometryConfig(
                snake_length=1.0,
                snake_radius=0.001,
                num_segments=10,
            ),
            prey_radius=0.1,
            prey_length=0.3,
            dt=5e-2,
        )

    def test_create_snake_geometry_deprecated(self, physics_config):
        """Test that create_snake_geometry raises deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            snake = create_snake_geometry(physics_config)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

        # Verify the geometry is still created correctly
        assert snake.num_segments == physics_config.num_segments
        assert snake.num_nodes == physics_config.num_segments + 1

    def test_create_prey_geometry(self, physics_config):
        """Test prey geometry creation."""
        prey = create_prey_geometry(physics_config)

        assert prey.radius == physics_config.prey_radius
        assert prey.length == physics_config.prey_length
        assert prey.mass > 0

    def test_prey_distance_to_point(self, physics_config):
        """Test distance computation from point to prey."""
        prey = create_prey_geometry(
            physics_config,
            position=np.array([0.0, 0.0, 0.5]),
            orientation=np.array([0.0, 0.0, 1.0]),
        )

        # Point on surface should have distance ~0
        surface_point = np.array([prey.radius, 0.0, 0.5])
        assert np.isclose(prey.distance_to_point(surface_point), 0.0, atol=1e-6)

        # Point outside should have positive distance
        outside_point = np.array([prey.radius + 0.1, 0.0, 0.5])
        assert prey.distance_to_point(outside_point) > 0

        # Point inside should have negative distance
        inside_point = np.array([0.0, 0.0, 0.5])
        assert prey.distance_to_point(inside_point) < 0


class TestSnakeRobot:
    """Tests for SnakeRobot physics simulation with DisMech."""

    @pytest.fixture
    def config(self):
        """Create a DisMech physics configuration for testing."""
        return DismechConfig(
            geometry=GeometryConfig(
                num_segments=10,
                snake_radius=0.001,
            ),
            dt=5e-2,
            density=1200.0,
        )

    @pytest.fixture
    def robot(self, config):
        """Create a snake robot for testing."""
        return SnakeRobot(config)

    def test_robot_initialization(self, robot):
        """Test robot is initialized correctly."""
        assert robot.snake is not None
        assert robot.prey is not None
        assert robot.time == 0.0

    def test_robot_reset(self, robot):
        """Test robot reset."""
        # Step a few times
        for _ in range(3):
            robot.step()

        # Reset
        robot.reset()

        assert robot.time == 0.0
        np.testing.assert_array_equal(
            robot.velocities, np.zeros_like(robot.velocities)
        )

    def test_robot_step(self, robot):
        """Test single simulation step."""
        initial_time = robot.time
        state = robot.step()

        assert robot.time > initial_time
        assert "positions" in state
        assert "velocities" in state
        assert "prey_distance" in state

    def test_robot_curvature_control(self, robot, config):
        """Test curvature control input."""
        # Apply curvature control
        curvatures = np.ones(config.num_segments - 1) * 0.5
        robot.set_curvature_control(curvatures)

        # Step simulation
        for _ in range(10):
            robot.step()

        # Get state (snake should have some configuration)
        state = robot.get_state()
        assert "curvatures" in state
        assert len(state["curvatures"]) == config.num_segments - 1

    def test_robot_get_observation(self, robot):
        """Test observation vector generation."""
        obs = robot.get_observation()

        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert len(obs.shape) == 1

    def test_robot_get_energy(self, robot):
        """Test energy computation."""
        energy = robot.get_energy()

        assert "kinetic" in energy
        assert "gravitational" in energy
        assert "elastic" in energy
        assert "total" in energy

        # Total should be sum of components
        assert np.isclose(
            energy["total"],
            energy["kinetic"] + energy["gravitational"] + energy["elastic"],
        )

    def test_robot_state_dict(self, robot):
        """Test state dictionary contents."""
        state = robot.get_state()

        required_keys = [
            "positions",
            "velocities",
            "curvatures",
            "prey_position",
            "prey_orientation",
            "prey_distance",
            "contact_mask",
            "contact_fraction",
            "wrap_angle",
            "wrap_count",
            "time",
        ]

        for key in required_keys:
            assert key in state, f"Missing key: {key}"


class TestPhysicsStability:
    """Tests for physics simulation stability with DisMech."""

    def test_basic_simulation_runs(self):
        """Test that simulation runs without errors."""
        config = DismechConfig(
            geometry=GeometryConfig(
                num_segments=10,
                snake_radius=0.001,
            ),
            dt=5e-2,
            density=1200.0,
        )
        robot = SnakeRobot(config)

        # Step a few times
        for _ in range(5):
            state = robot.step()

            # Check for NaN/Inf
            assert np.all(np.isfinite(state["positions"]))

    def test_simulation_with_curvature_control(self):
        """Test simulation with applied curvature control."""
        config = DismechConfig(
            geometry=GeometryConfig(
                num_segments=10,
                snake_radius=0.001,
            ),
            dt=5e-2,
        )
        robot = SnakeRobot(config)

        # Apply sinusoidal curvature pattern
        np.random.seed(42)
        for i in range(10):
            # Serpenoid-like curvature
            s = np.linspace(0, 2 * np.pi, config.num_segments - 1)
            curvatures = np.sin(s + i * 0.1) * 0.5
            robot.set_curvature_control(curvatures)
            state = robot.step()

            # Check for NaN/Inf
            assert np.all(np.isfinite(state["positions"]))
            assert np.all(np.isfinite(state["velocities"]))


mujoco_module = pytest.importorskip("mujoco")


class TestMujocoSnakeRobot:
    """Tests for MujocoSnakeRobot physics simulation with MuJoCo."""

    @pytest.fixture
    def config(self):
        """Create a MuJoCo physics configuration for testing."""
        return MujocoPhysicsConfig(
            geometry=GeometryConfig(
                num_segments=10,
                snake_radius=0.005,
                snake_length=1.0,
            ),
            dt=5e-2,
            density=1200.0,
            prey_radius=0.1,
            prey_length=0.3,
        )

    @pytest.fixture
    def robot(self, config):
        """Create a MuJoCo snake robot for testing."""
        from physics.mujoco_snake_robot import MujocoSnakeRobot
        return MujocoSnakeRobot(config)

    def test_robot_initialization(self, robot, config):
        """Test robot is initialized correctly."""
        assert robot.snake is not None
        assert robot.prey is not None
        assert robot.time == 0.0
        assert robot.snake.num_nodes == config.num_segments + 1
        assert robot.snake.positions.shape == (config.num_segments + 1, 3)

    def test_robot_step(self, robot):
        """Test single simulation step returns all state keys."""
        initial_time = robot.time
        state = robot.step()

        assert robot.time > initial_time

        required_keys = [
            "positions", "velocities", "curvatures",
            "prey_position", "prey_orientation", "prey_distance",
            "contact_mask", "contact_fraction",
            "wrap_angle", "wrap_count", "time",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"

    def test_robot_curvature_control(self, robot, config):
        """Test curvature control input."""
        curvatures = np.ones(config.num_segments - 1) * 0.5
        robot.set_curvature_control(curvatures)

        for _ in range(5):
            robot.step()

        state = robot.get_state()
        assert "curvatures" in state
        assert len(state["curvatures"]) == config.num_segments - 1

    def test_robot_get_observation(self, robot, config):
        """Test observation vector generation."""
        obs = robot.get_observation()

        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert len(obs.shape) == 1

        # Full observation: positions (11*3) + velocities (11*3) + curvatures (9) + prey (6)
        num_nodes = config.num_segments + 1
        num_joints = config.num_segments - 1
        expected_dim = 3 * num_nodes + 3 * num_nodes + num_joints + 6
        assert obs.shape[0] == expected_dim

    def test_robot_reset(self, robot):
        """Test reset produces valid state."""
        for _ in range(3):
            robot.step()

        robot.reset()

        assert robot.time == 0.0
        np.testing.assert_array_equal(
            robot.velocities, np.zeros_like(robot.velocities)
        )

        state = robot.get_state()
        assert np.all(np.isfinite(state["positions"]))

    def test_robot_get_energy(self, robot):
        """Test energy computation."""
        energy = robot.get_energy()

        assert "kinetic" in energy
        assert "gravitational" in energy
        assert "elastic" in energy
        assert "total" in energy

        assert np.isclose(
            energy["total"],
            energy["kinetic"] + energy["gravitational"] + energy["elastic"],
        )

    def test_simulation_stability(self, robot, config):
        """Test simulation stays stable with sinusoidal curvature over 20 steps."""
        for i in range(20):
            s = np.linspace(0, 2 * np.pi, config.num_segments - 1)
            curvatures = np.sin(s + i * 0.1) * 0.5
            robot.set_curvature_control(curvatures)
            state = robot.step()

            assert np.all(np.isfinite(state["positions"])), f"NaN/Inf at step {i}"
            assert np.all(np.isfinite(state["velocities"])), f"NaN/Inf vel at step {i}"

    def test_factory_creates_mujoco_robot(self, config):
        """Test factory dispatch creates MujocoSnakeRobot."""
        from physics import create_snake_robot
        from physics.mujoco_snake_robot import MujocoSnakeRobot

        robot = create_snake_robot(config)
        assert isinstance(robot, MujocoSnakeRobot)


_has_py_dismech = True
try:
    import py_dismech
except ImportError:
    _has_py_dismech = False


@pytest.mark.skipif(not _has_py_dismech, reason="py_dismech not installed")
class TestDismechRodsSnakeRobot:
    """Tests for DismechRodsSnakeRobot physics simulation with dismech-rods (C++)."""

    @pytest.fixture
    def config(self):
        """Create a dismech-rods physics configuration for testing."""
        return DismechRodsConfig(
            geometry=GeometryConfig(
                num_segments=10,
                snake_radius=0.001,
            ),
            dt=5e-2,
            density=1200.0,
        )

    @pytest.fixture
    def robot(self, config):
        """Create a dismech-rods snake robot for testing."""
        from physics.dismech_rods_snake_robot import DismechRodsSnakeRobot
        return DismechRodsSnakeRobot(config)

    def test_robot_initialization(self, robot, config):
        """Test robot is initialized correctly."""
        assert robot.snake is not None
        assert robot.prey is not None
        assert robot.time == 0.0
        assert robot.snake.num_nodes == config.num_segments + 1
        assert robot.snake.positions.shape == (config.num_segments + 1, 3)

    def test_robot_step(self, robot):
        """Test single simulation step."""
        initial_time = robot.time
        state = robot.step()

        assert robot.time > initial_time
        assert "positions" in state
        assert "velocities" in state
        assert "curvatures" in state
        assert "prey_position" in state
        assert "prey_distance" in state
        assert "contact_mask" in state
        assert "contact_fraction" in state
        assert "wrap_angle" in state
        assert "wrap_count" in state
        assert "time" in state

    def test_robot_curvature_control(self, robot, config):
        """Test curvature control input."""
        curvatures = np.ones(config.num_segments - 1) * 0.5
        robot.set_curvature_control(curvatures)

        # Step simulation
        for _ in range(5):
            robot.step()

        state = robot.get_state()
        assert "curvatures" in state
        assert len(state["curvatures"]) == config.num_segments - 1

    def test_robot_get_observation(self, robot, config):
        """Test observation vector generation."""
        obs = robot.get_observation()

        assert isinstance(obs, np.ndarray)
        assert obs.dtype == np.float32
        assert len(obs.shape) == 1

        # Full observation: positions (11*3) + velocities (11*3) + curvatures (9) + prey (6)
        num_nodes = config.num_segments + 1
        num_joints = config.num_segments - 1
        expected_dim = 3 * num_nodes + 3 * num_nodes + num_joints + 6
        assert obs.shape[0] == expected_dim

    def test_robot_reset(self, robot):
        """Test reset produces valid state."""
        # Step a few times
        for _ in range(3):
            robot.step()

        # Reset
        robot.reset()

        assert robot.time == 0.0
        np.testing.assert_array_equal(
            robot.velocities, np.zeros_like(robot.velocities)
        )

        # State should be valid after reset
        state = robot.get_state()
        assert np.all(np.isfinite(state["positions"]))
