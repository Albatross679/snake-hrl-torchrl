"""Tests for physics simulation module with DisMech integration."""

import pytest
import numpy as np
import warnings

from snake_hrl.physics.geometry import (
    SnakeGeometry,
    PreyGeometry,
    create_snake_geometry,
    create_prey_geometry,
    compute_contact_points,
    compute_wrap_angle,
)
from snake_hrl.physics.snake_robot import SnakeRobot
from snake_hrl.configs.env import PhysicsConfig


class TestGeometry:
    """Tests for geometry creation and manipulation."""

    @pytest.fixture
    def physics_config(self):
        """Create a physics configuration for testing."""
        return PhysicsConfig(
            snake_length=1.0,
            snake_radius=0.001,  # DisMech compatible radius
            num_segments=10,
            prey_radius=0.1,
            prey_length=0.3,
            dt=5e-2,  # DisMech timestep
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
        """Create a physics configuration for testing."""
        return PhysicsConfig(
            num_segments=10,
            dt=5e-2,  # DisMech timestep
            max_iter=25,
            tol=1e-4,
            ftol=1e-4,
            dtol=1e-2,
            snake_radius=0.001,
            youngs_modulus=2e6,
            density=1200.0,
            poisson_ratio=0.5,
            use_rft=True,
            rft_ct=0.01,
            rft_cn=0.1,
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
        config = PhysicsConfig(
            num_segments=10,
            dt=5e-2,
            max_iter=25,
            snake_radius=0.001,
            youngs_modulus=2e6,
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
        config = PhysicsConfig(
            num_segments=10,
            dt=5e-2,
            max_iter=25,
            snake_radius=0.001,
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
