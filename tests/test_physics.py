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
    FrictionModel,
    FrictionConfig,
    DismechConfig,
    DismechRodsConfig,
    ElasticaConfig,
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


# ---------------------------------------------------------------------------
# Friction model tests
# ---------------------------------------------------------------------------


class TestFrictionConfig:
    """Tests for FrictionConfig and backward compatibility."""

    def test_default_friction_is_rft(self):
        """RodConfig defaults to RFT friction."""
        config = DismechConfig()
        assert config.friction.model == FrictionModel.RFT
        assert config.friction.rft_ct == 0.01
        assert config.friction.rft_cn == 0.1

    def test_dismech_rods_default_is_native(self):
        """DismechRodsConfig defaults to NATIVE friction."""
        config = DismechRodsConfig()
        assert config.friction.model == FrictionModel.NATIVE

    def test_mujoco_default_is_native(self):
        """MujocoPhysicsConfig defaults to NATIVE friction."""
        config = MujocoPhysicsConfig()
        assert config.friction.model == FrictionModel.NATIVE

    def test_backward_compat_use_rft_false(self):
        """Setting use_rft=False migrates to FrictionModel.NONE."""
        config = DismechConfig(use_rft=False)
        assert config.friction.model == FrictionModel.NONE

    def test_backward_compat_rft_params(self):
        """Custom rft_ct/rft_cn migrate into FrictionConfig."""
        config = DismechConfig(rft_ct=0.05, rft_cn=0.5)
        assert config.friction.model == FrictionModel.RFT
        assert config.friction.rft_ct == 0.05
        assert config.friction.rft_cn == 0.5

    def test_explicit_friction_overrides_deprecated(self):
        """Explicit FrictionConfig takes precedence over deprecated fields."""
        fc = FrictionConfig(model=FrictionModel.COULOMB, mu_kinetic=0.4)
        config = DismechConfig(friction=fc, use_rft=False)
        # Explicit friction wins — it's not the default so __post_init__ won't override
        assert config.friction.model == FrictionModel.COULOMB
        assert config.friction.mu_kinetic == 0.4


class TestFrictionForceComputation:
    """Tests for standalone friction force computation functions."""

    def test_barrier_normal_force_above_ground(self):
        """No normal force when well above ground."""
        from physics.friction import compute_barrier_normal_force
        z = np.array([1.0, 0.5, 0.1])
        f = compute_barrier_normal_force(z, stiffness=50000.0, delta=0.01)
        np.testing.assert_allclose(f, 0.0, atol=1e-6)

    def test_barrier_normal_force_at_ground(self):
        """Positive normal force when near/below ground."""
        from physics.friction import compute_barrier_normal_force
        z = np.array([0.0, -0.005, -0.01])
        f = compute_barrier_normal_force(z, stiffness=50000.0, delta=0.01)
        assert np.all(f > 0)
        # Force should increase as z decreases
        assert f[1] > f[0]
        assert f[2] > f[1]

    def test_coulomb_force_shape(self):
        """Coulomb force returns correct shape."""
        from physics.friction import compute_coulomb_force
        config = FrictionConfig(model=FrictionModel.COULOMB)
        positions = np.array([[0, 0, 0.0], [0.1, 0, 0.0], [0.2, 0, 0.0]])
        velocities = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0]])
        forces = compute_coulomb_force(positions, velocities, config)
        assert forces.shape == (3, 3)
        # Normal forces should be positive (upward) at z=0
        assert np.all(forces[:, 2] > 0)

    def test_stribeck_force_shape(self):
        """Stribeck force returns correct shape."""
        from physics.friction import compute_stribeck_force
        config = FrictionConfig(model=FrictionModel.STRIBECK)
        positions = np.array([[0, 0, 0.0], [0.1, 0, 0.0]])
        velocities = np.array([[0.1, 0.05, 0], [-0.1, 0, 0]])
        forces = compute_stribeck_force(positions, velocities, config)
        assert forces.shape == (2, 3)

    def test_stribeck_higher_friction_at_low_speed(self):
        """Stribeck model has higher friction at low speed (mu_static > mu_kinetic)."""
        from physics.friction import compute_stribeck_force
        config = FrictionConfig(
            model=FrictionModel.STRIBECK,
            mu_kinetic=0.3,
            mu_static=0.5,
            stribeck_velocity=0.01,
        )
        pos = np.array([[0, 0, 0.0]])

        # Low speed
        vel_low = np.array([[0.001, 0, 0]])
        f_low = compute_stribeck_force(pos, vel_low, config)

        # High speed
        vel_high = np.array([[1.0, 0, 0]])
        f_high = compute_stribeck_force(pos, vel_high, config)

        # Friction force per unit speed should be higher at low speed
        # (due to mu_static > mu_kinetic)
        friction_ratio_low = np.abs(f_low[0, 0]) / (np.abs(vel_low[0, 0]) + 1e-12)
        friction_ratio_high = np.abs(f_high[0, 0]) / (np.abs(vel_high[0, 0]) + 1e-12)
        assert friction_ratio_low > friction_ratio_high


class TestDismechFrictionModels:
    """Tests for DisMech backend with different friction models."""

    @pytest.fixture
    def base_geom(self):
        return GeometryConfig(num_segments=10, snake_radius=0.001)

    def test_rft_default(self, base_geom):
        """Default RFT friction works (backward compat)."""
        config = DismechConfig(geometry=base_geom, dt=5e-2)
        robot = SnakeRobot(config)
        for _ in range(5):
            state = robot.step()
        assert np.all(np.isfinite(state["positions"]))

    def test_coulomb_friction(self, base_geom):
        """Coulomb friction model runs without errors."""
        config = DismechConfig(
            geometry=base_geom,
            dt=5e-2,
            friction=FrictionConfig(model=FrictionModel.COULOMB),
        )
        robot = SnakeRobot(config)
        for _ in range(5):
            state = robot.step()
        assert np.all(np.isfinite(state["positions"]))

    def test_stribeck_friction(self, base_geom):
        """Stribeck friction model runs without errors."""
        config = DismechConfig(
            geometry=base_geom,
            dt=5e-2,
            friction=FrictionConfig(model=FrictionModel.STRIBECK),
        )
        robot = SnakeRobot(config)
        for _ in range(5):
            state = robot.step()
        assert np.all(np.isfinite(state["positions"]))

    def test_none_friction(self, base_geom):
        """NONE friction model runs without errors."""
        config = DismechConfig(
            geometry=base_geom,
            dt=5e-2,
            friction=FrictionConfig(model=FrictionModel.NONE),
        )
        robot = SnakeRobot(config)
        for _ in range(5):
            state = robot.step()
        assert np.all(np.isfinite(state["positions"]))

    def test_energy_finite_all_models(self, base_geom):
        """Energy is finite for all supported DisMech friction models."""
        for model in [FrictionModel.RFT, FrictionModel.COULOMB, FrictionModel.STRIBECK, FrictionModel.NONE]:
            config = DismechConfig(
                geometry=base_geom,
                dt=5e-2,
                friction=FrictionConfig(model=model),
            )
            robot = SnakeRobot(config)
            for _ in range(3):
                robot.step()
            energy = robot.get_energy()
            assert np.isfinite(energy["total"]), f"Non-finite energy for {model}"


class TestMujocoFrictionModels:
    """Tests for MuJoCo backend with different friction models."""

    @pytest.fixture
    def base_geom(self):
        return GeometryConfig(num_segments=10, snake_radius=0.005, snake_length=1.0)

    def test_native_default(self, base_geom):
        """NATIVE friction works (default for MuJoCo)."""
        from physics.mujoco_snake_robot import MujocoSnakeRobot
        config = MujocoPhysicsConfig(geometry=base_geom, dt=5e-2)
        robot = MujocoSnakeRobot(config)
        for _ in range(5):
            state = robot.step()
        assert np.all(np.isfinite(state["positions"]))

    def test_none_friction(self, base_geom):
        """NONE friction sets zero friction in MuJoCo."""
        from physics.mujoco_snake_robot import MujocoSnakeRobot
        config = MujocoPhysicsConfig(
            geometry=base_geom,
            dt=5e-2,
            friction=FrictionConfig(model=FrictionModel.NONE),
        )
        robot = MujocoSnakeRobot(config)
        for _ in range(5):
            state = robot.step()
        assert np.all(np.isfinite(state["positions"]))

    def test_coulomb_mapped(self, base_geom):
        """Coulomb friction maps mu_kinetic to MuJoCo friction parameter."""
        from physics.mujoco_snake_robot import MujocoSnakeRobot
        config = MujocoPhysicsConfig(
            geometry=base_geom,
            dt=5e-2,
            friction=FrictionConfig(model=FrictionModel.COULOMB, mu_kinetic=0.5),
        )
        robot = MujocoSnakeRobot(config)
        for _ in range(5):
            state = robot.step()
        assert np.all(np.isfinite(state["positions"]))

    def test_energy_finite_all_models(self, base_geom):
        """Energy is finite for all supported MuJoCo friction models."""
        from physics.mujoco_snake_robot import MujocoSnakeRobot
        for model in [FrictionModel.NATIVE, FrictionModel.COULOMB, FrictionModel.STRIBECK, FrictionModel.NONE]:
            config = MujocoPhysicsConfig(
                geometry=base_geom,
                dt=5e-2,
                friction=FrictionConfig(model=model, mu_kinetic=0.3),
            )
            robot = MujocoSnakeRobot(config)
            for _ in range(3):
                robot.step()
            energy = robot.get_energy()
            assert np.isfinite(energy["total"]), f"Non-finite energy for {model}"


@pytest.mark.skipif(not _has_py_dismech, reason="py_dismech not installed")
class TestDismechRodsFrictionModels:
    """Tests for dismech-rods backend friction model support."""

    @pytest.fixture
    def base_geom(self):
        return GeometryConfig(num_segments=10, snake_radius=0.001)

    def test_native_default(self, base_geom):
        """NATIVE friction works (default for dismech-rods)."""
        from physics.dismech_rods_snake_robot import DismechRodsSnakeRobot
        config = DismechRodsConfig(geometry=base_geom, dt=5e-2)
        robot = DismechRodsSnakeRobot(config)
        for _ in range(5):
            state = robot.step()
        assert np.all(np.isfinite(state["positions"]))

    def test_none_friction(self, base_geom):
        """NONE friction runs without ground forces."""
        from physics.dismech_rods_snake_robot import DismechRodsSnakeRobot
        config = DismechRodsConfig(
            geometry=base_geom,
            dt=5e-2,
            friction=FrictionConfig(model=FrictionModel.NONE),
        )
        robot = DismechRodsSnakeRobot(config)
        for _ in range(5):
            state = robot.step()
        assert np.all(np.isfinite(state["positions"]))

    def test_rft_raises_not_implemented(self, base_geom):
        """RFT raises NotImplementedError for dismech-rods."""
        from physics.dismech_rods_snake_robot import DismechRodsSnakeRobot
        config = DismechRodsConfig(
            geometry=base_geom,
            dt=5e-2,
            friction=FrictionConfig(model=FrictionModel.RFT),
        )
        with pytest.raises(NotImplementedError):
            DismechRodsSnakeRobot(config)

    def test_coulomb_raises_not_implemented(self, base_geom):
        """Coulomb raises NotImplementedError for dismech-rods."""
        from physics.dismech_rods_snake_robot import DismechRodsSnakeRobot
        config = DismechRodsConfig(
            geometry=base_geom,
            dt=5e-2,
            friction=FrictionConfig(model=FrictionModel.COULOMB),
        )
        with pytest.raises(NotImplementedError):
            DismechRodsSnakeRobot(config)
