# Testing Patterns

**Analysis Date:** 2026-03-09

## Test Framework

**Runner:**
- pytest >= 8.0.0
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`

**Assertion Library:**
- pytest assertions (`assert`, `pytest.approx`, `pytest.raises`)
- numpy testing: `np.testing.assert_allclose`, `np.testing.assert_array_equal`
- torch testing: `torch.testing.assert_close`

**Run Commands:**
```bash
python3 -m pytest                    # Run all tests
python3 -m pytest -v --tb=short      # Verbose with short tracebacks (default via addopts)
python3 -m pytest tests/test_physics.py  # Run specific test file
python3 -m pytest -k "TestGeometry"  # Run by class/name pattern
python3 -m pytest --cov=src          # With coverage (requires pytest-cov)
```

## Test File Organization

**Location:**
- Separate `tests/` directory at project root (not co-located with source)
- Test config in `pyproject.toml`: `testpaths = ["tests"]`

**Naming:**
- Files: `test_<module>.py` (e.g., `test_physics.py`, `test_envs.py`, `test_ddpg.py`)
- Classes: `Test<ComponentName>` (e.g., `TestGeometry`, `TestSnakeRobot`, `TestBaseSnakeEnv`)
- Methods: `test_<behavior>` (e.g., `test_robot_initialization`, `test_reward_finite`)
- Pattern: `python_files = ["test_*.py"]`

**Structure:**
```
tests/
├── __init__.py
├── test_physics.py         # Physics backends, geometry, friction models
├── test_envs.py            # TorchRL environment implementations
├── test_locomotion.py      # Locomotion env (bing2019 package)
├── test_navigation.py      # COBRA navigation env (jiang2024 package)
├── test_ddpg.py            # DDPG trainer, configs, logging utils
├── test_choi2025.py        # Choi et al. soft manipulator package
├── test_zheng2022.py       # Zheng et al. underwater snake package
├── test_liu2023.py         # Liu et al. CPG locomotion package
├── test_pipe_geometry.py   # Pipe wall geometry for DisMech
└── test_run_dir.py         # Run directory setup and console logging
```

## Test Structure

**Suite Organization:**
```python
"""Tests for physics simulation module with DisMech integration."""

import pytest
import numpy as np

from src.physics.snake_robot import SnakeRobot
from src.configs.physics import DismechConfig


class TestSnakeRobot:
    """Tests for SnakeRobot physics simulation with DisMech."""

    @pytest.fixture
    def config(self):
        """Create a DisMech physics configuration for testing."""
        return DismechConfig(
            geometry=GeometryConfig(num_segments=10, snake_radius=0.001),
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
        assert robot.time == 0.0

    def test_robot_step(self, robot):
        """Test single simulation step."""
        state = robot.step()
        assert "positions" in state
        assert np.all(np.isfinite(state["positions"]))
```

**Patterns:**
- **Class-based grouping**: Tests are grouped by component into `Test*` classes
- **Instance fixtures**: Config and object fixtures defined as instance methods with `self`
- **Fixture chaining**: `robot` fixture depends on `config` fixture
- **Docstrings on all test methods**: Brief description of what is being tested
- **Module-level docstring**: Describes the test file's purpose
- **Lightweight configs for tests**: Use small `num_segments=10` to keep tests fast

## Fixture Patterns

**Config Fixtures:**
```python
@pytest.fixture
def config(self):
    """Create a physics configuration for testing."""
    return DismechConfig(
        geometry=GeometryConfig(num_segments=10, snake_radius=0.001),
        dt=5e-2,
        density=1200.0,
    )
```

**Object Fixtures:**
```python
@pytest.fixture
def robot(self, config):
    """Create a snake robot for testing."""
    return SnakeRobot(config)
```

**Environment Fixtures with Cleanup:**
```python
@pytest.fixture
def env(self):
    config = LocomotionEnvConfig(task="power_velocity")
    e = PlanarSnakeEnv(config=config)
    yield e
    e.close()
```

**Temporary Directory Fixtures:**
```python
def test_creates_directory_structure(self, tmp_path):
    cfg = _DummyConfig(output=Output(base_dir=str(tmp_path)))
    run_dir = setup_run_dir(cfg, timestamp="20260101_120000")
    assert run_dir.is_dir()
```

**Dummy Configs for Isolated Tests:**
```python
@dataclass
class _DummyConfig:
    """Minimal config for testing setup_run_dir."""
    name: str = "test_run"
    output: Output = field(default_factory=Output)
```

## Conditional Test Skipping

**Optional Backend Skipping:**
```python
# Skip entire class if py_dismech not installed
_has_py_dismech = True
try:
    import py_dismech
except ImportError:
    _has_py_dismech = False

@pytest.mark.skipif(not _has_py_dismech, reason="py_dismech not installed")
class TestDismechRodsSnakeRobot:
    ...
```

**importorskip for Required Dependencies:**
```python
# Skip if mujoco not installed (assigns module or skips)
mujoco_module = pytest.importorskip("mujoco")
```

**Boolean Flag Pattern for Package Availability:**
```python
try:
    from jiang2024.configs_jiang2024 import CobraEnvConfig
    has_jiang2024 = True
except ImportError:
    has_jiang2024 = False

@pytest.mark.skipif(not has_jiang2024, reason="jiang2024 not available")
class TestDDPGTrainer:
    ...
```

**Module-Level pytestmark:**
```python
pytestmark = pytest.mark.skipif(
    not has_locomotion, reason="locomotion package not importable"
)
```

**Device-Conditional Tests:**
```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_env(self):
    env = BaseSnakeEnv(device="cuda")
    ...
```

## Mocking

**Framework:** No dedicated mocking framework. Tests use real objects, not mocks.

**Approach:**
- Tests instantiate actual physics simulations, environments, and networks
- Small configs (10 segments, short episodes) keep tests fast
- No mock objects for physics or neural networks
- W&B disabled in trainer tests: `config.wandb.enabled = False`
- Environment headless rendering: `os.environ.setdefault("MUJOCO_GL", "egl")`

**What NOT to Mock:**
- Physics simulations (tests verify actual physics behavior)
- Neural networks (tests verify shapes, bounds, initialization)
- Config dataclasses (tests verify defaults and inheritance)

## Assertion Patterns

**Numeric Precision:**
```python
# pytest.approx for scalar comparisons
assert obs[-1] == pytest.approx(0.1, abs=0.01)

# np.testing.assert_allclose for array comparisons
np.testing.assert_allclose(ring.center, center)
np.testing.assert_allclose(dists, 0.5, atol=1e-10)

# np.isclose for single values
assert np.isclose(energy["total"], energy["kinetic"] + energy["gravitational"] + energy["elastic"])

# torch.testing.assert_close for tensor comparisons
torch.testing.assert_close(td1["observation"], td2["observation"])
```

**Finiteness/NaN Checks:**
```python
assert np.all(np.isfinite(state["positions"]))
assert np.all(np.isfinite(state["velocities"])), f"NaN/Inf vel at step {i}"
assert np.isfinite(reward)
assert not torch.any(torch.isnan(obs))
```

**Shape Assertions:**
```python
assert ring.nodes.shape == (12, 3)
assert obs.shape == (27,)
assert action.shape == (1, 7)
assert td["observation"].shape == (21,)
```

**Dictionary Key Presence:**
```python
required_keys = ["positions", "velocities", "curvatures", ...]
for key in required_keys:
    assert key in state, f"Missing key: {key}"
```

**Dtype and Device Assertions:**
```python
assert obs.dtype == torch.float32
assert isinstance(obs, np.ndarray)
assert obs.dtype == np.float32
assert td["observation"].device.type == "cpu"
```

**Bounds Checking:**
```python
assert torch.all(actions >= -1.5)
assert torch.all(actions <= 1.5)
assert float(action_spec.space.low.min()) == -1.0
```

**Exception Testing:**
```python
with pytest.raises(NotImplementedError):
    DismechRodsSnakeRobot(config)

with pytest.raises(ValueError, match="Unknown track type"):
    gen.step("unknown", 0, 0, 4, 0, 0.05)
```

**Deprecation Warning Testing:**
```python
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    snake = create_snake_geometry(physics_config)
    assert len(w) == 1
    assert issubclass(w[0].category, DeprecationWarning)
```

## Test Types

**Unit Tests:**
- Config defaults and inheritance: `TestDDPGConfig`, `TestConfigs`
- Standalone computations: `TestFrictionForceComputation`, `TestRewardFunctions`
- Network output shapes and bounds: `TestDeterministicActor`, `TestComputeGradNorm`
- Geometry computations: `TestPerpendicularVector`, `TestMakeRing`
- Pure functions: `compute_barrier_normal_force()`, `compute_power_velocity_reward()`

**Integration Tests:**
- Physics simulation multi-step: `TestPhysicsStability` (run 5-20 steps, check finite)
- Environment reset/step cycles: `TestBaseSnakeEnv`, `TestApproachEnv`
- Trainer initialization with real environment: `TestDDPGTrainer`
- Factory function dispatch: `test_factory_creates_mujoco_robot()`
- Config save/load round-trip: `test_json_round_trip()`
- Pipe geometry added to simulation: `TestAddToSimulation`

**Smoke Tests:**
- `script/smoke_test_collect.py`: Pre-flight check for data collection pipeline
- Verifies small collection runs, validates data shapes, estimates full-run time

**E2E Tests:**
- Not used (no full training E2E tests due to long runtimes)

## Common Patterns

**Multi-Step Simulation Testing:**
```python
def test_simulation_stability(self, robot, config):
    """Test simulation stays stable with sinusoidal curvature over 20 steps."""
    for i in range(20):
        s = np.linspace(0, 2 * np.pi, config.num_segments - 1)
        curvatures = np.sin(s + i * 0.1) * 0.5
        robot.set_curvature_control(curvatures)
        state = robot.step()
        assert np.all(np.isfinite(state["positions"])), f"NaN/Inf at step {i}"
```

**Environment Step Loop Testing:**
```python
def test_episode_truncation(self, env):
    td = env.reset()
    for i in range(env.config.max_episode_steps):
        action = torch.zeros(8, dtype=torch.float32)
        td["action"] = action
        td = env.step(td)
        td = td["next"]
    assert td["truncated"].item() is True
```

**Parametric Model Testing (all friction models):**
```python
def test_energy_finite_all_models(self, base_geom):
    for model in [FrictionModel.RFT, FrictionModel.COULOMB, FrictionModel.STRIBECK, FrictionModel.NONE]:
        config = DismechConfig(
            geometry=base_geom, dt=5e-2,
            friction=FrictionConfig(model=model),
        )
        robot = SnakeRobot(config)
        for _ in range(3):
            robot.step()
        energy = robot.get_energy()
        assert np.isfinite(energy["total"]), f"Non-finite energy for {model}"
```

**Reproducibility Testing:**
```python
def test_deterministic_reset(self):
    config = EnvConfig(randomize_initial_state=False, randomize_prey_position=False)
    env1 = BaseSnakeEnv(config=config)
    env2 = BaseSnakeEnv(config=config)
    td1 = env1.reset()
    td2 = env2.reset()
    torch.testing.assert_close(td1["observation"], td2["observation"])
```

**Convergence/Dynamics Testing:**
```python
def test_amplitude_convergence(self):
    """Amplitude should converge to target R."""
    cpg = BingCPG(n=3, a=20.0)
    R_target = 0.8
    for _ in range(5000):
        output = cpg.step(dt=0.01, R=R_target, omega=2.0, theta=0.5, delta=0.0)
    assert np.allclose(cpg.r, R_target, atol=0.1)
```

## Coverage

**Requirements:** No enforced coverage target. `pytest-cov >= 6.0.0` available as dev dependency.

**View Coverage:**
```bash
python3 -m pytest --cov=src --cov-report=term-missing
python3 -m pytest --cov=src --cov-report=html
```

## Test Gaps and Guidance

**What Has Good Coverage:**
- Physics backends (DisMech, MuJoCo, dismech-rods): stability, state keys, initialization, reset
- Friction models: all combinations tested across backends
- TorchRL environments: reset, step, specs, truncation, seeding, batching, devices
- Config dataclasses: defaults, inheritance, backward compatibility
- Reference implementations: each has dedicated test file

**What Lacks Coverage:**
- Training loops (PPO, DDPG, SAC, HRL): only DDPG has a trainer test, and it only tests init/select_action/soft_update
- Reward shaping (PBRS): `src/rewards/shaping.py` has no dedicated tests
- Observation extractors: `src/observations/` has no dedicated tests
- Behavioral cloning: `src/behavioral_cloning/` has no tests
- Surrogate model training: `aprx_model_elastica/` has no pytest tests (only smoke script)

**When Writing New Tests:**
- Place in `tests/test_<module>.py`
- Group related tests in `Test<Component>` classes
- Use instance fixtures (`self`) for config and object setup
- Use small configs (10 segments, short episodes) for speed
- Always check `np.isfinite()` after physics steps
- Skip tests for optional backends with `pytest.importorskip()` or `@pytest.mark.skipif`
- Clean up MuJoCo envs with `yield` + `env.close()` pattern
- Set `MUJOCO_GL=egl` at module level for headless rendering

---

*Testing analysis: 2026-03-09*
