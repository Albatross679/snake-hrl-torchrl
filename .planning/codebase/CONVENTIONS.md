# Coding Conventions

**Analysis Date:** 2026-03-09

## Naming Patterns

**Files:**
- Use `snake_case.py` for all Python modules: `snake_robot.py`, `base_env.py`, `logging_utils.py`
- Config files named by domain: `physics.py`, `training.py`, `network.py`, `env.py`, `geometry.py`
- Test files prefixed with `test_`: `test_physics.py`, `test_envs.py`, `test_ddpg.py`
- Reference implementation packages use author/year naming: `bing2019/`, `jiang2024/`, `liu2023/`, `zheng2022/`, `choi2025/`
- Within reference packages, files use `<topic>_<author><year>.py` suffix: `env_jiang2024.py`, `cpg_liu2023.py`, `rewards_zheng2022.py`

**Classes:**
- Use `PascalCase`: `SnakeRobot`, `PPOTrainer`, `BaseSnakeEnv`, `ActorNetwork`
- Config dataclasses end with `Config`: `PhysicsConfig`, `PPOConfig`, `EnvConfig`, `GeometryConfig`
- Enums use `PascalCase`: `SolverFramework`, `FrictionModel`, `ControlMethod`, `StateRepresentation`
- Adapter/wrapper classes describe their role: `SnakeGeometryAdapter`, `CPGEnvWrapper`
- Abstract base classes use `ABC` mixin: `PotentialFunction(ABC)`

**Functions:**
- Use `snake_case`: `create_snake_robot()`, `compute_barrier_normal_force()`, `get_activation()`
- Factory functions prefixed with `create_`: `create_actor()`, `create_critic()`, `create_snake_geometry()`
- Private methods prefixed with `_`: `_init_dismech()`, `_apply_curvature_to_dismech()`, `_make_spec()`
- Computation functions prefixed with `compute_`: `compute_wrap_angle()`, `compute_grad_norm()`, `compute_coulomb_force()`

**Variables:**
- Use `snake_case`: `obs_dim`, `action_dim`, `hidden_dims`, `num_segments`
- Private instance variables prefixed with `_`: `self._device`, `self._current_state`, `self._step_count`
- Constants use `UPPER_SNAKE_CASE`: `TEMP_DIR`, `NUM_WORKERS`, `EXPECTED_STATE_DIM`
- Physics parameters use domain-specific abbreviations: `dt`, `rft_ct`, `rft_cn`, `mu_kinetic`

**Types/Enums:**
- Enum values use `UPPER_SNAKE_CASE`: `SolverFramework.DISMECH`, `FrictionModel.COULOMB`
- String enums inherit from `(str, Enum)` for JSON serialization: `class SolverFramework(str, Enum)`
- Type variables use single uppercase: `T = TypeVar("T")`

## Code Style

**Formatting:**
- Black formatter, 100-character line length
- Target Python 3.12: `target-version = ['py312']`
- Config in `pyproject.toml` under `[tool.black]`

**Linting:**
- Ruff linter, 100-character line length
- Rules: E (pycodestyle errors), F (pyflakes), I (isort), N (naming), W (warnings)
- E501 (line too long) explicitly ignored (defers to Black)
- Config in `pyproject.toml` under `[tool.ruff]`

## Import Organization

**Order:**
1. Standard library imports (`os`, `sys`, `time`, `math`, `json`, `signal`, `tempfile`, `shutil`)
2. Third-party libraries (`torch`, `numpy`, `scipy`, `tqdm`, `wandb`)
3. TorchRL/TensorDict imports (`torchrl.envs`, `torchrl.data`, `torchrl.modules`, `tensordict`)
4. Project imports (`src.configs`, `src.physics`, `src.networks`, `src.trainers`)
5. Relative imports (`.geometry`, `.actor`, `.logging_utils`)

**Path Aliases:**
- No path aliases configured. All imports use full dotted paths from project root.
- `src.configs.physics`, `src.physics.snake_robot`, `src.networks.actor`

**Conditional/Lazy Imports:**
- Use try/except for optional dependencies:
  ```python
  try:
      from torchrl.data import BoundedTensorSpec, CompositeSpec
  except ImportError:
      from torchrl.data import Bounded as BoundedTensorSpec, Composite as CompositeSpec
  ```
- Use `pytest.importorskip()` in tests for optional backends: `mujoco_module = pytest.importorskip("mujoco")`
- Use module-level `__getattr__` for lazy loading heavy dependencies:
  ```python
  def __getattr__(name):
      if name == "SnakeRobot":
          from .snake_robot import SnakeRobot
          globals()["SnakeRobot"] = SnakeRobot
          return SnakeRobot
      raise AttributeError(...)
  ```
- Test files use top-level try/except with boolean flags:
  ```python
  try:
      from jiang2024.configs_jiang2024 import CobraEnvConfig
      has_jiang2024 = True
  except ImportError:
      has_jiang2024 = False
  ```

## Error Handling

**Patterns:**
- Physics simulation errors caught with try/except and fallback:
  ```python
  try:
      elastic = float(self._time_stepper.compute_total_elastic_energy(...))
  except Exception:
      # Fallback computation
      elastic = 0.5 * stiffness * np.sum(stretch**2)
  ```
- Convergence failures print warnings rather than raising:
  ```python
  except ValueError as e:
      self._last_residual_norm = None
      print(f"DisMech step warning: {e}")
  ```
- Assertions for input validation in performance-critical code:
  ```python
  assert len(curvatures) == self.config.num_segments - 1
  ```
- `raise ValueError` for invalid enum/string selections:
  ```python
  if name.lower() not in activations:
      raise ValueError(f"Unknown activation: {name}")
  ```
- `raise NotImplementedError` for unsupported backend+feature combinations (tested with `pytest.raises`)

**Anti-pattern to avoid:**
- Do NOT use bare `except:` -- always catch specific exceptions or `Exception`
- Do NOT silently swallow errors -- at minimum print a warning

## Logging

**Framework:** `print()` statements and W&B (`wandb`) / TensorBoard for metrics

**Patterns:**
- Physics simulation uses `print(f"DisMech step warning: {e}")` for non-fatal errors
- Training loops use `tqdm` for progress bars
- Metrics logged via W&B and/or TensorBoard through `src/trainers/logging_utils.py`
- System metrics (CPU, RAM, GPU) collected via `collect_system_metrics()` in `src/trainers/logging_utils.py`
- Console output can be tee'd to file via `ConsoleLogger` context manager in `src/configs/console.py`

## Comments

**When to Comment:**
- Module-level docstrings describe purpose, supported backends, and architecture decisions
- Complex physics formulas get inline explanations:
  ```python
  # Curvature from angle between consecutive segments
  cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
  ```
- Section dividers use comment blocks with dashes:
  ```python
  # ---------------------------------------------------------------------------
  # Base physics config
  # ---------------------------------------------------------------------------
  ```
- Enum values have inline comments for brief descriptions:
  ```python
  save_frequency: int = 0  # epochs or steps; 0 = disabled
  ```

**Docstrings:**
- Use Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections
- Module docstrings describe purpose and list key exports
- Class docstrings describe the class role and key behaviors
- Method docstrings describe parameters and return values
- Example from `src/physics/snake_robot.py`:
  ```python
  def get_state(self) -> Dict[str, Any]:
      """Get current simulation state.

      Returns:
          Dictionary containing:
              - positions: Snake node positions (n_nodes, 3)
              - velocities: Snake node velocities (n_nodes, 3)
              ...
      """
  ```

## Function Design

**Size:** Functions generally 10-50 lines. Longer methods are split with private helpers (e.g., `_init_dismech()`, `_apply_curvature_to_dismech()`, `_update_snake_adapter()`).

**Parameters:**
- Use `Optional[X] = None` for optional parameters
- Config dataclasses preferred over many individual parameters:
  ```python
  def __init__(self, config: PhysicsConfig, initial_snake_position=None, ...)
  ```
- Default to sensible values: `config = config or PPOConfig()`

**Return Values:**
- Dict returns for multi-value physics state: `Dict[str, Any]` with documented keys
- numpy arrays for physics computations (float32 for observations)
- torch tensors for RL components
- Named tuple or dataclass for structured results in reference implementations

## Module Design

**Exports:**
- Use `__all__` in `__init__.py` to define public API:
  ```python
  __all__ = ["create_snake_robot", "SnakeRobot", "SnakeGeometry", ...]
  ```
- Factory functions as primary entry points: `create_snake_robot()`, `create_actor()`, `create_critic()`

**Barrel Files:**
- `__init__.py` files re-export key symbols from submodules
- `src/physics/__init__.py` re-exports geometry functions and CPG modules
- `src/observations/__init__.py` re-exports all feature extractors
- `src/configs/__init__.py` exists but is minimal

**Config Hierarchy:**
- Dataclass inheritance for config specialization:
  ```
  MLBaseConfig -> RLConfig -> PPOConfig / DDPGConfig / SACConfig / HRLConfig
  PhysicsConfig -> RodConfig -> DERConfig -> DismechConfig / DismechRodsConfig
  ```
- Composable config pieces via fields: `WandB`, `Output`, `Console`, `FrictionConfig`
- `__post_init__` for backward-compatibility migration of deprecated fields
- Backward-compat aliases: `TrainingConfig = RLConfig`
- Properties for derived values: `@property def dt_substep`, `@property def obs_dim`

## Type Annotations

**Patterns:**
- Full type annotations on function signatures (parameters and return types)
- `Optional[X]` for nullable parameters
- `Dict[str, Any]` for state dictionaries
- `Tuple[float, float, float]` for fixed-size tuples
- `List[int]` for variable-length lists
- `Type[T]` with `TypeVar` for generic factory functions
- Modern union syntax `str | Path` used in some newer code (`save_config`)

---

*Convention analysis: 2026-03-09*
