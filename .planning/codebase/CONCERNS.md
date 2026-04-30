# Codebase Concerns

**Analysis Date:** 2026-03-09

## Tech Debt

**Duplicated SnakeGeometryAdapter across 4 physics backends:**
- Issue: The `get_curvatures()`, `get_segment_vectors()`, `get_segment_lengths()`, and property implementations are copy-pasted identically across 4 separate adapter classes with no shared base class or mixin.
- Files:
  - `src/physics/snake_robot.py` (`SnakeGeometryAdapter`, lines 84-166)
  - `src/physics/elastica_snake_robot.py` (`ElasticaSnakeGeometryAdapter`, lines 96-179)
  - `src/physics/dismech_rods_snake_robot.py` (`DismechRodsSnakeGeometryAdapter`, lines 30+)
  - `src/physics/mujoco_snake_robot.py` (`MujocoSnakeGeometryAdapter`, lines 151+)
- Impact: A curvature computation bug or feature change must be fixed in 4 places. The `get_curvatures()` method is ~30 lines duplicated 4 times. A 5th copy exists in `src/physics/geometry.py` (line 73).
- Fix approach: Extract a shared `BaseSnakeGeometryAdapter` base class in `src/physics/geometry.py` with common implementations. Each backend adapter inherits and only overrides the constructor (which differs by input format).

**Duplicated SnakeSimulator class definition:**
- Issue: `SnakeSimulator` (PyElastica simulator container) is defined separately in two files with slightly different base class mixins.
- Files:
  - `src/physics/elastica_snake_robot.py:91` - `SnakeSimulator(BaseSystemCollection, Constraints, Forcing, Damping)`
  - `locomotion_elastica/env.py:54` - `SnakeSimulator(BaseSystemCollection, Constraints, Forcing, Damping, Connections, CallBacks)`
- Impact: Bug fixes or changes to the simulator must be applied in both locations. The `locomotion_elastica` version has extra mixins (`Connections`, `CallBacks`) that the `src` version lacks.
- Fix approach: Consolidate into a single shared `SnakeSimulator` in `src/physics/elastica_snake_robot.py` with all mixins, and import it in `locomotion_elastica/env.py`.

**Duplicated observation extraction logic:**
- Issue: The observation computation (curvature modes via FFT, heading, CoM velocity, goal-relative features) is reimplemented in 3 separate places with different backends (NumPy in `src/physics/`, NumPy in `locomotion_elastica/env.py`, vectorized PyTorch in `aprx_model_elastica/env.py`).
- Files:
  - `src/physics/elastica_snake_robot.py` - `_get_reduced_observation()` and variants
  - `locomotion_elastica/env.py` - `_compute_obs()` (lines ~300+)
  - `aprx_model_elastica/env.py` - `_compute_obs_batch()` (lines 249-323)
- Impact: Observation semantics can drift between the real environment and surrogate environment, causing silent RL training bugs. Any change to observation layout must be synchronized across all three.
- Fix approach: Create a shared observation computation module that both environments import. The surrogate env can have a PyTorch-native vectorized version but must be validated against the NumPy reference.

**No abstract interface for physics backends:**
- Issue: The 4 physics backends (`SnakeRobot`, `ElasticaSnakeRobot`, `DismechRodsSnakeRobot`, `MujocoSnakeRobot`) share the same API surface (`reset()`, `step()`, `set_curvature_control()`, `get_state()`, `get_observation()`, `get_energy()`) but there is no abstract base class or Protocol defining this contract.
- Files:
  - `src/physics/snake_robot.py` (676 lines)
  - `src/physics/elastica_snake_robot.py` (625 lines)
  - `src/physics/dismech_rods_snake_robot.py` (517 lines)
  - `src/physics/mujoco_snake_robot.py` (647 lines)
  - `src/physics/__init__.py` - factory `create_snake_robot()` (lines 20-46)
- Impact: No compile-time or static analysis guarantee that new backends implement all required methods. Changes to the interface require manual checking across all 4 implementations.
- Fix approach: Define a `SnakeRobotProtocol` (runtime-checkable `Protocol` or ABC) in `src/physics/base.py` and have each backend inherit or satisfy it.

**HRL environment modules referenced but missing:**
- Issue: `src/envs/__init__.py` imports `ApproachEnv`, `CoilEnv`, and `HRLEnv` from modules that do not exist as files (`approach_env.py`, `coil_env.py`, `hrl_env.py`). The import is wrapped in a `try/except ImportError: pass` that silently swallows the failure.
- Files:
  - `src/envs/__init__.py` (lines 5-10)
  - `src/envs/` directory contains only `__init__.py` and `base_env.py`
  - `src/trainers/hrl.py` (lines 22-24) imports these non-existent modules
- Impact: The HRL trainer (`src/trainers/hrl.py`) will fail at runtime when instantiated because `ApproachEnv`, `CoilEnv`, and `HRLEnv` are imported directly (not wrapped in try/except). The entire HRL training pipeline is non-functional.
- Fix approach: Either implement the missing environment files or stub them with clear "not yet implemented" errors. The `HRLTrainer` is 671 lines of code that cannot be run.

**`_update_skill()` in HRL trainer is a no-op stub:**
- Issue: The `_update_skill()` method returns `{"updated": True}` without performing any actual skill policy update. The joint training mode silently does nothing for skill updates.
- Files: `src/trainers/hrl.py` (lines 435-443)
- Impact: Joint HRL training strategy silently fails to update skill policies. Only the sequential training strategy actually trains skills (via separate PPOTrainer instances).
- Fix approach: Implement proper skill-specific data filtering and PPO updates within `_update_skill()`, or explicitly raise `NotImplementedError` so the joint strategy is clearly marked as incomplete.

## Known Bugs

**`torch.load()` without `weights_only=True` in 4 trainers:**
- Symptoms: PyTorch 2.6+ emits `FutureWarning` on every checkpoint load. Will eventually become an error in future PyTorch versions.
- Files:
  - `src/trainers/ppo.py:518` - `torch.load(path, map_location=self.device)` (no `weights_only`)
  - `src/trainers/hrl.py:621` - same
  - `src/trainers/ddpg.py:458` - same
  - `src/trainers/behavioral_cloning.py:269` - same
- Trigger: Any call to `load_checkpoint()` in these trainers.
- Workaround: `src/trainers/sac.py:481` already uses `weights_only=False` explicitly. All should use `weights_only=True` where possible for security, or `weights_only=False` explicitly to suppress the warning.

**File descriptor leak in atomic save on failure:**
- Symptoms: If `torch.save()` succeeds but `os.rename()` fails, `os.close(fd)` runs in the `finally` block AFTER `os.unlink(temp_path)` has already removed the file. The fd is closed correctly, but the ordering of unlink vs close is reversed from the expected pattern. More critically, if `torch.save()` itself raises and the except block's `os.unlink()` also raises, the fd leaks.
- Files:
  - `src/trainers/ppo.py` (lines 502-510)
  - `src/trainers/hrl.py` (lines 607-615)
- Trigger: Disk full conditions, permissions errors during checkpoint save.
- Workaround: None. Edge case but can cause resource exhaustion in long-running training.

**Global `np.random.seed()` in `_set_seed()`:**
- Symptoms: Setting seed via `np.random.seed()` affects the global NumPy RNG state, which can cause unexpected seed interactions between parallel environments or other modules using NumPy random functions.
- Files:
  - `src/envs/base_env.py:325` - `np.random.seed(seed)`
  - `src/behavioral_cloning/generators.py:188` - `np.random.seed(seed)`
  - `src/behavioral_cloning/curvature_action_experiences.py:684` - `np.random.seed(seed)`
- Trigger: Any call to `_set_seed()` or `generate_*()` methods.
- Workaround: The `locomotion_elastica/env.py` already uses the correct pattern: `np.random.default_rng(seed)` with a local generator. Apply the same pattern to all other seed sites.

## Security Considerations

**Unsafe `torch.load()` enables arbitrary code execution:**
- Risk: `torch.load()` without `weights_only=True` uses Python `pickle` deserialization, which can execute arbitrary code embedded in checkpoint files. Loading a malicious `.pt` file could compromise the system.
- Files:
  - `src/trainers/ppo.py:518`
  - `src/trainers/hrl.py:621`
  - `src/trainers/ddpg.py:458`
  - `src/trainers/behavioral_cloning.py:269`
  - Several files in `bing2019/`, `zheng2022/`, `locomotion_elastica/` also use `weights_only=False`
- Current mitigation: Checkpoints are only loaded from local `output/` directory, not from external sources.
- Recommendations: Add `weights_only=True` to all `torch.load()` calls in core trainers. For checkpoints that include non-tensor data (like config dataclasses), serialize configs separately as JSON and only load tensor state dicts with `weights_only=True`.

**Credentials referenced in CLAUDE.md and `.netrc`:**
- Risk: W&B credentials are stored in `~/.netrc`. Docker Hub token referenced as env var `DOCKER_TOKEN`. HuggingFace token as `HF_TOKEN`.
- Files: `CLAUDE.md` (credentials section), `~/.netrc` (W&B API key)
- Current mitigation: `.netrc` is not in the repo. Env vars are not committed.
- Recommendations: Document required secrets in a non-committed file. Ensure `.netrc` is in `.gitignore` (it is not currently listed there, though the file is outside the repo root).

## Performance Bottlenecks

**RFT friction Python loop in `elastica_snake_robot.py`:**
- Problem: The `RFTForcing.apply_forces()` method uses a Python `for` loop (line 74) to interpolate tangent vectors to nodes, which is called at every substep (500 substeps per RL step by default).
- Files: `src/physics/elastica_snake_robot.py` (lines 59-88, especially 74-76)
- Cause: The loop `for i in range(1, n_nodes - 1)` iterates over interior nodes in pure Python. With 21 nodes and 500 substeps per RL step, this loop executes ~9,500 times per step.
- Improvement path: Replace the Python loop with vectorized NumPy: `node_tangents[:, 1:-1] = 0.5 * (tangents[:, :-1] + tangents[:, 1:])`. The `locomotion_elastica/env.py` already implements a fully vectorized version (`AnisotropicRFTForce`, lines 61-94) that avoids this loop entirely.

**PyElastica simulation is CPU-bound, single-threaded:**
- Problem: Each RL step requires 500 PyElastica substeps (configurable), each involving force computation, position updates, and damping. This dominates wall-clock time at ~57 FPS with 16 parallel environments.
- Files:
  - `src/physics/elastica_snake_robot.py` (lines 411-428, the substep loop)
  - `locomotion_elastica/env.py` (lines ~350+, same pattern)
- Cause: PyElastica's explicit symplectic integrator is written in Python/NumPy without GPU acceleration or JIT compilation. 500 substeps per RL step is the minimum for numerical stability with the current rod parameters.
- Improvement path: The `aprx_model_elastica/` surrogate model replaces the 500-substep loop with a single MLP forward pass. This is the correct long-term approach. Short-term: reduce substeps if stability permits, or use the C++ `dismech-rods` backend.

**Checkpoint save overhead on every best reward:**
- Problem: `save_checkpoint("best")` is called every time `mean_episode_reward > self.best_reward`, which can be frequent early in training. Each save includes full actor, critic, and optimizer state dicts plus backup copy.
- Files: `src/trainers/ppo.py` (lines 246-248)
- Cause: No cooldown or minimum improvement threshold before saving.
- Improvement path: Add a minimum improvement delta (e.g., save only if `reward > best_reward + threshold`) or a time-based cooldown between best checkpoint saves.

## Fragile Areas

**Physics parameter sensitivity:**
- Files:
  - `src/configs/physics.py` (all parameter defaults)
  - `locomotion_elastica/config.py` (separate defaults for locomotion)
  - `issues/elastica-rod-radius-too-small.md`, `issues/elastica-damping-too-high.md`
- Why fragile: The physics simulation has known sensitivities documented in the issues directory. Rod radius 0.001m (vs correct 0.02m) causes zero/chaotic motion because EI scales as r^4. Young's modulus 2e6 (vs correct 1e5) causes instability. Damping too high freezes the rod. These are all interrelated and changing one parameter can require retuning others.
- Safe modification: When changing any physics parameter, run the diagnostic script first (`locomotion_elastica/diagnose.py`) and verify locomotion speed is non-zero. Always cross-reference with the verified working parameters documented in CLAUDE.md memory.
- Test coverage: `tests/test_physics.py` (757 lines) covers basic physics operations but does not test parameter sensitivity boundaries.

**Observation dimension coupling:**
- Files:
  - `src/envs/base_env.py:66-67` - `obs_shape = (self.config.obs_dim,)` — hardcoded from config
  - `src/physics/elastica_snake_robot.py` - observation builders return variable-length arrays
  - `src/configs/env.py` - `obs_dim` field that must match the physics backend's output
- Why fragile: The observation dimension is specified in the config but computed dynamically by the physics backend. If `num_segments` changes, or a new feature is added to an observation representation, the config `obs_dim` must be manually updated. A mismatch causes silent shape errors in network construction.
- Safe modification: Always verify `obs_dim` matches the actual output of `sim.get_observation()` after any change. Add an assertion in `BaseSnakeEnv.__init__()` that checks `obs_dim == sim.get_observation().shape[0]`.
- Test coverage: `tests/test_envs.py` tests basic env operations but does not verify observation dimension consistency across representations.

**Surrogate model-environment consistency:**
- Files:
  - `aprx_model_elastica/env.py` - `SurrogateLocomotionEnv` (14-dim obs, GPU-batched)
  - `locomotion_elastica/env.py` - `LocomotionElasticaEnv` (14-dim obs, CPU single)
  - `aprx_model_elastica/state.py` - state layout constants (`POS_X`, `POS_Y`, etc.)
- Why fragile: The surrogate environment must produce identical observations and rewards as the real environment for RL transfer to work. The observation is reimplemented in vectorized PyTorch vs scalar NumPy, with different code paths for curvature mode extraction (FFT), heading computation, and velocity calculation. Any bug in one but not the other causes silent distribution shift.
- Safe modification: After any change to observation computation, run `aprx_model_elastica/validate.py` to compare surrogate predictions against ground truth. Add automated regression tests that compare observations from both environments given identical state inputs.
- Test coverage: `aprx_model_elastica/validate.py` exists but is a manual script, not an automated test.

## Scaling Limits

**PyElastica single-run throughput:**
- Current capacity: ~57 FPS with 16 parallel environments
- Limit: ~16 workers optimal due to L3 cache contention (see `issues/parallel-collection-scaling-bottleneck.md`). Beyond 16 workers, throughput plateaus.
- Scaling path: The neural surrogate model (`aprx_model_elastica/`) sidesteps this entirely by replacing physics simulation with an MLP forward pass on GPU. Target: 1000x+ FPS with GPU batching.

**Surrogate data collection for training:**
- Current capacity: ~270 FPS at 16 workers for data collection
- Limit: Collecting diverse training data is slow because each trajectory requires full PyElastica simulation
- Scaling path: Use space-filling sampling (Sobol sequences) instead of policy-dependent rollouts for better coverage with fewer samples (see `issues/surrogate-model-data-imbalance.md`).

## Dependencies at Risk

**PyElastica version pinning:**
- Risk: `pyelastica>=0.3.0` is a loose lower bound. PyElastica's API has changed across versions (import paths, class names, stepper interfaces). The codebase imports from `elastica` directly and uses internal APIs like `extend_stepper_interface` and `CosseratRod.straight_rod` which may change.
- Impact: A PyElastica upgrade could break the physics simulation without warning.
- Migration plan: Pin to a specific version (e.g., `pyelastica==0.3.2`). Add a CI test that runs physics simulation to catch API breaks early.

**TorchRL API instability:**
- Risk: TorchRL is a rapidly evolving library. The codebase already has compatibility shims for v0.11 API changes (import renames: `BoundedTensorSpec` vs `Bounded`, `CompositeSpec` vs `Composite`). Six resolved issues in `issues/torchrl-v011-*.md` document past breakages.
- Impact: TorchRL upgrades can break environment specs, collector behavior, and loss module interfaces.
- Migration plan: The existing try/except import pattern in `locomotion_elastica/env.py` (lines 22-32) handles the known rename. Apply the same pattern to `src/envs/base_env.py` which currently uses the old names only.

**DisMech Python dependency from Git:**
- Risk: `dismech-python` is installed from a Git repository (`github.com/StructuresComp/dismech-python.git`). No version pinning. The upstream repo could introduce breaking changes at any time.
- Impact: `pip install` may pull incompatible versions. The package is used by the primary DisMech physics backend.
- Migration plan: Pin to a specific commit hash in `pyproject.toml`: `dismech-python = {git = "...", rev = "abc123"}`.

## Missing Critical Features

**No automated test for surrogate-environment parity:**
- Problem: The surrogate environment (`aprx_model_elastica/env.py`) is hand-validated against the real environment, but there is no automated test ensuring they produce equivalent observations and rewards given identical inputs.
- Blocks: Confident RL training on the surrogate model. Observation drift between real and surrogate environments can cause policy transfer failure.

**No checkpoint resume for PPO training:**
- Problem: `PPOTrainer.load_checkpoint()` loads model weights but does not resume the data collector's internal state, step counter for the training loop, or LR scheduler state. Training cannot be cleanly resumed from a checkpoint.
- Files: `src/trainers/ppo.py` (lines 512-525)
- Blocks: Long training runs that need to survive interruptions.

**No type checking or linting CI:**
- Problem: `pyproject.toml` configures `ruff` and `black` but there is no CI pipeline configuration. No `mypy` or `pyright` configuration for type checking despite extensive use of type hints.
- Files: `pyproject.toml` (lines 73-85)
- Blocks: Catching type errors, import issues, and style violations before they reach production.

## Test Coverage Gaps

**No tests for surrogate model (`aprx_model_elastica/`):**
- What's not tested: The entire surrogate model pipeline: data collection, dataset loading, model training, state normalization, and the surrogate environment.
- Files: `aprx_model_elastica/*.py` (10 modules, ~2,700 lines total)
- Risk: The surrogate is a critical new capability (neural replacement for physics simulation). Bugs in state normalization or delta prediction cause silent RL training failures.
- Priority: High

**No tests for locomotion_elastica environment:**
- What's not tested: The actively-used locomotion training environment (`locomotion_elastica/env.py`) has test scaffolding in `tests/test_locomotion.py` but the tests require the physics backend to run and are not validated in any CI.
- Files: `locomotion_elastica/env.py` (656 lines), `tests/test_locomotion.py` (237 lines)
- Risk: Regression in reward computation, observation layout, or episode termination logic.
- Priority: High

**HRL trainer is untested and non-functional:**
- What's not tested: `src/trainers/hrl.py` (671 lines) has no test file. The environments it depends on (`ApproachEnv`, `CoilEnv`, `HRLEnv`) do not exist.
- Files: `src/trainers/hrl.py`, missing `src/envs/approach_env.py`, `src/envs/coil_env.py`, `src/envs/hrl_env.py`
- Risk: 671 lines of dead code that will fail at runtime. The `_update_skill()` method is a stub.
- Priority: Medium (not actively used, but represents significant wasted investment)

**No integration tests for training pipeline:**
- What's not tested: End-to-end training (env creation -> data collection -> PPO update -> checkpoint save/load) is not tested as an integrated flow.
- Files: `src/trainers/ppo.py`, `src/envs/base_env.py`, `src/networks/actor.py`, `src/networks/critic.py`
- Risk: Interface mismatches between components (TensorDict key naming, spec shapes, loss module expectations) cause failures only discovered during actual training runs.
- Priority: High

---

*Concerns audit: 2026-03-09*
