# Codebase Structure

**Analysis Date:** 2026-03-09

## Directory Layout

```
snake-hrl-torchrl/
├── src/                        # Core shared infrastructure (importable as `src.*`)
│   ├── __init__.py             # Package docstring, subpackage listing
│   ├── configs/                # Dataclass-based hierarchical configuration
│   ├── envs/                   # TorchRL base environment (approach, coil, HRL)
│   ├── networks/               # Actor-critic neural network architectures
│   ├── observations/           # Feature extractors for state representations
│   ├── physics/                # Snake physics simulation backends
│   │   └── cpg/                # Central Pattern Generator actuators
│   ├── rewards/                # Potential-based reward shaping
│   ├── trainers/               # Training loops (PPO, SAC, DDPG, HRL)
│   └── behavioral_cloning/     # Demo generation, buffers, BC pretraining
├── locomotion_elastica/        # Active: PyElastica locomotion experiment
├── aprx_model_elastica/        # Active: Neural surrogate for PyElastica
├── bing2019/                   # Reference impl: Bing et al. IJCAI 2019
├── choi2025/                   # Reference impl: Choi & Tong 2025
├── jiang2024/                  # Reference impl: Jiang et al. 2024 (HRL navigation)
├── liu2023/                    # Reference impl: Liu et al. 2023 (CPG locomotion)
├── zheng2022/                  # Reference impl: Zheng et al. 2022
├── locomotion/                 # Older locomotion experiment (DisMech backend)
├── tests/                      # Unit and integration tests
├── script/                     # Standalone shell scripts and benchmarks
├── scripts/ralph/              # Developer-specific scripts
├── data/                       # Datasets (surrogate training data, benchmarks)
├── output/                     # Training outputs (timestamped run directories)
├── model/                      # Saved model weights
├── figures/                    # Generated plots and figures
├── media/                      # Images, videos, GIFs
├── logs/                       # Documentation: change logs (one .md per entry)
├── experiments/                # Documentation: experiment records
├── issues/                     # Documentation: bug/issue tracking
├── knowledge/                  # Documentation: domain knowledge
├── references/                 # Documentation: external citations
├── tasks/                      # Documentation: PRDs and task specs
├── dismech-python/             # External dependency (git submodule, DO NOT MODIFY)
├── dismech-python-src/         # DisMech Python source (for development)
├── snakebot-gym/               # Custom gym environment (modify only if asked)
├── wandb/                      # W&B local run data (gitignored)
├── .planning/                  # GSD planning documents
├── pyproject.toml              # Package config, dependencies, tool settings
├── requirements.txt            # Pinned dependencies
├── CLAUDE.md                   # Project instructions for Claude
├── Dockerfile                  # Container build
├── docker-compose.yml          # Container orchestration
└── overview.tex                # Detailed project reference document
```

## Directory Purposes

**`src/configs/`:**
- Purpose: All configuration as composable Python dataclasses
- Contains: Base config, physics config hierarchy, env configs, network configs, training configs, project-level composed configs
- Key files:
  - `base.py`: `MLBaseConfig`, `WandB`, `Output`, `Console`, `resolve_device()`, `save_config()`, `load_config()`
  - `physics.py`: `SolverFramework` enum, `PhysicsConfig`, `ElasticaConfig`, `FrictionConfig`, `FrictionModel`
  - `env.py`: `EnvConfig`, `ControlMethod` enum, `StateRepresentation` enum, `ApproachEnvConfig`, `CoilEnvConfig`, `HRLEnvConfig`, `CPGConfig`
  - `training.py`: `RLConfig`, `PPOConfig`, `SACConfig`, `DDPGConfig`, `HRLConfig`
  - `network.py`: `ActorConfig`, `CriticConfig`, `NetworkConfig`, `HRLNetworkConfig`
  - `project.py`: `SnakeApproachConfig`, `SnakeCoilConfig`, `SnakeHRLConfig`
  - `geometry.py`: `GeometryConfig` (snake length, radius, segments)
  - `run_dir.py`: `setup_run_dir()` creates `output/<name>_<YYYYMMDD_HHMMSS>/`
  - `console.py`: `ConsoleLogger` context manager, `TeeStream`
  - `__init__.py`: Re-exports everything; conditionally imports configs from reference packages

**`src/physics/`:**
- Purpose: Physics simulation backends for snake robot dynamics
- Contains: Four interchangeable snake robot simulators, friction models, geometry utilities
- Key files:
  - `__init__.py`: `create_snake_robot()` factory, lazy `SnakeRobot` import
  - `elastica_snake_robot.py`: `ElasticaSnakeRobot` (PyElastica Cosserat rod, primary backend)
  - `snake_robot.py`: `SnakeRobot` (DisMech, implicit Euler DER)
  - `mujoco_snake_robot.py`: `MujocoSnakeRobot` (MuJoCo rigid-body chain)
  - `dismech_rods_snake_robot.py`: `DismechRodsSnakeRobot` (C++ DER via pybind11)
  - `friction.py`: `compute_barrier_normal_force()`, `CoulombForcing`, `StribeckForcing`
  - `geometry.py`: `SnakeGeometry`, `PreyGeometry`, `compute_contact_points()`, `compute_wrap_angle()`
  - `pipe_geometry.py`: `PipeGeometry`, `PipeRing` for tunnel environments

**`src/physics/cpg/`:**
- Purpose: Central Pattern Generator for action space reduction
- Contains: Neural oscillators and action transforms
- Key files:
  - `oscillators.py`: `MatsuokaOscillator`, `HopfOscillator`, `CPGNetwork`, `AdaptiveCPGNetwork`
  - `action_wrapper.py`: `DirectSerpenoidTransform`, `DirectSerpenoidSteeringTransform`, `CPGActionTransform`, `CPGEnvWrapper`

**`src/envs/`:**
- Purpose: Base TorchRL environment for snake simulation
- Contains: Abstract base class; concrete task envs imported conditionally
- Key files:
  - `base_env.py`: `BaseSnakeEnv(EnvBase)` with `_reset()`, `_step()`, `_make_spec()`, `_compute_reward()`, `_check_terminated()`
  - `__init__.py`: Conditionally imports `ApproachEnv`, `CoilEnv`, `HRLEnv`

**`src/networks/`:**
- Purpose: Actor-critic MLP architectures wrapped in TorchRL modules
- Contains: Actor (Gaussian/Categorical policy) and Critic (state-value/action-value) networks
- Key files:
  - `actor.py`: `ActorNetwork` MLP, `create_actor()` -> `ProbabilisticActor`, `CategoricalActorNetwork`
  - `critic.py`: `CriticNetwork` MLP, `create_critic()` -> `ValueOperator`

**`src/observations/`:**
- Purpose: Feature extractors for compact state representations
- Contains: Curvature modes (FFT), virtual chassis (body frame), goal-relative features, contact features
- Key files:
  - `extractors.py`: `FeatureExtractor` ABC, `CompositeFeatureExtractor`
  - `curvature_modes.py`: `CurvatureModeExtractor`, `ExtendedCurvatureModeExtractor`
  - `virtual_chassis.py`: `VirtualChassisExtractor`, `GoalRelativeExtractor`, `BodyFrameGoalExtractor`
  - `contact_features.py`: `ContactFeatureExtractor`, `ExtendedContactFeatureExtractor`

**`src/rewards/`:**
- Purpose: Potential-based reward shaping preserving optimal policies
- Contains: Abstract potentials, approach/coil task potentials, gait matching potentials
- Key files:
  - `shaping.py`: `PotentialFunction` ABC, `PotentialBasedRewardShaping`, `ApproachPotential`, `CoilPotential`, factory functions
  - `gait_potential.py`: `GaitPotential`, `CurriculumGaitPotential`, `AdaptiveGaitPotential`

**`src/trainers/`:**
- Purpose: RL training loop implementations with logging and checkpointing
- Contains: PPO, SAC, DDPG trainers and HRL orchestrator
- Key files:
  - `ppo.py`: `PPOTrainer` (TorchRL ClipPPOLoss + GAE + SyncDataCollector)
  - `sac.py`: `SACTrainer` (Soft Actor-Critic with replay buffer)
  - `ddpg.py`: `DDPGTrainer` (DDPG with replay buffer)
  - `hrl.py`: `HRLTrainer` (manager + worker PPO, sequential/joint training)
  - `logging_utils.py`: `compute_grad_norm()`, `collect_system_metrics()`

**`src/behavioral_cloning/`:**
- Purpose: Demonstration generation, experience storage, and BC pretraining
- Contains: Demo generators, KDTree-indexed buffers, fitness evaluation, I/O
- Key files:
  - `generators.py`: `SerpenoidGenerator`, `LateralUndulationGenerator`, `SidewindingGenerator`
  - `buffer.py`: `DemonstrationBuffer` (KDTree for nearest-neighbor)
  - `fitness.py`: Trajectory evaluation functions
  - `io.py`: Save/load utilities (pickle, JSON)
  - `approach_experiences.py`: `ApproachExperienceBuffer`, `ApproachExperienceGenerator`
  - `curvature_action_experiences.py`: `CurvatureActionExperienceBuffer`

**`locomotion_elastica/`:**
- Purpose: Active experiment for free-body snake locomotion with PyElastica
- Contains: Self-contained env, config, training, evaluation, recording
- Key files:
  - `config.py`: `LocomotionElasticaConfig(PPOConfig)`, `LocomotionElasticaEnvConfig`, `LocomotionElasticaPhysicsConfig(ElasticaConfig)`
  - `env.py`: `LocomotionElasticaEnv(EnvBase)` (standalone, does NOT subclass BaseSnakeEnv)
  - `train.py`: Training script entry point (`python -m locomotion_elastica.train`)
  - `evaluate.py`: Policy evaluation script
  - `record.py`: Video recording script
  - `rewards.py`: `compute_goal_reward()` (distance-based potential)
  - `diagnose.py`: Diagnostic physics checks

**`aprx_model_elastica/`:**
- Purpose: Neural surrogate model to replace PyElastica inner loop
- Contains: Data collection, supervised training, validation, GPU-batched env, RL training
- Key files:
  - `__main__.py`: CLI dispatcher (`collect`, `train`, `validate`, `rl`)
  - `state.py`: `RodState2D` pack/unpack, `StateNormalizer`, 124-dim state vector
  - `model.py`: `SurrogateModel` MLP (3x512 with LayerNorm + SiLU)
  - `dataset.py`: `SurrogateDataset` PyTorch Dataset
  - `collect_data.py`: Data collection from real PyElastica env with W&B monitoring
  - `train_surrogate.py`: Supervised training loop
  - `env.py`: `SurrogateLocomotionEnv(EnvBase)` GPU-batched TorchRL env
  - `train_rl.py`: RL training with surrogate env
  - `validate.py`: Accuracy validation vs real physics
  - `collect_config.py`: Data collection configuration
  - `train_config.py`: Model and training configuration

**Reference Implementation Packages (read-only examples):**

Each follows the same convention: `<authorYear>/` with `__init__.py`, `configs_*.py`, `env_*.py`, `rewards_*.py`, `train_*.py`.

- `bing2019/`: Bing et al. IJCAI 2019 planar locomotion (`PlanarSnakeEnv`, `TracksGenerator`)
- `jiang2024/`: Jiang et al. 2024 HRL navigation (`CobraNavigationEnv`, `AStarPlanner`, `KruskalMazeGenerator`)
- `liu2023/`: Liu et al. 2023 CPG locomotion with curriculum
- `zheng2022/`: Zheng et al. 2022 stiffness-tuned locomotion
- `choi2025/`: Choi & Tong 2025 implicit time-stepping

**`locomotion/`:**
- Purpose: Older locomotion experiment using DisMech backend
- Contains: Same structure as `locomotion_elastica/` but with DisMech physics
- Key files: `config.py`, `env.py`, `train.py`, `evaluate.py`, `record.py`

**`tests/`:**
- Purpose: Unit and integration tests
- Contains: Test files for physics, envs, trainers, reference implementations
- Key files:
  - `test_physics.py`: Physics backend tests (27K lines)
  - `test_envs.py`: Environment tests
  - `test_locomotion.py`: Locomotion-specific tests
  - `test_navigation.py`: Navigation tests
  - `test_ddpg.py`: DDPG trainer tests
  - `test_run_dir.py`: Run directory tests
  - `test_pipe_geometry.py`: Pipe geometry tests
  - `test_choi2025.py`, `test_liu2023.py`, `test_zheng2022.py`: Reference impl tests

**`script/`:**
- Purpose: Standalone shell scripts and Python utilities
- Key files:
  - `benchmark_physics.py`: Physics backend benchmarking
  - `benchmark_collect.sh`: Data collection benchmarking
  - `smoke_test_collect.py`: Pre-flight smoke test
  - `validate_surrogate_data.py`: Post-collection data validation
  - `install_dismech_rods.sh`: C++ dependency install script
  - `setup_macos.sh`: macOS development setup

**`data/`:**
- Purpose: Datasets for training
- Contains:
  - `surrogate/`: Surrogate model training data (state-action-next_state transitions)
  - `benchmark_collect/`: Benchmark data from collection runs

**`output/`:**
- Purpose: Training run outputs (auto-generated, gitignored)
- Structure: `output/<name>_<YYYYMMDD_HHMMSS>/config.json, console.log, checkpoints/`

## Key File Locations

**Entry Points:**
- `locomotion_elastica/train.py`: Primary training entry point (`python -m locomotion_elastica.train`)
- `aprx_model_elastica/__main__.py`: Surrogate pipeline (`python -m aprx_model_elastica {collect,train,validate,rl}`)
- `bing2019/train_locomotion.py`: Bing 2019 training
- `jiang2024/train_navigation.py`: Cobra navigation training

**Configuration:**
- `pyproject.toml`: Package metadata, dependencies, tool settings (black, ruff, pytest)
- `src/configs/__init__.py`: Central config re-exports (imports from all config modules + optional reference packages)
- `locomotion_elastica/config.py`: Active experiment configuration
- `aprx_model_elastica/train_config.py`: Surrogate model/training/env configs
- `Dockerfile`: Container build configuration
- `docker-compose.yml`: Container orchestration

**Core Logic:**
- `src/physics/elastica_snake_robot.py`: Primary physics simulation (PyElastica Cosserat rod)
- `src/physics/cpg/action_wrapper.py`: Action space transforms (serpenoid, CPG)
- `src/trainers/ppo.py`: Main training algorithm
- `locomotion_elastica/env.py`: Active environment implementation
- `aprx_model_elastica/model.py`: Surrogate neural network

**Testing:**
- `tests/test_physics.py`: Comprehensive physics backend tests
- `tests/test_envs.py`: Environment interface tests
- `tests/test_locomotion.py`: Locomotion-specific integration tests

## Naming Conventions

**Files:**
- Source modules: `snake_case.py` (e.g., `base_env.py`, `snake_robot.py`)
- Config modules: `snake_case.py` named by domain (e.g., `physics.py`, `training.py`)
- Reference implementations: `<purpose>_<author><year>.py` (e.g., `env_bing2019.py`, `configs_liu2023.py`)
- Test files: `test_<module>.py` (e.g., `test_physics.py`, `test_envs.py`)
- Scripts: `snake_case.py` or `snake_case.sh`

**Directories:**
- Source packages: `snake_case` (e.g., `behavioral_cloning`, `locomotion_elastica`)
- Reference packages: `<author><year>` (e.g., `bing2019`, `liu2023`)
- Documentation dirs: `lowercase` plural (e.g., `logs`, `experiments`, `issues`)

**Classes:**
- PascalCase: `ElasticaSnakeRobot`, `PPOTrainer`, `LocomotionElasticaEnv`
- Config classes: `<Domain>Config` suffix (e.g., `PhysicsConfig`, `PPOConfig`)
- Enums: PascalCase with UPPER_CASE members (e.g., `SolverFramework.ELASTICA`)

**Functions:**
- snake_case: `create_snake_robot()`, `compute_wrap_angle()`, `setup_run_dir()`
- Factory functions: `create_*` prefix (e.g., `create_actor()`, `create_critic()`, `create_snake_robot()`)
- Private methods: `_` prefix (e.g., `_make_spec()`, `_compute_reward()`)

## Where to Add New Code

**New Physics Backend:**
1. Create `src/physics/<backend>_snake_robot.py` implementing `reset()`, `step()`, `get_state()`, `set_curvature_control()`, `get_observation()`
2. Add enum value to `SolverFramework` in `src/configs/physics.py`
3. Add config dataclass inheriting from `PhysicsConfig` in `src/configs/physics.py`
4. Add factory branch in `src/physics/__init__.py` `create_snake_robot()`
5. Add tests in `tests/test_physics.py`

**New Experiment Package:**
1. Create `<name>/` at repo root with: `__init__.py`, `config.py`, `env.py`, `train.py`, `rewards.py`
2. Config should inherit from appropriate trainer config (e.g., `PPOConfig`)
3. Env should subclass `torchrl.envs.EnvBase` (or `src.envs.BaseSnakeEnv`)
4. Add package to `[tool.setuptools.packages.find]` in `pyproject.toml`
5. Optionally add config imports to `src/configs/__init__.py` with `try/except ImportError`

**New Trainer Algorithm:**
1. Create `src/trainers/<algorithm>.py` following `ppo.py` pattern
2. Include: constructor with env/config/network_config, `train()` method, W&B/checkpoint support
3. Export from `src/trainers/__init__.py`
4. Add config dataclass inheriting from `RLConfig` in `src/configs/training.py`

**New Observation Feature:**
1. Create `src/observations/<feature>.py` with class inheriting `FeatureExtractor` ABC from `src/observations/extractors.py`
2. Implement `extract(state) -> np.ndarray` and `dim` property
3. Export from `src/observations/__init__.py`

**New Reward Function:**
1. Create class inheriting `PotentialFunction` in `src/rewards/shaping.py` or a new file
2. Implement `__call__(state) -> float`
3. Add factory function (e.g., `create_<task>_shaper()`)
4. Export from `src/rewards/__init__.py`

**New Test:**
1. Place in `tests/test_<module>.py`
2. Follow pytest conventions: `test_` prefix for functions, descriptive names
3. Run with `pytest tests/test_<module>.py -v`

**New Script:**
1. Place in `script/<name>.py` or `script/<name>.sh`
2. Include argparse CLI for Python scripts
3. Make shell scripts executable (`chmod +x`)

**New Documentation:**
1. Place in appropriate directory (`logs/`, `experiments/`, `issues/`, `knowledge/`, `references/`, `tasks/`)
2. Use frontmatter with required properties (see CLAUDE.md)
3. Name as `<topic>.md`

## Special Directories

**`output/`:**
- Purpose: Timestamped training run directories
- Generated: Yes (by `setup_run_dir()`)
- Committed: No (gitignored)
- Structure: `output/<name>_<YYYYMMDD_HHMMSS>/config.json, console.log, checkpoints/*.pt`

**`wandb/`:**
- Purpose: W&B local run data
- Generated: Yes (by W&B SDK)
- Committed: No (gitignored)

**`dismech-python/`:**
- Purpose: External DisMech Python dependency (git submodule)
- Generated: No
- Committed: Yes (submodule reference)
- DO NOT MODIFY

**`dismech-python-src/`:**
- Purpose: DisMech Python source for development
- Generated: No
- Committed: Partially

**`data/`:**
- Purpose: Training datasets (surrogate transitions, benchmarks)
- Generated: Yes (by data collection scripts)
- Committed: Partially (small files only)

**`.planning/`:**
- Purpose: GSD planning and codebase analysis documents
- Generated: Yes (by GSD commands)
- Committed: Yes

**`model/`:**
- Purpose: Saved model weights
- Generated: Yes
- Committed: Selectively

---

*Structure analysis: 2026-03-09*
