# Architecture

**Analysis Date:** 2026-03-09

## Pattern Overview

**Overall:** Layered modular architecture with configuration-driven dependency injection. Each layer (configs, physics, environments, networks, trainers) is a Python package under `src/`. Multiple self-contained experiment packages (e.g., `locomotion_elastica/`, `aprx_model_elastica/`, `bing2019/`) sit at the repo root and compose `src/` components.

**Key Characteristics:**
- Dataclass-based hierarchical configuration with composition and inheritance
- Factory pattern for physics backends (four interchangeable simulators)
- TorchRL `EnvBase` as the universal environment interface
- Clear separation: physics simulation is decoupled from RL environment logic
- Reference paper implementations as standalone top-level packages that reuse `src/` infrastructure
- Neural surrogate pattern: MLP replaces physics inner loop for GPU-batched RL

## Layers

**Configuration Layer:**
- Purpose: Define all experiment parameters as composable Python dataclasses
- Location: `src/configs/`
- Contains: Hierarchical dataclass trees for physics, environment, network, and training
- Depends on: Nothing (leaf layer)
- Used by: Every other layer
- Key hierarchy:
  - `MLBaseConfig` -> `RLConfig` -> `PPOConfig` / `SACConfig` / `DDPGConfig` / `HRLConfig`
  - `PhysicsConfig` -> `RodConfig` -> `DERConfig` / `CosseratConfig` -> `ElasticaConfig`
  - `PhysicsConfig` -> `MujocoPhysicsConfig`
  - `NetworkConfig` = `ActorConfig` + `CriticConfig`
- Key files:
  - `src/configs/base.py`: `MLBaseConfig`, `Checkpointing`, `WandB`, `Output`, `Console`, `resolve_device()`, `save_config()`, `load_config()`
  - `src/configs/physics.py`: `SolverFramework` enum, `PhysicsConfig`, `ElasticaConfig`, `FrictionConfig`, `FrictionModel`
  - `src/configs/env.py`: `EnvConfig`, `ControlMethod` enum, `StateRepresentation` enum, `CPGConfig`
  - `src/configs/training.py`: `RLConfig`, `PPOConfig`, `SACConfig`, `DDPGConfig`, `HRLConfig`
  - `src/configs/network.py`: `ActorConfig`, `CriticConfig`, `NetworkConfig`, `HRLNetworkConfig`
  - `src/configs/project.py`: Top-level composed configs `SnakeApproachConfig`, `SnakeCoilConfig`, `SnakeHRLConfig`
  - `src/configs/run_dir.py`: `setup_run_dir()` creates timestamped output directories
  - `src/configs/console.py`: `ConsoleLogger` context manager for stdout/stderr capture

**Physics Layer:**
- Purpose: Simulate soft snake robot dynamics with multiple backend options
- Location: `src/physics/`
- Contains: Snake robot simulators, friction models, geometry, CPG actuators
- Depends on: `src/configs/` (physics and env configs)
- Used by: `src/envs/`, experiment packages (`locomotion_elastica/`, etc.)
- Key abstraction: `create_snake_robot(config)` factory dispatches on `SolverFramework` enum:
  - `DISMECH` -> `SnakeRobot` in `src/physics/snake_robot.py` (DisMech implicit Euler)
  - `ELASTICA` -> `ElasticaSnakeRobot` in `src/physics/elastica_snake_robot.py` (PyElastica Cosserat)
  - `DISMECH_RODS` -> `DismechRodsSnakeRobot` in `src/physics/dismech_rods_snake_robot.py` (C++ DER)
  - `MUJOCO` -> `MujocoSnakeRobot` in `src/physics/mujoco_snake_robot.py` (MuJoCo rigid-body)
- Shared interface: `reset()`, `step()`, `get_state()`, `get_observation()`, `set_curvature_control()`
- Lazy imports: `SnakeRobot` uses `__getattr__` to avoid heavy DisMech/numba initialization in parallel workers
- Key files:
  - `src/physics/__init__.py`: Factory function `create_snake_robot()`, lazy import
  - `src/physics/elastica_snake_robot.py`: Primary physics backend (22K lines), PyElastica Cosserat rod
  - `src/physics/snake_robot.py`: DisMech DER backend (25K lines)
  - `src/physics/mujoco_snake_robot.py`: MuJoCo rigid-body backend (23K lines)
  - `src/physics/dismech_rods_snake_robot.py`: C++ DER backend (19K lines)
  - `src/physics/friction.py`: Coulomb and Stribeck friction force implementations
  - `src/physics/geometry.py`: `SnakeGeometry`, `PreyGeometry`, contact/wrap utilities
  - `src/physics/pipe_geometry.py`: Pipe tunnel geometry for rod-to-rod contact

**CPG/Actuator Sublayer:**
- Purpose: Transform low-dimensional RL actions into joint curvatures
- Location: `src/physics/cpg/`
- Contains: Neural oscillators and action transforms
- Key abstraction: Control methods reduce action space from 19-dim (direct) to 4-5-dim (serpenoid)
- Key files:
  - `src/physics/cpg/oscillators.py`: `MatsuokaOscillator`, `HopfOscillator`, `CPGNetwork`, `AdaptiveCPGNetwork`
  - `src/physics/cpg/action_wrapper.py`: `DirectSerpenoidTransform` (4-dim), `DirectSerpenoidSteeringTransform` (5-dim), `CPGActionTransform`, `CPGEnvWrapper`

**Observation Layer:**
- Purpose: Extract compact features from high-dimensional simulation state
- Location: `src/observations/`
- Contains: Feature extractors that compose into pipelines
- Depends on: NumPy (pure computation, no framework dependency)
- Used by: `src/physics/` (snake robot classes), `src/envs/`
- Key pattern: `CompositeFeatureExtractor` combines multiple extractors into a single vector
- Key files:
  - `src/observations/extractors.py`: `FeatureExtractor` ABC, `CompositeFeatureExtractor`
  - `src/observations/curvature_modes.py`: `CurvatureModeExtractor` (amplitude, wave_number, phase from FFT)
  - `src/observations/virtual_chassis.py`: `VirtualChassisExtractor` (body-frame state), `GoalRelativeExtractor`
  - `src/observations/contact_features.py`: `ContactFeatureExtractor` (contact points, wrap angle)

**Environment Layer:**
- Purpose: Wrap physics simulation as TorchRL environments with TensorDict I/O
- Location: `src/envs/` (base) + experiment packages (concrete implementations)
- Contains: Base environment class and task-specific subclasses
- Depends on: `src/physics/`, `src/configs/`, TorchRL
- Used by: `src/trainers/`, experiment training scripts
- Key files:
  - `src/envs/base_env.py`: `BaseSnakeEnv(EnvBase)` -- core TorchRL interface with `_reset()`, `_step()`, `_make_spec()`
  - `locomotion_elastica/env.py`: `LocomotionElasticaEnv(EnvBase)` -- standalone locomotion env (does NOT subclass BaseSnakeEnv; embeds PyElastica directly)
  - `aprx_model_elastica/env.py`: `SurrogateLocomotionEnv(EnvBase)` -- GPU-batched env using neural surrogate
- Important: Some experiment envs (e.g., `locomotion_elastica/env.py`) directly subclass `EnvBase` rather than `BaseSnakeEnv`, embedding physics inline for performance

**Network Layer:**
- Purpose: Actor-critic neural network architectures for policy and value function
- Location: `src/networks/`
- Contains: MLP-based actor and critic with configurable architecture
- Depends on: `src/configs/network.py`, PyTorch, TorchRL modules
- Used by: `src/trainers/`
- Key files:
  - `src/networks/actor.py`: `ActorNetwork` MLP, `create_actor()` factory (returns TorchRL `ProbabilisticActor`)
  - `src/networks/critic.py`: `CriticNetwork` MLP, `create_critic()` factory (returns TorchRL `ValueOperator`)
- Actor outputs: Gaussian (TanhNormal) for continuous control, Categorical for HRL manager
- Both support orthogonal initialization, layer normalization, configurable activations

**Reward Layer:**
- Purpose: Potential-based reward shaping (PBRS) that preserves optimal policies
- Location: `src/rewards/`
- Contains: Abstract potential functions, task-specific potentials, composite shaping
- Depends on: NumPy (pure computation)
- Used by: `src/envs/`, experiment envs
- Key files:
  - `src/rewards/shaping.py`: `PotentialFunction` ABC, `PotentialBasedRewardShaping`, `ApproachPotential`, `CoilPotential`
  - `src/rewards/gait_potential.py`: `GaitPotential`, `CurriculumGaitPotential` (demo-based Gaussian potential)

**Trainer Layer:**
- Purpose: RL training loops with data collection, optimization, logging, and checkpointing
- Location: `src/trainers/`
- Contains: Algorithm-specific trainers (PPO, SAC, DDPG) and HRL orchestrator
- Depends on: All other layers, TorchRL, W&B
- Used by: Training scripts in experiment packages
- Key files:
  - `src/trainers/ppo.py`: `PPOTrainer` -- TorchRL `ClipPPOLoss` + `GAE` + `SyncDataCollector`, graceful shutdown, W&B logging
  - `src/trainers/sac.py`: `SACTrainer` -- Soft Actor-Critic with replay buffer
  - `src/trainers/ddpg.py`: `DDPGTrainer` -- DDPG with replay buffer
  - `src/trainers/hrl.py`: `HRLTrainer` -- orchestrates manager (discrete skill selection) + worker PPO trainers
  - `src/trainers/logging_utils.py`: `compute_grad_norm()`, `collect_system_metrics()`
- Training loop pattern: `SyncDataCollector` collects batches -> `GAE` computes advantages -> minibatch PPO updates -> W&B/checkpoint logging

**Behavioral Cloning Layer:**
- Purpose: Demonstration generation, experience buffers, and behavioral cloning pretraining
- Location: `src/behavioral_cloning/`
- Contains: Demo generators, KDTree-indexed buffers, fitness evaluation, I/O utilities
- Depends on: `src/physics/`, `src/observations/`, `src/configs/`
- Used by: BC pretraining scripts
- Key files:
  - `src/behavioral_cloning/generators.py`: `SerpenoidGenerator`, `LateralUndulationGenerator`, `SidewindingGenerator`
  - `src/behavioral_cloning/buffer.py`: `DemonstrationBuffer` with KDTree for nearest-neighbor queries
  - `src/behavioral_cloning/fitness.py`: Trajectory evaluation (displacement, direction coverage)
  - `src/behavioral_cloning/io.py`: Save/load demonstrations (pickle, JSON)
  - `src/behavioral_cloning/approach_experiences.py`: `ApproachExperienceBuffer`
  - `src/behavioral_cloning/curvature_action_experiences.py`: `CurvatureActionExperienceBuffer`

**Neural Surrogate Layer:**
- Purpose: Replace physics inner loop with a trained MLP for GPU-batched RL training
- Location: `aprx_model_elastica/`
- Contains: Data collection, model training, validation, surrogate environment, RL training
- Depends on: `src/physics/cpg/`, `src/configs/`, `locomotion_elastica/` (for data collection)
- Key files:
  - `aprx_model_elastica/state.py`: `RodState2D` pack/unpack, `StateNormalizer`, 124-dim flat state vector
  - `aprx_model_elastica/model.py`: `SurrogateModel` MLP (131 -> 3x512 -> 124)
  - `aprx_model_elastica/dataset.py`: `SurrogateDataset` PyTorch Dataset
  - `aprx_model_elastica/collect_data.py`: Data collection from real PyElastica env
  - `aprx_model_elastica/train_surrogate.py`: Supervised training of surrogate model
  - `aprx_model_elastica/env.py`: `SurrogateLocomotionEnv` GPU-batched TorchRL env
  - `aprx_model_elastica/train_rl.py`: RL training using surrogate env
  - `aprx_model_elastica/validate.py`: Accuracy validation vs real env
  - `aprx_model_elastica/__main__.py`: CLI dispatcher (`collect`, `train`, `validate`, `rl`)

## Data Flow

**RL Training Loop (PPO):**

1. `LocomotionElasticaConfig` (dataclass) configures all parameters
2. `setup_run_dir(config)` creates timestamped output directory with config snapshot
3. `LocomotionElasticaEnv(config)` creates env with inline PyElastica physics
4. `ParallelEnv(num_envs, factory)` wraps in multiprocessing for parallel rollouts
5. `TransformedEnv` + `StepCounter` + `RewardSum` adds episode tracking
6. `PPOTrainer(env, config, network_config)` creates actor, critic, `ClipPPOLoss`, `GAE`, `SyncDataCollector`
7. Collector gathers `frames_per_batch` transitions using actor policy
8. `GAE` computes advantages over the batch
9. PPO minibatch updates with clipped surrogate objective
10. Metrics logged to W&B and checkpoints saved to `run_dir/checkpoints/`

**Neural Surrogate Pipeline:**

1. `python -m aprx_model_elastica collect` -- runs `LocomotionElasticaEnv` to collect (state, action, next_state) transitions
2. `python -m aprx_model_elastica train` -- trains `SurrogateModel` on collected data via supervised learning
3. `python -m aprx_model_elastica validate` -- compares surrogate predictions to real physics
4. `python -m aprx_model_elastica rl` -- trains RL policy using `SurrogateLocomotionEnv` (GPU-batched)

**HRL Training (Manager + Workers):**

1. `HRLTrainer` creates `ApproachEnv` and `CoilEnv` skill environments
2. In sequential mode: trains approach skill PPO, then coil skill PPO, then manager
3. Manager outputs discrete skill selection (categorical distribution)
4. Selected worker policy executes for a fixed number of steps
5. Manager receives reward based on worker outcomes

**State Management:**
- Physics state is a dictionary with keys: `positions`, `velocities`, `curvatures`, `prey_position`, `prey_distance`
- TorchRL uses `TensorDict` for all env I/O (observations, actions, rewards, done flags)
- Config state is immutable dataclasses (serialized to JSON in run directory)
- Training state (weights, optimizer) saved via PyTorch `state_dict` checkpointing

## Key Abstractions

**Physics Backend (Strategy Pattern):**
- Purpose: Interchangeable simulation backends via `SolverFramework` enum
- Examples: `src/physics/elastica_snake_robot.py`, `src/physics/snake_robot.py`, `src/physics/mujoco_snake_robot.py`
- Pattern: Factory function `create_snake_robot(config)` dispatches on `config.solver_framework`
- Common interface: `reset()`, `step()`, `get_state()`, `set_curvature_control()`

**Control Method (Strategy Pattern):**
- Purpose: Reduce RL action space from direct joint control to parameterized gaits
- Examples: `src/physics/cpg/action_wrapper.py`
- Pattern: `ControlMethod` enum selects transform; transform maps low-dim action to joint curvatures
- Options: DIRECT (19-dim), CPG (4-dim oscillators), SERPENOID (4-dim formula), SERPENOID_STEERING (5-dim)

**Feature Extractor (Composite Pattern):**
- Purpose: Extract compact observation from high-dimensional simulation state
- Examples: `src/observations/extractors.py`, `src/observations/curvature_modes.py`
- Pattern: `CompositeFeatureExtractor` composes multiple `FeatureExtractor` instances

**Reward Shaping (Template Method + Strategy):**
- Purpose: Shape rewards while preserving optimal policy
- Examples: `src/rewards/shaping.py`, `src/rewards/gait_potential.py`
- Pattern: `PotentialBasedRewardShaping` computes `gamma * Phi(s') - Phi(s)` using pluggable `PotentialFunction`

**Config Hierarchy (Template + Composition):**
- Purpose: Hierarchical configuration that composes env, physics, network, and training params
- Examples: `src/configs/project.py`, `locomotion_elastica/config.py`
- Pattern: Top-level configs (e.g., `LocomotionElasticaConfig`) inherit from trainer config (`PPOConfig`) and compose env/network configs as dataclass fields

## Entry Points

**Primary Training (Locomotion):**
- Location: `locomotion_elastica/train.py`
- Triggers: `python -m locomotion_elastica.train --gait forward`
- Responsibilities: Parse CLI args, build config, create ParallelEnv, create PPOTrainer, run training loop

**Surrogate Model Pipeline:**
- Location: `aprx_model_elastica/__main__.py`
- Triggers: `python -m aprx_model_elastica {collect,train,validate,rl}`
- Responsibilities: Route to data collection, model training, validation, or RL training subcommand

**Reference Implementation Entry Points:**
- `bing2019/train_locomotion.py`: `python -m bing2019.train_locomotion` (planar locomotion)
- `jiang2024/train_navigation.py`: Cobra navigation with A* planning
- `liu2023/train_liu2023.py`: CPG-regulated locomotion with curriculum
- `zheng2022/train_zheng2022.py`: Stiffness-tuned locomotion
- `choi2025/train.py`: Implicit time-stepping

**Evaluation/Recording:**
- `locomotion_elastica/evaluate.py`: Evaluate trained policy
- `locomotion_elastica/record.py`: Record video of locomotion
- `locomotion_elastica/diagnose.py`: Diagnostic physics checks

**Scripts:**
- `script/benchmark_physics.py`: Benchmark physics backends
- `script/smoke_test_collect.py`: Pre-flight smoke test for data collection
- `script/validate_surrogate_data.py`: Post-collection data validation

## Error Handling

**Strategy:** Defensive with graceful degradation and signal handling

**Patterns:**
- Graceful shutdown: `PPOTrainer` and `HRLTrainer` install SIGINT/SIGTERM handlers, save checkpoint on interrupt
- Optional imports: Reference packages use `try/except ImportError: pass` for optional dependencies
- TorchRL API compatibility: `try/except ImportError` blocks handle v0.11 API renames (`BoundedTensorSpec` vs `Bounded`)
- Config validation: `__post_init__` methods on dataclasses for derived field computation
- Atomic checkpointing: PPO saves to temp file then moves to final path

## Cross-Cutting Concerns

**Logging:**
- W&B integration in trainers (`config.wandb.enabled`)
- `ConsoleLogger` context manager captures stdout/stderr to `console.log` in run directory
- `collect_system_metrics()` tracks CPU/RAM/GPU usage at configurable intervals
- tqdm progress bars in training loops

**Validation:**
- Config validation via dataclass `__post_init__`
- Physics state validation implicit in simulation (NaN/instability detection)
- Pre-flight smoke tests in `script/smoke_test_collect.py`
- Post-collection data validation in `script/validate_surrogate_data.py`

**Parallelism:**
- TorchRL `ParallelEnv` for multiprocessing-based vectorized environments
- OpenBLAS/OMP/MKL thread limiting (`os.environ` in training scripts) to prevent thread exhaustion
- Picklable `_EnvFactory` classes for multiprocessing compatibility
- Lazy imports to avoid heavy initialization in worker processes

**Serialization:**
- Config: JSON via `save_config()` / `load_config()` with recursive dataclass reconstruction
- Models: PyTorch `state_dict` checkpointing (best, last, interrupted)
- Demonstrations: pickle and JSON via `src/behavioral_cloning/io.py`
- Surrogate data: NumPy `.npz` files

---

*Architecture analysis: 2026-03-09*
