# External Integrations

**Analysis Date:** 2026-03-09

## APIs & External Services

**Experiment Tracking:**
- Weights & Biases (W&B) - Training metric logging, hyperparameter tracking, run comparison
  - SDK/Client: `wandb` 0.25.0
  - Auth: `WANDB_API_KEY` env var or `~/.netrc` credentials
  - Projects: `snake-hrl` (PPO training), `snake-hrl-surrogate` (surrogate training), `surrogate-data-collection` (data collection monitoring)
  - Usage locations:
    - `src/trainers/ppo.py` - PPOTrainer logs train/episode/gradient/timing/system metrics per batch
    - `aprx_model_elastica/train_surrogate.py` - Surrogate training logs epoch/val/component losses
    - `aprx_model_elastica/collect_data.py` - Data collection logs FPS/progress/disk/schedule metrics
  - Config: `src/configs/base.py:WandB` dataclass (project, entity, group, tags)
  - Init pattern:
    ```python
    import wandb
    wandb_run = wandb.init(
        project=wandb_cfg.project,
        entity=wandb_cfg.entity or None,
        group=wandb_cfg.group or None,
        tags=wandb_cfg.tags or None,
        name=config.name,
        config=asdict(config),
        dir=str(run_dir),
    )
    ```

**Model Registry:**
- HuggingFace Hub - Model weight storage (configured, not heavily used in current code)
  - Auth: `HF_TOKEN` env var

**Container Registry:**
- Docker Hub - Container image publishing
  - Auth: `DOCKER_TOKEN` env var (user `albatross679`)
  - Image: CPU-only `python:3.13-slim` base

## Data Storage

**Databases:**
- None. All data is file-based.

**File Storage:**
- Local filesystem only
  - Training checkpoints: `output/<run_name>/checkpoints/*.pt` (PyTorch `torch.save`)
  - Surrogate training data: `data/surrogate/batch_*.pt` or `batch_*.parquet`
  - Trained surrogate models: `output/surrogate/model.pt`, `normalizer.pt`, `config.json`
  - Console logs: `output/<run_name>/console.log`
  - Run configs: `output/<run_name>/config.json`
  - Validation plots: `figures/surrogate_validation/*.png`

**Data Formats:**
- `.pt` (PyTorch tensors) - Primary format for checkpoints and surrogate data
  - Checkpoint schema: `{actor_state_dict, critic_state_dict, optimizer_state_dict, total_frames, total_episodes, best_reward, config}`
  - Surrogate data schema: `{states(T,124), actions(T,5), serpenoid_times(T,), next_states(T,124), episode_ids(T,), step_indices(T,), forces?}`
- `.parquet` (Apache Parquet via PyArrow) - Alternative columnar format for surrogate data
  - Uses `FixedSizeListArray` for multi-dimensional arrays, zstd compression
  - Read via `pyarrow.parquet.read_table()`

**Caching:**
- None (no Redis/Memcached). NumPy/PyTorch tensor operations are computed fresh each run.

## Authentication & Identity

**Auth Provider:**
- None (no user authentication). This is a research/ML training codebase.
- External service auth is purely API-key-based:
  - W&B: `WANDB_API_KEY` or `~/.netrc`
  - HuggingFace: `HF_TOKEN`
  - Docker Hub: `DOCKER_TOKEN`

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry/Datadog). Errors surface as Python exceptions and training loop console output.
- `sentry-sdk` 2.54.0 is installed but not configured in project code (likely a system dependency).

**Logs:**
- Console logging via `tqdm.write()` in training loops
- Console tee to file: `src/configs/console.py:ConsoleLogger` writes stdout/stderr to `console.log` in run directory
- W&B dashboard for training metrics (when enabled)
- TensorBoard configured (`src/configs/base.py:TensorBoard`) but W&B is the primary monitoring tool

**System Metrics:**
- `src/trainers/logging_utils.py:collect_system_metrics()` - CPU%, RAM usage (via `psutil`), GPU memory (via `torch.cuda`)
- Logged to W&B every N batches (configurable via `MetricGroups.system_interval`)

## CI/CD & Deployment

**Hosting:**
- Single GPU server (Tesla V100-PCIE-16GB, 48 CPUs)
- Docker support via `Dockerfile` + `docker-compose.yml`

**CI Pipeline:**
- None detected. No `.github/workflows/`, `.gitlab-ci.yml`, or similar CI config files.

**Build:**
- `Dockerfile` - Multi-arch CPU-only image for reproducible environments
- `script/setup_macos.sh` - macOS development setup
- `script/install_dismech_rods.sh` - C++ dependency build from source

## Environment Configuration

**Required env vars (for full functionality):**
- `WANDB_API_KEY` - Weights & Biases authentication (or `~/.netrc`)
- `OPENBLAS_NUM_THREADS=1` - Prevent thread contention in parallel workers
- `OMP_NUM_THREADS=1` - Prevent OpenMP thread explosion
- `MKL_NUM_THREADS=1` - Prevent MKL thread contention

**Optional env vars:**
- `HF_TOKEN` - HuggingFace Hub access
- `DOCKER_TOKEN` - Docker Hub publishing
- `CUDA_VISIBLE_DEVICES` - GPU selection (standard PyTorch)

**Secrets location:**
- `~/.netrc` - W&B credentials (host `api.wandb.ai`)
- Environment variables (not committed to repo)
- `.env` files: None detected in repo

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- W&B API calls during training (metric logging via `wandb.log()`)

## Physics Simulation Backends

Four physics backends are integrated, selected via `SolverFramework` enum in `src/configs/physics.py`:

**PyElastica (primary, active):**
- Package: `pyelastica` 0.3.3.post2
- Client code: `src/physics/elastica_snake_robot.py`, `locomotion_elastica/env.py`
- Purpose: Cosserat rod dynamics with symplectic integration (PositionVerlet/PEFRL)
- Used by: `locomotion_elastica` training, surrogate data collection

**DisMech Python:**
- Package: `dismech-python` 0.1.0 (local install from `dismech-python-src/`)
- Client code: `src/physics/snake_robot.py`
- Purpose: Discrete elastic rod with implicit Euler integration
- Used by: `src/envs/base_env.py` (approach/coil environments)

**dismech-rods C++:**
- Package: `dismech-rods` (built from source, optional)
- Client code: `src/physics/dismech_rods_snake_robot.py`
- Purpose: High-performance C++ DER simulation via pybind11

**MuJoCo:**
- Package: `mujoco` 3.2.0+
- Client code: `src/physics/mujoco_snake_robot.py`
- Purpose: Rigid-body chain simulation with hinge joints and position actuators

## Neural Surrogate Pipeline

A complete surrogate model pipeline replaces physics simulation with a learned MLP:

1. **Data Collection** (`aprx_model_elastica/collect_data.py`):
   - Runs PyElastica env with random/Sobol/policy actions
   - Multiprocess collection (1 env per worker, `forkserver` start method)
   - Saves to `.pt` or `.parquet` in `data/surrogate/`

2. **Training** (`aprx_model_elastica/train_surrogate.py`):
   - MLP: 3x512 with LayerNorm + SiLU, delta prediction
   - Two-phase: single-step MSE (epochs 1-20), then +multi-step rollout loss (8-step BPTT)
   - Density-based sample weighting for rare state coverage
   - Output: `output/surrogate/model.pt`, `normalizer.pt`

3. **Validation** (`aprx_model_elastica/validate.py`):
   - Compares autoregressive surrogate rollouts against real PyElastica trajectories
   - Reports per-step RMSE, CoM drift, heading drift at multiple horizons

4. **GPU-Batched RL Env** (`aprx_model_elastica/env.py`):
   - `SurrogateLocomotionEnv(EnvBase)` - runs N envs on GPU simultaneously
   - Single MLP forward pass replaces 500 integration substeps

---

*Integration audit: 2026-03-09*
