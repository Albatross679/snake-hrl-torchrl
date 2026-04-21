# Technology Stack

**Analysis Date:** 2026-03-09

## Languages

**Primary:**
- Python 3.12+ (`requires-python = ">=3.12"` in `pyproject.toml`; system install is 3.12.3)

**Secondary:**
- C++ (via `dismech-rods` submodule built from source with pybind11 bindings)
- XML (MJCF scene descriptions generated programmatically in `src/physics/mujoco_snake_robot.py`)

## Runtime

**Environment:**
- Python 3.12.3 (system install at `/usr/bin/python3`, no virtualenv)
- CUDA 12.8 (PyTorch GPU backend)
- GPU: Tesla V100-PCIE-16GB (16 GB VRAM)
- 48 CPUs available

**Package Manager:**
- pip (primary installer)
- uv 0.10.8 (available, used for source dependencies in `pyproject.toml` `[tool.uv.sources]`)
- Lockfile: `requirements.txt` present but partially stale (contains freeform notes at bottom, version ranges are looser than actual installs)

## Frameworks

**Core:**
- PyTorch 2.10.0 - Deep learning framework, tensor operations, model training
- TorchRL 0.11.1 - Reinforcement learning framework (environments, collectors, losses, transforms)
- TensorDict 0.11.0 - Data container for TorchRL environment I/O

**Physics Simulation (4 backends):**
- PyElastica 0.3.3.post2 - Cosserat rod dynamics (primary active backend for locomotion)
- DisMech (Python) 0.1.0 - Discrete elastic rod simulation (installed from local `dismech-python-src/`)
- dismech-rods (C++) - C++ DER simulation with pybind11 (optional, built from source)
- MuJoCo 3.2.0+ - Rigid-body physics simulation

**RL Environment:**
- Gymnasium 1.2.3 - Environment API standard (used via TorchRL wrappers)

**Testing:**
- pytest 9.0.2 - Test runner
- pytest-cov 7.0.0 - Coverage reporting

**Build/Dev:**
- setuptools 80.9.0 - Package build backend (`pyproject.toml`)
- black 26.1.0 - Code formatter (line-length=100, target py312)
- ruff 0.15.4 - Linter (rules: E, F, I, N, W; ignores E501)

## Key Dependencies

**Critical (core functionality):**
- `torch` 2.10.0 - All neural network training, GPU compute, tensor operations
- `torchrl` 0.11.1 - EnvBase, ClipPPOLoss, GAE, SyncDataCollector, ParallelEnv, ReplayBuffer
- `tensordict` 0.11.0 - TensorDict data container used everywhere as env step I/O
- `pyelastica` 0.3.3.post2 - Primary physics engine (CosseratRod, PositionVerlet, damping, forcing)
- `numpy` 2.4.0 - Array operations, physics state packing, data collection
- `scipy` 1.17.1 - Scientific computing (FFT for curvature modes, optimization)

**Training Infrastructure:**
- `wandb` 0.25.0 - Experiment tracking, metric logging, hyperparameter visualization
- `tqdm` 4.67.1 - Progress bars for training loops
- `matplotlib` 3.10.8 - Validation plots, figure generation

**Data Pipeline:**
- `pyarrow` 23.0.1 - Parquet file I/O for surrogate training data
- `torch.save`/`torch.load` - Primary data format (`.pt` files)

**Performance:**
- `numba` 0.64.0 - JIT compilation (listed in requirements, available for compute-heavy loops)
- `psutil` 5.9.0+ - System metrics monitoring (CPU, RAM usage in training loops)
- `triton` 3.6.0 - GPU kernel compilation for PyTorch

**Optional:**
- `plotly` 6.6.0 - Interactive visualization
- `mujoco` - MuJoCo rigid-body backend (imported conditionally in `src/physics/mujoco_snake_robot.py`)
- `tensorboard` 2.20.0 - TensorBoard logging (configured but W&B is primary)

## Configuration

**Environment:**
- Thread control env vars set at import time in training scripts and Docker:
  - `OPENBLAS_NUM_THREADS=1`
  - `OMP_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
- W&B credentials: `WANDB_API_KEY` env var (also `~/.netrc`)
- HuggingFace: `HF_TOKEN` env var
- Docker Hub: `DOCKER_TOKEN` env var (user `albatross679`)
- Device resolution: `"auto"` -> CUDA when available, else CPU (`src/configs/base.py:resolve_device()`)

**Build:**
- `pyproject.toml` - Package metadata, dependencies, tool config (black, ruff, pytest)
- `Dockerfile` - Multi-arch CPU-only image based on `python:3.13-slim`
- `docker-compose.yml` - Service definition with volume mounts for `output/` and `data/`

**Config System:**
- Pure Python dataclasses (no YAML/JSON config files at rest)
- Hierarchical inheritance: `MLBaseConfig` -> `RLConfig` -> `PPOConfig`/`SACConfig`/`DDPGConfig`/`HRLConfig`
- Entry point: `src/configs/base.py` (`MLBaseConfig`, `save_config()`, `load_config()`)
- Physics config hierarchy: `PhysicsConfig` -> `RodConfig` -> `DERConfig`/`CosseratConfig` -> backend-specific
- All configs serializable to JSON via `dataclasses.asdict()`

## Platform Requirements

**Development:**
- Python 3.12+
- CUDA-capable GPU recommended (Tesla V100 or better for training)
- 48+ CPUs optimal for 16 parallel environment workers
- CMake + build-essential (for dismech-rods C++ compilation)
- OpenGL libraries (for MuJoCo rendering): `libgl1-mesa-glx`, `libglew-dev`, `libosmesa6-dev`

**Production/Training:**
- Docker image: `python:3.13-slim` base with CPU-only PyTorch
- GPU training: Use host PyTorch with CUDA (not Docker image)
- Volumes: `./output` (checkpoints, logs), `./data` (surrogate training data)
- Minimum disk: ~1.5 KB per transition for surrogate data collection

**Entry Points:**
- `python -m locomotion_elastica.train` - PPO training with PyElastica
- `python -m aprx_model_elastica.collect_data` - Surrogate data collection
- `python -m aprx_model_elastica.train_surrogate` - Surrogate model training
- `python -m aprx_model_elastica.validate` - Surrogate validation
- `python -m aprx_model_elastica` - Surrogate RL training (`__main__.py`)

---

*Stack analysis: 2026-03-09*
