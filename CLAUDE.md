# CLAUDE.md

## Project Overview

Hierarchical Reinforcement Learning (HRL) for snake robot predation using TorchRL. The snake robot learns to approach and coil around prey using a two-level hierarchy: an approach policy and a coil policy, coordinated by a meta-controller.

## Project Reference

For detailed project architecture, physics, and background, refer to `doc/overview.tex`.

## Directory Structure

```
snake-hrl/
├── .claude/           # Claude Code settings, skills, and memory
├── .git/              # Git version control
├── .gitignore
├── CLAUDE.md          # This file — project instructions for Claude
├── pyproject.toml     # Package config and dependencies
├── requirements.txt   # Pinned dependencies
├── Bing et al. (IJCAI 2019)/  # Reference locomotion code (not importable)
├── data/              # Datasets, demos, and saved experiences
├── doc/               # Documentation, logs, experiments, issues, knowledge
│   ├── logs/          # One file per log entry (<topic>.md)
│   ├── experiments/   # One file per experiment (<topic>.md)
│   ├── issues/        # One file per issue (<topic>.md)
│   └── knowledge/     # Domain knowledge and reference (<topic>.md)
├── figures/           # Generated plots and figures
├── media/             # Images, videos, and GIFs
├── model/             # Saved model weights
├── output/            # Training outputs and error logs
├── script/            # Standalone shell scripts and examples
├── src/
│   ├── configs/       # Dataclass-based experiment configs
│   ├── cpg/           # Central Pattern Generator modules
│   ├── demonstrations/ # Demo generation and loading
│   ├── envs/          # Gymnasium environments
│   ├── observations/  # Observation feature extractors
│   ├── networks/      # Policy and value network architectures
│   ├── physics/       # Snake physics and simulation
│   ├── rewards/       # Reward function definitions
│   └── trainers/      # Training loop implementations
├── snakebot-gym/      # Custom gym environment package
├── dismech-python/    # DER simulation dependency (submodule)
├── dismech-rods/      # C++ DER simulation (built from source, .gitignored)
└── tests/             # Unit and integration tests
```

## Do Not Modify

- `dismech-python/` — external dependency (submodule)
- `dismech-rods/` — external dependency (built from source)
- `snakebot-gym/` — custom gym environment (modify only if explicitly asked)

## Files You Can Modify

- `src/` — all source modules (configs, cpg, demonstrations, envs, observations, networks, physics, rewards, trainers)
- `tests/` — test files
- `script/` — standalone scripts
- `data/` — data files
- `doc/` — documentation files

## Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Training
```bash
train-approach    # Train approach policy
train-coil        # Train coil policy
train-hrl         # Train full HRL system
```

### Evaluation
```bash
evaluate          # Run evaluation
```

### Testing
```bash
pytest            # Run all tests
pytest tests/test_envs.py    # Run specific test file
```

## Architecture

- **Configs** (`src/configs/`): Dataclass-based hierarchical config for env, network, and training parameters
- **Environments** (`src/envs/`): TorchRL environments for approach, coil, and HRL tasks
- **Physics** (`src/physics/`): Snake body dynamics and contact simulation (four backends: DisMech, PyElastica, dismech-rods, MuJoCo)
- **Networks** (`src/networks/`): Actor-critic network architectures
- **Observations** (`src/observations/`): Feature extractors for compact state representations
- **Rewards** (`src/rewards/`): Shaped reward functions for approach and coil tasks
- **Trainers** (`src/trainers/`): TorchRL-based training loops for PPO and HRL
- **CPG** (`src/cpg/`): Central Pattern Generator for locomotion gaits
- **Demonstrations** (`src/demonstrations/`): Demo generation and loading

## Todoist

- Project ID: `6fxH85hJ3hvWq8hh`

## Documentation Requirements

IMPORTANT: You MUST document all work in `doc/` after completing any task. Each log, experiment, and issue is a **separate file** in its own subdirectory.

Follow the skill at `.claude/skills/markdown-for-project` for formatting, frontmatter, naming conventions, and templates.

| What | Where | Naming | When |
|---|---|---|---|
| Logs | `doc/logs/` | `<topic>.md` | After ANY code change or task |
| Experiments | `doc/experiments/` | `<topic>.md` | When running simulations/tests |
| Issues | `doc/issues/` | `<topic>.md` | When encountering bugs/errors |
| Knowledge | `doc/knowledge/` | `<topic>.md` | When capturing domain knowledge or reference material |
