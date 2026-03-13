# CLAUDE.md

## Project Overview

Hierarchical Reinforcement Learning (HRL) for snake robot predation using TorchRL. The snake robot learns to approach and coil around prey using a two-level hierarchy: an approach policy and a coil policy, coordinated by a meta-controller.

## Project Reference

For detailed project architecture, physics, and background, refer to `overview.tex`.

## Directory Structure

```
snake-hrl/
├── .claude/           # Claude Code settings, skills, and memory
├── .git/              # Git version control
├── .gitignore
├── CLAUDE.md          # This file — project instructions for Claude
├── pyproject.toml     # Package config and dependencies
├── requirements.txt   # Pinned dependencies
├── overview.tex       # Detailed project architecture and physics reference
├── bing2019/          # Locomotion env package (Bing et al., IJCAI 2019)
├── data/              # Datasets, demos, and saved experiences
├── logs/              # One file per log entry (<topic>.md)
├── experiments/       # One file per experiment (<topic>.md)
├── issues/            # One file per issue (<topic>.md)
├── knowledge/         # Domain knowledge and reference (<topic>.md)
├── references/        # One file per reference (<topic>.md)
├── tasks/             # PRDs and task specs (prd-<feature>.md)
├── figures/           # Generated plots and figures
├── media/             # Images, videos, and GIFs
├── model/             # Saved model weights
├── output/            # Training outputs and error logs
├── script/            # Standalone shell scripts and examples
├── src/
│   ├── behavioral_cloning/ # Demo generation, buffers, and BC pretraining data
│   ├── configs/       # Dataclass-based experiment configs
│   ├── envs/          # Gymnasium environments
│   ├── observations/  # Observation feature extractors
│   ├── networks/      # Policy and value network architectures
│   ├── physics/       # Snake physics, simulation, and CPG actuators
│   │   └── cpg/       # Central Pattern Generator modules
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

- `src/` — all source modules (behavioral_cloning, configs, envs, observations, networks, physics, rewards, trainers)
- `tests/` — test files
- `script/` — standalone scripts
- `data/` — data files
- `logs/`, `experiments/`, `issues/`, `knowledge/`, `references/`, `tasks/` — documentation files

## LaTeX

- The research report lives in `report/report.tex`
- Always use the **LaTeX Workshop** VS Code extension to compile — do not invoke `pdflatex`, `tectonic`, or other CLI compilers directly

## Architecture

- **Configs** (`src/configs/`): Dataclass-based hierarchical config for env, network, and training parameters
- **Environments** (`src/envs/`): TorchRL environments for approach, coil, and HRL tasks
- **Physics** (`src/physics/`): Snake body dynamics, contact simulation (four backends: DisMech, PyElastica, dismech-rods, MuJoCo), and CPG actuators (`physics/cpg/`)
- **Networks** (`src/networks/`): Actor-critic network architectures
- **Observations** (`src/observations/`): Feature extractors for compact state representations
- **Rewards** (`src/rewards/`): Shaped reward functions for approach and coil tasks
- **Trainers** (`src/trainers/`): TorchRL-based training loops for PPO and HRL
- **Behavioral Cloning** (`src/behavioral_cloning/`): Demo generation, experience buffers, fitness evaluation, and BC pretraining data

## Credentials

- GitHub: `Albatross679`, email `qifan_wen@outlook.com`
- HuggingFace access token: set `HF_TOKEN` env var
- WandB API key: set `WANDB_API_KEY` env var
- Docker Hub (user `albatross679`): set `DOCKER_TOKEN` env var

## Todoist

- Project ID: `6fxH85hJ3hvWq8hh`

## Documentation (IMPORTANT)

Claude Code MUST document **as it goes** — immediately after each change, not batched at the end of the session. Each entry is a **separate file** in its subdirectory.

Every Markdown documentation file MUST include a `type` property in its frontmatter, set to the file's document type:

| What | Where | Naming | When | Type |
|---|---|---|---|---|
| Logs | `logs/` | `<topic>.md` | After any code change that adds, fixes, or modifies functionality | `log` |
| Experiments | `experiments/` | `<topic>.md` | After running a simulation, test, or investigation | `experiment` |
| Issues | `issues/` | `<topic>.md` | When encountering a bug or error (before or alongside the fix) | `issue` |
| Knowledge | `knowledge/` | `<topic>.md` | When capturing domain knowledge or reference material | `knowledge` |
| References | `references/` | `<topic>.md` | When capturing external references or citations | `reference` |
| Tasks | `tasks/` | `prd-<feature>.md` | When planning a feature or task (PRDs) | `task` |

### Required properties by type

All types share these **common properties**: `name`, `description`, `type`, `created`, `updated`, `tags`, `aliases`.

In addition, each type has specific properties that MUST be set:

- **`log`**: `status` (draft | complete), `subtype` (fix | training | tuning | research | refactor | setup | feature)
- **`experiment`**: `status` (planned | running | complete | failed), `subtype` (training | architecture | hyperparameter | physics | data | ablation)
- **`issue`**: `status` (open | investigating | resolved | wontfix), `severity` (low | medium | high | critical), `subtype` (training | physics | compatibility | system | performance)
- **`knowledge`**: `subtype` (domain | implementation | infrastructure | physics | ml)
- **`reference`**: `source`, `url`, `authors`, `subtype` (paper | blog | documentation | tutorial | library)
- **`task`**: `status` (planned | in-progress | complete | cancelled)

**Threshold for logging:** A change warrants a log if it modifies behavior, fixes a bug, or changes configuration. Trivial edits (typos, whitespace, comment-only changes) do not need a log entry.

## Active Mode ("Stay Active" / Long-Running Tasks)

When the user asks Claude Code to **stay active** (e.g., "keep going", "stay active and fix everything", "run until done"), Claude Code MUST:

1. **Scope** — only address issues related to the current task or explicitly requested area. Do not "fix" unrelated code that appears to work.
2. **Prioritize** — fix errors and blockers first, then warnings, then improvements.
3. **Autonomously iterate** — continuously identify and resolve issues without waiting for further prompts.
4. **Document each resolution** — write a log entry immediately after each fix, before moving to the next problem.
5. **Check in every ~5 fixes** — briefly summarize progress and remaining issues so the user can redirect if needed.
6. **Stop when** there are no remaining in-scope issues or the user asks to stop.
