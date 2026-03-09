---
name: init-project
description: |
  Set up standard project structure with directories, git, .gitignore, README.md, CLAUDE.md, and Python environment.
  Use when: (1) Starting a new project and need to create the initial directory hierarchy, (2) Initializing a project with standard boilerplate files, (3) Setting up a new Python project environment with venv and dependencies.
---

# Project Initialization

Use this skill when starting a new project to set up the standard structure and files.

## Project Hierarchy

Create the following directory structure:

```
project/
├── .claude/        # Claude Code settings and memory
├── .git/           # Git version control
├── .gitignore
├── README.md
├── CLAUDE.md
├── pyproject.toml    # project metadata and dependencies
├── temp.md           # user scratch notes (temporary, not committed)
├── data/           # datasets and data files
├── doc/            # documentation (markdown, PDFs, LaTeX)
├── media/          # images, videos, and GIFs
├── model/          # saved model weights
├── output/         # outputs and errors from running
├── script/         # standalone scripts, examples, and tests
├── src/            # core source code (shared modules)
│   └── utils/      # generic reusable helper modules
├── <experiment>/   # experiment folders (one per model/approach)
│   ├── __init__.py
│   ├── config.py   # experiment-specific config (inherits from src config)
│   ├── model.py    # model definition
│   └── train.py    # training script
```

## Initialization Steps

1. **Create directories** (only those needed for the project type):
   ```bash
   mkdir -p data doc media model output script src/utils
   ```

2. **Initialize git**:
   ```bash
   git init
   ```

3. **Create .gitignore** with common exclusions:
   - Python: `__pycache__/`, `*.pyc`, `.venv/`, `*.egg-info/`
   - Environment: `.env`, `.env.local`
   - IDE: `.vscode/`, `.idea/`
   - OS: `.DS_Store`, `Thumbs.db`
   - Project: `output/`, `model/`, `.claude/`, `temp.md`

4. **Create README.md** with:
   - Project title and description
   - Setup instructions
   - Usage examples
   - License (if applicable)

5. **Create CLAUDE.md** with:
   - Project overview
   - Project directory structure (tree showing all created dirs/files with comments)
   - Do Not Modify section (list driver files)
   - Files You Can Modify section

6. **Scan existing files and set up Python environment**:
   - Scan all `.py` files in the directory for import statements
   - Identify third-party packages (exclude stdlib modules)
   - If `pyproject.toml` does not exist, create one with:
     - `[project]` section: name (from directory name), version `"0.1.0"`, `requires-python >= "3.10"`
     - `[project.dependencies]`: list all detected third-party packages
     - `[project.optional-dependencies]`: `dev = ["pytest", "tensorboard"]`
   - If `pyproject.toml` already exists, leave it as-is
   - Install the environment:
     ```bash
     pip install -e ".[dev]"
     ```

## CLAUDE.md Template

```markdown
# CLAUDE.md

## Project Overview

[Brief description of what this project does]

## Project Structure

\`\`\`
project/
├── .claude/        # Claude Code settings and memory
├── .gitignore
├── README.md
├── CLAUDE.md
├── pyproject.toml  # project metadata and dependencies
├── temp.md         # user scratch notes (temporary, not committed)
├── data/           # datasets and data files
├── doc/            # documentation (markdown, PDFs, LaTeX)
├── media/          # images, videos, and GIFs
├── model/          # saved model weights
├── output/         # outputs and errors from running
├── script/         # standalone scripts, examples, and tests
├── src/            # core source code (shared modules)
│   └── utils/      # generic reusable helper modules
├── <experiment>/   # experiment folders (one per model/approach)
│   ├── __init__.py
│   ├── config.py   # experiment-specific config (inherits from src config)
│   ├── model.py    # model definition
│   └── train.py    # training script
\`\`\`

### `src/` — shared source code

Reusable modules imported by experiment folders. Code here is **not run directly** —
it is imported. Typical contents:

- `data_loader.py` — load and preprocess raw data
- `feature_engineer.py` — shared feature engineering pipelines
- `scoring.py` — evaluation metrics, residual computation
- `optimizer.py` — ranking, selection, or optimization logic
- `explainer.py` — explainability and visualization helpers
- `config.py` — base configuration dataclasses that experiments inherit from
- `utils/` — small generic helpers (io, logging, plotting)

### `script/` — standalone scripts and tests

One-off or utility scripts that are **run directly** from the command line.
Not imported by other code. Examples:

- `prepare_data.py` — pre-compute features or download datasets
- `evaluate_all.py` — run scoring across all experiments
- `generate_report.py` — produce summary tables or plots
- `test_scoring.py` — unit and integration tests

### `<experiment>/` — experiment folders

Created at the project root, named after the model or approach
(e.g. `xgb/`, `lstm/`, `transformer/`). Each contains its own config, model, and
training script. Configs inherit from base configs in `src/`. Shared data loading,
feature engineering, and scoring logic lives in `src/`.

### `output/` — run outputs

Each training run creates a timestamped subdirectory under `output/`.

\`\`\`
output/<run>/
├── config.json            # frozen config snapshot for reproducibility
├── console.log            # stdout/stderr captured during training
├── metrics.json           # final evaluation metrics
├── predictions.parquet    # model predictions on test set
├── checkpoints/           # saved model weights (best, last)
├── tensorboard/           # TensorBoard event files
└── plots/                 # generated visualizations (PNG/SVG)
\`\`\`

## Do Not Modify

- [List driver/framework files here]

## Files You Can Modify

- [List implementation files here]
```

## Important Notes

- Only create directories that are needed for the specific project
- Do NOT modify CLAUDE.md after initial creation unless explicitly asked
- Keep existing files in root directory; don't move them unless user specifies
- Ask user about project type and requirements before initializing
