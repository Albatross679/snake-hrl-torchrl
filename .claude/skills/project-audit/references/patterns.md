# Audit Pattern Catalog

This file defines patterns for the project-audit skill. To extend for new project types, add a new section following the existing format.

## 1. Gitignore Patterns

Organized by ecosystem. Check which are missing from the project's `.gitignore`.

### Python

| Pattern | Description |
|---------|-------------|
| `__pycache__/` | Bytecode cache directories |
| `*.pyc` | Compiled bytecode files |
| `*.pyo` | Optimized bytecode files |
| `.eggs/` | Egg build artifacts |
| `*.egg-info/` | Egg metadata directories |
| `dist/` | Distribution packages |
| `build/` | Build output |
| `.venv/` | Virtual environment |
| `venv/` | Virtual environment (alt name) |
| `.env` | Environment variables file |
| `.tox/` | Tox test runner cache |
| `.pytest_cache/` | Pytest cache |
| `.mypy_cache/` | Mypy type-checker cache |
| `.ruff_cache/` | Ruff linter cache |
| `htmlcov/` | Coverage HTML reports |
| `coverage.xml` | Coverage XML output |
| `.coverage` | Coverage data file |

### Node.js

| Pattern | Description |
|---------|-------------|
| `node_modules/` | Dependencies |
| `dist/` | Build output |
| `.next/` | Next.js build |
| `.nuxt/` | Nuxt.js build |
| `.output/` | Nitro/Nuxt output |
| `.parcel-cache/` | Parcel bundler cache |
| `.turbo/` | Turborepo cache |

### ML / Data Science

| Pattern | Description |
|---------|-------------|
| `wandb/` | Weights & Biases run data |
| `mlruns/` | MLflow run data |
| `output/` | Training output directory |
| `checkpoints/` | Model checkpoints |
| `*.pt` | PyTorch model files |
| `*.pth` | PyTorch model files (alt) |
| `*.ckpt` | Checkpoint files |
| `*.h5` | HDF5 model/data files |
| `*.safetensors` | SafeTensors model files |
| `model/` | Model weight directories (if large binaries) |

### IDE / Editor

| Pattern | Description |
|---------|-------------|
| `.idea/` | JetBrains IDE |
| `.vscode/settings.json` | VS Code user settings |
| `*.swp` | Vim swap files |
| `*.swo` | Vim swap files (alt) |
| `*~` | Editor backup files |
| `.project` | Eclipse project |
| `.classpath` | Eclipse classpath |

### OS Files

| Pattern | Description |
|---------|-------------|
| `.DS_Store` | macOS directory metadata |
| `Thumbs.db` | Windows thumbnail cache |
| `desktop.ini` | Windows folder settings |
| `*.lnk` | Windows shortcut files |

### Build / Cache

| Pattern | Description |
|---------|-------------|
| `.cache/` | Generic cache directory |
| `target/` | Rust/Maven build output |
| `vendor/` | Go vendored dependencies |

### LaTeX

| Pattern | Description |
|---------|-------------|
| `*.aux` | Auxiliary data |
| `*.log` | Compilation log |
| `*.out` | Hyperref output |
| `*.toc` | Table of contents |
| `*.synctex.gz` | SyncTeX data |
| `*.fls` | File list |
| `*.fdb_latexmk` | Latexmk database |
| `missfont.log` | Missing font log |
| `*.bbl` | Bibliography output |
| `*.blg` | BibTeX log |

## 2. Clutter Patterns (Delete Candidates)

### Temp Files

| Pattern | Description |
|---------|-------------|
| `*.tmp` | Temporary files |
| `*.temp` | Temporary files (alt) |
| `*.bak` | Backup files |
| `*.orig` | Original/merge conflict files |
| `*.swp` | Swap files |
| `temp.*` | Temp-prefixed files |
| `tmp.*` | Tmp-prefixed files |

### Misplaced Archives

Flag `*.zip`, `*.tar.gz`, `*.tar.bz2`, `*.rar`, `*.7z` in the project root (not inside `assets/`, `vendor/`, or `data/` dirs where they may be intentional).

### Empty Files

Flag 0-byte files. Exceptions (do not flag):
- `__init__.py`
- `.gitkeep`
- `.keep`
- `.gitignore` (empty but intentional)

### Duplicate Indicators

Flag files matching these naming patterns:
- `*(1)*`, `*(2)*` -- OS copy suffix
- `*copy*`, `*Copy*` -- Manual copy
- `*-old*`, `*-backup*` -- Manual backup
- `*_bak*` -- Backup suffix

## 3. Stale Patterns (Archive Candidates)

### Stale Documentation

Markdown files (`.md`) outside core project docs not modified in 30+ days, when the project has commits within the last 14 days.

Directories to check: `docs/`, `experiments/`, `issues/`, `logs/`, `knowledge/`, `references/`, `tasks/`.

Exclude from staleness checks: `README.md`, `CHANGELOG.md`, `LICENSE.md`, `CLAUDE.md`, `CONTRIBUTING.md`.

### Stale Logs

Log files (`*.log`) older than 14 days.

### Draft / WIP Files

| Pattern | Description |
|---------|-------------|
| `draft-*` | Draft documents |
| `wip-*` | Work in progress |
| `old-*` | Explicitly marked old |

### Superseded Files

Files matching `*-v1.*`, `*-v2.*`, etc. when a higher version exists in the same directory. Flag the lower versions.

## 4. Large Directory Thresholds

| Level | Threshold | Action |
|-------|-----------|--------|
| Flag | > 100 MB | Include in report, suggest gitignore or cleanup |
| Warn | > 500 MB | Highlight prominently, recommend immediate action |
| Alert | > 1 GB | Top of report, likely needs gitignore + cleanup |

Common large directory offenders:
- `node_modules/` -- Often 500MB+
- `.git/` -- Flag if > 1GB (suggest `git gc` or shallow clone)
- `wandb/` -- ML run logs accumulate quickly
- `model/`, `checkpoints/` -- Model weights
- `data/` -- Dataset directories (verify if intentionally tracked)
- `venv/`, `.venv/` -- Python virtual environments (200MB+)

## 5. Zombie Patterns (Delete Candidates)

### Accidental Creation Artifacts

Files created by shell mishaps -- commands that redirected or captured output into unintended filenames.

| Pattern | Cause | Example |
|---------|-------|---------|
| `=*` | Unquoted `>=` in pip/shell | `=0.3.0` from `pip install pkg>=0.3.0` |
| `>*`, `<*`, `\|*` | Shell redirect/pipe chars | `>output` from mistyped command |
| `nohup.out` | `nohup` without redirect | Background process stdout capture |
| `core.*` | Core dump files | Process crash artifacts |
| `MUJOCO_LOG.TXT` | MuJoCo logging | Physics sim log leaked to root |
| `*.log` in project root | Runtime log leaks | `training_output.log`, `debug.log` tracked in git |

Detection: `find . -maxdepth 1 -name "=*" -o -name ">*" -o -name "<*" -o -name "nohup.out" -o -name "core.*"`

### Empty Directories

Directories containing only `.` and `..`. Common causes:
- Uninitialized git submodules (`dismech-python/` with no content)
- Leftover from `git rm -r` that didn't clean the parent dir
- Abandoned work directories

Detection: `find . -maxdepth 2 -type d -empty -not -path "./.git/*"`

Exceptions (do not flag): `.git/` internals, directories with `.gitkeep`.

### Orphaned Build Artifacts

| Pattern | Description |
|---------|-------------|
| `*.egg-info/` in project root | Stale editable install metadata |
| `*.whl` in project root | Built wheel not in `dist/` |
| `*.egg` in project root | Legacy egg format |
| `*.pth` in project root | Path configuration files (usually accidental) |

### Runtime Log Leaks

Log files that are git-tracked but should not be. Check `git ls-files "*.log"` and flag any that are not explicitly documentation.

## 6. Duplicate Detection Patterns

### Same-Name Files

Find files with identical basenames in different directories. High-signal extensions:
- `*.pdf` -- papers/references often duplicated
- `*.csv`, `*.json` -- datasets copied to multiple locations
- `*.png`, `*.jpg` -- images in both `media/` and `figures/`

Detection: `find . -type f \( -name "*.pdf" -o -name "*.csv" -o -name "*.png" \) | awk -F/ '{print $NF}' | sort | uniq -d`

Then: `find . -type f -name "<duplicate_basename>"` to locate all copies.

### Canonical Location Rules

When deduplicating, keep the copy in the canonical location:
- PDFs → `references/`
- Images → `media/` or `figures/`
- Data files → `data/`
- Config files → `src/configs/` or project root

## 7. Structural Analysis Patterns

### Cross-Project Contamination

Flag directories that appear to be separate projects embedded in this repo:

| Indicator | Evidence |
|-----------|----------|
| Own `.git/` dir | Independent git history |
| Own `package.json` or `pyproject.toml` | Independent dependency management |
| Own `CLAUDE.md` or `README.md` | Independent documentation |
| Size > 50 MB and untracked | Large unrelated content |

These should be moved to their own repo or removed.

### Directory Purpose Overlap

Common overlapping directory pairs to flag:

| Pair | Recommendation |
|------|---------------|
| `script/` + `scripts/` | Consolidate into one |
| `doc/` + `docs/` | Consolidate into one |
| `test/` + `tests/` | Consolidate into one |
| `lib/` + `libs/` | Consolidate into one |
| `util/` + `utils/` | Consolidate into one |
| `config/` + `configs/` | Consolidate into one |

### Root Clutter from Packages

When the project root has many importable Python packages (dirs with `__init__.py`), suggest grouping under a parent. Threshold: 5+ package dirs at root level.

Common grouping targets:
- Reference implementations → `papers/` or `external/`
- Utility packages → `lib/` or `packages/`
- Plugin/extension packages → `plugins/`

**Import safety checklist** before moving Python packages:
1. Map all imports: `grep -r "from <pkg>\|import <pkg>" --include="*.py" src/ tests/`
2. Check `pyproject.toml` or `setup.py` for package configuration
3. After move: update `where` and `package-dir` in `pyproject.toml`
4. Reinstall: `rm -rf *.egg-info && pip install -e .`
5. Verify: `python -c "import importlib.util; assert importlib.util.find_spec('<pkg>')"`

## 8. Project Type Detection

Detect project type to load the relevant gitignore patterns.

| Indicator File | Project Type |
|---------------|--------------|
| `pyproject.toml` | Python |
| `setup.py` / `setup.cfg` | Python |
| `requirements.txt` / `Pipfile` | Python |
| `package.json` | Node.js |
| `tsconfig.json` | TypeScript |
| `Cargo.toml` | Rust |
| `go.mod` | Go |
| `pom.xml` | Java (Maven) |
| `build.gradle` | Java (Gradle) |
| `Gemfile` | Ruby |
| `*.tex` (any) | LaTeX |
| `wandb/` or `train*.py` or `*.pt` | ML project |
| `docker-compose.yml` | Docker |
| `Dockerfile` | Docker |

A project may match multiple types (e.g., Python + ML + LaTeX). Apply all matching pattern sets.
