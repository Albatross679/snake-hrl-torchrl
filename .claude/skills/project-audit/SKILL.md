---
name: project-audit
description: Audit any project directory for clutter, unused files, uncommitted changes, stale artifacts, large runtime directories, zombie files, duplicate content, and structural disorganization -- then present categorized recommendations the user can selectively execute. Use when the user asks to "organize", "clean up", "audit", "declutter", or "tidy up" a project or directory, or asks about unused files, stale artifacts, zombie files, duplicate files, what to gitignore, how to reduce project size, or "what's all this junk".
---

# Project Audit

Scan a project directory, categorize findings, present an actionable report, and execute user-approved cleanup actions.

This skill works for any project directory. Pattern definitions in [references/patterns.md](references/patterns.md) are extensible -- load them during the Scan phase for the full pattern catalog.

## Audit Workflow

### Phase 1: Scan

Gather raw data about the project directory. Run each step and collect results.

1. **Git status** -- Run `git status` to identify modified, deleted, and untracked files. If not a git repo, note this and skip git-related checks.

2. **Runtime/cache directory sizes** -- Run `du -sh` on common runtime and cache directories that exist:
   - `output/`, `wandb/`, `mlruns/`, `checkpoints/`, `model/`
   - `__pycache__/`, `.cache/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`
   - `node_modules/`, `dist/`, `build/`, `.next/`, `.nuxt/`
   - `venv/`, `.venv/`, `.tox/`, `htmlcov/`
   - `target/` (Rust), `vendor/` (Go)

3. **Root-level clutter** -- List root-level files and flag potential clutter:
   - Temp files: `*.tmp`, `*.temp`, `*.bak`, `*.orig`, `*.swp`, `temp.*`, `tmp.*`
   - Archives: `*.zip`, `*.tar.gz`, `*.rar` in project root
   - OS files: `.DS_Store`, `Thumbs.db`, `desktop.ini`
   - Compiled artifacts: `*.pyc` outside `__pycache__/`

4. **Gitignore gaps** -- Check `.gitignore` for missing common patterns. Detect the project type (see patterns.md "Project Type Detection") and verify ecosystem-specific patterns are present.

5. **Stale file detection** -- Check documentation directories for stale files. A file is "stale" if not modified in 30+ days AND the project has commits within the last 14 days. Directories to check: `docs/`, `experiments/`, `issues/`, `logs/`, `knowledge/`, `references/`, `tasks/`.

6. **Zombie file detection** -- Identify files that should not exist. See [references/patterns.md](references/patterns.md) "Zombie Patterns" for the full catalog:
   - **Accidental creation artifacts**: Files created by redirected command output (e.g., `=0.3.0` from `pip install pyelastica>=0.3.0` without quotes). Look for files whose names start with `=`, `>`, `<`, `|`, or contain shell metacharacters.
   - **Empty directories**: Dirs containing only `.` and `..` (empty submodules, leftover from deleted content). Exclude `.git`-managed dirs.
   - **Orphaned build artifacts**: `*.egg-info/` dirs, `*.egg` files, `*.whl` in project root, stale `.pth` files.
   - **Runtime log leaks**: `training_output.log`, `nohup.out`, `*.log` in project root that are git-tracked but should not be.

7. **Duplicate file detection** -- Find files with identical names (or content) in multiple locations:
   - Run `find . -type f -name "*.pdf" -o -name "*.png" -o -name "*.csv"` and group by basename. Flag basenames appearing in 2+ locations.
   - For tracked duplicates, compare file sizes to confirm. Suggest keeping one canonical copy.

8. **Structural analysis** -- Detect organizational issues:
   - **Cross-project contamination**: Large untracked directories that appear to be separate projects (contain their own `.git/`, `package.json`, `pyproject.toml`, or `CLAUDE.md`). Flag with size.
   - **Directory purpose overlap**: Multiple dirs serving the same role (e.g., `script/` and `scripts/`, `doc/` and `docs/`). Suggest consolidation.
   - **Misplaced files**: Files in wrong directories (e.g., an experiment log in `doc/experiments/` when `experiments/` exists at root).
   - **Root clutter from packages**: Python packages or reference dirs that clutter the project root. Check if many top-level dirs are importable packages (have `__init__.py`) that could be grouped under a parent directory.
   - **Import dependency mapping**: Before recommending any move of a Python package directory, run `grep -r "from <pkg>" --include="*.py" src/ tests/` to identify all import sites. Track which packages have active imports so the reorganization plan accounts for them.

9. **Load pattern catalog** -- Read [references/patterns.md](references/patterns.md) for the full list of patterns organized by category and ecosystem. Cross-reference against scan results.

### Phase 2: Categorize

Group all findings into exactly five action categories.

**Gitignore** -- Items that should be added to `.gitignore`:
- Build artifacts and compiled output
- Runtime directories (caches, virtual environments, node_modules)
- IDE and editor files
- OS-generated files
- Large model weights or dataset directories that should not be tracked

For each item, show the gitignore pattern to add.

**Delete** -- Files safe to remove:
- Temp files, backup files, swap files
- Empty files (except `__init__.py`, `.gitkeep`, `.keep`)
- Empty directories (dead submodules, leftover from deletions)
- Duplicate indicators: files with `(1)`, `copy`, `-old`, `-backup` in name
- Duplicate content: same file in 2+ locations (keep canonical copy, delete others)
- Archives in project root (unless they serve a purpose)
- Compiled artifacts already covered by source
- Zombie files: accidental creation artifacts (see scan step 6)
- Cross-project contamination: separate projects nested inside this repo (flag with size, confirm before removing)

**Reorganize** -- Structural moves to clean up project layout:
- Consolidate duplicate-purpose directories (e.g., `script/` + `scripts/` → one)
- Move misplaced files to correct directories (e.g., experiment log in `doc/` → `experiments/`)
- Group root clutter under parent directories (e.g., 14 paper-implementation dirs → `papers/`)
- **Import safety**: For any Python package move, first map all imports with `grep -r "from <pkg>" --include="*.py"`. Update `pyproject.toml` (`package-dir`, `where`, `include`) and reinstall with `pip install -e .`. Verify with `importlib.util.find_spec()` for each moved package.

**Archive** -- Files to move to an `archive/` folder:
- Stale documentation (not modified in 30+ days, project recently active)
- Old experiment logs and superseded files
- Draft/WIP files that appear abandoned: `draft-*`, `wip-*`, `old-*`
- Versioned files where a newer version exists: `*-v1.*` when `*-v2.*` exists

**Commit** -- Meaningful uncommitted changes that should be tracked:
- Modified tracked files with substantive changes
- Untracked documentation or code files that should be version-controlled
- Deleted files that should be committed as deletions

### Phase 3: Report

Present findings as a structured report.

**Summary line:** Total files flagged across all categories, total reclaimable disk space.

**Per-category section format:**

```
## [Category Name] ([N] files, [size])

| File | Size | Reason |
|------|------|--------|
| path/to/file | 1.2 MB | [why flagged] |

Suggested action: [specific command or description]
```

Rules:
- Show all five categories even if empty. For empty categories: "No items found."
- Sort files within each category by size (largest first).
- Show individual file sizes in human-readable format (KB, MB, GB).
- End each category with a concrete "Suggested action" line.

### Phase 4: Execute

After presenting the report, ask the user which categories to act on. Offer options like "Execute all", "Just Gitignore and Delete", "Skip Archive", or individual category selection.

**NEVER auto-execute. Always present the report first, then wait for explicit user approval per category.**

**Gitignore execution:**
- Show the patterns to append to `.gitignore`.
- Show a diff preview of the `.gitignore` changes.
- Append patterns (additive only -- never overwrite existing entries).
- Deduplicate: skip patterns already present.

**Delete execution:**
- Show the exact file list with sizes.
- For git-tracked files: use `git rm`.
- For untracked files: use `rm`.
- For empty directories: use `rmdir` (safe -- fails if non-empty).
- For cross-project contamination dirs: use `rm -rf` after explicit confirmation including the dir size.
- Ask for confirmation before deleting.

**Reorganize execution:**
- Present the full move plan before executing. Show source → destination for each item.
- For git-tracked files/dirs: use `git mv`.
- For untracked files/dirs: use `mv`.
- **Python package moves** (dirs with `__init__.py` that are imported):
  1. Create the target parent directory (e.g., `mkdir -p papers/`).
  2. `git mv` each package dir to the new location.
  3. Update `pyproject.toml`: add the new parent to `[tool.setuptools.packages.find] where`, and add explicit `[tool.setuptools.package-dir]` mappings for each moved package.
  4. `rm -rf *.egg-info && pip install -e .` to reinstall.
  5. Verify with `python -c "import importlib.util; spec = importlib.util.find_spec('<pkg>'); assert spec and '<new_parent>/' in spec.origin"` for each moved package.
  6. If verification fails, debug the editable finder mappings before continuing.
- **Directory consolidation** (e.g., `scripts/` → `script/`): `git mv` contents from source into target, then `rmdir` the empty source.
- **Misplaced file moves**: `git mv` for tracked files, `mv` for untracked.

**Archive execution:**
- Create `archive/` directory if it does not exist.
- For git-tracked files: use `git mv`.
- For untracked files: use `mv`.
- Show the full file list before moving. Wait for confirmation.

**Commit execution:**
- Stage the relevant files with `git add` (individual files, not `-A`).
- Show `git diff --cached --stat` for review.
- Ask the user for a commit message or suggest one based on the changes.
- Create the commit only after user confirms the message.

## Design Notes

- **Project-agnostic**: This skill works for any project directory. Detect project type and adapt patterns accordingly.
- **Stale threshold**: Files not modified in 30+ days when the project has commits within the last 14 days. Adjust threshold based on project activity level.
- **Large directory threshold**: Flag directories over 100MB. Warn at 500MB+.
- **Gitignore priority**: If a file is already in `.gitignore`, lower its priority in Delete recommendations -- it is already excluded from version control.
- **Monorepo caution**: For monorepos or unfamiliar project structures, ask the user about directory purposes before categorizing. Do not assume.
- **Extensible patterns**: Load [references/patterns.md](references/patterns.md) for the full pattern catalog. To support new project types, add a new section to that file.
