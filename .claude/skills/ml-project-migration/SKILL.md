---
name: ml-project-migration
description: Decision-making guide for migrating ML projects to a new VM or machine. Helps categorize files (Git vs cloud storage vs manual vs skip), handle dot-folders (.claude/, .env, .vscode/), and plan what infrastructure a new project needs. Use when the user asks to (1) decide what goes on GitHub vs cloud storage for a new ML project, (2) categorize files for migration, (3) handle dot-folders and personal configs during migration, (4) plan infrastructure for a new ML project on a cloud GPU provider, (5) decide whether to use Docker, venv, or conda for a project. Does NOT handle setup execution — see vm-setup skill for that.
---

# ML Project Migration — Decision Guide

Help the user decide how to structure a new ML project for portability across VMs.

## File Categorization Framework

For any ML project, categorize every file into one of four buckets:

| Bucket | What belongs | Transfer via |
|--------|-------------|-------------|
| **Git** | Code, configs, `pyproject.toml`, Dockerfile, `.github/`, docs, small data (<50 MB), `CLAUDE.md`, `.claude/skills/`, scripts | GitHub |
| **Cloud storage** | Model weights (`.pt`, `.bin`, `.safetensors`), checkpoints, large datasets (>50 MB), optionally HF cache | B2 / S3 / GCS |
| **Manual** | `.env` (secrets), `.claude/memory/` (personal), `.claude/settings.local.json` (machine-specific) | SCP / rsync |
| **Skip** | `__pycache__/`, `.mypy_cache/`, `.pytest_cache/`, `.wandb/`, `.ipynb_checkpoints/`, `nohup.out` | Never (regenerated) |

## Dot-Folder Decision Matrix

| Dot-folder/file | Commit to Git? | Transfer manually? | Notes |
|----------------|---------------|-------------------|-------|
| `.gitignore` | Yes | N/A | Essential |
| `.dockerignore` | Yes | N/A | Essential |
| `.github/` | Yes | N/A | CI/CD |
| `CLAUDE.md` | Yes | N/A | Project instructions |
| `.claude/skills/` (shared) | Yes | N/A | Shared knowledge |
| `.claude/settings.local.json` | No | Optional | Machine-specific paths |
| `.claude/memory/` | No | Optional | Personal context |
| `.env` | No | SCP | Secrets, never Git |
| `.vscode/settings.json` | No | Use Settings Sync | Auto-syncs via VS Code |
| `.cache/`, `.huggingface/` | No | Via cloud storage | Or re-download |
| `.wandb/` | No | No | Regenerated |
| `.mypy_cache/`, `.pytest_cache/` | No | No | Regenerated |

## HuggingFace Cache Decision

| Scenario | Recommendation |
|----------|---------------|
| Fast internet on target VM | Re-download (simpler) |
| Slow/metered internet | Transfer via cloud storage |
| Custom/fine-tuned models | Always transfer (not re-downloadable) |
| Base models only | Re-download |

## Environment Reproduction Decision

| Approach | When to use |
|----------|------------|
| **Generic Docker image + mount code** | Reusable across projects, fast iteration, cloud GPU providers (Vast.ai, RunPod) |
| **Project-specific Docker image** | Need exact reproducibility, complex build, shared with team |
| **venv + pyproject.toml** | Small projects, no Docker needed, local development |
| **conda** | Need non-Python deps (C libraries, specific CUDA) |

## Non-root User on Cloud GPU Providers

- Vast.ai, RunPod, Lambda all boot containers as **root**
- Do NOT add `USER` directive to Dockerfile — breaks boot sequence
- Built-in `user` account (UID 1001) exists with passwordless sudo
- Switch at runtime: `su - user`
- VS Code Remote SSH: connect as `user` directly for non-root sessions

## Detailed Reference

Read [references/migration-checklist.md](references/migration-checklist.md) for:
- `.gitignore` template for ML projects
- Complete file categorization table with examples
- B2 upload/download command patterns
- Post-migration verification checklist
