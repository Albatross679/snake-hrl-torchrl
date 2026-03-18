# CLAUDE.md

## Project Overview

Hierarchical Reinforcement Learning (HRL) for snake robot predation using TorchRL. The snake robot learns to approach and coil around prey using a two-level hierarchy: an approach policy and a coil policy, coordinated by a meta-controller.

## Project Reference

For detailed project architecture, physics, and background, refer to `overview.tex`.

## Directory Structure

Source code lives in `src/` (configs, envs, networks, observations, physics, rewards, trainers, behavioral_cloning). Physics simulation has four backends: DisMech, PyElastica, dismech-rods, and MuJoCo — with CPG actuators in `src/physics/cpg/`. External dependencies `dismech-python/` (submodule) and `dismech-rods/` (built from source, .gitignored) should not be modified. Documentation is spread across `logs/`, `experiments/`, `issues/`, `knowledge/`, `references/`, and `tasks/` — each as one-file-per-entry Markdown.

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

## Architecture

- Configs use **dataclass-based hierarchical config** (`src/configs/`), not YAML/JSON.
- Three TorchRL environment types: **approach**, **coil**, and **HRL** (meta-controller).
- Behavioral cloning pipeline in `src/behavioral_cloning/` generates demos, stores in experience buffers, and evaluates fitness — it is separate from the RL training loop.

## Todoist

- Project ID: `6fxH85hJ3hvWq8hh`

## LaTeX Report

Report is in `report/report.tex`, compiled with Tectonic. Plots go in `media/` and are included with `\includegraphics`.

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
- **`experiment`**: `status` (planned | running | complete | failed)
- **`issue`**: `status` (open | investigating | resolved | wontfix), `severity` (low | medium | high | critical), `subtype` (training | physics | compatibility | system | performance)
- **`knowledge`**: common properties only
- **`reference`**: `source`, `url`, `authors`
- **`task`**: `status` (planned | in-progress | complete | cancelled)

**Threshold for logging:** A change warrants a log if it modifies behavior, fixes a bug, or changes configuration. Trivial edits (typos, whitespace, comment-only changes) do not need a log entry.

## GPU Task Safety

Before launching any task that requires GPU (training, validation, inference, sweeps), Claude Code MUST:

1. **Check for existing GPU processes** — run `nvidia-smi` to see if any GPU-consuming process is already running.
2. **If a process is running:**
   - **Zombie/stale process** (e.g., defunct, no parent, stuck at 0% utilization for extended time): kill it with `kill -9 <PID>` and proceed.
   - **Legitimate running task**: do NOT kill it. Wait for it to finish before launching the new task. Inform the user that a GPU task is already in progress.
3. **Only then** launch the new GPU task.

## Active Mode ("Stay Active" / Long-Running Tasks)

When the user asks Claude Code to **stay active** (e.g., "keep going", "stay active and fix everything", "run until done"), Claude Code MUST:

1. **Scope** — only address issues related to the current task or explicitly requested area. Do not "fix" unrelated code that appears to work.
2. **Prioritize** — fix errors and blockers first, then warnings, then improvements.
3. **Autonomously iterate** — continuously identify and resolve issues without waiting for further prompts.
4. **Document each resolution** — write a log entry immediately after each fix, before moving to the next problem.
5. **Check in every ~5 fixes** — briefly summarize progress and remaining issues so the user can redirect if needed.
6. **Stop when** there are no remaining in-scope issues or the user asks to stop.
