---
name: project-preference
description: Set up project documentation system, CLAUDE.md structure, and doc-db skills for a new project. Use when the user wants to initialize a new project with their preferred documentation conventions, transfer their markdown management system, set up CLAUDE.md, create doc-db skills, or mentions "project preferences", "transfer setup", "init docs", or "set up like my other project". Also use when starting any new project that needs structured documentation.
---

# Project Preference - Portable Project Setup

This skill captures the user's preferred project conventions for transfer to new projects. It bundles four things:

1. **Documentation system** - structured markdown files with typed frontmatter
2. **CLAUDE.md layout** - standard sections and conventions for project instructions
3. **Doc-db skills** - SQLite query and development guide for the Doc Database VSCode extension
4. **VS Code settings** - workspace settings for Claude Code, LaTeX Workshop, and other extensions
5. **GPU Task Safety** - pre-launch GPU check to avoid conflicts with running tasks

## Working Example

This project (`snake-hrl-torchrl`) is a complete working example. See its `CLAUDE.md` and `.claude/skills/` for the target output.

## How to Use

When setting up a new project, read the relevant reference files and adapt them:

### 1. Documentation System

Read [references/documentation-system.md](references/documentation-system.md) for the full specification of:
- Directory structure (`logs/`, `experiments/`, `issues/`, `knowledge/`, `references/`, `tasks/`)
- Frontmatter schema per document type
- Required properties and allowed values
- Naming conventions and logging thresholds

**To apply:** Create the directories, then add the Documentation section to the project's CLAUDE.md. Adapt the `subtype` values in `log` and `issue` to match the new project's domain (e.g., replace `training | physics` with domain-appropriate subtypes).

### 2. CLAUDE.md Template

Read [references/claude-md-template.md](references/claude-md-template.md) for the standard CLAUDE.md layout:
- Project Overview
- Directory Structure
- Do Not Modify / Files You Can Modify
- Architecture
- Credentials
- Documentation (the doc system from step 1)
- GPU Task Safety (for GPU projects)
- Active Mode instructions

**To apply:** Copy the template, fill in project-specific sections. The Documentation and Active Mode sections are reusable as-is.

### 3. Doc-db Skills (Optional)

If the project uses the Doc Database VSCode extension:

- Read [references/doc-db-query.md](references/doc-db-query.md) - transfer instructions for the query skill
- Read [references/doc-db-schema.md](references/doc-db-schema.md) - transfer instructions for the schema reference
- Read [references/doc-db-guide.md](references/doc-db-guide.md) - transfer instructions for the extension dev guide

**To apply:** Copy the `doc-db-query` and `doc-db-guide` skills from the source project, then customize table definitions (`subtype` values, table list) to match the new project's domain.

### 4. GPU Task Safety (Optional — for GPU projects)

If the project runs GPU tasks (training, inference, sweeps), add the GPU Task Safety section to CLAUDE.md. This instructs Claude to run `nvidia-smi` before launching any GPU task, kill zombie processes, and wait for legitimate running tasks to finish.

**To apply:** Copy the GPU Task Safety section from the template in [references/claude-md-template.md](references/claude-md-template.md) into the project's CLAUDE.md.

### 5. LaTeX Report (Optional)

If the project includes a LaTeX report compiled via LaTeX Workshop:

- Add a `## LaTeX Report` section to CLAUDE.md with the report directory tree, build rules, and constraints.
- Key convention: LaTeX Workshop compiles on save — Claude should never run `pdflatex`/`latexmk` from the terminal.
- Reference the template in [references/claude-md-template.md](references/claude-md-template.md) for the standard section format.

### 6. VS Code Settings

Read [references/vscode-settings.md](references/vscode-settings.md) for workspace settings:
- Base settings (Claude Code reasoning effort)
- LaTeX Workshop with Tectonic build engine
- Extension-specific settings

**To apply:** Create `.vscode/settings.json` in the new project. Always include the base settings; merge in conditional sections (LaTeX, etc.) as needed.

## Adaptation Notes

- **Document types are customizable**: The six types (log, experiment, issue, knowledge, reference, task) are a starting point. Add or remove types based on the project's needs.
- **Subtypes should match the domain**: `log.subtype` and `issue.subtype` values should reflect the project's concern areas.
- **The doc-db extension** needs to be installed separately. The skills just provide Claude with query and development knowledge.
- **Active Mode** section is project-agnostic and can be copied verbatim.
