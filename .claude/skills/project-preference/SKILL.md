---
name: project-preference
description: Set up project documentation system, CLAUDE.md structure, and doc-db skills for a new project. Use when the user wants to initialize a new project with their preferred documentation conventions, transfer their markdown management system, set up CLAUDE.md, create doc-db skills, or mentions "project preferences", "transfer setup", "init docs", or "set up like my other project". Also use when starting any new project that needs structured documentation.
---

# Project Preference - Portable Project Setup

This skill captures the user's preferred project conventions for transfer to new projects. It bundles three things:

1. **Documentation system** - structured markdown files with typed frontmatter
2. **CLAUDE.md layout** - standard sections and conventions for project instructions
3. **Doc-db skills** - SQLite query and development guide for the Doc Database VSCode extension

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
- Active Mode instructions

**To apply:** Copy the template, fill in project-specific sections. The Documentation and Active Mode sections are reusable as-is.

### 3. Doc-db Skills (Optional)

If the project uses the Doc Database VSCode extension:

- Read [references/doc-db-query.md](references/doc-db-query.md) - creates the `doc-db-query` skill for generating SQLite queries
- Read [references/doc-db-schema.md](references/doc-db-schema.md) - the SQLite schema reference
- Read [references/doc-db-guide.md](references/doc-db-guide.md) - creates the `doc-db-guide` skill for extension development

**To apply:** Create `.claude/skills/doc-db-query/` and `.claude/skills/doc-db-guide/` in the new project, adapting table definitions to match the project's document types.

## Adaptation Notes

- **Document types are customizable**: The six types (log, experiment, issue, knowledge, reference, task) are a starting point. Add or remove types based on the project's needs.
- **Subtypes should match the domain**: `log.subtype` and `issue.subtype` values should reflect the project's concern areas.
- **The doc-db extension** needs to be installed separately. The skills just provide Claude with query and development knowledge.
- **Active Mode** section is project-agnostic and can be copied verbatim.
