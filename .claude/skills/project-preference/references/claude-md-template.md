# CLAUDE.md Template

Use this template when setting up a new project. Fill in project-specific sections (marked with `{PLACEHOLDER}`), and copy reusable sections (Documentation, Active Mode) as-is.

---

```markdown
# CLAUDE.md

## Project Overview

{Brief description of the project — what it does, its purpose, key technologies.}

## Project Reference

{Optional: pointer to detailed architecture docs, e.g., "For detailed project architecture, refer to `docs/architecture.md`."}

## Directory Structure

{Tree of the project's directory structure with brief comments. Example:}

\`\`\`
project-name/
├── .claude/           # Claude Code settings, skills, and memory
├── .git/              # Git version control
├── .gitignore
├── CLAUDE.md          # This file — project instructions for Claude
├── src/               # Source code
├── tests/             # Tests
├── logs/              # One file per log entry (<topic>.md)
├── experiments/       # One file per experiment (<topic>.md)
├── issues/            # One file per issue (<topic>.md)
├── knowledge/         # Domain knowledge and reference (<topic>.md)
├── references/        # One file per reference (<topic>.md)
├── tasks/             # PRDs and task specs (prd-<feature>.md)
└── ...
\`\`\`

## Do Not Modify

{List directories/files that should not be touched, e.g.:}
- `vendor/` — external dependencies
- `generated/` — auto-generated code

## Files You Can Modify

{List directories/files Claude is free to edit, e.g.:}
- `src/` — all source modules
- `tests/` — test files
- `logs/`, `experiments/`, `issues/`, `knowledge/`, `references/`, `tasks/` — documentation files

## Architecture

{Brief description of key modules and their responsibilities. Example:}
- **API** (`src/api/`): REST endpoints and request handlers
- **Models** (`src/models/`): Data models and database schemas
- **Services** (`src/services/`): Business logic layer

## Credentials

- GitHub: `Albatross679`, email `qifan_wen@outlook.com`
{List required environment variables for external services, e.g.:}
- API key: set `API_KEY` env var
- Database URL: set `DATABASE_URL` env var

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

- **`log`**: `status` (draft | complete), `subtype` ({CUSTOMIZE: e.g., fix | feature | refactor | setup})
- **`experiment`**: `status` (planned | running | complete | failed)
- **`issue`**: `status` (open | investigating | resolved | wontfix), `severity` (low | medium | high | critical), `subtype` ({CUSTOMIZE: e.g., backend | frontend | performance})
- **`knowledge`**: common properties only
- **`reference`**: `source`, `url`, `authors`
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
```
