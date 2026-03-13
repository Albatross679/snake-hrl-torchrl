# Documentation System Specification

A structured markdown documentation system where Claude Code documents as it goes - immediately after each change, not batched at the end of the session. Each entry is a separate file in its subdirectory.

## Directory Structure

```
project-root/
├── logs/           # One file per log entry (<topic>.md)
├── experiments/    # One file per experiment (<topic>.md)
├── issues/         # One file per issue (<topic>.md)
├── knowledge/      # Domain knowledge and reference (<topic>.md)
├── references/     # One file per reference (<topic>.md)
├── tasks/          # PRDs and task specs (prd-<feature>.md)
```

## Document Types

Every Markdown documentation file MUST include a `type` property in its frontmatter, set to the file's document type:

| What | Where | Naming | When | Type |
|---|---|---|---|---|
| Logs | `logs/` | `<topic>.md` | After any code change that adds, fixes, or modifies functionality | `log` |
| Experiments | `experiments/` | `<topic>.md` | After running a simulation, test, or investigation | `experiment` |
| Issues | `issues/` | `<topic>.md` | When encountering a bug or error (before or alongside the fix) | `issue` |
| Knowledge | `knowledge/` | `<topic>.md` | When capturing domain knowledge or reference material | `knowledge` |
| References | `references/` | `<topic>.md` | When capturing external references or citations | `reference` |
| Tasks | `tasks/` | `prd-<feature>.md` | When planning a feature or task (PRDs) | `task` |

## Required Properties by Type

All types share these **common properties**: `name`, `description`, `type`, `created`, `updated`, `tags`, `aliases`.

In addition, each type has specific properties that MUST be set:

- **`log`**: `status` (draft | complete), `subtype` (CUSTOMIZE per project domain)
- **`experiment`**: `status` (planned | running | complete | failed)
- **`issue`**: `status` (open | investigating | resolved | wontfix), `severity` (low | medium | high | critical), `subtype` (CUSTOMIZE per project domain)
- **`knowledge`**: common properties only
- **`reference`**: `source`, `url`, `authors`
- **`task`**: `status` (planned | in-progress | complete | cancelled)

## Subtype Customization

The `subtype` field for `log` and `issue` should be tailored to each project's domain. Examples:

**ML/Robotics project:**
- log subtypes: fix | training | tuning | research | refactor | setup | feature
- issue subtypes: training | physics | compatibility | system | performance

**Web application:**
- log subtypes: fix | feature | refactor | setup | deploy | migration
- issue subtypes: frontend | backend | database | auth | performance | security

**Data pipeline:**
- log subtypes: fix | feature | refactor | setup | etl | monitoring
- issue subtypes: ingestion | transformation | storage | scheduling | quality

## Frontmatter Example

```yaml
---
name: fix-login-redirect
description: Fixed redirect loop on login page when session expired
type: log
status: complete
subtype: fix
created: 2026-03-11
updated: 2026-03-11
tags:
  - auth
  - bugfix
aliases:
  - login-redirect-fix
---
```

## Logging Threshold

A change warrants a log if it modifies behavior, fixes a bug, or changes configuration. Trivial edits (typos, whitespace, comment-only changes) do not need a log entry.

## CLAUDE.md Section

Add this to the project's CLAUDE.md under `## Documentation (IMPORTANT)`:

```markdown
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

- **`log`**: `status` (draft | complete), `subtype` (fix | feature | refactor | setup | ...)
- **`experiment`**: `status` (planned | running | complete | failed)
- **`issue`**: `status` (open | investigating | resolved | wontfix), `severity` (low | medium | high | critical), `subtype` (...)
- **`knowledge`**: common properties only
- **`reference`**: `source`, `url`, `authors`
- **`task`**: `status` (planned | in-progress | complete | cancelled)

**Threshold for logging:** A change warrants a log if it modifies behavior, fixes a bug, or changes configuration. Trivial edits (typos, whitespace, comment-only changes) do not need a log entry.
```
