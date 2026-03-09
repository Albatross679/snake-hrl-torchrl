# AgentInstruction FileClass Schema

Store in: `Agent Instructions/`
FileClass: `AgentInstruction`

## Properties

| Property | Type | Options | Required |
|----------|------|---------|----------|
| status | Select | Draft, Ready, In Progress, Completed, Archived | Yes |
| scope | Select | VS Code Extension, Script, Obsidian Plugin, CLI Tool, Full Stack, Other | Yes |
| priority | Select | High, Medium, Low | Yes |
| summary | Input | Free text (one-line) | Yes |
| related | Multi | Wikilinks to related notes | No |
| date_created | Date | ISO 8601 | Yes |
| date_modified | Date | ISO 8601 | Yes |
| tags | Multi | agent-instruction, vscode, database, obsidian, automation, tooling | Yes |

## Example Frontmatter

```yaml
---
fileClass: AgentInstruction
status: Ready
scope: VS Code Extension
priority: High
summary: "Build a SQLite-backed PKM index over Obsidian frontmatter"
related:
  - "[[Some Related Note]]"
date_created: 2026-03-06
date_modified: 2026-03-06
tags:
  - agent-instruction
  - vscode
  - database
---
```

## Status Lifecycle

- **Draft** -- still being refined, not ready for execution
- **Ready** -- fully specified, can be executed as-is
- **In Progress** -- currently being worked on
- **Completed** -- built and verified
- **Archived** -- no longer relevant
