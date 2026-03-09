---
name: todoist-tasks
description: >
  Create Todoist tasks from natural language. Use when the user mentions a task, to-do,
  reminder, or something they need to do. Triggers on phrases like "I need to", "remind me to",
  "add task", or any actionable item the user wants to track. Also use when the user asks to
  check, update, complete, or organize tasks on Todoist.
---

# Todoist Task Management

## Access

Todoist is a **managed MCP server** — already authenticated via OAuth. No local config needed.
Load tools with `ToolSearch` using query `+todoist <action>` (e.g., `+todoist find tasks`).

### Key tools

| Action | Tool |
|--------|------|
| Find projects | `mcp__todoist__find-projects` |
| Find tasks | `mcp__todoist__find-tasks` |
| Add tasks | `mcp__todoist__add-tasks` |
| Update tasks | `mcp__todoist__update-tasks` |
| Complete tasks | `mcp__todoist__complete-tasks` |
| Find sections | `mcp__todoist__find-sections` |
| Overview | `mcp__todoist__get-overview` |

Always load a tool via `ToolSearch` before calling it.

## User Preferences

### Hierarchy

Organize tasks in a **project > parent task > subtask** structure:
- **Project** = course or major initiative (e.g., `nlp_as3`)
- **Parent tasks** = top-level milestones or phases (e.g., `part1`, `part2`)
- **Subtasks** = concrete, actionable items under each parent

### Task style

- **Concise, imperative titles**: "Implement eval_epoch in train_t5.py" not "We need to work on implementing the eval_epoch function"
- **Use priorities**: p1 = must-do implementation, p2 = tuning/evaluation, p4 = optional/stretch
- **Due dates** via `dueString` (natural language): "March 11 2026", "tomorrow", "next Friday"
- **Descriptions** only when extra context is needed (e.g., submission instructions)

### Workflow

1. Find the relevant project first: `find-projects` with `search`
2. Find existing parent tasks: `find-tasks` with `projectId`
3. Add subtasks using `parentId` of the parent task, with `order` to control sequence
4. When creating from a spec/document, group by logical phase and set priorities by dependency order

## Known Projects

| Project | ID | Description |
|---------|----|-------------|
| nlp_as3 | `6g5H7Vvvwhx95Pvv` | CSE 5525 Assignment 3: NL-to-SQL |
