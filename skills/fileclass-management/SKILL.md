---
name: fileclass-management
description: |
  Manage Obsidian notes using Metadata Menu fileClasses.
  Use when creating, modifying, or organizing notes with structured properties.
---

# FileClass Management Skill

## Core Principle

**ALWAYS define `fileClass` explicitly in frontmatter** when creating or modifying notes.

## FileClass Definitions

All fileClass definitions are stored in: `fileClasses/`

Before creating or modifying a note:
1. Read the appropriate fileClass definition file (e.g., `fileClasses/Contact.md`)
2. Use the properties and valid values defined there
3. Always include `fileClass: ClassName` as the first frontmatter property

## Priority Rules

When multiple fileClasses could apply:
1. **Explicit `fileClass` in frontmatter** (highest - always use this)
2. Folder-based query
3. Tag-based mapping (lowest)

Always use explicit frontmatter declaration to avoid ambiguity.

## Temporary FileClass Rules

Temporary files live in `Temporary/`. When creating files with `fileClass: Temporary`:
- The `type` property is **required** — always set it in frontmatter
- Valid values: `Message`, `Status Update`, `3P Update`, `FAQ`, `Newsletter`, `Other`
- Place `fileClass` first, then `type` immediately after in the frontmatter

For drop-in and disposal workflows (delete, reclassify, create Todoist tasks, modify existing files), see [references/temporary-files.md](references/temporary-files.md).

## Creating New FileClasses

When user needs a new note type:
1. Create definition file in `fileClasses/NewType.md`
2. Define properties with types (Input, Select, Multi, Date, etc.)
3. Update Metadata Menu config if folder-based auto-assignment needed
