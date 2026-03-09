---
name: instruction-improver
description: "Improve, create, and refine task instructions for AI agents, storing them as AgentInstruction notes in the Obsidian vault. Use when the user asks to: (1) improve or refine a rough task description or instruction, (2) create a new agent instruction from scratch, (3) revise or enhance an existing AgentInstruction note, (4) convert a planning conversation into a stored instruction spec, or mentions 'improve instruction', 'write a spec', 'task spec', 'agent instruction', 'refine this task', 'make this actionable'."
---

# Instruction Improver

Transform rough task descriptions into precise, actionable agent instructions and store them as `AgentInstruction` notes in `Agent Instructions/`.

## Workflow

### 1. Gather Context

Before improving an instruction, gather the context needed to make it precise:

- Read the vault's fileClass definitions in `fileClasses/` to understand the data schema.
- Scan existing files relevant to the instruction (e.g., if the task references "frontmatter," read actual frontmatter from vault notes).
- Identify the tech stack the user works with (check existing configs, `package.json`, installed tools).
- Check for related AgentInstruction notes in `Agent Instructions/` to avoid duplication.

### 2. Apply Improvement Principles

Transform the input by applying these principles:

**Ground in reality.** Replace vague references with concrete details from the codebase. Instead of "parse the frontmatter," specify which fields, their types, and the exact file locations.

**Resolve ambiguity.** When the user expresses uncertainty (e.g., "should it update on change or on query? I'm not sure"), evaluate both options and recommend one with justification.

**Specify the schema.** Map abstract data descriptions to concrete types, constraints, and relationships. Reference actual field names and allowed values from fileClass definitions.

**Name the tools.** Replace generic descriptions ("a script") with specific libraries and technologies (e.g., "`gray-matter` for YAML parsing, `better-sqlite3` for SQLite").

**Add verification.** Append a checklist of testable assertions that confirm the task is complete. Each item should be a concrete check, not a vague "it works."

**Structure consistently.** Use numbered top-level sections for major components. Use sub-bullets for details within each section.

**Fix errors.** Correct typos, grammatical issues, and technical inaccuracies silently.

### 3. Structure the Output

Every improved instruction must follow this structure:

```
# Title

One-line objective.

**Stay active until you finish verification.**

## 1. [Component Name]
Requirements and details...

## 2. [Component Name]
...

## N. Implementation Details
Tech stack, file locations, dependencies...

## N+1. Verification
- [ ] Testable assertion 1
- [ ] Testable assertion 2
...
```

### 4. Store as AgentInstruction

Save the result to `Agent Instructions/<Title>.md` with frontmatter. See [references/agentinstruction-schema.md](references/agentinstruction-schema.md) for the full fileClass property reference and example frontmatter.

## Modes of Operation

**Improve rough instructions:** User pastes a rough description. Read the vault for grounding context, apply improvement principles, store result.

**Create from scratch:** User describes what they want built. Ask clarifying questions if scope is unclear, then draft the full instruction spec.

**Improve existing note:** User points to an existing `Agent Instructions/*.md` file. Read it, identify gaps (missing verification? vague tech stack? unresolved ambiguity?), and revise in place.

**Convert conversation to spec:** After a planning discussion, distill the key decisions, requirements, and implementation details into a new AgentInstruction note.
