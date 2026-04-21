# Doc-db-guide Skill (Portable)

## How to Transfer

1. Copy the `doc-db-guide` skill from the source project's `.claude/skills/doc-db-guide/` to the new project's `.claude/skills/doc-db-guide/`
2. Update the `doc-db` extension source path if it differs in the new project
3. If the new project has different document types, update references to table definitions accordingly

The skill itself is project-agnostic — it describes the extension's architecture, commands, and development workflows. No table-specific customization is needed in the guide skill.
