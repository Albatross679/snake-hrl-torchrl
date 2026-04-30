# Doc-db-query Skill (Portable)

## How to Transfer

1. Copy the `doc-db-query` skill from the source project's `.claude/skills/doc-db-query/` to the new project's `.claude/skills/doc-db-query/`
2. Customize the **Tables** section in `SKILL.md` to match the new project's document types:
   - Update `subtype` allowed values for `log` and `issue` to match the project's domain
   - Add or remove table rows if the project uses different document types
3. Customize `references/schema.md`:
   - Update `subtype` Select values under `log` and `issue` tables
   - Add/remove table definitions to match the project's document types

## Key Customization Points

- `log.subtype`: e.g., `fix | feature | refactor | setup | deploy` (varies by domain)
- `issue.subtype`: e.g., `frontend | backend | database | performance` (varies by domain)
- Table list: the default six types (log, experiment, issue, knowledge, reference, task) may need additions or removals

The query rules, SQL patterns, and examples in SKILL.md are project-agnostic and can be kept as-is.
