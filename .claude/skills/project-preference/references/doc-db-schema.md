# Doc DB Schema Reference (Portable)

When transferring the `doc-db-query` skill, copy its `references/schema.md` from the source project and customize:

1. Update `subtype` Select values for `log` and `issue` tables to match the new project's domain
2. Add/remove table definitions if the project uses different document types
3. Update junction table listings accordingly

The schema structure (common columns, field type mapping, meta tables) is project-agnostic and needs no changes.
