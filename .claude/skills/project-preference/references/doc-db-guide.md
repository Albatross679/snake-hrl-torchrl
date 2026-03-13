# Doc-db-guide Skill (Portable)

Use this as the basis for a `doc-db-guide` skill in a new project. Create `.claude/skills/doc-db-guide/SKILL.md` with the content below, adapting the extension path to the new project.

---

## SKILL.md Template

```yaml
---
name: pkm-guide
description: Guide for the Doc Database VSCode extension. Use when the user asks to modify, extend, debug, or understand the doc-db extension, add new commands or features, change the webview UI, update the database schema, or fix extension bugs. Also use when asked about "doc-db", "doc database", or "doc plugin" development.
---
```

## Source Layout

```
doc-db/src/
├── extension.ts   - Command registration, message handlers, file watcher
├── schema.ts      - Hardcoded table definitions, SQL DDL generation, frontmatter templates
├── database.ts    - DocDatabase class (sql.js wrapper, CRUD, junction table ops)
├── sync.ts        - SyncEngine (full sync, single file sync, file watching)
├── webview.ts     - HTML renderers (renderTable, renderEditableTable, renderDashboard)
├── parser.ts      - Frontmatter parsing and value extraction
├── types.ts       - TypeScript interfaces (TableDef, FieldDef, etc.)
└── sql.js.d.ts    - Type declarations for sql.js
```

## Key Design

Doc-db uses **hardcoded table definitions** in `schema.ts` (`getTableDefs()`) instead of reading fileClass files. Table type is resolved by:
1. Frontmatter `type` field
2. Frontmatter `fileClass` field (legacy)
3. Directory path inference via `dirToTableType()` (`logs/` -> `log`, etc.)

## Commands

| Command | ID | Description |
|---|---|---|
| Rebuild Database | `doc-db.rebuildDatabase` | Drop + recreate all tables, full sync |
| Run Query | `doc-db.runQuery` | SQL input box, results in editable webview |
| Validation Errors | `doc-db.showValidationErrors` | Show type mismatches from indexing |
| New Entry | `doc-db.newEntry` | Pick table type, create file with frontmatter template |
| Dashboard | `doc-db.showDashboard` | Overview of all tables with counts and recent files |

## Architecture Summary

- **Schema** (`schema.ts`): Tables defined in `getTableDefs()`. Main tables have `_file_path TEXT PRIMARY KEY` + scalar columns. Junction tables for Multi fields: `{tableName}_{fieldName}` with `(file_path, value)`.
- **Database** (`database.ts`): sql.js wrapper. `upsertFile()` deletes then re-inserts. `updateField()` validates columns. `addJunctionValue()`/`removeJunctionValue()` for multi-field edits.
- **Sync** (`sync.ts`): `fullSync()` walks folders, `syncSingleFile()` indexes one file. File watchers auto-sync on create/change/delete.
- **Webview** (`webview.ts`): `renderTable()` (read-only), `renderEditableTable()` (full editing with chips), `renderDashboard()` (card grid).
- **Frontmatter editing** (`extension.ts`): Regex-based scalar replacement, YAML array add/remove.

## Adding a New Table

1. Add entry to `getTableDefs()` array in `schema.ts`
2. Add folder->type mapping in `dirToTableType()` in `schema.ts`
3. Recompile, repackage, reinstall
4. Run `Doc DB: Rebuild Database`
5. Update the `doc-db-query` skill's schema reference

## Adding a New Command

1. Add to `contributes.commands` in `package.json`
2. Register handler in `extension.ts` `activate()` with `vscode.commands.registerCommand()`
3. Use `ensureInitialized()` to get `db` and `syncEngine`
4. For webview output: use `renderTable()` or `renderEditableTable()`
5. For editable webviews: add message handlers in `panel.webview.onDidReceiveMessage()`

## Build & Install

```bash
cd {PROJECT_ROOT}/doc-db
npx tsc -p ./                                          # TypeScript -> out/
npx @vscode/vsce package --allow-missing-repository    # -> doc-db-X.X.X.vsix
code --install-extension doc-db-X.X.X.vsix --force     # Install
```

After install, reload VSCode window to activate. After modifying source, you MUST run all three steps.

## Configuration

| Setting | Default | Description |
|---|---|---|
| `doc-db.databasePath` | `.doc-db/doc-db.sqlite` | Database file location |
| `doc-db.autoWatch` | `true` | Auto-watch for file changes |
