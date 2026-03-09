---
name: pkm-guide
description: Guide for the Doc Database VSCode extension. Use when the user asks to modify, extend, debug, or understand the doc-db extension, add new commands or features, change the webview UI, update the database schema, or fix extension bugs. Also use when asked about "doc-db", "doc database", or "doc plugin" development.
---

# Doc Database - Agent Guide

VSCode extension at `/home/coder/snake-hrl-torchrl/doc-db/` that indexes project documentation frontmatter into SQLite for querying, editing, and validation.

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
3. Directory path inference via `dirToTableType()` (`logs/` → `log`, etc.)

## Commands

| Command | ID | Description |
|---|---|---|
| Rebuild Database | `doc-db.rebuildDatabase` | Drop + recreate all tables, full sync |
| Run Query | `doc-db.runQuery` | SQL input box, results in editable webview |
| Validation Errors | `doc-db.showValidationErrors` | Show type mismatches from indexing |
| New Entry | `doc-db.newEntry` | Pick table type, create file with frontmatter template |
| Dashboard | `doc-db.showDashboard` | Overview of all tables with counts and recent files |

## Architecture

### Schema (`schema.ts`)

Tables are defined in `getTableDefs()`. Each table has:
- **Main table**: `{tableName}` with `_file_path TEXT PRIMARY KEY` + scalar columns
- **Junction tables**: `{tableName}_{fieldName}` for Multi fields with `(file_path, value)` composite PK

Common fields on all tables: `id`, `name`, `description`, `created`, `updated`
Common junction tables: `{table}_tags`, `{table}_aliases`

Field type mapping:
- `Input` → TEXT
- `Select` → TEXT with CHECK constraint
- `Number` → REAL
- `Boolean` → INTEGER CHECK(0,1)
- `Date` → TEXT CHECK GLOB YYYY-MM-DD
- `Multi` → junction table

Key functions: `getTableDefs()`, `generateSchema()`, `generateDropStatements()`, `generateFrontmatterTemplate(td)`, `getScalarFields(td)`, `getJunctionFields(td)`, `dirToTableType(relDir)`

### Database (`database.ts`)

`DocDatabase` wraps sql.js (in-memory SQLite compiled to WASM).
- `open()` loads from disk, `save()` writes back, `close()` saves + closes
- `upsertFile()` deletes then re-inserts (cascades to junction tables)
- `updateField()` validates against allowed columns before UPDATE
- `addJunctionValue()` / `removeJunctionValue()` for multi-field edits
- `query(sql)` accepts SELECT/PRAGMA/EXPLAIN (read-only) or write ops
- `getTableType(filePath)` returns which table a file belongs to

### Sync Engine (`sync.ts`)

- `fullSync()` walks each table folder, indexes `.md` files into matching tables
- `syncSingleFile()` indexes one file (parse frontmatter, resolve table type, upsert, log errors)
- `resolveTableType(data, relPath)` determines table by frontmatter `type`/`fileClass` or directory
- File watchers auto-sync on create/change/delete within each table folder (`logs/**/*.md`, `experiments/**/*.md`, etc.)
- Periodic save every 30 seconds if dirty

### Webview (`webview.ts`)

Three renderers:
1. **`renderTable()`** - Read-only HTML table (validation errors)
2. **`renderEditableTable()`** - Full editing: contenteditable cells, chip UI for multi fields, file path links, CSV export
3. **`renderDashboard()`** - Card grid showing table counts and recent files

All edits flow: webview `postMessage` → extension handler → `updateField()`/`addJunctionValue()`/`removeJunctionValue()` → `db.save()` → update markdown file on disk

### Frontmatter Editing (in `extension.ts`)

- Scalar fields: regex `^(fieldName:\s*).*$` replacement
- YAML quoting: auto-quotes values containing `,:[]{}` etc.
- Array add: `addYamlArrayItem()` finds field, inserts `  - value` after last item
- Array remove: `removeYamlArrayItem()` finds and splices matching `  - value` line

## Configuration

| Setting | Default | Description |
|---|---|---|
| `doc-db.databasePath` | `.doc-db/doc-db.sqlite` | Database file location |
| `doc-db.autoWatch` | `true` | Auto-watch for file changes |

## Build & Install

```bash
cd /home/coder/snake-hrl-torchrl/doc-db
npx tsc -p ./                                          # TypeScript → out/
npx @vscode/vsce package --allow-missing-repository    # → doc-db-0.1.0.vsix
code --install-extension doc-db-0.1.0.vsix --force     # Install
```

After install, reload VSCode window to activate.

**IMPORTANT:** After modifying source code, you MUST run all three steps (compile, package, install) for changes to take effect.

## Adding a New Table

1. Add a new entry to the array in `getTableDefs()` in `schema.ts` with `name`, `folder`, and `fields`
2. Add the folder→type mapping in `dirToTableType()` in `schema.ts`
3. Recompile, repackage, reinstall
4. Run `Doc DB: Rebuild Database`
5. Update `references/schema.md` in the `doc-db-query` skill

## Adding a New Command

1. Add to `contributes.commands` in `package.json`
2. Register handler in `extension.ts` `activate()` with `vscode.commands.registerCommand()`
3. Use `ensureInitialized()` to get `db` and `syncEngine`
4. For webview output: use `renderTable()` (read-only) or `renderEditableTable()` (editable)
5. For editable webviews: add message handlers in `panel.webview.onDidReceiveMessage()`

## Common Patterns

### Query with junction tables (use correlated subqueries)
```sql
SELECT l._file_path, l.name, l.created,
  (SELECT GROUP_CONCAT(value, ' | ') FROM log_tags WHERE file_path = l._file_path) AS tags
FROM log l
ORDER BY l.created DESC
```

### Opening files from webview
```typescript
vscode.postMessage({ type: 'openFile', filePath: link.dataset.file });
// In extension: opens via vscode.workspace.openTextDocument()
```
