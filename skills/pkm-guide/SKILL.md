---
name: pkm-guide
description: Guide for the PKM SQLite Index VSCode extension. Use when the user asks to modify, extend, debug, or understand the pkm-index extension, add new commands or features, change the webview UI, update the database schema, or fix extension bugs. Also use when asked about "pkm plugin", "pkm extension", or "pkm-index" development.
---

# PKM SQLite Index - Agent Guide

VSCode extension at `/Users/qifanwen/Desktop/pkm-index/` that indexes Obsidian vault frontmatter into SQLite for querying, editing, and validation.

## Source Layout

```
src/
├── extension.ts   - Command registration, message handlers, file watcher
├── schema.ts      - FileClass parsing, SQL DDL generation, frontmatter templates
├── database.ts    - PkmDatabase class (sql.js wrapper, CRUD, junction table ops)
├── sync.ts        - SyncEngine (full sync, single file sync, file watching)
├── webview.ts     - HTML renderers (renderTable, renderActivityTracker, renderEditableTable)
├── parser.ts      - Frontmatter parsing and value extraction
└── types.ts       - TypeScript interfaces (FileClassDef, FieldDef, etc.)
```

## Commands

| Command | ID | Description |
|---|---|---|
| Rebuild Database | `pkm-index.rebuildDatabase` | Drop + recreate all tables, full sync |
| Run Query | `pkm-index.runQuery` | SQL input box, results in editable webview |
| Validation Errors | `pkm-index.showValidationErrors` | Show type mismatches from indexing |
| Activity Tracker | `pkm-index.showActivityTracker` | Journal boolean grid with checkboxes |
| Broken Links | `pkm-index.findBrokenLinks` | Show unresolved MultiLink targets |
| New Note | `pkm-index.newFromFileClass` | Pick fileClass, auto-title (Journal=today), create file |
| Apply Template | `pkm-index.applyTemplate` | Add missing fileClass fields to current file |
| Recalibrate | `pkm-index.recalibrateFiles` | Move files to their fileClass-defined folders |

## Architecture

### Schema Generation (`schema.ts`)

FileClass `.md` files in `fileClasses/` are parsed. Each fileClass becomes:
- **Main table**: `{ClassName}` with `_file_path TEXT PRIMARY KEY` + scalar columns
- **Junction tables**: `{ClassName}_{fieldName}` for Multi/MultiLink fields with `(file_path, value)` composite PK

Field type mapping:
- `Input/Select/Link/Date` -> TEXT (Select has CHECK constraint for allowed values)
- `Number` -> REAL
- `Boolean` -> INTEGER CHECK(0,1)
- `Multi/MultiLink` -> junction table

Key functions: `generateSchema()`, `generateDropStatements()`, `generateFrontmatterTemplate(fc, overrides?)`, `getScalarFields(fc)`, `getJunctionFields(fc)`

### Database (`database.ts`)

`PkmDatabase` wraps sql.js (in-memory SQLite compiled to WASM).
- `open()` loads from disk, `save()` writes back, `close()` saves + closes
- `upsertFile()` deletes then re-inserts (cascades to junction tables)
- `updateField()` validates against allowed columns before UPDATE
- `addJunctionValue()` / `removeJunctionValue()` for multi-field edits
- `query(sql)` accepts SELECT/PRAGMA/EXPLAIN (read-only returns rows) or write ops

### Sync Engine (`sync.ts`)

- `fullSync()` walks vault folders, indexes files matching fileClass folders
- `syncSingleFile()` indexes one file (parse frontmatter, upsert, log errors)
- File watcher auto-syncs on create/change/delete (excludes `.obsidian/` and `fileClasses/`)
- Periodic save every 30 seconds if dirty

### Webview (`webview.ts`)

Three renderers:
1. **`renderTable()`** - Read-only HTML table (validation errors, broken links)
2. **`renderActivityTracker()`** - Journal boolean grid with checkbox toggles
3. **`renderEditableTable()`** - Full editing: contenteditable cells, chip UI for multi fields, file path links, export buttons

Editable table features:
- Scalar cells: click to edit, Enter to save, Escape to revert
- Boolean cells: checkbox toggle
- Multi fields (short): horizontal chips with x-remove + add input
- Multi fields (long, avg >60 chars): vertical blocks with x-remove + add input
- `_file_path` column: clickable links that open file via `code --reuse-window --goto`
- Export: CSV (native) or Parquet (via Python pandas)
- New Row: creates file from fileClass template, re-runs query

All edits flow: webview `postMessage` -> extension handler -> `updateField()`/`addJunctionValue()`/`removeJunctionValue()` -> `db.save()` -> update markdown file on disk (regex replace in frontmatter)

### Frontmatter Editing (in `extension.ts`)

- Scalar fields: regex `^(fieldName:\s*).*$` replacement
- YAML quoting: auto-quotes values containing `,:[]{}` etc.
- Array add: `addYamlArrayItem()` finds field, inserts `  - value` after last item
- Array remove: `removeYamlArrayItem()` finds and splices matching `  - value` line

## Build & Install

```bash
cd /Users/qifanwen/Desktop/pkm-index
npm run compile                    # TypeScript -> out/
npx vsce package                   # -> pkm-index-0.1.0.vsix
"/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code" --install-extension pkm-index-0.1.0.vsix --force
```

After install, reload VSCode window to activate.

**IMPORTANT:** VSCode loads the extension from `~/.vscode/extensions/qifanwen.pkm-index-0.1.0/`, NOT from the source directory. After modifying source code, you MUST run all three steps (compile, package, install) for changes to take effect. Simply compiling is not enough.

## Adding a New Command

1. Add to `contributes.commands` in `package.json`
2. Register handler in `extension.ts` `activate()` with `vscode.commands.registerCommand()`
3. Use `ensureInitialized()` to get `db` and `syncEngine`
4. For webview output: use `renderTable()` (read-only) or `renderEditableTable()` (editable)
5. For editable webviews: add message handlers in `panel.webview.onDidReceiveMessage()`

## Adding a FileClass Field

1. Edit the fileClass `.md` in `fileClasses/` (add to `options:` array)
2. Run `PKM: Rebuild Database` to regenerate schema
3. Update `references/schema.md` in the `pkm-query` skill if it exists

## Common Patterns

### Query with junction tables (use correlated subqueries)
```sql
SELECT p._file_path, p.title,
  (SELECT GROUP_CONCAT(value, ' | ') FROM Table_field WHERE file_path = p._file_path) AS field
FROM TableName p
```

### Opening files from webview
```typescript
// In webview HTML: send message with file path
vscode.postMessage({ type: 'openFile', filePath: link.dataset.file });
// In extension: handle with code CLI
execFile(codePath, ['--reuse-window', '--goto', fullPath], ...);
```

### Auto-generated titles for new notes
```typescript
if (picked.fc.name === 'Journal') {
  defaultTitle = today;  // YYYY-MM-DD
  overrides['date'] = today;
}
```
