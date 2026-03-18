# Doc DB SQLite Schema Reference

## Schema Rules

- Each doc type → one main table with `_file_path TEXT PRIMARY KEY`
- `Multi` fields → junction table `{table}_{field}` with columns `(file_path, value)`
- `Select` fields have CHECK constraints for allowed values
- `Boolean` fields stored as INTEGER (0/1)
- All timestamps stored as TEXT in `YYYY-MM-DDTHH:MM:SS` format
- Table type resolved by: frontmatter `type` field, frontmatter `fileClass` field, or directory path

## Common Columns (all tables)

| Column | Type |
|--------|------|
| `_file_path` | TEXT PK |
| `id` | TEXT |
| `name` | TEXT |
| `description` | TEXT |
| `created` | TEXT (datetime) |
| `updated` | TEXT (datetime) |

Common junctions: `{table}_tags (file_path, value)`, `{table}_aliases (file_path, value)`

## Tables

### log
Folder: `logs/`
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| id, name, description, created, updated | TEXT |
| status | Select: draft, complete |
| subtype | Select: fix, feature, refactor, setup, training, tuning, prompting |
Junction: `log_tags`, `log_aliases`

### experiment
Folder: `experiments/`
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| id, name, description, created, updated | TEXT |
| status | Select: planned, running, complete, failed |
Junction: `experiment_tags`, `experiment_aliases`

### issue
Folder: `issues/`
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| id, name, description, created, updated | TEXT |
| status | Select: open, investigating, resolved, wontfix |
| severity | Select: low, medium, high, critical |
| subtype | Select: training, data, model, evaluation, compatibility, performance |
Junction: `issue_tags`, `issue_aliases`

### knowledge
Folder: `knowledge/`
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| id, name, description, created, updated | TEXT |
Junction: `knowledge_tags`, `knowledge_aliases`

### reference
Folder: `references/`
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| id, name, description, created, updated | TEXT |
| source | TEXT |
| url | TEXT |
Junction: `reference_tags`, `reference_aliases`, `reference_authors`

### task
Folder: `tasks/`
| Column | Type |
|--------|------|
| _file_path | TEXT PK |
| id, name, description, created, updated | TEXT |
| status | Select: planned, in-progress, complete, cancelled |
Junction: `task_tags`, `task_aliases`

## Meta Tables

- `_file_metadata` (file_path, table_type, last_modified, last_indexed)
- `_validation_errors` (id, file_path, field_name, invalid_value, expected_type, timestamp)
