---
name: pkm-query
description: Generate SQLite queries for the Doc Database VSCode extension. Use when the user asks to query, search, filter, or list documentation entries using SQL, mentions "doc-db query", "run query", "sqlite", or wants to find/filter docs by frontmatter properties. Also use when the user asks "show me all X" or "find docs where Y" about their project documentation.
---

# Doc DB Query Skill

Generate correct SQLite queries for the Doc Database plugin (`doc-db.runQuery` command).

## Schema

Read [references/schema.md](references/schema.md) for the full table/column reference.

## Key Rules

1. **Multi fields live in junction tables**, not the main table.
   Junction table name: `{tableName}_{fieldName}` with columns `(file_path, value)`.

2. **Include multi fields via correlated subqueries** (not JOINs) to avoid cross-join duplication:

   ```sql
   SELECT
     l._file_path,
     l.name,
     (SELECT GROUP_CONCAT(value, ' | ') FROM log_tags WHERE file_path = l._file_path) AS tags
   FROM log l
   ```

3. **Always include `_file_path`** to enable inline editing in the result table.

4. **SQLite quirk**: `GROUP_CONCAT(DISTINCT x, separator)` with two args is not supported.
   Use correlated subqueries instead (rule 2).

5. **Boolean fields** are INTEGER 0/1. Filter with `= 1` or `= 0`.

6. **Select fields** use exact string values (case-sensitive). Check schema for allowed values.

7. **Filtering by a multi field value**: use EXISTS subquery:
   ```sql
   WHERE EXISTS (
     SELECT 1 FROM issue_tags WHERE file_path = i._file_path AND value = 'torchrl'
   )
   ```

8. **Cross-table queries**: use `_file_metadata` to find which table a file belongs to.

## Tables

| Table | Folder | Extra Columns |
|---|---|---|
| `log` | `logs/` | `status` (draft, complete), `subtype` (fix, training, tuning, research, refactor, setup, feature) |
| `experiment` | `experiments/` | `status` (planned, running, complete, failed) |
| `issue` | `issues/` | `status` (open, investigating, resolved, wontfix), `severity` (low, medium, high, critical), `subtype` (training, physics, compatibility, system, performance) |
| `knowledge` | `knowledge/` | — |
| `reference` | `references/` | `source`, `url` + junction: `reference_authors` |
| `idea` | `ideas/` | `status` (draft, exploring, implemented, abandoned), `priority` (low, medium, high) |

All tables share: `_file_path` (PK), `id`, `name`, `description`, `created`, `updated` + junctions `{table}_tags`, `{table}_aliases`.

## Examples

### All logs with tags
```sql
SELECT l._file_path, l.name, l.created, l.status,
  (SELECT GROUP_CONCAT(value, ' | ') FROM log_tags WHERE file_path = l._file_path) AS tags
FROM log l
ORDER BY l.created DESC
```

### Open issues by severity
```sql
SELECT i._file_path, i.name, i.severity, i.created,
  (SELECT GROUP_CONCAT(value, ' | ') FROM issue_tags WHERE file_path = i._file_path) AS tags
FROM issue i
WHERE i.status = 'open'
ORDER BY CASE i.severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 ELSE 4 END
```

### Recent experiments
```sql
SELECT e._file_path, e.name, e.status, e.created
FROM experiment e
ORDER BY e.created DESC
LIMIT 10
```

### Knowledge entries tagged with a keyword
```sql
SELECT k._file_path, k.name, k.description,
  (SELECT GROUP_CONCAT(value, ' | ') FROM knowledge_tags WHERE file_path = k._file_path) AS tags
FROM knowledge k
WHERE EXISTS (SELECT 1 FROM knowledge_tags WHERE file_path = k._file_path AND value = 'physics')
```

### References with authors
```sql
SELECT r._file_path, r.name, r.source, r.url,
  (SELECT GROUP_CONCAT(value, ' | ') FROM reference_authors WHERE file_path = r._file_path) AS authors,
  (SELECT GROUP_CONCAT(value, ' | ') FROM reference_tags WHERE file_path = r._file_path) AS tags
FROM reference r
ORDER BY r.created DESC
```

### Cross-table: all recent entries regardless of type
```sql
SELECT file_path, table_type, last_modified
FROM _file_metadata
ORDER BY last_modified DESC
LIMIT 20
```

### Ideas by priority
```sql
SELECT i._file_path, i.name, i.status, i.priority,
  (SELECT GROUP_CONCAT(value, ' | ') FROM idea_tags WHERE file_path = i._file_path) AS tags
FROM idea i
WHERE i.status != 'abandoned'
ORDER BY CASE i.priority WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END
```
