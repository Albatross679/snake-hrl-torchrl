# Doc-db-query Skill (Portable)

Use this as the basis for a `doc-db-query` skill in a new project. Create `.claude/skills/doc-db-query/SKILL.md` with the content below, adapting table names and columns to the project's document types.

---

## SKILL.md Template

```yaml
---
name: pkm-query
description: Generate SQLite queries for the Doc Database VSCode extension. Use when the user asks to query, search, filter, or list documentation entries using SQL, mentions "doc-db query", "run query", "sqlite", or wants to find/filter docs by frontmatter properties. Also use when the user asks "show me all X" or "find docs where Y" about their project documentation.
---
```

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

## Default Tables

Adapt these to your project's document types:

| Table | Folder | Extra Columns |
|---|---|---|
| `log` | `logs/` | `status` (draft, complete), `subtype` (CUSTOMIZE) |
| `experiment` | `experiments/` | `status` (planned, running, complete, failed) |
| `issue` | `issues/` | `status` (open, investigating, resolved, wontfix), `severity` (low, medium, high, critical), `subtype` (CUSTOMIZE) |
| `knowledge` | `knowledge/` | -- |
| `reference` | `references/` | `source`, `url` + junction: `reference_authors` |
| `task` | `tasks/` | `status` (planned, in-progress, complete, cancelled) |

All tables share: `_file_path` (PK), `id`, `name`, `description`, `created`, `updated` + junctions `{table}_tags`, `{table}_aliases`.

## Example Queries

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

### Cross-table: all recent entries regardless of type
```sql
SELECT file_path, table_type, last_modified
FROM _file_metadata
ORDER BY last_modified DESC
LIMIT 20
```
