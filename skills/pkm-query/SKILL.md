---
name: pkm-query
description: Generate SQLite queries for the PKM Index VSCode extension. Use when the user asks to query, search, filter, or list notes from their Obsidian vault using SQL, mentions "pkm query", "run query", "sqlite", or wants to find/filter notes by frontmatter properties. Also use when the user asks "show me all X" or "find notes where Y" about their vault.
---

# PKM Query Skill

Generate correct SQLite queries for the PKM Index plugin (`pkm-index.runQuery` command).

## Schema

Read [references/schema.md](references/schema.md) for the full table/column reference.

## Key Rules

1. **Multi/MultiLink fields live in junction tables**, not the main table.
   Junction table name: `{TableName}_{fieldName}` with columns `(file_path, value)`.

2. **Include multi fields via correlated subqueries** (not JOINs) to avoid cross-join duplication
   and to use a custom separator:

   ```sql
   SELECT
     p._file_path,
     p.title,
     (SELECT GROUP_CONCAT(value, ' | ') FROM ProjectResumeSession_skills WHERE file_path = p._file_path) AS skills
   FROM ProjectResumeSession p
   ```

3. **Always include `_file_path`** to enable inline editing in the result table.

4. **SQLite quirk**: `GROUP_CONCAT(DISTINCT x, separator)` with two args is not supported.
   Use correlated subqueries instead (rule 2).

5. **Boolean fields** are INTEGER 0/1. Filter with `= 1` or `= 0`.

6. **Select fields** use exact string values (case-sensitive). Check schema for allowed values.

7. **Filtering by a multi field value**: use EXISTS subquery:
   ```sql
   WHERE EXISTS (
     SELECT 1 FROM ProjectResumeSession_tags WHERE file_path = p._file_path AND value = 'ml'
   )
   ```

8. **Cross-table queries**: use `_file_metadata` to find which table a file belongs to,
   or join tables that share Link/MultiLink references (e.g., MeetingNote.contact -> Contact._file_path).

## Examples

### All projects with all fields
```sql
SELECT
  p._file_path, p.title, p.institution, p.start_date, p.end_date,
  p.project_type, p.status, p.github_link,
  (SELECT GROUP_CONCAT(value, ' | ') FROM ProjectResumeSession_advisors WHERE file_path = p._file_path) AS advisors,
  (SELECT GROUP_CONCAT(value, ' | ') FROM ProjectResumeSession_description WHERE file_path = p._file_path) AS description,
  (SELECT GROUP_CONCAT(value, ' | ') FROM ProjectResumeSession_skills WHERE file_path = p._file_path) AS skills,
  (SELECT GROUP_CONCAT(value, ' | ') FROM ProjectResumeSession_other_links WHERE file_path = p._file_path) AS other_links,
  (SELECT GROUP_CONCAT(value, ' | ') FROM ProjectResumeSession_tags WHERE file_path = p._file_path) AS tags
FROM ProjectResumeSession p
ORDER BY p.start_date DESC
```

### Journal activity for a date range
```sql
SELECT * FROM Journal WHERE date BETWEEN '2026-03-01' AND '2026-03-07' ORDER BY date
```

### Contacts filtered by department
```sql
SELECT _file_path, name, email, type, department, status FROM Contact WHERE department = 'CSE'
```

### Meetings with a specific participant
```sql
SELECT m._file_path, m.date, m.type, m.summary
FROM MeetingNote m
WHERE EXISTS (
  SELECT 1 FROM MeetingNote_participants WHERE file_path = m._file_path AND value LIKE '%ContactName%'
)
ORDER BY m.date DESC
```

### Concepts tagged with a specific label
```sql
SELECT c._file_path, c.note_type, c.domain, c.sophistication,
  (SELECT GROUP_CONCAT(value, ' | ') FROM Concept_tags WHERE file_path = c._file_path) AS tags
FROM Concept c
WHERE EXISTS (SELECT 1 FROM Concept_tags WHERE file_path = c._file_path AND value = 'attention')
```
