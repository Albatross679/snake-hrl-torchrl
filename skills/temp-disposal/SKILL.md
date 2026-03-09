---
name: temp-disposal
description: Review and dispose of temporary files in the Temporary/ folder. Generates an interactive HTML checklist for the user to approve disposal actions (delete, archive, move, reclassify). Use when the user says "clean up temporary", "dispose temp files", "review temp files", "clean up inbox", "clear temporary folder", or asks to manage/triage temporary files.
---

# Temporary File Disposal

Scan the `Temporary/` folder, build an interactive disposal checklist, and execute user-approved actions.

## Workflow

### Step 1: Scan Temporary/

List ALL files in `Temporary/` (both .md and non-.md). For each .md file, read its frontmatter to extract `fileClass`, `type`, `status`, `subject`, and `tags`.

### Step 2: Classify Each File

For each file, determine a suggested disposal action:

**Auto-check** (pre-select for disposal) if:
- `.md` with `status: Sent` or `status: Archived`
- Non-`.md` files with no wikilinks referencing them from other vault notes

**Suggested action mapping for .md files by `type`:**

| type | Default suggestion |
|------|-------------------|
| Message | Delete |
| Status Update | Delete |
| 3P Update | Delete |
| FAQ | Reclassify as Concept, move to Knowledge/ |
| Newsletter | Delete |
| Other | Delete (user should override if needed) |

**For non-.md files:** Default suggestion is "Delete". If the file is a PDF that looks academic (author names, year in filename), suggest "Move to a project folder or Archives/".

**For .md files without `fileClass: Temporary`:** Suggest "Add fileClass: Temporary or reclassify".

### Step 3: Generate the Checklist HTML

1. Read the template from [assets/disposal-checklist.html](assets/disposal-checklist.html)
2. Build a JSON array of items with these fields per file:
   ```
   { id, file, ext, type, status, suggestion, autoCheck, isNonMd }
   ```
3. Replace the `/*__DATA__*/[]` placeholder in the template with the actual JSON array
4. Write the populated HTML to `scripts/temp-disposal-review.html` in the vault root's `scripts/` folder
5. Open it with: `open scripts/temp-disposal-review.html`

### Step 4: Execute Approved Actions

When the user pastes the generated task list back, execute each item:

- **Delete**: Remove the file with `rm`
- **Archive**: Add `fileClass: Archive` frontmatter (read `fileClasses/Archive.md` for properties), move to `Archives/`
- **Move**: Move to specified folder
- **Reclassify**: Change `fileClass` and properties to match target fileClass (read the target fileClass definition first), move to the target folder

After all actions, check for broken `[[wikilinks]]` referencing disposed files.

### Step 5: Report

```
Temporary: disposed N entries
- filename.md -> deleted
- filename.md -> archived (moved to Archives/)
- filename.md -> reclassified as Concept (moved to Knowledge/)
- filename.pdf -> deleted
```
