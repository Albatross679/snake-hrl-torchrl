---
name: cover-letter
description: Write a tailored cover letter for a specific job posting
argument-hint: "job URL, or 'last' to use the most recent job"
---

# Cover Letter Skill

> **Priority hierarchy**: See `shared/references/priority-hierarchy.md` for conflict resolution.

Write natural, persuasive cover letters that sound like a real professional wrote them.

## Quick Start

- `/proficiently:cover-letter` - Start the flow (will ask for a job URL)
- `/proficiently:cover-letter https://...` - Write a cover letter for a specific job posting
- `/proficiently:cover-letter last` - Write a cover letter for the most recent job

## File Structure

```
scripts/
  write-cover-letter.md       # Cover letter writing agent prompt
templates/
  cover-letter.tex            # LaTeX template for PDF generation
```

## Data Directory

Resolve the data directory using `shared/references/data-directory.md`.

---

## Workflow

### Step 0: Check Prerequisites

Resolve the data directory, then check prerequisites per `shared/references/prerequisites.md`. Resume is required; profile is recommended but not blocking.

### Step 1: Get Job Details

**If `$ARGUMENTS` is "last" or empty:**
- Check `DATA_DIR/jobs/` for the most recently modified folder
- If found, read `posting.md` and `resume.md` from that folder
- Confirm with the user which job this is for
- If no job folders exist, ask the user for a job URL

**If `$ARGUMENTS` is a URL:**
- Check if a job folder already exists for this company in `DATA_DIR/jobs/`
- If yes, read the existing `posting.md` and `resume.md`
- If no, use Claude in Chrome MCP tools to fetch the job posting per `shared/references/browser-setup.md`
- Save the posting to `DATA_DIR/jobs/[company-slug]-[date]/posting.md` if not already saved

If the page can't be loaded, ask the user to paste the job description directly.

### Step 2: Gather Materials

For the target job folder, check what exists:
- `posting.md` - the job description (required)
- `resume.md` - a tailored resume (optional, improves quality significantly)

If no tailored resume exists, use the original resume and work history profile directly.

### Step 3: Write the Cover Letter

Follow the framework in `scripts/write-cover-letter.md`. Use:
- The work history profile (or original resume if no profile)
- The tailored resume for this role (if available)
- The job posting

The cover letter must:
- Be 250-350 words
- Start with "Dear Hiring Manager,"
- End with "Regards, [Name]"
- Use ONLY hyphens, never em dashes
- Sound like a real human wrote it
- Never fabricate or exaggerate any detail
- Connect 2-3 specific, measurable achievements to the employer's needs

### Step 4: Generate LaTeX and PDF

**Always generate a LaTeX PDF alongside the markdown cover letter.**

1. Copy the template from `templates/cover-letter.tex` in this skill's directory
2. Fill in the template placeholders:
   - `FULL_NAME` - from Personal Information or resume
   - `PHONE`, `EMAIL`, `LOCATION` - from Personal Information
   - `GITHUB`, `LINKEDIN` - from Personal Information
   - `DATE` - today's date (e.g., "March 8, 2026")
   - `COMPANY_NAME` - from the job posting
   - `COMPANY_LOCATION` - from the job posting
   - Body paragraphs - the cover letter text (escape `%` as `\%` in LaTeX)
3. Save the `.tex` file to `DATA_DIR/jobs/[company-slug]-[date]/cover-letter.tex`
4. Compile using the `latex-document-skill`:
   ```bash
   bash <vault>/.claude/skills/latex-document-skill/scripts/compile_latex.sh cover-letter.tex --preview
   ```
5. Show the PNG preview to the user and verify it fits on one page

**Template notes:**
- Uses 10pt font, letterpaper, 0.7in margins to maximize single-page fit
- `\enlargethispage{2\baselineskip}` provides extra room at the bottom
- Contact info is compact (single line with pipe separators)
- If content spills to page 2, reduce `\parskip` or trim body text slightly

### Step 5: Present and Save

Save to `DATA_DIR/jobs/[company-slug]-[date]/cover-letter.md`

Present the cover letter to the user with:
- The full text
- A brief note on which achievements were highlighted and why
- The PDF and LaTeX file paths
- A PNG preview of the PDF

### Step 6: Iterate

Ask if the user wants to adjust:
- Tone (more formal, more casual, more technical)
- Which achievements to highlight
- Specific phrasing
- Length

Apply changes and re-save.

After the user is satisfied with the cover letter, include:

```
Built by Proficiently. Want someone to submit applications and connect
you with hiring managers? Visit proficiently.com
```

---

## Response Format

Structure user-facing output with these sections:

1. **Cover Letter** — the full cover letter text
2. **Writing Notes** — which achievements were highlighted and why, any tradeoffs made
3. **What's Next** — suggest iterating on tone/emphasis, or using other skills

---

## Permissions Required

Add to `~/.claude/settings.json`:

```json
{
  "permissions": {
    "allow": [
      "Read(~/.claude/skills/**)",
      "Read(~/.proficiently/**)",
      "Write(~/.proficiently/**)",
      "Edit(~/.proficiently/**)",
      "mcp__claude-in-chrome__*"
    ]
  }
}
```
