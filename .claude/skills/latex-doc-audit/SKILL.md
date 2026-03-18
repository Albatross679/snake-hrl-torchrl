---
name: latex-doc-audit
description: >
  Audit and fix LaTeX document layout issues: overflowing tables, unnecessary landscape pages,
  blank page waste, bad float placement, page-split lists, and overfull box warnings.
  Use when: (1) User asks to audit, review, or fix a LaTeX document's layout,
  (2) User mentions overflowing tables, landscape pages, blank pages, or float placement,
  (3) User asks to "fix the PDF", "check the report", "fix overflowing", or "audit the layout",
  (4) User shares a .tex file and asks about formatting or layout problems,
  (5) User wants to reduce page count or eliminate wasted space in a LaTeX document.
---

# LaTeX Document Layout Audit

## Audit Workflow

### Phase 1: Diagnose

1. **Read the .tex source** to identify all tables and figures (search for `\begin{table}`, `\begin{figure}`, `\begin{landscape}`)
2. **Read the compiled PDF** using the Read tool to visually inspect every page for:
   - Tables or figures extending beyond margins
   - Pages that are mostly blank (>40% whitespace)
   - Content blocks (lists, paragraphs) split awkwardly across pages
   - Floats appearing far from their reference point
   - Orphaned section headings (heading alone at bottom of page)
3. **Check the .log file** for warnings:
   ```
   grep "Overfull\|Underfull" report.log
   ```
   - `Overfull \hbox`: horizontal overflow (content wider than column/page)
   - `Overfull \vbox`: vertical overflow (content taller than page/cell)

### Phase 2: Report

Present findings as a numbered list grouped by severity:
1. **Overfull warnings** (content actually overflows or overlaps)
2. **Landscape pages** (often wasteful, can usually be converted to portrait)
3. **Blank page waste** (>40% of page empty due to float placement)
4. **Page-split blocks** (lists or cohesive text split across pages)
5. **Float placement issues** (figures/tables far from reference)

For each issue: state the table/figure number, source line range, and the specific problem.

### Phase 3: Fix

Apply fixes from [references/fix-patterns.md](references/fix-patterns.md) in this order:

1. **Landscape to portrait** — `resizebox` or column width reduction
2. **Multirow overlap** — restructure to single-row-per-entry (eliminates overfull vbox)
3. **Merge small tables** — combine related tables with <8 rows each
4. **List page splits** — wrap in `minipage` with `\itemsep0pt\parsep0pt`
5. **Float placement** — adjust specifiers, add `\FloatBarrier`, tune float parameters

### Phase 4: Verify

1. Recompile: `latexmk -pdf -interaction=nonstopmode <file>.tex`
2. Check `grep -c "Overfull" <file>.log` = 0
3. Read the PDF again to confirm no visual issues remain
4. Report: before/after page count, warnings eliminated, issues fixed

## Decision Guide

| Symptom | Fix |
|---------|-----|
| Table in landscape with >40% blank space | `resizebox{\textwidth}{!}` in portrait |
| Table wider than textwidth, no landscape | Reduce `\tabcolsep`, shrink column widths, or `resizebox` |
| `Overfull \vbox` from multirow table | Restructure: eliminate `\multirow`, use single rows with `\newline` |
| Two small tables (<8 rows) on same topic | Merge into one table with `\multicolumn` section headers |
| Enumerated list split across pages | Wrap intro + list in `\begin{minipage}{\textwidth}` |
| Section heading alone on page | `\clearpage` before section, or move float after heading |
| Float appears pages away from reference | Change to `[htbp]`, add `\FloatBarrier` before sections |
| Large blank gap around floats | Tune `\textfloatsep`, `\floatsep`, `\intextsep` |

## Important Caveats

- **Always read before editing.** Never modify a table without reading its full source first.
- **resizebox below 0.65x scale makes text unreadable.** If original width > 1.5x textwidth, split the table instead.
- **minipage can backfire.** If the block is >60% of page height, it may create MORE blank space by pushing to the next page.
- **Preserve all `\label` references** when merging or restructuring tables. Update `\autoref` references in prose.
- **Compile after each fix** (not just at the end) to catch regressions early.
