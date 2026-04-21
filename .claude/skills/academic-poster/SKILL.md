---
name: academic-poster
description: >
  Create and edit academic posters using tikzposter (LaTeX). Handles poster creation
  from scratch, content population from project results, theme/color customization,
  and compilation. Use when: (1) User asks to create an academic or conference poster,
  (2) User asks to make a poster for a course assignment or presentation,
  (3) User wants to update or edit an existing tikzposter .tex file,
  (4) User mentions "poster", "academic poster", "conference poster", or "course poster".
---

# Academic Poster

Generate academic posters using the `tikzposter` LaTeX document class. Posters live in `poster/` and compile with `latexmk -pdf poster.tex`.

## Workflow

### 1. Create New Poster

Copy template and fill content:

```bash
mkdir -p poster
cp .claude/skills/academic-poster/assets/poster-template.tex poster/poster.tex
```

Replace all `{{PLACEHOLDER}}` markers with actual content. Read the project's results files, report, and media to populate:
- Title, author, institute from project context
- Methods from `partN/` code and report
- Results from `results/` SQL files and `records/` pickles
- Figures from `media/` (use `../media/` relative paths, prefer `.pdf` vector files)

### 2. Choose Theme & Colors

Set in the preamble of `poster.tex`:

```latex
\usetheme{Default}        % Options: Default, Wave, Basic, Simple, Board, Envelope
\usecolorstyle{Denmark}   % Options: Default, Denmark, Britain, Sweden, Australia
```

For custom colors, define a `\definecolorstyle{Custom}{...}{...}` block. See [design-guide.md](references/design-guide.md#custom-color-palette) for the full pattern.

### 3. Compile & Verify

```bash
cd poster && latexmk -pdf poster.tex
```

After compilation, verify against the design checklist:
- Total text under 800 words
- No font below 24pt on A0
- All figures are vector PDF or 300+ DPI
- 3-5 colors max, consistent semantics
- Each block has 3-6 bullet points, no prose paragraphs
- Key Findings block has concrete quantitative claims

### 4. Edit Existing Poster

Read `poster/poster.tex`, make targeted edits. Common operations:
- Add/remove/reorder `\block{Title}{Content}` elements
- Update results table numbers
- Swap figures with `\includegraphics`
- Adjust column widths in `\column{0.XX}`

## Key Rules

- **3-column landscape A0** is the default layout
- **Bullet points only** — no prose paragraphs on posters
- **Vector figures** — always `../media/*.pdf` over `.png`
- **One idea per block** — don't combine unrelated content
- **Simplified tables** — use `booktabs`, fewer rows than paper
- **No `\figure` floats** — use `\begin{center}` + `\includegraphics` inside blocks

## Resources

- **Template:** [poster-template.tex](assets/poster-template.tex) — copy to `poster/poster.tex` to start
- **Design guide:** [design-guide.md](references/design-guide.md) — typography, color, layout rules, theme reference, pitfalls. Read when making design decisions or troubleshooting visual issues
