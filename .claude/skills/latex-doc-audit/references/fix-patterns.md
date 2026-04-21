# LaTeX Layout Fix Patterns

## Table of Contents
- [Landscape to Portrait with resizebox](#landscape-to-portrait-with-resizebox)
- [Multirow Cell Overlap After Scaling](#multirow-cell-overlap-after-scaling)
- [Merging Small Related Tables](#merging-small-related-tables)
- [Preventing List/Block Page Splits](#preventing-listblock-page-splits)
- [Float Placement Issues](#float-placement-issues)
- [Column Width Reduction](#column-width-reduction)
- [Blank Page Elimination](#blank-page-elimination)

---

## Landscape to Portrait with resizebox

**Problem:** Table uses `\begin{landscape}...\end{landscape}` (from `pdflscape`), wasting a full page when the table only fills part of it.

**Fix:** Remove landscape wrapper, wrap the tabular in `\resizebox{\textwidth}{!}{...}`.

```latex
% BEFORE (wastes a full landscape page)
\begin{landscape}
\begin{table}
\centering
\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{...}
  ...
\end{tabular}
\caption{...}
\end{table}
\end{landscape}

% AFTER (fits portrait, auto-scaled)
\begin{table}[h!]
\centering
\resizebox{\textwidth}{!}{%
\setlength{\tabcolsep}{5pt}
\begin{tabular}{...}
  ...
\end{tabular}%
}
\caption{...}
\end{table}
```

**Key details:**
- The `%` after `{` on the resizebox line and after `\end{tabular}` prevents unwanted whitespace
- `\resizebox` requires the `graphicx` package
- Font size commands like `\small` inside resizebox are redundant (resizebox scales everything)
- If the original table is extremely wide (>2x textwidth), the scaled text becomes unreadably small; consider splitting the table instead
- Scale factor check: if original width / textwidth > 1.5, the text will be below ~7pt and may be hard to read

**When NOT to use:** When the table content is already near textwidth and landscape was unnecessary, just remove landscape without adding resizebox.

---

## Multirow Cell Overlap After Scaling

**Problem:** `\multirow[t]{N}{width}{content}` cells assume N rows of height. When `resizebox` scales the table down, rows become shorter but multirow content retains its text volume, causing overlap between adjacent multirow groups.

**Symptoms:** Overfull `\vbox` warnings, visible text collision in the PDF.

**Fix:** Restructure the table to eliminate `\multirow`. Combine the multi-row content into a single row using `p{}` columns with `\newline` for internal line breaks.

```latex
% BEFORE (multirow, breaks under resizebox)
\multirow[t]{3}{2.5cm}{Error Type A}
& \multirow[t]{3}{1cm}{Model}
& NL   & "example query"
& \multirow[t]{3}{5cm}{Long description...}
& \multirow[t]{3}{3cm}{Stats} \\
& & GT   & \texttt{gold sql} & & \\
& & Pred & \texttt{pred sql} & & \\[8pt]

% AFTER (single row, no multirow)
Error Type A
& Model
& NL: "example query" \newline GT: \texttt{gold sql} \newline Pred: \texttt{pred sql}
& Long description...
& Stats \\
```

**Key details:**
- Use `\newline` (not `\\`) for line breaks within `p{}` columns
- `\raggedright\arraybackslash` on `p{}` columns prevents justification issues
- Adjust column widths so total fits within `\textwidth` directly (no resizebox needed)
- Use `\footnotesize` or `\scriptsize` if content is dense

---

## Merging Small Related Tables

**Problem:** Two small tables (3-6 rows each) on the same topic occupy separate pages with >50% blank space each.

**Fix:** Combine into one table with section headers via `\multicolumn`.

```latex
% BEFORE: two separate tables
\begin{table}[h!]
\begin{tabular}{lll} ... 4 rows ... \end{tabular}
\caption{Category A issues.}\label{tab:a}
\end{table}

\begin{table}[h!]
\begin{tabular}{lll} ... 6 rows ... \end{tabular}
\caption{Category B issues.}\label{tab:b}
\end{table}

% AFTER: one merged table
\begin{table}[h!]
\begin{tabular}{lll}
\toprule
\textbf{Col1} & \textbf{Col2} & \textbf{Col3} \\
\midrule
\multicolumn{3}{l}{\textbf{Category A}} \\
\midrule
... 4 rows ...
\midrule
\multicolumn{3}{l}{\textbf{Category B}} \\
\midrule
... 6 rows ...
\bottomrule
\end{tabular}
\caption{Category A and B issues.}
\label{tab:a}
\label{tab:b}
\end{table}
```

**Key details:**
- Both tables must share the same column structure
- Assign both `\label`s to the merged table so existing `\autoref` references still work
- Update any prose that referenced the tables separately (e.g., "Tables 5 and 6" becomes "Table 5")

---

## Preventing List/Block Page Splits

**Problem:** An enumerated/itemized list or a cohesive text block splits across pages, with only 1-2 items on the first page and the rest on the next.

**Fix:** Wrap the block in `\begin{minipage}{\textwidth}...\end{minipage}`.

```latex
% BEFORE (list splits across pages)
Four factors explain this:
\begin{enumerate}
  \item First point...
  \item Second point...
  \item Third point...
  \item Fourth point...
\end{enumerate}

% AFTER (list stays together)
\begin{minipage}{\textwidth}
Four factors explain this:
\begin{enumerate}\itemsep0pt\parsep0pt
  \item First point...
  \item Second point...
  \item Third point...
  \item Fourth point...
\end{enumerate}
\end{minipage}
```

**Key details:**
- `\itemsep0pt\parsep0pt` reduces inter-item spacing to help fit
- Only use minipage if the block fits on a single page; if it's too tall, LaTeX will push the entire minipage to the next page, potentially creating MORE blank space
- Rule of thumb: if the block is <60% of page height, minipage is safe
- Alternative for shorter blocks: `\nopagebreak` after the intro line and `\begin{enumerate}[nosep]` (requires `enumitem` package)

---

## Float Placement Issues

### Floats pushed far from reference point

**Problem:** A table or figure appears pages away from where it's referenced in the text.

**Fixes:**
1. Use `[h!]` or `[htbp]` placement specifiers instead of `[h]` alone
2. Add `\FloatBarrier` (from `placeins` package) before sections to prevent floats from crossing section boundaries
3. Relax float placement parameters:

```latex
\renewcommand{\topfraction}{0.9}      % allow floats to fill 90% of top
\renewcommand{\bottomfraction}{0.9}   % allow floats to fill 90% of bottom
\renewcommand{\textfraction}{0.1}     % require only 10% text on float pages
\renewcommand{\floatpagefraction}{0.7} % float pages must be 70% full
```

### Orphaned section heading

**Problem:** A section heading appears at the bottom of a page with no content following it, while the content (a float) is on the next page.

**Fixes:**
1. Add `\clearpage` before the section to start it on a fresh page
2. Move the float definition to appear right after the section heading in the source
3. Use `[h!]` on the float to encourage placement at the definition point

### Two-column figure/table spanning issues

**Problem:** A `figure*` or `table*` in two-column mode appears on the wrong page.

**Fix:** Place the float definition before the point where it should appear (LaTeX processes `*` floats on the next page in two-column mode).

### Excessive blank space around floats

**Problem:** Large vertical gaps appear above or below floats.

**Fixes:**
```latex
% Reduce float-to-text spacing
\setlength{\textfloatsep}{10pt plus 2pt minus 4pt}   % top/bottom floats
\setlength{\floatsep}{8pt plus 2pt minus 2pt}        % between floats
\setlength{\intextsep}{8pt plus 2pt minus 2pt}       % inline floats [h]
```

---

## Column Width Reduction

**Problem:** Table columns specified with `p{Xcm}` sum to more than `\textwidth`.

**Diagnosis:** Calculate: sum of all `p{}` widths + (2 * number_of_columns * `\tabcolsep`). Default `\tabcolsep` is 6pt (~0.21cm). If total > textwidth, the table overflows.

**Fixes (in order of preference):**
1. Reduce `\tabcolsep`: `\setlength{\tabcolsep}{4pt}` or `\setlength{\tabcolsep}{3pt}`
2. Reduce individual column widths proportionally
3. Use smaller font: `\footnotesize` or `\scriptsize`
4. Use `\resizebox{\textwidth}{!}{...}` as last resort

**Width calculation helper:**
```
textwidth (common values):
  letter, 1in margins:  6.5in = 16.51cm
  letter, 1.2in margins: 6.1in = 15.49cm
  A4, 2.5cm margins:    16.0cm
  A4, 1in margins:      17.0cm

tabcolsep overhead = 2 * num_columns * tabcolsep_value
  3 columns, 6pt sep: 36pt = 1.27cm
  5 columns, 5pt sep: 50pt = 1.76cm
  13 columns, 5pt sep: 130pt = 4.57cm
```

---

## Blank Page Elimination

**Problem:** A page is mostly blank because the next float is landscape or too large to fit.

**Diagnosis checklist:**
1. Is the float using `landscape`? → Convert to portrait with resizebox
2. Is the float `[h]` and too tall for remaining space? → Change to `[htbp]`
3. Is there a `\clearpage` or `\newpage` forcing a break? → Remove if unnecessary
4. Is a minipage too tall for the current page? → Split content or remove minipage
5. Multiple small floats refusing to stack? → Adjust float parameters or combine tables
