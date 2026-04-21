# Quick Task 3: Add structured notation explanations after equations in report - Context

**Gathered:** 2026-03-11
**Status:** Ready for planning

<domain>
## Task Boundary

Convert paragraph-style notation explanations after equations in `report/report.tex` to structured tabular where-blocks. This is a formatting/style change only — no content additions or equation changes.

</domain>

<decisions>
## Implementation Decisions

### Format
- **Tabular where-block**: A `\noindent where` line followed by an aligned tabular list — each symbol on its own row with description
- Uses `@{}>{$}l<{$} @{\quad} l@{}` column spec for clean math-mode symbol alignment

### Scope
- **All equations with inline explanations** — convert every equation that currently has a paragraph-style explanation into structured tabular form
- Skip equations that already have no inline explanation (e.g., standalone definitions)

### Claude's Discretion
- Exact tabular column spacing and formatting details
- Whether to add `\smallskip` or `\medskip` spacing around where-blocks for visual consistency

</decisions>

<specifics>
## Specific Ideas

- Phase 6 CONTEXT.md establishes: "Most equations get their own displayed line", "Number all displayed equations"
- The notation table at the top already defines many symbols — where-blocks should complement (not duplicate) the notation table by providing equation-local context
- Example target format (from discussion):
  ```
  \noindent where
  \begin{tabular}{@{}>{$}l<{$} @{\quad} l@{}}
    \mathbf{x}(s,t) & rod centerline position (m) \\
    \mathbf{F}       & internal force (shear + tension) \\
  \end{tabular}
  ```

</specifics>
