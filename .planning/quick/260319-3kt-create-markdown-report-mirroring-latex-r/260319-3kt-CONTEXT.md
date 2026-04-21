# Quick Task 260319-3kt: Create Markdown report mirroring LaTeX report — Context

**Gathered:** 2026-03-19
**Status:** Ready for planning

<domain>
## Task Boundary

Create a Markdown report (`report/report.md`) that mirrors the current LaTeX report (`report/report.tex`). The Markdown file becomes the base file for all future content edits — modify Markdown first, then propagate changes to LaTeX.

</domain>

<decisions>
## Implementation Decisions

### Content Completeness
- Full verbatim mirror: copy ALL prose text from LaTeX into Markdown
- Future edits happen in Markdown first, then propagate to LaTeX

### Math Notation
- Keep raw LaTeX math using `$...$` (inline) and `$$...$$` (display) delimiters
- Renders in most Markdown viewers (GitHub, VSCode, Obsidian)
- Easiest to convert back to LaTeX

### Figures and Diagrams
- Use Markdown image syntax: `![caption](path/to/figure.png)` for figures
- TikZ diagrams get a descriptive placeholder comment (not raw LaTeX)
- Tables converted to Markdown tables where feasible
- Complex tables that don't convert cleanly can use HTML or a descriptive note

### Claude's Discretion
- File location: `report/report.md` (same directory as `report.tex`)
- Bibliography handling: reference citations as `[Author, Year]` or keep `\cite{key}` notation for easy LaTeX round-trip

</decisions>

<specifics>
## Specific Ideas

- The LaTeX report has ~2970 lines with 9 main sections + appendix + bibliography
- Sections: Notation, Introduction, Related Work, Physics Simulation Backends, Data Collection, Neural Surrogate Model, Physics-Informed Surrogate, Reinforcement Learning, Discussion, Conclusion, Appendix (Issue Tracker)
- Custom LaTeX macros (\xhat, \avec, \fssm, \fnn, \phig) should be expanded or noted in the Markdown

</specifics>
