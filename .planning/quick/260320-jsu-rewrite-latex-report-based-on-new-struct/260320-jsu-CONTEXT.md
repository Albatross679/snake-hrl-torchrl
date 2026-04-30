# Quick Task 260320-jsu: Rewrite LaTeX report based on new structure with concise, structured style - Context

**Gathered:** 2026-03-20
**Status:** Ready for planning

<domain>
## Task Boundary

Rewrite report/report.tex following the new structure in report/structure.md. Replace wordy paragraphs with tables, pseudo-code algorithms, plots, and concise language throughout.

</domain>

<decisions>
## Implementation Decisions

### Content Migration
- Condense physics (Elastica + DisMech) into a single Ch.3 section ("Neural Network Surrogate Model for Elastica"). Move detailed math derivations to appendix.
- Issue tracker: keep as an appendix section, condensed into a summary table (issue, status, resolution).

### New Sections Scope (Ch.4 & Ch.5)
- Ch.4 (RL on Elastica): populate with existing RL content from old report.
- Ch.5 (RL on DisMech/Choi2025 follow task): write with whatever DisMech/Choi2025 RL content exists in the codebase (configs, logs, experiments). Leave placeholders for missing results.

### Visual Assets
- Reuse all existing figures (~10 in figures/ and media/) in appropriate sections.
- Add \placeholder{} macros where new figures would strengthen a section (architecture diagrams, PINN loss curves, etc.).

### Claude's Discretion
- None — all areas discussed.

</decisions>

<specifics>
## Specific Ideas

- User strongly prefers structured descriptions: tables, pseudo-code, bullet lists over prose paragraphs.
- Use concise language throughout — no filler or verbose explanations.
- Existing figures: coupling_pattern, staggered_grid, surrogate_component_r2, surrogate_r2_evolution, surrogate_training_curves, rl_training_curves, rl_evaluation, sampling_comparison, rl_algorithms_pseudocode.
- New structure has 6 main chapters + appendix (see report/structure.md).

</specifics>

<canonical_refs>
## Canonical References

- report/structure.md — new report structure (authoritative)
- report/report.tex — old report content (source material)
- figures/ — existing plot assets
- media/ — additional media assets

</canonical_refs>
