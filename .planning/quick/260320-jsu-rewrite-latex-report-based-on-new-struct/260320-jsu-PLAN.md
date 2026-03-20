---
phase: quick-260320-jsu
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - report/report.tex
autonomous: true
requirements: [REWRITE-01]

must_haves:
  truths:
    - "Report compiles with tectonic without errors"
    - "Report follows the 6-chapter structure from report/structure.md"
    - "Physics math derivations are in appendix, not in main chapters"
    - "Issue tracker is an appendix summary table, not subsections"
    - "All existing figures are referenced in appropriate sections"
    - "Prose is replaced with tables, bullet lists, and pseudo-code where possible"
  artifacts:
    - path: "report/report.tex"
      provides: "Complete rewritten report"
      contains: "\\section{Introduction}"
  key_links:
    - from: "report/report.tex"
      to: "figures/*.pdf"
      via: "\\includegraphics"
      pattern: "includegraphics"
---

<objective>
Rewrite report/report.tex following the new 6-chapter structure in report/structure.md, converting verbose prose to concise structured format (tables, bullet lists, pseudo-code).

Purpose: The old report has grown organically with verbose prose and misaligned structure. The new structure condenses physics into one chapter, adds two RL chapters (Elastica and DisMech), and moves detailed math to an appendix.

Output: A complete, compilable report/report.tex with new structure and concise style.
</objective>

<execution_context>
@/home/user/snake-hrl-torchrl/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/snake-hrl-torchrl/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@report/structure.md
@report/report.tex (old — source material, read in chunks)
@experiments/choi2025-full-results.md (Ch.5 content)
@experiments/choi2025-quick-validation.md (Ch.5 content)
@experiments/otpg-100k-validation-follow-target.md (Ch.5 content)
@.planning/quick/260320-jsu-rewrite-latex-report-based-on-new-struct/260320-jsu-CONTEXT.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Read old report and source materials</name>
  <files>report/report.tex</files>
  <action>
Read the ENTIRE old report/report.tex in chunks (lines 1-500, 501-1000, 1001-1500, 1501-2000, 2001-2500, 2501-2969). Extract and organize all reusable content:

1. **Preamble** (lines 1-152): Keep ALL packages, macros, title block, notation table. Preserve exactly.
2. **Introduction** (lines 187-198): Reuse, tighten language.
3. **Related Work** (lines 199-301): Reuse both subsections, convert prose to bullet lists where appropriate.
4. **Physics** (lines 302-1152): Extract KEY content for condensed Ch.3 (Elastica physics only — algorithm summary, CPG control, state representation). Move all detailed math derivations (Cosserat PDE, DisMech formulations, staggered grid details, force handling details) to an appendix.
5. **Data Collection** (lines 1153-1247): Reuse for Ch.3 data collection section.
6. **Surrogate Model** (lines 1248-1607): Reuse for Ch.3 surrogate section.
7. **PINN** (lines 1609-2443): Reuse for Ch.6 (PINN chapter).
8. **RL** (lines 2444-2593): Split into Ch.4 (Elastica RL) content.
9. **Discussion** (lines 2595-2737): Distribute relevant challenges into respective chapters.
10. **Conclusion** (lines 2739-2749): Reuse framework, update for new structure.
11. **Issue Tracker** (lines 2750-2960): Convert ALL subsections into a single summary table.
12. **Bibliography** (lines 2961-2969): Keep.

Also read:
- experiments/choi2025-full-results.md — for Ch.5 (RL on DisMech) content
- experiments/choi2025-quick-validation.md — for Ch.5 content
- experiments/otpg-100k-validation-follow-target.md — for Ch.5 OTPG content

Do NOT write anything yet. This task is read-only to build context for Task 2.
  </action>
  <verify>All 2969 lines of report.tex have been read. Experiment files read.</verify>
  <done>Complete mental model of old report content and source materials for rewrite.</done>
</task>

<task type="auto">
  <name>Task 2: Write complete rewritten report.tex</name>
  <files>report/report.tex</files>
  <action>
Write the COMPLETE new report/report.tex following this structure. Use the Write tool to create the full file in one pass.

**CRITICAL STYLE RULES (apply everywhere):**
- Prefer tables over prose for comparisons, parameter lists, results
- Prefer bullet lists over paragraphs for enumerating items
- Use algorithm/algpseudocode environments for procedures
- Concise sentences — cut filler words, hedging, redundant explanations
- Keep \Cref for cross-references (not \autoref or manual refs)

**STRUCTURE:**

### Preamble (PRESERVE from old report)
- ALL \usepackage declarations, macros, title block, \begin{document}, abstract, notation table
- Add \graphicspath{{../figures/}{../media/}} (both directories)

### Chapter 1: Introduction
- Reuse old intro content. Tighten to 3-4 concise paragraphs.
- Update \cref references to match new section labels.

### Chapter 2: Related Work
- \subsection{Neural Surrogates for ODE/PDE} — reuse old content, convert to structured format
- \subsection{RL Control for Snake and Soft Robots} — reuse old content, convert to structured format

### Chapter 3: Neural Network Surrogate Model for Elastica
This is the BIG restructure. Condense physics + data + surrogate into one chapter.

- \subsection{Elastica Physics} — Condensed overview of Cosserat rod model as it pertains to surrogate data. Include:
  - Brief rod formulation (state vector, forces, moments) as a summary TABLE, not full derivation
  - CPG control section (reuse from old 2.2, keep equations)
  - Backend comparison TABLE (reuse from old 2.5)
  - Reference appendix for full derivations: "See \Cref{app:physics} for detailed PDE derivations."

- \subsection{Data Collection} — Reuse old Section 4 content. Convert pipeline description to bullet list or numbered steps. Keep the quality metrics table.

- \subsection{Model Architecture and Training} — Merge old Sections 5.1-5.4:
  - State representation (brief, table format)
  - Phase encoding (keep equation, shorten explanation)
  - Architecture (keep diagram description, hyperparameter TABLE)
  - Two-phase training (table comparing Phase 1 vs Phase 2)
  - Density-weighted sampling (brief paragraph + equation)

- \subsection{Results} — Reuse old Section 5.6. Keep figures (surrogate_component_r2, surrogate_r2_evolution, surrogate_training_curves). Convert prose results to table.

### Chapter 4: Reinforcement Learning on Elastica
- \subsection{Environment and Training Setup} — Reuse old Section 7.1 content. Hyperparameters as TABLE.
- \subsection{Direct Simulation Results} — Reuse old 7.1.2. Keep rl_training_curves and rl_evaluation figures.
- \subsection{Surrogate Environment Results} — Reuse old 7.2. Keep sampling_comparison figure.
- \subsection{Challenges} — Reuse old 7.3, convert to bullet list.

### Chapter 5: Reinforcement Learning on DisMech (Choi2025 Follow Task)
Write using content from experiment files. Structure:

- \subsection{Task Description} — Describe the follow_target task from Choi2025 paper. Brief paragraph.
- \subsection{Algorithm Comparison} — Table comparing PPO, SAC, OTPG configurations and results.
  - PPO: from choi2025-full-results.md
  - SAC: from choi2025-full-results.md and choi2025-quick-validation.md
  - OTPG: from otpg-100k-validation-follow-target.md
- \subsection{Results} — Write available results. Use \placeholder{} for missing long-run results.
- Include rl_algorithms_pseudocode figure from media/ if appropriate.

### Chapter 6: Physics-Informed Neural Network
Reuse old Section 6 content (lines 1609-2443). This is already well-structured. Main changes:

- \subsection{Mathematical Formulation} — How to formulate PDE systems for PINNs. Reuse old 6.1 Standard PINN content.
  - \subsubsection{Standard PINN} — reuse, tighten
  - \subsubsection{Physics Regularizer} — reuse constraint descriptions, convert to TABLE
  - \subsubsection{DD-PINN Method} — reuse, keep equations
  - \subsubsection{Limitations} — add brief note on what PINNs cannot handle

- \subsection{PDE System for Snake Robot} — Reuse old 6.3 implementation content.
  - Differentiable Cosserat RHS
  - Physics-regularized loss
  - DD-PINN ansatz
  - Nondimensionalization (brief)
  - ReLoBRaLo (brief)

- \subsection{Results and Comparison} — Reuse old 6.4. Keep the four-way comparison content and any tables.

### Conclusion
- Reuse old conclusion framework. Update to reference 6 chapters. Concise.

### Appendix A: Detailed Physics Derivations
Move here from old report:
- Full Cosserat rod PDE derivation (old 2.1)
- PyElastica integration details (old 2.3 — staggered grid, method of lines, external forces)
- DisMech formulation details (old 2.4 — system formulation, DER algorithm, implicit integration)
- Label as \appendix \section{Detailed Physics Derivations} \label{app:physics}

### Appendix B: Issue Tracker
Convert ALL issue tracker subsections (Physics Calibration, Training, TorchRL Compatibility, Performance/System) into ONE summary table:
| ID | Category | Issue | Status | Resolution |
Extract issue name, status, one-line resolution from each \paragraph in old report.

### Bibliography
- Keep \bibliographystyle{plainnat} and \bibliography{references}

**FIGURE REFERENCES — use these exact filenames:**
- coupling_pattern (figures/)
- staggered_grid (figures/)
- surrogate_component_r2 (figures/)
- surrogate_r2_evolution (figures/)
- surrogate_training_curves (figures/)
- rl_training_curves (figures/)
- rl_evaluation (figures/)
- sampling_comparison (figures/)
- rl_algorithms_pseudocode (media/)

**PLACEHOLDER MACROS — add where new figures would help:**
- \placeholder{Surrogate architecture diagram}
- \placeholder{PINN loss curves}
- \placeholder{DisMech/Choi2025 training curves}
- \placeholder{OTPG vs PPO vs SAC comparison plot}

**WHAT TO AVOID:**
- Do NOT drop any existing equations that are central to understanding (CPG, loss functions, PINN losses). Move derivation steps to appendix but keep final-form equations in main text.
- Do NOT change \label names for sections that have existing \cref references — or update ALL \cref references to match.
- Do NOT remove any \usepackage or macro definitions from preamble.
- Do NOT use \bm{} for bold vectors — use \mathbf{} per project convention (except existing \bm uses in notation table which can stay).
  </action>
  <verify>
    <automated>cd /home/user/snake-hrl-torchrl/report && tectonic report.tex 2>&1 | tail -20</automated>
  </verify>
  <done>
- report/report.tex compiles without errors
- Structure matches report/structure.md (6 chapters + appendix)
- Physics derivations are in appendix, not main chapters
- Issue tracker is a single appendix table
- All existing figures referenced with \includegraphics
- Prose converted to tables/bullets/algorithms throughout
- Concise language, no filler
  </done>
</task>

</tasks>

<verification>
1. `cd /home/user/snake-hrl-torchrl/report && tectonic report.tex` compiles without errors
2. New report has 6 \section{} commands (chapters) plus \appendix sections
3. All existing figures referenced: coupling_pattern, staggered_grid, surrogate_component_r2, surrogate_r2_evolution, surrogate_training_curves, rl_training_curves, rl_evaluation, sampling_comparison
4. Issue tracker is a single longtable/tabular, not multiple subsections
5. No verbose prose paragraphs where a table or list would suffice
</verification>

<success_criteria>
- Compilable report.tex with zero tectonic errors
- 6-chapter structure matching report/structure.md
- All old content migrated (nothing lost, only reorganized and condensed)
- Structured format: tables for comparisons/parameters/issues, bullets for enumerations, algorithms for procedures
- Physics derivations in appendix with forward references from Ch.3
- Ch.5 populated with available Choi2025/OTPG data, placeholders for gaps
</success_criteria>

<output>
After completion, create `.planning/quick/260320-jsu-rewrite-latex-report-based-on-new-struct/260320-jsu-SUMMARY.md`
</output>
