# Phase 6: Write Research Report in LaTeX - Context

**Gathered:** 2026-03-10 (updated 2026-03-11 — writing & maths style)
**Status:** Ready for planning

<domain>
## Phase Boundary

Write a comprehensive LaTeX research report documenting the neural surrogate modeling pipeline (data collection → training → validation), RL training using the surrogate, and comparison against direct Elastica RL (Phase 8 baseline). The report is a living document — writing begins now (background, related work, methods) and results sections are filled in as Phases 3–8 complete. The final report is a graduation project deliverable.

Does NOT cover: rerunning experiments, new data collection, new model training, or any code changes.

</domain>

<decisions>
## Implementation Decisions

### Document format
- **New standalone `.tex` file** — not added to `overview.tex` (which stays as the project architecture reference)
- **Single-column, paper-like layout** — NeurIPS or TMLR style (clean, wide margins, ML-standard)
- **BibTeX + natbib, author-year citations** — e.g. `(Stolzle et al., 2025)` in-text
- **Table of Contents** — matches the internal report / thesis chapter style; document will grow over time
- **No institution template mandate** — free to choose; NeurIPS/TMLR style recommended

### Document structure (sections)
1. **Abstract**
2. **Introduction** — problem statement, contribution summary, paper roadmap
3. **Background** — separate section: 2D Cosserat rod PDEs, RFT friction contact, CPG actuation, PyElastica simulator
4. **Related Work** — neural surrogates for soft robots, RL with learned simulators; sourced from `knowledge/` files
5. **Methods** — data collection pipeline, surrogate model architecture, training procedure, per-element CPG encoding
6. **Experiments & Results** — surrogate accuracy (Phase 4), RL with surrogate (Phase 5), comparison vs Elastica RL (Phase 8)
7. **Discussion** — challenges & lessons learned (see below)
8. **Conclusion**

### Primary contribution / headline result
- **"Neural surrogate enables faster RL"** — central claim is speedup achieved at acceptable accuracy loss
- Phase 8 (direct Elastica RL baseline) is the comparison anchor
- Per-component accuracy and rollout fidelity are supporting evidence

### Findings & Discussion section
- **Both inline + standalone discussion section:**
  - **Inline** — critical issues woven into the relevant section (e.g., rod radius calibration inside Background, omega_z coverage gap inside Methods)
  - **Discussion section** — "Lessons Learned" synthesis covering three categories:
    1. **Physics calibration issues:** rod radius (0.001 → 0.02 m), Young's modulus, friction coefficients, serpenoid wave direction sign error — these were critical blockers
    2. **Surrogate architecture experiments:** Phase 03.1 rollout loss variants, residual MLP vs baseline MLP, architecture comparison findings
    3. **Data collection pipeline challenges:** parallel scaling bottleneck, Numba thread deadlock, stall detection false positives, omega_z coverage gap and fix (perturb_omega_std: 0.05 → 1.5 rad/s)

### Iterative writing strategy
- Writing begins immediately — Background, Related Work, Methods can be written without Phase 5/8 results
- Results sections use **placeholders** (e.g., `\placeholder{Phase 4 validation figures here}`) until experiments complete
- Report updates incrementally as each phase completes: Phase 4 → validation results, Phase 5 → RL results, Phase 8 → baseline comparison
- Final assembly and abstract written last, after all results are in

### Mathematical depth
- **Cosserat rod:** Summary with equations (~1–2 pages) — state the linear and angular momentum PDEs, define the N=21 node discretization, introduce state vector notation. Content from `knowledge/surrogate-mathematical-formulation.md` already written.
- **Surrogate model:** Full formulation — define ground-truth transition operator T: (s_t, a_t) → s_{t+1}, surrogate approximation f_θ, delta prediction formulation, training loss (single-step MSE + multi-step rollout), and per-element CPG phase encoding φ_i
- **CPG actuation:** Equations — κ_i^target = A sin(k·s_i − ω·t + φ_0), ω derived from action frequency parameter, per-element encoding φ_i = (sin(ω·t_i), cos(ω·t_i), κ_i)

### Math notation conventions
- **Vectors/matrices:** Bold lowercase/uppercase — `\mathbf{x}` for vectors, `\mathbf{M}` for matrices
- **Time derivatives:** Leibniz notation throughout — `\frac{dx}{dt}`, `\frac{\partial}{\partial s}` for spatial derivatives
- **Subscripts:** Both time and node index as subscripts with comma separation — `\mathbf{x}_{t,i}` for position of node i at time t
- **Network parameters:** Single `\theta` for all learnable parameters — `f_\theta(\mathbf{s}_t, \mathbf{a}_t)`
- **Notation table:** Include a notation/glossary table near the beginning of the document so terms can be used freely after definition

### Equation presentation
- **Display generously:** Most equations get their own displayed line (`\begin{equation}`), including definitions and intermediate expressions — not just key results
- **Number all displayed equations:** Every displayed equation gets a number, not just cross-referenced ones
- **Cosserat rod derivation:** State the final PDEs directly, cite Antman/Gazzola for derivation — no step-by-step derivation in main text
- **Surrogate formulation:** Full formulation including per-element CPG encoding — define ground-truth operator T, surrogate f_θ, delta prediction, single-step MSE loss, rollout loss, and per-element phase encoding φ_{t,i} = (sin(ω·t_i), cos(ω·t_i), κ_i)

### Writing voice & tone
- **Voice:** Passive voice throughout — "A surrogate model is trained...", "The dataset was collected..."
- **Formality:** Readable academic — formal but not stiff, no jargon without definition, plain-language explanations where helpful, approachable to non-specialists
- **Audience assumption:** Reader knows ML (RL, PPO, neural nets) but not Cosserat rods or soft robotics — explain physics foundations, skip ML basics
- **Paragraphs:** Short (3-5 sentences), varied sentence length — modern, scannable style
- **Figure/table references:** Casual inline — "(see Fig. 3)" or "the results (Table 2) show..." — abbreviated, parenthetical when possible
- **Section numbering:** All numbered with deep hierarchy (1.1.1-level) for maximum navigability
- **Algorithm descriptions:** Use `\begin{algorithm}` pseudocode environment with numbered steps, inputs/outputs for key algorithms (data collection pipeline, training loop)
- **Numerical results:** Scientific notation for small/large values — "MSE of $1.7 \times 10^{-4}$"; include units with physical quantities
- **Code in report:** None — pure math and prose, no code snippets or `\texttt{}` references
- **Discussion style:** Brief bullet-style lessons learned — enumerated items with 1-2 sentence explanations each, scannable and compact

### Claude's Discretion
- Exact LaTeX package selection (NeurIPS vs TMLR template file)
- Figure selection and layout within each section
- Exact bibliography file organization (`.bib` entries from knowledge/ citations)
- Whether to include a Limitations subsection inside Discussion or as standalone
- Appendix content (full derivation, hyperparameter tables, extended figures)
- Precise placeholder macro definition
- Exact notation table content and placement (after abstract or after introduction)

</decisions>

<specifics>
## Specific Ideas

- The math formulation is already written and can be adapted from `knowledge/surrogate-mathematical-formulation.md` directly
- The related work coverage is already deep in `knowledge/neural-surrogate-cosserat-rod.md`, `knowledge/neural-surrogates-cosserat-rod-dynamics.md`, `knowledge/surrogate-architecture-comparison.md`, and `knowledge/simulator-comparison-soft-robot-rl.md`
- Issues documented in `issues/` and experiments in `experiments/` are the source material for the Discussion section
- The report is a graduation project deliverable — it should be self-contained and readable by a thesis committee unfamiliar with PyElastica

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `overview.tex`: Existing LaTeX preamble with package setup (amsmath, amssymb, graphicx, minted, tikz, booktabs, hyperref, palatino). Can copy/adapt the preamble for the new document.
- `knowledge/surrogate-mathematical-formulation.md`: Full Cosserat rod PDE derivation, surrogate formulation, state space definition — content ready to convert to LaTeX.
- `knowledge/neural-surrogate-cosserat-rod.md`: Literature review of DD-PINN, KNODE-Cosserat, SoRoLEX, PINN surrogates — direct source for Related Work.
- `knowledge/surrogate-architecture-comparison.md`: Architecture comparison content for Discussion.
- `figures/data_validation/`: Existing figures from Phase 2 validation.
- `figures/surrogate_training/`, `figures/surrogate_validation/`: Will be populated by Phases 3–4.
- `issues/*.md`, `experiments/*.md`: Source material for Discussion section "lessons learned".

### Established Patterns
- LaTeX uses Palatino font, custom geometry from `overview.tex` — can reuse preamble patterns
- Figures saved to `figures/<section>/` subdirectories at dpi=150 with Matplotlib Agg backend
- W&B sweeps produce comparison charts saved to `figures/surrogate_training/`

### Integration Points
- Report file location: `report/report.tex` (new directory) or `report.tex` at project root — Claude decides
- Figures imported from `figures/` relative paths
- Bibliography: new `report/references.bib` file built from citations in `knowledge/` files
- Compiled PDF: `report/report.pdf` (or at root)

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-write-research-report-in-latex*
*Context gathered: 2026-03-10*
