# Phase 6: Write Research Report in LaTeX - Research

**Researched:** 2026-03-10
**Domain:** LaTeX academic writing, research report structure, scientific document authoring
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Document format:** New standalone `.tex` file — not added to `overview.tex` (which stays as architecture reference)
- **Layout:** Single-column, paper-like — NeurIPS or TMLR style (clean, wide margins, ML-standard)
- **Bibliography:** BibTeX + natbib, author-year citations — e.g. `(Stolzle et al., 2025)` in-text
- **Table of Contents:** Yes — matches internal report / thesis chapter style; document grows over time
- **No institution template mandate** — NeurIPS/TMLR style recommended
- **Document sections (required order):**
  1. Abstract
  2. Introduction — problem statement, contribution summary, paper roadmap
  3. Background — Cosserat rod PDEs, RFT friction contact, CPG actuation, PyElastica simulator
  4. Related Work — neural surrogates for soft robots, RL with learned simulators
  5. Methods — data collection pipeline, surrogate architecture, training procedure, per-element CPG encoding
  6. Experiments & Results — surrogate accuracy (Phase 4), RL with surrogate (Phase 5), comparison vs Elastica RL (Phase 8)
  7. Discussion — challenges & lessons learned
  8. Conclusion
- **Primary contribution:** "Neural surrogate enables faster RL" — speedup at acceptable accuracy loss; Phase 8 direct Elastica RL baseline is the comparison anchor
- **Discussion section structure:** Both inline + standalone:
  - Inline: critical issues woven into relevant section
  - Standalone: "Lessons Learned" synthesis in three categories: (1) physics calibration, (2) surrogate architecture experiments, (3) data collection pipeline challenges
- **Iterative writing strategy:** Write Background/Related Work/Methods immediately; use `\placeholder{}` macros in Results; fill in as Phases 4/5/8 complete
- **Mathematical depth:**
  - Cosserat rod: ~1-2 pages with PDEs, N=21 discretization, state vector notation
  - Surrogate model: Full formulation — T, f_θ, delta prediction, MSE + rollout loss, per-element CPG phase encoding φ_i
  - CPG actuation: κ_i^target = A sin(k·s_i − ω·t + φ_0), per-element encoding φ_i = (sin(ω·t_i), cos(ω·t_i), κ_i)

### Claude's Discretion

- Exact LaTeX package selection (NeurIPS vs TMLR template file)
- Figure selection and layout within each section
- Exact bibliography file organization (.bib entries from knowledge/ citations)
- Whether to include a Limitations subsection inside Discussion or as standalone
- Appendix content (full derivation, hyperparameter tables, extended figures)
- Precise placeholder macro definition

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope
</user_constraints>

---

## Summary

Phase 6 writes a standalone graduation-project research report documenting the complete neural surrogate pipeline: data collection (Phases 1–2.2), model training and architecture experiments (Phases 3–3.1), validation (Phase 4), RL training (Phase 5), and comparison against direct Elastica RL (Phase 8). The report is a living document — Background, Related Work, and Methods sections can be written immediately from existing `knowledge/` files; Results sections use placeholder macros until upstream phases complete.

The primary authoring challenge is not LaTeX syntax — it is organizing already-rich project knowledge (27 knowledge files, 30 issue documents, 9 experiment files) into a coherent academic narrative. The project has unusually complete source material: full mathematical formulation in `knowledge/surrogate-mathematical-formulation.md`, literature review in `knowledge/neural-surrogate-cosserat-rod.md` and `knowledge/neural-surrogates-cosserat-rod-dynamics.md`, and architecture comparison in `knowledge/surrogate-architecture-comparison.md`.

LaTeX is not installed on this machine. The document must be written and compiled in a Docker environment or exported to Overleaf. The `report/` directory does not yet exist and must be created. The existing `overview.tex` preamble (Palatino font, fancyhdr, amsmath, booktabs, tikz, minted) is a verified working template that can be adapted; the project's `latex` skill defines the general academic style matching this preamble.

**Primary recommendation:** Create `report/report.tex` using the project's existing general-academic LaTeX preamble (Palatino, natbib, booktabs, hyperref), write all content sections immediately except Results, and define `\newcommand{\placeholder}[1]{\textcolor{gray}{[PLACEHOLDER: #1]}}` for forward-reference sections.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| LaTeX (pdflatex) | TeX Live 2023+ | Document compilation | Industry standard for academic ML papers |
| natbib | 8.31b | Author-year citations `\citep{}` / `\citet{}` | Required by NeurIPS/TMLR; compatible with plainnat.bst |
| amsmath | 2.17+ | Equation environments (align, equation, cases) | Standard for multi-line PDEs and matrix notation |
| amssymb | 3.01+ | Math symbols (∂, ∈, ⊙, ∥) | Required for Cosserat rod notation |
| booktabs | 1.618 | Professional table rules (`\toprule`, `\midrule`, `\bottomrule`) | Matches project skill and overview.tex pattern |
| graphicx | 1.2c | Figure inclusion (`\includegraphics`) | Required; figures in `figures/` subdirs |
| hyperref | 7.00u | Clickable links, PDF bookmarks | Already in overview.tex as `[hidelinks]` |
| geometry | 5.9 | Page margins and layout | Already in overview.tex |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| palatino | — | Serif body font | Project standard — matches overview.tex |
| fancyhdr | 4.0 | Custom headers/footers | For page numbering and title header |
| tikz + positioning | pgf 3.1.10 | Architecture diagrams | For surrogate pipeline flow diagrams |
| subcaption | 1.3 | Side-by-side subfigures | For multi-panel result figures |
| xcolor | 2.12 | Colored text for placeholder macro | Define `\placeholder` command |
| microtype | 2.8 | Microtypography (better line breaks) | Always include in polished docs |
| titlesec | 2.13 | Section spacing | Matches overview.tex compact style |
| enumitem | 3.9 | List spacing control | Matches project skill |
| setspace | 6.7b | Line spacing control | For 1.1 line stretch (matches overview.tex) |
| todonotes | 1.1.4 | Alternative to placeholder macro | Optional — xcolor approach is simpler |
| cleveref | 0.21.4 | Smart cross-references (`\cref{}`) | Recommended for paper-length documents |

### Bibliography Backend
| Choice | When to Use |
|--------|-------------|
| `plainnat.bst` + `natbib` | Standard author-year; works with `\citep{}` / `\citet{}` |
| `abbrvnat.bst` | Same but abbreviated journal names (saves space) |
| NeurIPS `.bst` | If submitting to NeurIPS; download from NeurIPS style files |

**Installation (if TeX Live not present):**
```bash
# Docker approach (recommended since LaTeX not installed on this machine)
docker run --rm -v $(pwd):/workspace -w /workspace \
  texlive/texlive:latest \
  latexmk -pdf -shell-escape report/report.tex

# Or install minimal TeX Live
apt-get install -y texlive-latex-extra texlive-science texlive-fonts-recommended latexmk
```

---

## Architecture Patterns

### Recommended Project Structure
```
report/
├── report.tex          # Main document (single file, no \input splits yet)
├── references.bib      # BibTeX entries built from knowledge/ citations
├── report.pdf          # Compiled output (gitignored or committed)
└── figures -> ../figures/   # Symlink or relative paths to project figures
```

Note: The `latex` skill says output goes to `doc/[filename].tex`. However, CONTEXT.md specifies `report/report.tex` or `report.tex` at root — Claude decides. Prefer `report/report.tex` to keep the root clean.

### Pattern 1: Placeholder Macro for Incomplete Results

**What:** Define a command that renders a gray inline note, making missing Results sections visible without breaking compilation.
**When to use:** All forward-referencing sections until Phases 4/5/8 complete.
**Example:**
```latex
% In preamble
\newcommand{\placeholder}[1]{\textcolor{gray}{\textit{[Placeholder: #1]}}}
\newcommand{\todo}[1]{\textcolor{red}{\textbf{TODO:} #1}}

% In text
\placeholder{Phase 4 validation figures: per-component RMSE table}
\placeholder{Phase 5 RL reward curves: surrogate vs Elastica}
```

### Pattern 2: natbib Author-Year Citations

**What:** Using `natbib` with `\citep{}` for parenthetical and `\citet{}` for textual references.
**When to use:** Throughout the document per locked decision.
**Example:**
```latex
% In preamble
\usepackage[round, sort, authoryear]{natbib}

% In text
\citet{Stolzle2025} demonstrated 44,000x speedup for Cosserat rod MPC\ldots
The surrogate approach follows \citet{SoRoLEX2024}, who trained an LSTM\ldots
Multi-step rollout loss has been shown to reduce drift \citep{Janner2019}.

% Bibliography
\bibliographystyle{plainnat}
\bibliography{references}
```

### Pattern 3: Cosserat Rod PDE Equations

**What:** The mathematical formulation from `knowledge/surrogate-mathematical-formulation.md` translated to LaTeX. The PDEs are already written in LaTeX-ready notation.
**When to use:** Background section, ~1–2 pages.
**Example:**
```latex
% Source: knowledge/surrogate-mathematical-formulation.md
\subsection{Cosserat Rod Dynamics}

The continuous Cosserat rod model \citep{Till2019} represents the snake body as a
1D continuum parameterized by arc length $s \in [0, L]$. The governing PDEs per
unit arc length are:
\begin{align}
\rho A \frac{\partial^2 \mathbf{x}}{\partial t^2}
  &= \frac{\partial \mathbf{F}}{\partial s} + \mathbf{f}_\text{ext}
  \label{eq:linear-momentum} \\
\rho \mathbf{I} \frac{\partial^2 \mathbf{d}}{\partial t^2}
  &= \frac{\partial \mathbf{M}}{\partial s} + \mathbf{m}_\text{ext}
    + \mathbf{x}' \times \mathbf{F}
  \label{eq:angular-momentum}
\end{align}
where $\mathbf{x}(s,t)$ is the centerline position, $\mathbf{d}(s,t)$ the director
frame, $\mathbf{F}$ internal forces (shear + tension), $\mathbf{M}$ internal moments
(bending + torsion), $\mathbf{f}_\text{ext}$ RFT anisotropic friction contact forces,
and $\mathbf{m}_\text{ext}$ CPG-driven muscle moments.
```

### Pattern 4: Surrogate Model Formulation

**What:** Formal definition of the surrogate as approximation of the ground-truth transition operator.
**When to use:** Methods section.
**Example:**
```latex
% Source: knowledge/surrogate-mathematical-formulation.md
\subsection{Surrogate Model Definition}

Let $T: (\mathbf{s}_t, \mathbf{a}_t) \mapsto \mathbf{s}_{t+1}$ denote the
ground-truth transition operator implemented by PyElastica.
We approximate $T$ with a neural network:
\begin{equation}
  f_\theta : \mathbf{z}_t \mapsto \overline{\Delta\mathbf{s}}, \quad
  \hat{\mathbf{s}}_{t+1} = \mathbf{s}_t + \sigma_\Delta \odot f_\theta(\mathbf{z}_t) + \mu_\Delta
\end{equation}
where $\mathbf{z}_t = [\bar{\mathbf{s}}_t \| \mathbf{a}_t \| \boldsymbol{\phi}_i]
\in \mathbb{R}^{189}$, with $\bar{\mathbf{s}}_t$ the z-score normalized state,
$\mathbf{a}_t$ the 5-dim action, and $\boldsymbol{\phi}_i$ the per-element CPG
phase encoding (60-dim: $\sin(\omega t_i),\cos(\omega t_i),\kappa_i$ for each of
20 elements).
```

### Pattern 5: Table for Results (with Placeholder)

**What:** Use `booktabs` for professional tables; leave data cells as placeholders.
**When to use:** Experiments & Results section.
**Example:**
```latex
\begin{table}[t]
\centering
\caption{Per-component surrogate prediction error on held-out validation set.}
\label{tab:surrogate-accuracy}
\begin{tabular}{lrr}
\toprule
Component & RMSE & Units \\
\midrule
Position $x_i$     & \placeholder{Phase 4} & mm \\
Position $y_i$     & \placeholder{Phase 4} & mm \\
Velocity $\dot{x}_i$ & \placeholder{Phase 4} & mm/s \\
Angular velocity $\omega_{z,e}$ & \placeholder{Phase 4} & rad/s \\
\bottomrule
\end{tabular}
\end{table}
```

### Anti-Patterns to Avoid

- **Splitting into `\input{}` files prematurely:** The document will grow to ~30 pages max; a single `.tex` file is simpler to manage while the report is still evolving. Split only if the file exceeds 1000 lines.
- **Using raw time as input feature:** The surrogate input uses `sin(ωt)/cos(ωt)` not raw `t` — this distinction must be clearly explained in Methods.
- **Inconsistent notation:** Define all symbols in their first use and maintain consistency. E.g., `\mathbf{s}_t` for state vector throughout, never `s_t` (scalar) in the same document.
- **Committing `report.pdf` to git:** PDF files are large binary blobs. Either gitignore or commit only on major milestones.
- **Using `figure!` float specifier:** Use `[t]` or `[H]` — the project skill specifies this.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Author-year citations | Custom citation format | `natbib` + `\citep{}`/`\citet{}` | Handles formatting, sorting, bibliography styles automatically |
| Bibliography entries | Manually typed text | `.bib` file + `\bibliography{}` | Consistent formatting, reusable across documents |
| Multi-panel figures | Separate figure environments | `subcaption` with `subfigure` | Handles shared labels (a), (b), (c) and captions |
| Cross-references | Hard-coded "Section 3" | `\label{}` + `\cref{}` | Automatically updates when sections renumber |
| Table rules | `\hline` | `booktabs` `\toprule`/`\midrule`/`\bottomrule` | Professional look; project standard |
| Equation alignment | `equation` with manual spacing | `align` environment | Proper multi-line equation alignment with `&` |

**Key insight:** The math content is already 90% written in `knowledge/surrogate-mathematical-formulation.md` using LaTeX math notation. The primary work is wrapping it in document structure, not re-deriving equations.

---

## Common Pitfalls

### Pitfall 1: BibTeX Key Collision and Missing Entries
**What goes wrong:** `bibtex` silently ignores unknown `\cite{}` keys, leaving `[?]` in the compiled PDF.
**Why it happens:** `.bib` entries are added incrementally; easy to cite a paper before adding its `.bib` entry.
**How to avoid:** Build `references.bib` upfront from all cited papers in `knowledge/` files; run `bibtex report` and check for warnings after each compilation.
**Warning signs:** `[?]` citations in the output PDF; bibtex output shows "I didn't find a database entry for ..."

### Pitfall 2: natbib/hyperref Incompatibility
**What goes wrong:** `hyperref` loaded after `natbib` can cause duplicate anchor warnings or broken links.
**Why it happens:** Package load order matters; `hyperref` must be loaded last among these packages.
**How to avoid:** Always load `hyperref` last in the preamble (or use `\usepackage{hyperref}` after `natbib`). The existing `overview.tex` pattern is correct — follow it.
**Warning signs:** "pdfTeX warning: ... has been referenced but does not exist" in compilation log.

### Pitfall 3: Placeholder Content Breaking Compilation
**What goes wrong:** Forgetting to define `\placeholder` causes `Undefined control sequence` errors.
**Why it happens:** Using `\placeholder{}` without the `\newcommand` definition in the preamble.
**How to avoid:** Define the macro in the preamble before `\begin{document}`. Use the pattern above exactly.
**Warning signs:** LaTeX errors on the first `\placeholder{}` use.

### Pitfall 4: Figure Path Issues
**What goes wrong:** `\includegraphics` fails silently or throws "File not found" on figures from `figures/`.
**Why it happens:** LaTeX resolves paths relative to the `.tex` file location. If `report.tex` is in `report/`, figure paths must be `../figures/data_validation/action_coverage_heatmap.png`.
**How to avoid:** Set `\graphicspath{{../figures/}}` in the preamble. Or use a symlink from `report/figures` to `../figures`.
**Warning signs:** Blank boxes in PDF where figures should appear; "LaTeX Warning: File ... not found."

### Pitfall 5: Long Math Lines Overflowing Text Width
**What goes wrong:** Multi-term equations (like the 189-dim input vector definition) overflow the text column.
**Why it happens:** Single `equation` environment does not wrap.
**How to avoid:** Use `align` with `\\` line breaks; break at `+` or `\|` (concatenation) operators; use `\begin{multline}` for single long equations.
**Warning signs:** "Overfull \hbox" warnings in the compilation log.

### Pitfall 6: minted Package Requiring --shell-escape
**What goes wrong:** `minted` package fails to compile without `--shell-escape` flag.
**Why it happens:** `minted` uses Pygments (Python) to syntax-highlight code; needs shell access during compilation.
**How to avoid:** For the research report, code listings are not needed (math-heavy paper); use `verbatim` or `lstlistings` if code snippets are needed. Or always pass `-shell-escape` to `latexmk`.
**Warning signs:** "Package minted Error: You must invoke LaTeX with the -shell-escape flag."

---

## Code Examples

Verified patterns from project's existing LaTeX (overview.tex) and skill definition:

### Full Preamble for report.tex
```latex
% Source: project latex skill + overview.tex preamble (adapted)
\documentclass[11pt]{article}

% Font (project standard)
\usepackage{palatino}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}

% Page layout — NeurIPS/TMLR-style wide margins
\usepackage[top=1in, bottom=1in, left=1.25in, right=1.25in]{geometry}
\usepackage{setspace}
\setstretch{1.1}
\setlength{\parindent}{1em}
\setlength{\parskip}{0.4em}

% Math
\usepackage{amsmath}
\usepackage{amssymb}

% Figures and tables
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{float}
\usepackage{booktabs}
\usepackage{multirow}
\graphicspath{{../figures/}}

% Cross-references and links (load last)
\usepackage{cleveref}
\usepackage[hidelinks]{hyperref}
\usepackage{microtype}

% Colors (for placeholder macro)
\usepackage{xcolor}

% Bibliography (natbib for author-year)
\usepackage[round, sort, authoryear]{natbib}

% Header/footer (project standard)
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Neural Surrogate for Snake Robot RL}
\fancyfoot[R]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}

% Section spacing (project standard)
\usepackage{titlesec}
\titlespacing*{\section}{0pt}{1.2em}{0.6em}
\titlespacing*{\subsection}{0pt}{0.9em}{0.3em}

% List spacing
\usepackage{enumitem}
\setlist[itemize]{nosep, topsep=0.3em, itemsep=0.2em}

% Table of contents depth
\setcounter{tocdepth}{2}
\setcounter{secnumdepth}{3}

% Placeholder and todo macros
\newcommand{\placeholder}[1]{\textcolor{gray}{\textit{[Placeholder: #1]}}}
\newcommand{\mytodo}[1]{\textcolor{red}{\textbf{[TODO: #1]}}}

\title{Neural Surrogate Modeling for Snake Robot Locomotion:\\[0.3em]
\large Faster Reinforcement Learning via Learned Cosserat Rod Dynamics}
\author{[Author Name]}
\date{}
```

### BibTeX Entry Format for Key References
```bibtex
% Source: knowledge/neural-surrogate-cosserat-rod.md references
@article{Stolzle2025,
  author    = {Stolzle, Maximilian and others},
  title     = {Adaptive Model-Predictive Control of a Soft Continuum Robot
               Using a Physics-Informed Neural Network Based on Cosserat Rod Theory},
  journal   = {arXiv preprint arXiv:2508.12681},
  year      = {2025}
}

@article{Hsieh2024,
  author    = {Hsieh, Ching-An and others},
  title     = {Knowledge-based Neural Ordinary Differential Equations for
               Cosserat Rod-based Soft Robots},
  journal   = {arXiv preprint arXiv:2408.07776},
  year      = {2024}
}

@article{SoRoLEX2024,
  author    = {Uljad, Berdica and others},
  title     = {Towards Reinforcement Learning Controllers for Soft Robots
               using Learned Environments},
  journal   = {arXiv preprint arXiv:2410.18519},
  year      = {2024}
}

@inproceedings{Janner2019,
  author    = {Janner, Michael and Fu, Justin and Zhang, Marvin and Levine, Sergey},
  title     = {When to Trust Your Model: Model-Based Policy Optimization},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2019}
}

@article{Naughton2021,
  author    = {Naughton, Noel and others},
  title     = {Elastica: A compliant mechanics environment for soft robotic control},
  journal   = {IEEE Robotics and Automation Letters},
  year      = {2021}
}
```

### Discussion Section "Lessons Learned" Structure
```latex
% Source: CONTEXT.md Discussion section spec
\section{Discussion}

\subsection{Physics Calibration Challenges}
During development, several critical physics parameter calibration issues were encountered.
The rod radius was initially set to $r = 0.001$ m (the physical snake radius), producing
zero locomotion. The bending stiffness $EI \propto r^4$ rendered the rod completely
limp at this scale. Correcting to $r = 0.02$ m with $E = 10^5$ Pa restored serpentine
motion, illustrating that surrogate training data quality depends critically on the
correctness of the underlying simulator.

% ... (serpenoid wave direction sign error, omega_z coverage gap, friction coefficients)

\subsection{Surrogate Architecture Experiments}
\placeholder{Phase 03.1 results: rollout loss ablation, residual MLP vs baseline MLP}

\subsection{Data Collection Pipeline}
Parallel data collection using 16 workers exposed several infrastructure challenges.
A Numba thread pool deadlock caused worker stalls on the first forward pass of each
new worker process \citep{issues/numba-thread-pool-deadlock-worker-stalls}. This was
resolved by setting \texttt{NUMBA\_NUM\_THREADS=1} before worker initialization.
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Global phase encoding (2-dim sin/cos) | Per-element CPG phase encoding (60-dim) | Phase 02.1 | Input dim 131→189; captures spatial wave structure per rod segment |
| Single-step MSE only | Single-step MSE + rollout loss (8-step BPTT) | Phase 03.1 | Reduces autoregressive drift; `λ_r = 0.1` from epoch 20 |
| Uniform data sampling | Density-weighted sampling (inverse histogram) | Phase 03 | Compensates for Sobol clustering near center of action space |
| `\thebibliography{}` | BibTeX + natbib | Phase 6 | Proper author-year citations; matches ML paper conventions |

**Deprecated / outdated in this project's context:**
- `\thebibliography{}` (hardcoded bibliography): replaced by `.bib` file + `natbib`
- Global `serpenoid_time` as a raw float: replaced by per-element `(sin(ωt_i), cos(ωt_i), κ_i)` encoding
- `\hline` in tables: replaced by `booktabs` rules (project standard)

---

## Source Material Inventory

This section documents where each report section's content comes from — critical for the planner to create tasks that convert existing knowledge to LaTeX prose.

### Background Section Content Sources
| Topic | Source File | Status |
|-------|------------|--------|
| Cosserat rod PDEs (linear/angular momentum) | `knowledge/surrogate-mathematical-formulation.md` | Complete, LaTeX-ready |
| N=21 node discretization, state vector (124-dim) | `knowledge/surrogate-mathematical-formulation.md` | Complete, LaTeX-ready |
| RFT friction contact model | `knowledge/friction-contact-models.md` | Available |
| CPG actuation equations | `knowledge/control-and-actuation.md` | Available |
| PyElastica simulator description | `knowledge/simulator-comparison-soft-robot-rl.md` (Sec 1) | Available |

### Related Work Content Sources
| Topic | Source File | Status |
|-------|------------|--------|
| DD-PINN for Cosserat rod (Stolzle 2025) | `knowledge/neural-surrogate-cosserat-rod.md` Sec 1.1 | Complete |
| KNODE-Cosserat (Hsieh 2024) | `knowledge/neural-surrogate-cosserat-rod.md` Sec 1.2 | Complete |
| SoRoLEX learned RL environments (2024) | `knowledge/neural-surrogate-cosserat-rod.md` Sec 1.3 | Complete |
| RL with learned models (MBPO, DreamerV3) | `knowledge/neural-surrogates-cosserat-rod-dynamics.md` Sec 5 | Available |
| Simulator comparison for soft robot RL | `knowledge/simulator-comparison-soft-robot-rl.md` | Available |
| GNN surrogates (MeshGraphNets, GNS) | `knowledge/neural-surrogates-cosserat-rod-dynamics.md` Sec 2 | Available |

### Methods Content Sources
| Topic | Source File | Status |
|-------|------------|--------|
| Data collection pipeline (Sobol, perturbation) | `experiments/surrogate-parallel-data-collection.md` | Available |
| Per-element CPG phase encoding | `knowledge/surrogate-mathematical-formulation.md` | Complete |
| Surrogate architecture (MLP, LayerNorm, SiLU) | `knowledge/surrogate-mathematical-formulation.md` Sec "Architecture" | Complete |
| Training loss (single-step + rollout) | `knowledge/surrogate-mathematical-formulation.md` Sec "Training Objective" | Complete |
| Density-weighted sampling | `knowledge/surrogate-mathematical-formulation.md` Sec "Sample Weighting" | Complete |
| Input/output normalization | `knowledge/surrogate-mathematical-formulation.md` | Complete |

### Discussion Content Sources
| Topic | Source File |
|-------|------------|
| Rod radius calibration (0.001→0.02 m) | `issues/elastica-rod-radius-too-small.md` |
| Serpenoid wave direction sign error | `issues/serpenoid-wave-direction-sign-error.md` |
| Friction coefficient tuning | `issues/elastica-friction-coefficient-tuning.md` |
| omega_z coverage gap and fix | `issues/surrogate-omega-z-poor-prediction.md` |
| Numba thread deadlock | `issues/numba-thread-pool-deadlock-worker-stalls.md` |
| Stall detection false positives | `issues/stall-detection-false-positives-during-init.md` |
| Parallel collection bottleneck | `issues/parallel-collection-scaling-bottleneck.md` |
| Architecture comparison (Phase 03.1) | `knowledge/surrogate-architecture-comparison.md` |

### Existing Figures Available Now
| Figure | Path | Used In |
|--------|------|---------|
| Action coverage heatmap | `figures/data_validation/action_coverage_heatmap.png` | Methods: data collection |
| Action histograms | `figures/data_validation/action_histograms.png` | Methods: data collection |
| Episode length distribution | `figures/data_validation/episode_length_distribution.png` | Methods / Appendix |
| Sampling comparison | `figures/sampling_comparison.png` | Methods: Sobol vs uniform |

### Figures to Be Generated (Placeholders Now)
| Figure | Generated By | Used In |
|--------|-------------|---------|
| Surrogate training loss curves | Phase 3 W&B export | Methods / Experiments |
| Per-component RMSE table/plots | Phase 4 | Experiments |
| Multi-step rollout error vs steps | Phase 4 | Experiments |
| RL reward curves (surrogate vs Elastica) | Phase 5 + Phase 8 | Experiments |
| Architecture comparison chart | Phase 03.1 | Methods / Discussion |

---

## Open Questions

1. **LaTeX compilation environment**
   - What we know: LaTeX is not installed on this machine (no `pdflatex` or `xelatex` found)
   - What's unclear: Whether to use Docker, install TeX Live, or use Overleaf for compilation
   - Recommendation: Document should be written to compile under standard TeX Live 2023+; a Makefile or `latexmk` rule should be included; first compilation test should be done via Docker (`texlive/texlive:latest`)

2. **NeurIPS vs TMLR vs General Academic template**
   - What we know: User locked NeurIPS/TMLR style but left exact template to Claude's discretion; no conference submission planned (graduation project only); project already uses general-academic Palatino style in overview.tex
   - What's unclear: Whether to download NeurIPS `.sty` file or adapt existing preamble
   - Recommendation: Use the project's existing general-academic preamble (Palatino + geometry) rather than NeurIPS `.sty` — avoids a binary style file dependency, produces equivalent single-column paper look, and works with the existing LaTeX skill template exactly

3. **Per-element CPG encoding in report vs baseline model**
   - What we know: Phase 02.1 changed input from 131-dim (global phase) to 189-dim (per-element); existing `knowledge/surrogate-mathematical-formulation.md` describes the 131-dim baseline
   - What's unclear: Whether the Methods section should describe v1 (131-dim) or v2 (189-dim) as the primary architecture
   - Recommendation: Describe v2 (189-dim, per-element) as the primary Methods formulation; mention v1 as the baseline in Discussion / Architecture Experiments

4. **Report location: `report/report.tex` vs root `report.tex`**
   - What we know: CONTEXT.md left this to Claude's discretion
   - Recommendation: Use `report/report.tex` — keeps root clean, mirrors common academic project structure, and allows `report/references.bib` alongside

---

## Validation Architecture

> `nyquist_validation` is enabled in `.planning/config.json`.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | LaTeX compilation (latexmk) — not a unit test framework |
| Config file | `report/Makefile` or `report/.latexmkrc` — Wave 0 |
| Quick run command | `latexmk -pdf -shell-escape report/report.tex` |
| Full suite command | Same — LaTeX compilation is the only validation |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DOC-01 | report.tex compiles without errors | smoke | `latexmk -pdf report/report.tex 2>&1 \| grep -c "^!"` = 0 | Wave 0 |
| DOC-02 | All citations resolve (no [?]) | smoke | `bibtex report 2>&1 \| grep "I couldn't find"` = empty | Wave 0 |
| DOC-03 | All \label{} have matching \ref{} | smoke | `chktex report/report.tex` or manual grep | Wave 0 |
| DOC-04 | Background section complete (no \placeholder) | manual | `grep -c 'placeholder' report/report.tex` for Background only | ❌ Wave 0 |
| DOC-05 | Methods section complete (no \placeholder) | manual | `grep -c 'placeholder' report/report.tex` for Methods only | ❌ Wave 0 |
| DOC-06 | All currently-available figures included | manual | Visual inspection of PDF | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `latexmk -pdf report/report.tex && echo "COMPILED OK"`
- **Per wave merge:** Full compilation + bibtex + check for undefined citations
- **Phase gate:** Document compiles, Background + Related Work + Methods complete, no errors

### Wave 0 Gaps
- [ ] `report/report.tex` — main document (does not exist yet)
- [ ] `report/references.bib` — bibliography entries from knowledge/ citations
- [ ] `report/Makefile` — latexmk compilation rule
- [ ] LaTeX toolchain available for compilation (Docker or apt install)

---

## Sources

### Primary (HIGH confidence)
- `overview.tex` — Project's existing LaTeX preamble; exact package versions verified by inspection
- `.claude/skills/latex/SKILL.md` — Project-specific LaTeX style guide (general-academic template)
- `knowledge/surrogate-mathematical-formulation.md` — Complete mathematical formulation, LaTeX-notation throughout
- `knowledge/neural-surrogate-cosserat-rod.md` — Full related work literature review with arXiv citations
- `knowledge/surrogate-architecture-comparison.md` — Architecture comparison for Discussion section
- `knowledge/neural-surrogates-cosserat-rod-dynamics.md` — Additional survey content for Related Work

### Secondary (MEDIUM confidence)
- `knowledge/simulator-comparison-soft-robot-rl.md` — Simulator comparison data; cited with sources
- `knowledge/control-and-actuation.md` — CPG actuation math for Background
- `figures/data_validation/` — 7 existing figures verified present

### Tertiary (LOW confidence)
- natbib/hyperref load-order compatibility note — based on common LaTeX knowledge, not verified against specific TeX Live version on a test system

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — matches overview.tex preamble exactly; project skill confirmed
- Architecture (document structure): HIGH — locked in CONTEXT.md; preamble pattern verified working
- Source material inventory: HIGH — all files verified to exist
- Pitfalls: MEDIUM — LaTeX pitfalls are well-established; specific version interactions not tested on target system
- Compilation environment: LOW — LaTeX not installed; Docker approach unverified on this machine

**Research date:** 2026-03-10
**Valid until:** 2027-03-10 (LaTeX packages are very stable; source material is in-project)
