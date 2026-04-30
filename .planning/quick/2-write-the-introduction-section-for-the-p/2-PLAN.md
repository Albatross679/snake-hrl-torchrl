---
phase: quick-2
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - report/report.tex
autonomous: true
requirements: [QUICK-2]

must_haves:
  truths:
    - "Introduction section replaces the placeholder at line 128 of report.tex"
    - "Problem statement establishes slow PyElastica simulation as RL bottleneck"
    - "Contribution summary describes the neural surrogate pipeline (collect, train, RL)"
    - "Paper roadmap guides the reader through each section"
  artifacts:
    - path: "report/report.tex"
      provides: "Introduction section content"
      contains: "\\section{Introduction}"
  key_links:
    - from: "Introduction"
      to: "Background, Related Work, Methods, Discussion sections"
      via: "Section references using cref"
      pattern: "\\\\cref\\{sec:"
---

<objective>
Write the introduction section for the Phase 6 research report, replacing the placeholder at line 128 of report/report.tex.

Purpose: The introduction frames the entire report for a thesis committee — establishing the computational bottleneck problem, summarizing the neural surrogate contribution, and providing a roadmap through the existing sections.
Output: Complete introduction section in report/report.tex
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@report/report.tex
@.planning/quick/2-write-the-introduction-section-for-the-p/2-CONTEXT.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Write introduction section replacing placeholder</name>
  <files>report/report.tex</files>
  <action>
Replace the placeholder line (line 128: `\placeholder{Introduction: problem statement, contribution summary, paper roadmap.}`) with a complete introduction section containing three parts:

**Part 1 — Problem statement (1-2 paragraphs):**
- Snake robots modeled as Cosserat rods require high-fidelity physics simulation (PyElastica)
- Each RL environment step requires ~46ms due to symplectic Euler integration of the discretized PDE system
- At ~350 steps/second with 16 parallel workers, a 2M-step PPO run takes ~36 hours wall-clock
- This computational bottleneck motivates replacing the physics simulator with a learned model
- Reference the existing background section for Cosserat rod details: \cref{sec:background}

**Part 2 — Contribution summary (1 paragraph):**
- Present the neural surrogate pipeline as the central contribution
- Three stages: (1) parallel data collection from PyElastica with Sobol sampling and state perturbation, (2) MLP surrogate training with single-step MSE and multi-step rollout loss, (3) RL training using the surrogate as a drop-in environment
- State the central claim: neural surrogate enables faster RL at acceptable accuracy loss
- Mention the projected speedup (4-5 orders of magnitude from ~46ms/step to ~0.001ms/step)
- Note that Phase 8 direct Elastica RL baseline provides the controlled comparison anchor

**Part 3 — Paper roadmap (1 paragraph):**
- Brief section-by-section guide using \cref references
- Background (\cref{sec:background}): Cosserat rod dynamics, RFT, CPG actuation, PyElastica
- Related Work (\cref{sec:related}): neural surrogates for soft robots, RL with learned simulators
- Methods (\cref{sec:methods}): data collection, state representation, phase encoding, architecture, training
- Experiments (\cref{sec:experiments}): surrogate accuracy, RL training, baseline comparison
- Discussion (\cref{sec:discussion}): physics calibration challenges, architecture experiments, pipeline lessons
- Conclusion (\cref{sec:conclusion})

**Style constraints:**
- Use natbib citations where appropriate (\citet, \citep) — can reference existing bib entries (e.g., Naughton2021 for PyElastica, SoRoLEX2024, Stolzle2025)
- Keep the tone consistent with existing report sections (formal academic, thesis-committee audience)
- The report uses 1em parindent and 0.4em parskip — write standard LaTeX paragraphs (no explicit \par or \noindent unless needed after equations/lists)
- Total length: approximately 1-1.5 pages when compiled (roughly 4-5 paragraphs)
- Do NOT add any new \subsection commands — the introduction should be flowing prose paragraphs
  </action>
  <verify>
    <automated>cd /home/coder/snake-hrl-torchrl && grep -c "placeholder" report/report.tex | grep -v "newcommand" && python3 -c "
content = open('report/report.tex').read()
intro_start = content.find('\\\\section{Introduction}')
bg_start = content.find('\\\\section{Background}')
intro = content[intro_start:bg_start]
assert 'placeholder' not in intro.lower(), 'Placeholder still present'
assert 'cref{sec:background}' in intro, 'Missing background cref'
assert 'cref{sec:methods}' in intro, 'Missing methods cref'
assert 'cref{sec:experiments}' in intro, 'Missing experiments cref'
assert len(intro) > 500, f'Introduction too short: {len(intro)} chars'
print(f'Introduction length: {len(intro)} chars — OK')
print('All checks passed')
"</automated>
  </verify>
  <done>The placeholder at line 128 is replaced with a complete introduction containing problem statement (slow physics sim bottleneck), contribution summary (neural surrogate pipeline with speedup claim), and paper roadmap (section-by-section cref guide). No placeholder text remains in the introduction section.</done>
</task>

</tasks>

<verification>
- The placeholder text is completely removed from the introduction section
- The introduction references existing sections via \cref
- The content is consistent with the existing Background, Methods, and Discussion sections
- LaTeX compiles without errors (if pdflatex available)
</verification>

<success_criteria>
- Introduction section exists as flowing prose (no subsections, no placeholder)
- Problem statement quantifies the computational bottleneck (~46ms/step, ~36hr training)
- Contribution summary describes the three-stage surrogate pipeline
- Paper roadmap covers all major sections with \cref references
- Tone and style match the existing report sections
</success_criteria>

<output>
After completion, create `.planning/quick/2-write-the-introduction-section-for-the-p/2-01-SUMMARY.md`
</output>
