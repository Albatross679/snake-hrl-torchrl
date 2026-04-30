---
phase: quick-4
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: [report/report.tex]
autonomous: true
requirements: [QUICK-4]

must_haves:
  truths:
    - "Section 2.1 opens with a clear statement that the neural surrogate approximates a specific ODE system"
    - "The inputs (state vector s_t, action vector a_t) and output (next state s_{t+1}) are explicitly boxed or highlighted before any derivation"
    - "The ODE system equations are grouped together as THE system being approximated, not scattered across subsubsections"
    - "The approximation chain is preserved and connects back to the explicit I/O framing"
  artifacts:
    - path: "report/report.tex"
      provides: "Restructured section 2.1"
      contains: "subsection{Cosserat Rod Dynamics}"
  key_links:
    - from: "sec:background:cosserat"
      to: "sec:background:cosserat:chain"
      via: "I/O framing connects intro to approximation chain"
      pattern: "f_theta.*mathbf.s._t.*mathbf.a._t"
---

<objective>
Restructure section 2.1 (Cosserat Rod Dynamics) in report/report.tex so the reader immediately understands WHAT system the neural surrogate approximates, with explicit inputs and outputs.

Purpose: The current section 2.1 buries the "what are we approximating" framing deep in subsubsections. The reader should grasp the I/O structure (state + action -> next state) upfront, then see the governing equations as the system that produces this mapping.

Output: Revised section 2.1 in report/report.tex with reorganized subsubsection structure.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@.planning/phases/06-write-research-report-in-latex/06-CONTEXT.md
@report/report.tex

Key style constraints from 06-CONTEXT.md:
- Passive voice throughout
- Itemized list notation explanations after each equation (NOT tabular where-blocks, NOT inline prose)
- Display equations generously, number all displayed equations
- State final PDEs directly, cite Antman/Gazzola for derivation
- Short paragraphs (3-5 sentences)
- Reader knows ML but not Cosserat rods
</context>

<tasks>

<task type="auto">
  <name>Task 1: Restructure section 2.1 with upfront I/O framing</name>
  <files>report/report.tex</files>
  <action>
Restructure section 2.1 (lines ~201-463) with the following reorganization. All existing content is preserved and rearranged — no content is deleted, only moved and lightly rewritten for flow.

**New structure for section 2.1:**

1. **Opening paragraph** (revised): Keep the Cosserat rod intro sentence, then IMMEDIATELY state the key framing: "The governing equations define a transition operator $T$ that maps the current rod state $\mathbf{s}_t$ and a control action $\mathbf{a}_t$ to the next state $\mathbf{s}_{t+1}$. This operator is the target of the neural surrogate approximation." Then briefly preview the subsection structure.

2. **Subsubsection: "Governing PDEs"** (was the opening equations + 2D reduction paragraph): The continuous Cosserat PDEs (eq:linear-momentum, eq:angular-momentum) with their notation lists. Keep the 2D projection paragraph. This is short — just state the physics.

3. **Subsubsection: "Spatial Discretization and the Semi-Discrete ODE System"** (keep as-is, lines 236-303): The 21-node discretization, node equations, element equations, bending moment. This is the core ODE system.

4. **NEW addition at end of the ODE subsubsection**: Add a short "summary box" paragraph (or a small displayed equation block) that collects the I/O explicitly:
   - **Input:** State $\mathbf{s}_t \in \mathbb{R}^{124}$ (reference Table 1) and action $\mathbf{a}_t \in \mathbb{R}^5$ (amplitude, frequency, wave number, phase offset, direction bias — reference the Methods section for details)
   - **Output:** Next state $\mathbf{s}_{t+1} \in \mathbb{R}^{124}$
   - **Mapping:** $\mathbf{s}_{t+1} = T(\mathbf{s}_t, \mathbf{a}_t)$ where $T$ integrates the 124-dim ODE forward by one control step
   - Use a displayed equation with the explicit mapping, and reference the state vector table.
   - Keep the existing sentence at line 302-303 ("This is the exact system...") but move it here where it has more impact.

5. **Subsubsection: "Time Integration"** (keep as-is, lines 305-370): Verlet scheme, substeps, composed transition operator, state vector table.

6. **Subsubsection: "The Approximation Chain"** (keep as-is, lines 373-463): Error decomposition, neural approximation.

**Key changes summary:**
- Move the continuous PDEs into their own subsubsection "Governing PDEs" (they currently have no subsubsection heading)
- Add an explicit I/O summary after the ODE system derivation
- Revise the opening paragraph to front-load the "what are we approximating" framing
- Everything else stays in place — the subsubsections after the PDEs are already well-organized

**Style rules to follow:**
- Passive voice
- Itemized list notation after equations (not tabular, not inline prose)
- Number all displayed equations
- Short paragraphs (3-5 sentences)
  </action>
  <verify>
    <automated>cd /home/coder/snake-hrl-torchrl && python3 -c "
import re
with open('report/report.tex') as f:
    tex = f.read()
# Check new subsubsection exists
assert 'Governing PDEs' in tex or 'Governing Equations' in tex, 'Missing governing PDEs subsubsection'
# Check I/O framing in opening
assert 'transition operator' in tex[:tex.find('Governing')], 'Missing transition operator in opening'
# Check explicit I/O summary exists near ODE section
ode_pos = tex.find('Semi-Discrete ODE')
chain_pos = tex.find('Approximation Chain')
between = tex[ode_pos:chain_pos]
assert 'mathbb{R}^{124}' in between, 'Missing explicit state dimension in I/O summary'
assert 'mathbb{R}^5' in between or 'mathbb{R}^{5}' in between, 'Missing explicit action dimension in I/O summary'
# Check all original equations preserved
for label in ['eq:linear-momentum', 'eq:angular-momentum', 'eq:node-position', 'eq:node-velocity',
              'eq:element-yaw', 'eq:element-angular-velocity', 'eq:bending-moment',
              'eq:verlet-half-position', 'eq:transition-composed', 'eq:approximation-chain']:
    assert label in tex, f'Missing equation label: {label}'
print('All checks passed')
"
    </automated>
  </verify>
  <done>
    - Section 2.1 opens with explicit "transition operator T: (s_t, a_t) -> s_{t+1}" framing
    - Continuous PDEs have their own subsubsection heading
    - Explicit I/O summary appears after the ODE system, before time integration
    - All original equations, notation lists, and content preserved
    - Style matches 06-CONTEXT.md conventions (passive voice, itemized notation, numbered equations)
  </done>
</task>

</tasks>

<verification>
- All equation labels from the original section are present and unchanged
- No LaTeX compilation errors (verify via LaTeX Workshop extension)
- Section 2.1 subsubsection count increased by 1 (was 3, now 4: Governing PDEs, ODE System, Time Integration, Approximation Chain)
- The state vector table (tab:state-vector) remains in place
- Cross-references to other sections (sec:background:rft, sec:background:cpg, sec:methods:training) unchanged
</verification>

<success_criteria>
A reader encountering section 2.1 can immediately understand: (1) the neural surrogate approximates a specific ODE system, (2) the inputs are a 124-dim state and 5-dim action, (3) the output is the next 124-dim state, and (4) the ODE system governs how that mapping works physically.
</success_criteria>

<output>
After completion, create `.planning/quick/4-restructure-section-2-1-to-clearly-prese/4-SUMMARY.md`
</output>
