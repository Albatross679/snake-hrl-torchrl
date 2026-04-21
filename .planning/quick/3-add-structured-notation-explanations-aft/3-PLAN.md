---
phase: quick-3
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - report/report.tex
autonomous: true
requirements: [QUICK-3]

must_haves:
  truths:
    - "Every equation with a paragraph-style notation explanation now uses a structured tabular where-block"
    - "No equation content or numbering has changed"
    - "The notation table in sec:notation is unchanged"
    - "The document compiles without errors"
  artifacts:
    - path: "report/report.tex"
      provides: "Structured where-blocks after equations"
      contains: "noindent where"
  key_links: []
---

<objective>
Replace paragraph-style notation explanations after equations in report/report.tex with structured tabular where-blocks.

Purpose: Improve readability and consistency of mathematical notation explanations throughout the report.
Output: Updated report/report.tex with tabular where-blocks after all qualifying equations.
</objective>

<execution_context>
@./.claude/get-shit-done/workflows/execute-plan.md
@./.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@report/report.tex
@.planning/quick/3-add-structured-notation-explanations-aft/3-CONTEXT.md

<interfaces>
The where-block format (from user decision):
```latex
\noindent where
\begin{tabular}{@{}>{$}l<{$} @{\quad} l@{}}
  \mathbf{x}(s,t) & rod centerline position (m) \\
  \mathbf{F}       & internal force (shear + tension) \\
\end{tabular}
```
</interfaces>
</context>

<tasks>

<task type="auto">
  <name>Task 1: Convert all paragraph-style notation explanations to tabular where-blocks</name>
  <files>report/report.tex</files>
  <action>
Convert every equation that currently has a paragraph-style variable explanation into a structured tabular where-block. Use `\smallskip` before the `\noindent where` line for visual spacing.

The format for each where-block is:
```latex
\smallskip
\noindent where
\begin{tabular}{@{}>{$}l<{$} @{\quad} l@{}}
  SYMBOL & description \\
\end{tabular}
```

Equations to convert (identified by label):

1. **eq:linear-momentum and eq:angular-momentum** (lines 207-223): The paragraph starting "In these equations..." explains symbols for both equations. Convert to a single where-block after eq:angular-momentum covering: x(s,t), d(s,t), F, M, f_ext, m_ext, rho A, rho I. Keep the remaining prose sentence about ground contact forces referencing sec:background:rft and CPG referencing sec:background:cpg as a normal paragraph after the where-block.

2. **eq:rft-tangential and eq:rft-normal** (lines 262-274): The paragraph "Here $v_t$ and $v_n$..." Convert to where-block after eq:rft-normal covering: v_t, v_n, c_t, c_n. Keep the sentence about anisotropy ratio values and physics explanation as a following paragraph.

3. **eq:cpg-curvature** (lines 283-300): The paragraph "The variable $s_i \in [0,1]$..." Convert to where-block covering: s_i, A, k, f, phi_0, b. Keep the sentence introducing the action vector as a following paragraph.

4. **eq:action-mapping** (lines 411-416): The brief explanation "The normalized action..." is already concise and refers to denormalization context. Convert to where-block covering: a_t, a_phys, a_max/a_min, \odot. Keep the sentence about denormalization occurring inside PyElastica as a following paragraph.

5. **eq:effective-time and eq:per-element-phase** (lines 429-448): Convert the explanatory text into a where-block after eq:per-element-phase covering: t_e, k, s_e, f, sin/cos terms, kappa_e^actual. Keep the sentence about stacking into 60-dim vector as a following paragraph.

6. **eq:surrogate-input** (lines 452-460): Convert the explanation into a where-block covering: bar{s}_t, a_t, phi_1...phi_{N_e}, mu_s, sigma_s. Keep the dimensionality decomposition sentence as part of the where-block descriptions or as a following sentence.

7. **eq:surrogate-forward and eq:next-state-reconstruction** (lines 468-481): Convert the explanation "Here mu_Delta, sigma_Delta..." into a where-block after eq:next-state-reconstruction covering: f_theta, overline{Delta s}, mu_Delta, sigma_Delta, \odot.

8. **eq:single-loss** (lines 512-518): Brief — convert "normalized state delta" context to a where-block if there are symbols worth explaining (N = batch size, f_theta, overline{Delta s}_true).

9. **eq:combined-loss and eq:rollout-loss** (lines 528-543): Convert the explanation into a where-block after eq:rollout-loss covering: lambda_r, L, hat{s}_{t+1}. Keep the sentence about gradient backpropagation as a following paragraph.

10. **eq:density-weight and eq:density-features** (lines 552-566): Convert the explanation into a where-block after eq:density-features covering: w^(i), hat{p}, c^(i) components. Keep the sentence about histogram bins and weight clipping as a following paragraph.

IMPORTANT constraints:
- Do NOT modify equation content, labels, or numbering
- Do NOT modify the Notation table (sec:notation)
- Do NOT add or remove any \cite or \cref references
- Preserve all existing prose that provides context, reasoning, or cross-references — only restructure the "symbol = meaning" portions into tabular form
- Where-blocks should complement (not duplicate) the notation table — provide equation-local context rather than repeating global definitions
  </action>
  <verify>
    <automated>cd /home/coder/snake-hrl-torchrl && grep -c "noindent where" report/report.tex</automated>
  </verify>
  <done>At least 8 tabular where-blocks exist in report.tex. All equations retain their original content and labels. No paragraph-style "In these equations..." or "Here $X$ denotes..." patterns remain for symbol definitions.</done>
</task>

<task type="auto">
  <name>Task 2: Verify LaTeX compiles and no content lost</name>
  <files>report/report.tex</files>
  <action>
Verify the modified report.tex by:
1. Check that all equation labels still exist (grep for each label)
2. Check that all \cref references still have targets
3. Check that the Notation table is unchanged (the table between \section*{Notation} and \section{Introduction})
4. Count total \begin{equation} occurrences to confirm no equations were accidentally deleted
5. Verify no unclosed tabular environments by counting \begin{tabular} vs \end{tabular}

Expected equation count: 18 equations (eq:linear-momentum through eq:density-features).
Expected tabular count: at least 8 new where-block tabulars plus the existing Notation table and state-vector table.
  </action>
  <verify>
    <automated>cd /home/coder/snake-hrl-torchrl && echo "Equations:" && grep -c '\\begin{equation}' report/report.tex && echo "Tabular begin:" && grep -c '\\begin{tabular}' report/report.tex && echo "Tabular end:" && grep -c '\\end{tabular}' report/report.tex && echo "Where blocks:" && grep -c 'noindent where' report/report.tex && echo "Labels:" && grep -o 'label{eq:[^}]*}' report/report.tex | wc -l</automated>
  </verify>
  <done>Equation count is 18, tabular begin/end counts match, at least 8 where-blocks exist, all 18 equation labels present.</done>
</task>

</tasks>

<verification>
- grep "noindent where" report/report.tex shows at least 8 where-blocks
- grep -c "begin{equation}" report/report.tex returns 18
- Tabular begin/end counts are balanced
- All equation labels preserved
</verification>

<success_criteria>
All paragraph-style notation explanations after equations have been converted to structured tabular where-blocks using the @{}>{$}l<{$} @{\quad} l@{} column spec. The document retains all equations, labels, references, and contextual prose. Only the "symbol = meaning" portions have been restructured.
</success_criteria>

<output>
After completion, create `.planning/quick/3-add-structured-notation-explanations-aft/3-SUMMARY.md`
</output>
