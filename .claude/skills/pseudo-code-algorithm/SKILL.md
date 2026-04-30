---
name: pseudo-code-algorithm
description: >
  Generate publication-quality pseudo-code for algorithms in LaTeX (algpseudocode/algorithm2e) or markdown.
  Covers ML/deep learning (training loops, optimizers, backprop), reinforcement learning (Q-learning, PPO,
  actor-critic, REINFORCE), and evolutionary algorithms (GA, CMA-ES, NEAT, GP).
  Use when: (1) Writing pseudo-code for a paper, report, or assignment,
  (2) Converting Python/PyTorch code to pseudo-code,
  (3) Presenting an algorithm in LaTeX algorithm environment,
  (4) Creating pseudo-code for ML training, RL, or evolutionary methods,
  (5) User asks to "write pseudo-code", "algorithmize", or "present as algorithm".
---

# Pseudo-Code Algorithm Generator

Generate pseudo-code following established conventions from seminal ML/RL/EA papers.

- **Two-column template and layout conventions**: Read [references/two-column-template.md](references/two-column-template.md)
- **Domain-specific examples** (Adam, PPO, GA, CMA-ES, NEAT, Q-learning, etc.): Read [references/algorithms-research.md](references/algorithms-research.md)

## Workflow

### 1. Detect Domain

| Domain | Structural Pattern | Style |
|--------|-------------------|-------|
| **Neural Network / DL** | forward-backward-update loop | CLRS/Goodfellow style |
| **Reinforcement Learning** | interact-compute-update loop | Sutton-Barto or CLRS style |
| **Evolutionary** | initialize-evaluate-select-reproduce loop | Procedural with set notation |
| **General / Other** | varies | CLRS style (default) |

### 2. Select Output Format

- **LaTeX** (default): `algpseudocode` + `algorithm` package. Use two-column layout from template.
- **Markdown**: Plain-text pseudo-code blocks with consistent formatting.

### 3. Select Algorithm Variant

When complexity warrants it, produce a **conceptual + practical pair**:

- **Conceptual**: Pure math, scalar-level loops, named math operators (`\text{clip}`, `\textsc{StopGrad}`).
- **Practical / Vectorized**: Tensor-level ops, `\texttt{}` for implementation functions,
  `\hfill $\mathcal{O}(...)$` complexity annotations, footer summary.

### 4. Apply Formatting Rules

1. **Two-column layout**: Left minipage (0.28) for metadata, right minipage (0.68) for algorithmic body, `\vrule` separator.
   Always read [references/two-column-template.md](references/two-column-template.md) for the exact template.
2. **One statement per line**
3. **Indent to show hierarchy** (not braces)
4. **Mathematical notation in conceptual versions** (`\nabla`, `\sum`, `\argmax` -- not `for x in range(n)`).
   `\texttt{}` permitted in practical versions for implementation-level functions.
5. **Left-arrow for assignment** (`\gets`, not `=`). Equals for equality testing only.
6. **Number and caption algorithms** ("Algorithm 1: Name")
7. **Left panel for hyperparameters** -- Ensure, Hyperparameters, Notation/Tensors sections in the left minipage.
   Never bury hyperparameters in the loop body.
8. **Phase comments** via `\Statex \hspace{\algorithmicindent}\textit{// Description}`.
   In practical versions, append `\hfill $\mathcal{O}(...)$`.
9. **Consistent notation**: bold for vectors/params (`\boldsymbol{\theta}`), calligraphic for sets/losses (`\mathcal{L}`, `\mathcal{D}`),
   greek for scalars (`\alpha`), hat for estimates (`\hat{A}_t`).
10. **Self-contained**: readable from the algorithm box alone.

### 5. Follow Domain-Specific Structure

**Neural Networks** -- Four canonical steps in the inner loop:
```
forward pass -> loss computation -> backward pass (backprop) -> parameter update
```

**Reinforcement Learning** -- Three-phase skeleton:
```
Phase 1: Interact with environment (collect experience)
Phase 2: Compute targets (returns, advantages, TD errors)
Phase 3: Update parameters (policy, value function, or both)
```

**Evolutionary Algorithms** -- Population-level loop:
```
Phase 1: Evaluate fitness
Phase 2: Select parents, apply variation (crossover, mutation)
Phase 3: Select survivors
```

### 6. Avoid Anti-Patterns

- No programming-language syntax in conceptual versions (`loss.backward()`, `for x in range(n)`)
- No implementation details (data loading, logging, checkpointing, device management)
- No ambiguous `=` for assignment (use `\gets`)
- No missing initialization -- always show initial values before the loop
- No unlabeled algorithms -- always "Algorithm N: Name"
- No `\Require`/`\Ensure` inside `algorithmic` -- use the left panel instead

## LaTeX Package Reference

### algpseudocode (recommended)

Key commands: `\State`, `\If{} ... \EndIf`, `\While{} ... \EndWhile`,
`\For{} ... \EndFor`, `\ForAll{} ... \EndFor`, `\Call{Name}{args}`,
`\Statex` (unnumbered line), `\Comment{text}`.

### algorithm2e (alternative)

Key commands: `\KwIn{}`, `\KwOut{}`, `\KwData{}`, `\KwResult{}`, lines end with `\;`,
`\eIf{cond}{true}{false}`, `\If{cond}{body}`, `\While{cond}{body}`, `\For{cond}{body}`,
`\tcc{multi-line comment}`, `\tcp{single-line comment}`.

**Critical:** These two package families are mutually exclusive. Never load both.
