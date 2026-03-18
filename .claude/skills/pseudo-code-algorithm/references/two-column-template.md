# Two-Column Algorithm Template

Preferred layout for all algorithms. Left panel holds metadata (hyperparameters, notation, tensor shapes),
right panel holds the algorithmic body. Separated by a vertical rule.

## Conceptual vs Practical Pair

Each algorithm should have two versions when complexity warrants it:

1. **Conceptual** — Pure mathematical notation, scalar-level loops, no implementation details.
   Use `\text{clip}`, `\text{clamp}`, `\textsc{StopGrad}` for named operations.
2. **Practical / Vectorized** — Tensor-level operations, batch indexing, complexity annotations.
   Use `\texttt{}` for implementation-level functions (e.g., `\texttt{envs.step}`, `\texttt{randperm}`,
   `\texttt{mean}`, `\texttt{std}`, `\texttt{stop\_grad}`, `\texttt{token\_mask}`).

## Base Template (Conceptual)

```latex
\begin{algorithm}[p]\small
\caption{Algorithm Name (Full Title)}\label{alg:name}
\vspace{2pt}
\noindent
\begin{minipage}[t]{0.28\linewidth}
\renewcommand{\arraystretch}{1.12}
\textbf{Ensure} \\[2pt]
Output description

\vspace{6pt}
\textbf{Hyperparameters} \\[2pt]
$\boldsymbol{\theta}_0$ — init params \\
$\epsilon$ — clip ratio (typ.\ $0.2$) \\
$\gamma$ — discount factor \\
% ... one line per hyperparameter

\vspace{6pt}
\textbf{Notation} \\[2pt]
$s_t$ — state \\
$a_t$ — action \\
% ... one line per symbol
\end{minipage}%
\hfill
\vrule
\hfill
\begin{minipage}[t]{0.68\linewidth}
\begin{algorithmic}[1]
\For{$k = 0, 1, 2, \ldots$}
    \Statex \hspace{\algorithmicindent}\textit{// Phase 1: Description}
    % ... phase 1 steps
    \Statex \hspace{\algorithmicindent}\textit{// Phase 2: Description}
    % ... phase 2 steps
    \Statex \hspace{\algorithmicindent}\textit{// Phase 3: Description}
    % ... phase 3 steps
\EndFor
\end{algorithmic}
\end{minipage}
\end{algorithm}
```

## Base Template (Practical / Vectorized)

Extends the conceptual template with:
- **Tensors** section replaces or supplements **Notation** in left panel
- Complexity annotations via `\hfill $\mathcal{O}(...)$`
- `\texttt{}` for named implementation operations
- Footer with complexity summary

```latex
\begin{algorithm}[p]\small
\caption{Algorithm Name — Practical Vectorized Implementation}\label{alg:name-practical}
\vspace{2pt}
\noindent
\begin{minipage}[t]{0.28\linewidth}
\renewcommand{\arraystretch}{1.12}
\textbf{Ensure} \\[2pt]
Output description

\vspace{6pt}
\textbf{Hyperparameters} \\[2pt]
$\boldsymbol{\theta}_0$ — init params \\
% ... one line per hyperparameter

\vspace{6pt}
\textbf{Tensors} \\[2pt]
$B$ — batch size \\
$\mathbf{S} \in \mathbb{R}^{N \times T}$ \\
$\mathbf{D} \in \{0,1\}^{N \times T}$ — done flags \\
% ... one line per tensor with shape
\end{minipage}%
\hfill
\vrule
\hfill
\begin{minipage}[t]{0.68\linewidth}
\begin{algorithmic}[1]
\For{$k = 0, 1, 2, \ldots$}
    \Statex \hspace{\algorithmicindent}\textit{// Phase 1: Description} \hfill $\mathcal{O}(\ldots)$
    \For{$t = 0, \ldots, T{-}1$}
        \State $\mathbf{A}_{:,t} \sim \pi_{\boldsymbol{\theta}_k}(\cdot \mid \mathbf{S}_{:,t})$ \hfill $\mathcal{O}(N \cdot C_\pi)$
        \State $\mathbf{R}_{:,t},\, \mathbf{S}_{:,t+1} \gets \texttt{envs.step}(\mathbf{A}_{:,t})$ \hfill $\mathcal{O}(N)$
    \EndFor
    \Statex \hspace{\algorithmicindent}\textit{// Phase 2: Description} \hfill $\mathcal{O}(\ldots)$
    % ... vectorized operations
    \Statex \hspace{\algorithmicindent}\textit{// Phase 3: Description} \hfill $\mathcal{O}(\ldots)$
    % ... minibatch SGD loop
\EndFor
\end{algorithmic}
\vspace{4pt}
\noindent\rule{\linewidth}{0.4pt}
\vspace{2pt}
\textit{Per-iteration complexity:} $\mathcal{O}\big(\ldots\big)$ \\[2pt]
$C_\pi$: policy forward/backward cost. \\
Dominant term explanation.
\end{minipage}
\end{algorithm}
```

## Comparison Summary Box

For side-by-side algorithm comparison (no line numbers):

```latex
\begin{algorithm}[t]
\caption{Key Differences: A vs B vs C}\label{alg:compare}
\begin{algorithmic}[0]
\Statex \textbf{Algorithm A} \textit{(Author, Year):}
\Statex \hspace{\algorithmicindent} $L = \ldots$ \; (loss formula)
\Statex \hspace{\algorithmicindent} Key property description.
\Statex
\Statex \textbf{Algorithm B} \textit{(Author, Year):}
\Statex \hspace{\algorithmicindent} $L = \ldots$
\Statex \hspace{\algorithmicindent} Key property description.
\end{algorithmic}
\end{algorithm}
```

## Left Panel Conventions

| Section | When to use | Content |
|---------|------------|---------|
| **Ensure** | Always | One-line output description |
| **Hyperparameters** | Always | One param per line: `$symbol$ — description` |
| **Notation** | Conceptual | Symbols used in the algorithm body |
| **Tensors** | Practical | Tensor names with shapes: `$\mathbf{X} \in \mathbb{R}^{N \times T}$` |

Formatting: `\renewcommand{\arraystretch}{1.12}` for line spacing. Use `\\[2pt]` after section headers.
Use `\vspace{6pt}` between sections. Use `\\` (not `\\[Xpt]`) between entries within a section.

## Phase Comment Convention

Use `\Statex` with italic comments to label phases. Place before the phase's first statement:

```latex
\Statex \hspace{\algorithmicindent}\textit{// Phase description}
```

In practical versions, append per-phase complexity:

```latex
\Statex \hspace{\algorithmicindent}\textit{// Phase description} \hfill $\mathcal{O}(\ldots)$
```

## Complexity Annotation Convention (Practical only)

- **Per-phase**: On the `\Statex` comment line via `\hfill`
- **Per-statement**: On expensive operations (neural forward passes, env steps) via `\hfill`
- **Footer summary**: Below `\noindent\rule{\linewidth}{0.4pt}`, italic text with total per-iteration cost
  and explanation of cost variables ($C_\pi$, $C_V$, dominant term analysis)
