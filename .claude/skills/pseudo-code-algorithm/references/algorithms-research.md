# Pseudo-Code Algorithms Across ML Paradigms - Research

**Researched:** 2026-03-14
**Domain:** Pseudo-code conventions for Machine Learning, Reinforcement Learning, and Evolutionary Algorithms
**Confidence:** HIGH (well-established conventions from seminal papers and textbooks)

## Summary

Pseudo-code in machine learning spans three major paradigms -- supervised deep learning, reinforcement learning, and evolutionary computation -- each with distinct but overlapping conventions for presenting algorithms. The dominant style in ML/AI papers uses a structured "Algorithm box" format with numbered lines, explicit input/output specifications, and mathematical notation embedded within procedural control flow. Two influential textbooks have shaped modern conventions: Goodfellow, Bengio, and Courville's *Deep Learning* (2016) uses a `Require:`/`Ensure:` format derived from Cormen et al.'s *CLRS*, while Sutton and Barto's *Reinforcement Learning: An Introduction* (2018, 2nd ed.) uses a distinctive shaded-box format with natural-language headers. Evolutionary computation tends toward a more procedural style with explicit population-level operations.

For LaTeX typesetting, three package families dominate: **algorithmicx/algpseudocode** (the most common in ML papers, compatible with IEEE templates), **algorithm2e** (more flexible, more actively maintained, popular in European venues), and the older **algorithmic** package. These are mutually exclusive and cannot be loaded simultaneously. The choice is often dictated by venue/template requirements.

**Primary recommendation:** Use the `algpseudocode` + `algorithm` package combination (from the `algorithmicx` family) for maximum compatibility with ML venues, and follow the Goodfellow/CLRS-style `Require:`/`while`/`return` conventions with embedded LaTeX math for gradient updates and loss computations.

---

## Pseudo-code Conventions

### The Three Dominant Notation Styles

#### 1. CLRS / Goodfellow Style (Most Common in ML)
Used in: *Deep Learning* (Goodfellow et al., 2016), *Introduction to Algorithms* (CLRS), most NeurIPS/ICML/ICLR papers.

Characteristics:
- **`Require:`** for inputs/preconditions, **`Ensure:`** for outputs/postconditions
- Title-cased keywords: `while`, `for`, `if`, `then`, `else`, `return`
- Line numbers on every line
- Mathematical notation inline: `theta <- theta - alpha * nabla_theta L(theta)`
- Assignment uses left-arrow (`<-`) not equals
- Comments right-aligned in italics or gray
- Algorithm wrapped in a numbered, captioned float ("Algorithm 1: ...")

Example structure:
```
Algorithm 1: Stochastic Gradient Descent (SGD)
  Require: Learning rate alpha, initial parameters theta
  Require: Training set {(x_1, y_1), ..., (x_N, y_N)}
    while stopping criterion not met do
      Sample minibatch {x^(1), ..., x^(m)} from training set
      Compute gradient estimate: g_hat <- (1/m) nabla_theta sum L(f(x^(i); theta), y^(i))
      Apply update: theta <- theta - alpha * g_hat
    end while
  return theta
```

#### 2. Sutton-Barto Style (Standard in RL)
Used in: *Reinforcement Learning: An Introduction* (Sutton & Barto, 2018), many RL papers.

Characteristics:
- Shaded/boxed algorithm with bold title
- Natural-language section headers: "Initialize:", "Loop for each episode:", "Loop for each step:"
- Indentation-based nesting (no explicit `end` keywords)
- Greek letters for parameters (alpha, gamma, epsilon)
- Episode-centric structure with explicit environment interaction
- Tabular update rules displayed as assignment statements
- Uses epsilon-greedy notation inline

Example structure:
```
Q-learning (off-policy TD control)
  Algorithm parameters: step size alpha in (0, 1], small epsilon > 0
  Initialize Q(s, a), for all s in S+, a in A(s), arbitrarily except Q(terminal, .) = 0

  Loop for each episode:
    Initialize S
    Loop for each step of episode:
      Choose A from S using policy derived from Q (e.g., epsilon-greedy)
      Take action A, observe R, S'
      Q(S, A) <- Q(S, A) + alpha [R + gamma max_a Q(S', a) - Q(S, A)]
      S <- S'
    until S is terminal
```

#### 3. Algorithm2e / European Style
Used in: Many European conference papers, some AI venues.

Characteristics:
- Line terminators with `\;`
- Braces for block structure
- `KwData:` / `KwResult:` for input/output
- `eIf{}{}{}`-style conditionals
- Often uses `ruled` style (horizontal lines top and bottom)
- More compact, slightly more "code-like"

### LaTeX Package Comparison

| Package Family | Key Package | Syntax Style | Best For | Maintenance |
|---|---|---|---|---|
| **algorithmicx** | `algpseudocode` | Title-case (`\State`, `\If`, `\While`) | ML papers, IEEE venues, CLRS-style | Active |
| **algorithm2e** | `algorithm2e` | Brace-based with `\;` terminators | Flexible formatting, European venues | Most active |
| **algorithmic** | `algorithmic` | UPPERCASE (`\STATE`, `\IF`, `\WHILE`) | Legacy papers, IEEE default | Legacy |

**Critical constraint:** These packages are mutually exclusive. Loading more than one will produce LaTeX errors.

#### algorithmicx / algpseudocode (Recommended for ML)

```latex
\usepackage{algorithm}
\usepackage{algpseudocode}

\begin{algorithm}
\caption{Mini-Batch Stochastic Gradient Descent}\label{alg:sgd}
\begin{algorithmic}[1]
\Require Learning rate $\alpha$, initial parameters $\boldsymbol{\theta}$
\Require Training set $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_N, y_N)\}$
\While{stopping criterion not met}
    \State Sample minibatch $\mathcal{B} = \{(\mathbf{x}^{(1)}, y^{(1)}), \ldots, (\mathbf{x}^{(m)}, y^{(m)})\}$
    \State Compute gradient: $\hat{\mathbf{g}} \gets \frac{1}{m} \nabla_{\boldsymbol{\theta}} \sum_{i=1}^{m} \mathcal{L}(f(\mathbf{x}^{(i)}; \boldsymbol{\theta}), y^{(i)})$
    \State Update parameters: $\boldsymbol{\theta} \gets \boldsymbol{\theta} - \alpha \hat{\mathbf{g}}$
\EndWhile
\State \textbf{return} $\boldsymbol{\theta}$
\end{algorithmic}
\end{algorithm}
```

Key commands:
- `\Require` / `\Ensure` -- inputs / outputs
- `\State` -- a statement line
- `\If{cond} ... \ElsIf{cond} ... \Else ... \EndIf`
- `\While{cond} ... \EndWhile`
- `\For{$i = 1$ \textbf{to} $n$} ... \EndFor`
- `\ForAll{item \textbf{in} collection} ... \EndFor`
- `\Repeat ... \Until{cond}`
- `\Procedure{Name}{$params$} ... \EndProcedure`
- `\Function{Name}{$params$} ... \EndFunction`
- `\Call{FunctionName}{$args$}` -- function calls
- `\Comment{text}` -- right-aligned comments
- `[1]` after `\begin{algorithmic}` enables line numbering

#### algorithm2e

```latex
\usepackage[ruled,lined,linesnumbered,commentsnumbered,longend]{algorithm2e}

\begin{algorithm}[H]
\caption{Mini-Batch SGD}
\label{alg:sgd-a2e}
\KwIn{Learning rate $\alpha$, initial parameters $\boldsymbol{\theta}$, training set $\mathcal{D}$}
\KwOut{Optimized parameters $\boldsymbol{\theta}$}
\While{stopping criterion not met}{
    Sample minibatch $\mathcal{B} \sim \mathcal{D}$\;
    $\hat{\mathbf{g}} \gets \frac{1}{|\mathcal{B}|} \nabla_{\boldsymbol{\theta}} \sum_{(\mathbf{x},y) \in \mathcal{B}} \mathcal{L}(f(\mathbf{x}; \boldsymbol{\theta}), y)$\;
    $\boldsymbol{\theta} \gets \boldsymbol{\theta} - \alpha \hat{\mathbf{g}}$\;
}
\Return{$\boldsymbol{\theta}$}
\end{algorithm}
```

Key commands:
- `\KwIn{}` / `\KwOut{}` / `\KwData{}` / `\KwResult{}` -- input/output
- Lines must end with `\;`
- `\eIf{cond}{true}{false}` -- if/else
- `\If{cond}{body}` -- if only
- `\While{cond}{body}` / `\For{cond}{body}`
- `\tcc{text}` -- multi-line comment
- `\tcp{text}` -- single-line comment
- `\SetKwFunction{cmd}{display}` -- custom functions
- `\SetKwInOut{KwIn}{Input}` -- aligned keyword definitions

### Cross-Paradigm Formatting Rules

These rules apply across all three ML paradigms:

1. **One statement per line.** Each line should express exactly one operation.
2. **Indent to show hierarchy.** Indentation, not braces, communicates nesting to the human reader.
3. **Mathematical notation over code syntax.** Use `nabla`, `sum`, `argmax`, set notation -- not `for x in range(n)`.
4. **Left-arrow for assignment.** `theta <- theta - alpha * g` (not `=`). Equals is reserved for equality testing.
5. **Capitalize keywords consistently.** Either all title-case (`While`, `If`, `Return`) or all small-caps.
6. **Name algorithms with numbered captions.** "Algorithm 1: Adam Optimizer" allows cross-referencing.
7. **List hyperparameters in the Input/Require section.** Not buried in the loop body.
8. **Comments explain "why", not "what".** The pseudo-code itself explains "what."
9. **Use calligraphic/bold/greek consistently.** E.g., bold for vectors (`theta`), calligraphic for sets/losses (`L`, `D`), greek for scalars (`alpha`, `gamma`).
10. **Pseudo-code should be self-contained.** A reader should understand the algorithm from the box alone, without reading surrounding text.

---

## Neural Network Algorithms

### Standard Deep Learning Training Loop

The canonical presentation follows the Goodfellow textbook pattern (Chapter 8). This is the most common pseudo-code structure in supervised learning papers.

```
Algorithm 1: Neural Network Training with Mini-Batch SGD
  Require: Learning rate schedule {alpha_t}, initial parameters theta_0
  Require: Training dataset D = {(x_i, y_i)}_{i=1}^N
  Require: Mini-batch size m, number of epochs T

  1:  for epoch = 1 to T do
  2:    Shuffle D
  3:    for each mini-batch B = {(x^(1), y^(1)), ..., (x^(m), y^(m))} from D do
  4:      // Forward pass
  5:      y_hat^(i) <- f(x^(i); theta)  for i = 1, ..., m
  6:      // Compute loss
  7:      L <- (1/m) sum_{i=1}^m l(y_hat^(i), y^(i))
  8:      // Backward pass (backpropagation)
  9:      g <- nabla_theta L
  10:     // Parameter update
  11:     theta <- theta - alpha_t * g
  12:   end for
  13:   // Evaluate on validation set (optional)
  14:   if validation_metric has not improved for p epochs then
  15:     break  // Early stopping
  16:   end if
  17: end for
  return theta
```

Key conventions:
- The **forward pass**, **loss computation**, **backward pass**, and **update** are the four canonical steps
- Early stopping is shown as a conditional break within the epoch loop
- Validation evaluation sits between epochs, not within the batch loop
- The learning rate may be a schedule (subscripted by t)

### Adam Optimizer (Kingma & Ba, 2015)

The Adam paper (arXiv:1412.6980) set a template widely imitated in ML papers. Algorithm 1 from the paper:

```
Algorithm 1: Adam
  Require: alpha: Stepsize
  Require: beta_1, beta_2 in [0, 1): Exponential decay rates for moment estimates
  Require: f(theta): Stochastic objective function with parameters theta
  Require: theta_0: Initial parameter vector

  1:  m_0 <- 0  (Initialize 1st moment vector)
  2:  v_0 <- 0  (Initialize 2nd moment vector)
  3:  t <- 0    (Initialize timestep)
  4:  while theta_t not converged do
  5:    t <- t + 1
  6:    g_t <- nabla_theta f_t(theta_{t-1})     (Get gradients w.r.t. stochastic objective at timestep t)
  7:    m_t <- beta_1 * m_{t-1} + (1 - beta_1) * g_t    (Update biased first moment estimate)
  8:    v_t <- beta_2 * v_{t-1} + (1 - beta_2) * g_t^2  (Update biased second raw moment estimate)
  9:    m_hat_t <- m_t / (1 - beta_1^t)         (Compute bias-corrected first moment estimate)
  10:   v_hat_t <- v_t / (1 - beta_2^t)         (Compute bias-corrected second raw moment estimate)
  11:   theta_t <- theta_{t-1} - alpha * m_hat_t / (sqrt(v_hat_t) + epsilon)  (Update parameters)
  12: end while
  return theta_t  (Resulting parameters)
```

Notable conventions from this seminal paper:
- **Parenthesized comments** on each line explaining the operation
- **Subscript-t notation** for time-indexed variables
- **Greek letters** for all hyperparameters
- **Explicit initialization** of all state variables before the loop
- **"Good default values"** stated in the paper text, not the algorithm box

### Architecture Description in Pseudo-code

Papers like *Attention Is All You Need* (Vaswani et al., 2017) typically do NOT present the architecture as pseudo-code. Instead, they use:
- **Diagrams** (the famous encoder-decoder figure)
- **Mathematical equations** for individual components (multi-head attention formula)
- **Table of hyperparameters** (model dimension, number of heads, etc.)

When architectures ARE presented as pseudo-code (less common), they follow a functional decomposition:
```
Function MultiHeadAttention(Q, K, V, d_k, h):
  for i = 1 to h do
    head_i <- Attention(Q W_i^Q, K W_i^K, V W_i^V)
  end for
  return Concat(head_1, ..., head_h) W^O

Function Attention(Q, K, V):
  return softmax(Q K^T / sqrt(d_k)) V
```

### Backpropagation

The backpropagation algorithm in textbooks is typically presented in two forms:
1. **Per-layer forward/backward** as a general framework (Goodfellow Ch. 6)
2. **Computational graph** with chain rule applied to each node

The pseudo-code form usually looks like:
```
Algorithm: Backpropagation (for a feedforward network)
  // Forward pass
  h_0 <- x  (input)
  for l = 1 to L do
    a_l <- W_l h_{l-1} + b_l    (pre-activation)
    h_l <- sigma_l(a_l)          (activation)
  end for
  L <- loss(h_L, y)

  // Backward pass
  delta_L <- nabla_{h_L} L * sigma_L'(a_L)
  for l = L-1 to 1 do
    delta_l <- (W_{l+1}^T delta_{l+1}) * sigma_l'(a_l)
  end for

  // Gradient computation
  for l = 1 to L do
    nabla_{W_l} L <- delta_l h_{l-1}^T
    nabla_{b_l} L <- delta_l
  end for
```

---

## Reinforcement Learning Algorithms

### Tabular Q-Learning (Sutton & Barto Style)

This is the canonical presentation from *Reinforcement Learning: An Introduction* (2nd ed., 2018). The Sutton-Barto style uses natural language headers and a distinctive boxed format.

```
Q-learning (off-policy TD control) for estimating pi ~= pi*

  Algorithm parameters: step size alpha in (0, 1], small epsilon > 0
  Initialize Q(s, a), for all s in S+, a in A(s), arbitrarily except that Q(terminal, .) = 0

  Loop for each episode:
    Initialize S
    Loop for each step of episode:
      Choose A from S using policy derived from Q (e.g., epsilon-greedy)
      Take action A, observe R, S'
      Q(S, A) <- Q(S, A) + alpha [R + gamma * max_a Q(S', a) - Q(S, A)]
      S <- S'
    until S is terminal
```

Key conventions:
- **No line numbers** -- uses indentation and natural language headers
- **Uppercase single letters** for random variables (S, A, R, S')
- **"Loop for each..."** instead of `for` or `while`
- **Epsilon-greedy** described inline parenthetically
- **TD error** expressed as the bracketed quantity `[R + gamma max_a Q(S', a) - Q(S, A)]`
- **Terminal state** handled as initialization condition

### SARSA (On-Policy TD Control)

```
Sarsa (on-policy TD control) for estimating Q ~= q_pi

  Algorithm parameters: step size alpha in (0, 1], small epsilon > 0
  Initialize Q(s, a), for all s in S+, a in A(s), arbitrarily except that Q(terminal, .) = 0

  Loop for each episode:
    Initialize S
    Choose A from S using policy derived from Q (e.g., epsilon-greedy)
    Loop for each step of episode:
      Take action A, observe R, S'
      Choose A' from S' using policy derived from Q (e.g., epsilon-greedy)
      Q(S, A) <- Q(S, A) + alpha [R + gamma * Q(S', A') - Q(S, A)]
      S <- S'; A <- A'
    until S is terminal
```

Note the key structural difference from Q-learning: action A' is selected BEFORE the update (on-policy), and the update uses `Q(S', A')` instead of `max_a Q(S', a)`.

### REINFORCE (Williams, 1992)

The REINFORCE algorithm is the foundational policy gradient method. Modern presentations (e.g., Sutton & Barto Ch. 13) use this style:

```
REINFORCE: Monte-Carlo Policy-Gradient Control (episodic) for pi*

  Input: a differentiable policy parameterization pi(a|s, theta)
  Algorithm parameter: step size alpha > 0

  Initialize policy parameter theta in R^d (e.g., to 0)

  Loop forever (for each episode):
    Generate an episode S_0, A_0, R_1, ..., S_{T-1}, A_{T-1}, R_T following pi(.|., theta)
    Loop for each step of the episode t = 0, 1, ..., T-1:
      G <- sum_{k=t+1}^{T} gamma^{k-t-1} R_k        (return from step t)
      theta <- theta + alpha * gamma^t * G * nabla_theta ln pi(A_t|S_t, theta)
```

With baseline (variance reduction):
```
REINFORCE with Baseline (episodic) for pi*

  Input: a differentiable policy parameterization pi(a|s, theta)
  Input: a differentiable state-value parameterization v_hat(s, w)
  Algorithm parameters: step sizes alpha^theta > 0, alpha^w > 0

  Initialize policy parameter theta and state-value weights w

  Loop forever (for each episode):
    Generate episode following pi(.|., theta)
    Loop for each step t = 0, ..., T-1:
      G <- sum_{k=t+1}^T gamma^{k-t-1} R_k
      delta <- G - v_hat(S_t, w)                    (advantage estimate)
      w <- w + alpha^w * delta * nabla_w v_hat(S_t, w)
      theta <- theta + alpha^theta * gamma^t * delta * nabla_theta ln pi(A_t|S_t, theta)
```

### PPO (Schulman et al., 2017)

From the original paper (arXiv:1707.06347) and OpenAI Spinning Up:

```
Algorithm 1: PPO-Clip

  for iteration = 1, 2, ... do
    for actor = 1, 2, ..., N do
      Run policy pi_{theta_old} in environment for T timesteps
      Compute advantage estimates A_hat_1, ..., A_hat_T
    end for
    Optimize surrogate L w.r.t. theta, with K epochs and minibatch size M <= NT:
      L^CLIP(theta) = E_t [min(r_t(theta) A_hat_t, clip(r_t(theta), 1-epsilon, 1+epsilon) A_hat_t)]
    where r_t(theta) = pi_theta(a_t|s_t) / pi_{theta_old}(a_t|s_t)
    theta_old <- theta
  end for
```

Expanded form from OpenAI Spinning Up:

```
Algorithm: PPO-Clip (Expanded)

  Input: initial policy parameters theta_0, initial value function parameters phi_0
  Hyperparameters: clip ratio epsilon, GAE parameter lambda, discount gamma

  for k = 0, 1, 2, ... do
    // Rollout phase
    Collect set of trajectories D_k by running pi_k = pi(theta_k) in the environment
    Compute rewards-to-go R_hat_t
    Compute advantage estimates A_hat_t (using GAE-lambda) based on current V_{phi_k}

    // Policy update phase (multiple epochs over D_k)
    for epoch = 1, ..., K do
      for each minibatch from D_k do
        r_t(theta) <- pi_theta(a_t|s_t) / pi_{theta_k}(a_t|s_t)
        L^CLIP <- (1/|B|) sum_t min(r_t A_hat_t, clip(r_t, 1-eps, 1+eps) A_hat_t)
        Update theta by maximizing L^CLIP via SGD (Adam)
      end for
    end for

    // Value function update
    Fit V_{phi_{k+1}} by regression on mean-squared error:
      phi_{k+1} = argmin_phi (1/|D_k|T) sum_{tau in D_k} sum_{t=0}^T (V_phi(s_t) - R_hat_t)^2
  end for
```

Key RL pseudo-code conventions:
- **Probability ratio** `r_t(theta)` is a standard notation across policy gradient papers
- **Advantage estimates** `A_hat` always use hat notation
- **Environment interaction** is explicitly shown as a distinct phase from optimization
- **Multiple epochs** over collected data (unique to PPO-style algorithms)
- **Subscript k** for iteration index vs **subscript t** for timestep

### Actor-Critic (A2C)

```
Algorithm: Advantage Actor-Critic (A2C)

  Initialize policy network pi_theta and value network V_phi
  Hyperparameters: learning rates alpha_pi, alpha_V; discount gamma; GAE lambda; n-step T

  repeat
    // Collect T-step rollout
    for t = 0, ..., T-1 do
      a_t ~ pi_theta(.|s_t)
      s_{t+1}, r_t <- env.step(a_t)
    end for

    // Bootstrap final value
    if s_T is not terminal then
      R <- V_phi(s_T)
    else
      R <- 0
    end if

    // Compute advantages (GAE)
    for t = T-1, ..., 0 do
      delta_t <- r_t + gamma * V_phi(s_{t+1}) - V_phi(s_t)    (TD error)
      A_t <- delta_t + gamma * lambda * A_{t+1}                 (GAE)
    end for

    // Update critic
    L_V <- (1/T) sum_t (V_phi(s_t) - (A_t + V_phi(s_t)))^2
    phi <- phi - alpha_V * nabla_phi L_V

    // Update actor
    L_pi <- -(1/T) sum_t log pi_theta(a_t|s_t) * A_t
    theta <- theta - alpha_pi * nabla_theta L_pi
  until convergence
```

### RL Pseudo-code Structural Pattern

All RL algorithms share a common skeleton:

```
1. Initialize (parameters, value estimates, environment)
2. Loop (episodes or iterations):
   a. Interact with environment (collect experience)
   b. Compute targets (returns, advantages, TD errors)
   c. Update parameters (policy, value function, or both)
3. Return (policy, value function)
```

The key differentiator between RL algorithms in pseudo-code is:
- **What is updated:** Q-table (Q-learning), policy parameters (REINFORCE), both (actor-critic)
- **How targets are computed:** Monte Carlo returns (REINFORCE), TD targets (Q-learning), GAE (PPO/A2C)
- **When updates happen:** Per-step (Q-learning), per-episode (REINFORCE), per-rollout (PPO)

---

## Evolutionary Algorithms

### Generic Evolutionary Algorithm Framework

The standard EA pseudo-code template used across textbooks (Eiben & Smith, De Jong):

```
Algorithm: Generic Evolutionary Algorithm

  Input: Population size mu, offspring count lambda, fitness function f
  Output: Best individual x*

  1:  P_0 <- InitializePopulation(mu)      // Random initialization
  2:  Evaluate(P_0, f)                      // Compute fitness for all individuals
  3:  t <- 0
  4:  while termination condition not met do
  5:    Parents <- SelectParents(P_t)        // Parent selection (tournament, roulette, etc.)
  6:    Offspring <- empty set
  7:    for i = 1 to lambda do
  8:      if random() < p_c then             // Crossover probability
  9:        (p1, p2) <- ChooseTwo(Parents)
  10:       child <- Crossover(p1, p2)
  11:     else
  12:       child <- Copy(ChooseOne(Parents))
  13:     end if
  14:     if random() < p_m then             // Mutation probability
  15:       child <- Mutate(child)
  16:     end if
  17:     Offspring <- Offspring union {child}
  18:   end for
  19:   Evaluate(Offspring, f)
  20:   P_{t+1} <- SelectSurvivors(P_t union Offspring)  // Survivor selection
  21:   t <- t + 1
  22: end while
  return BestOf(P_t)
```

Key EA pseudo-code conventions:
- **Population-level operations** (Initialize, Evaluate, Select) are treated as black-box functions
- **Operator probabilities** (p_c, p_m) appear as conditions in the loop
- **Set notation** for population manipulation (union, empty set)
- **(mu, lambda)** or **(mu + lambda)** notation indicates selection strategy: comma = offspring only, plus = parents + offspring

### Genetic Algorithm (Standard/Canonical GA)

```
Algorithm: Simple Genetic Algorithm (SGA)

  Input: Population size N, crossover rate p_c, mutation rate p_m
  Input: Fitness function f, chromosome length L
  Output: Best individual found

  1:  Initialize population P of N random binary strings of length L
  2:  Evaluate fitness f(x) for each x in P
  3:  while not converged do
  4:    // Selection (fitness-proportionate / roulette wheel)
  5:    P' <- empty
  6:    for i = 1 to N do
  7:      Select individual x from P with probability f(x) / sum_j f(x_j)
  8:      P' <- P' union {x}
  9:    end for
  10:   // Crossover (single-point)
  11:   for each consecutive pair (x_i, x_{i+1}) in P' do
  12:     if random() < p_c then
  13:       Choose random crossover point k in {1, ..., L-1}
  14:       Swap x_i[k+1:L] and x_{i+1}[k+1:L]
  15:     end if
  16:   end for
  17:   // Mutation (bit-flip)
  18:   for each x in P' do
  19:     for each bit j in x do
  20:       if random() < p_m then
  21:         x[j] <- 1 - x[j]          // Flip bit
  22:       end if
  23:     end for
  24:   end for
  25:   Evaluate fitness f(x) for each x in P'
  26:   P <- P'
  27: end while
  return argmax_{x in P} f(x)
```

### Evolution Strategies (ES)

#### (mu/rho, lambda)-ES

From Clever Algorithms (Brownlee, 2011) and Lilian Weng's survey:

```
Algorithm: (mu, lambda)-Evolution Strategy

  Input: mu (parent pop size), lambda (offspring count, lambda > mu)
  Input: Problem size n, fitness function f
  Output: Best solution S_best

  1:  Population <- InitializePopulation(mu)
  2:  Evaluate(Population, f)
  3:  S_best <- GetBest(Population)
  4:  while termination condition not met do
  5:    Children <- empty set
  6:    for i = 1 to lambda do
  7:      Parent_i <- RandomSelect(Population)
  8:      // Mutate strategy parameters
  9:      sigma_i' <- Parent_i.sigma * exp(tau * N(0,1))
  10:     // Mutate problem variables
  11:     x_i' <- Parent_i.x + sigma_i' * N(0, I)
  12:     Child_i <- (x_i', sigma_i')
  13:     Children <- Children union {Child_i}
  14:   end for
  15:   Evaluate(Children, f)
  16:   // Comma selection: parents do NOT survive
  17:   Population <- SelectBest(Children, mu)
  18:   S_best <- GetBest(Population)
  19: end while
  return S_best
```

#### CMA-ES (Covariance Matrix Adaptation)

```
Algorithm: CMA-ES

  Input: Initial mean m_0 in R^n, initial step-size sigma_0 > 0, population size lambda
  Initialize: C_0 <- I (covariance matrix), p_sigma <- 0 (step-size evolution path),
              p_c <- 0 (covariance evolution path), t <- 0

  while not converged do
    // 1. Sample offspring
    for k = 1 to lambda do
      z_k ~ N(0, I)
      x_k <- m_t + sigma_t * C_t^{1/2} z_k
    end for
    Evaluate f(x_k) for k = 1, ..., lambda

    // 2. Sort and select mu best
    x_{1:lambda}, ..., x_{mu:lambda} <- sort by fitness (ascending for minimization)

    // 3. Update mean (weighted recombination)
    m_{t+1} <- sum_{i=1}^mu w_i * x_{i:lambda}
    where w_1 >= w_2 >= ... >= w_mu > 0, sum w_i = 1

    // 4. Update evolution paths
    p_sigma <- (1 - c_sigma) p_sigma + sqrt(c_sigma(2-c_sigma) mu_eff) * C_t^{-1/2} (m_{t+1} - m_t)/sigma_t
    p_c <- (1 - c_c) p_c + h_sigma * sqrt(c_c(2-c_c) mu_eff) * (m_{t+1} - m_t)/sigma_t

    // 5. Update covariance matrix
    C_{t+1} <- (1 - c_1 - c_mu) C_t
               + c_1 * p_c p_c^T                                    (rank-one update)
               + c_mu sum_{i=1}^mu w_i y_{i:lambda} y_{i:lambda}^T  (rank-mu update)
    where y_{i:lambda} <- (x_{i:lambda} - m_t) / sigma_t

    // 6. Update step size (cumulative step-size adaptation)
    sigma_{t+1} <- sigma_t * exp((c_sigma / d_sigma)(||p_sigma|| / E||N(0,I)|| - 1))

    t <- t + 1
  end while
  return m_t
```

CMA-ES pseudo-code conventions:
- **Heavy mathematical notation** -- more equations than procedural steps
- **Subscript notation** for sorted individuals (`x_{i:lambda}` means the i-th best out of lambda)
- **Hyperparameters** (c_sigma, c_c, c_1, c_mu, d_sigma) have established default formulas based on n
- **Evolution paths** are a unique concept requiring explicit tracking

#### OpenAI ES (Salimans et al., 2017)

```
Algorithm 1: OpenAI Evolution Strategy (from arXiv:1703.03864)

  Input: Learning rate alpha, noise standard deviation sigma, initial parameters theta_0
  for t = 0, 1, 2, ... do
    Sample epsilon_1, ..., epsilon_n ~ N(0, I)
    Compute returns F_i = F(theta_t + sigma * epsilon_i) for i = 1, ..., n
    theta_{t+1} <- theta_t + alpha * (1/(n*sigma)) sum_{i=1}^n F_i * epsilon_i
  end for
```

This is notably compact -- the simplicity is the point. Key conventions:
- **Embarrassingly parallel** evaluation is implicit in the "Compute returns" line
- **No population management** -- just perturbation and gradient estimation
- **Mirror sampling** variant adds antithetic pairs: also evaluate F(theta_t - sigma * epsilon_i)

### NEAT (Stanley & Miikkulainen, 2002)

NEAT uses a unique pseudo-code style that emphasizes structural operations:

```
Algorithm: NEAT (NeuroEvolution of Augmenting Topologies)

  Initialize: Minimal-topology population (all inputs connected directly to all outputs)
  global_innovation_number <- 0

  Loop for each generation:
    // 1. Evaluate fitness
    for each genome g in population do
      Decode g into neural network
      fitness(g) <- Evaluate(network, task)
    end for

    // 2. Speciation (compatibility distance)
    for each genome g do
      Assign g to species based on delta(g, representative) < delta_t
      delta = (c_1 * E)/N + (c_2 * D)/N + c_3 * W_bar
      // E = excess genes, D = disjoint genes, W_bar = avg weight diff of matching genes
    end for

    // 3. Adjusted fitness (fitness sharing)
    for each genome g in species s do
      adjusted_fitness(g) <- fitness(g) / |s|
    end for

    // 4. Reproduction
    Allocate offspring per species proportional to sum of adjusted fitnesses
    for each species s do
      // Elitism: copy champion if |s| >= 5
      Sort members by fitness
      Remove bottom half
      // Crossover
      for each offspring slot do
        if random() < p_crossover then
          Select two parents from s (fitness-proportional)
          child <- CrossoverByInnovationNumber(parent1, parent2)
        else
          child <- Copy(random parent from s)
        end if
        // Mutation
        With prob p_weight: perturb connection weights
        With prob p_add_node: split existing connection, add node
        With prob p_add_link: add new connection between unconnected nodes
        Assign new innovation numbers to structural mutations
      end for
    end for
```

NEAT-specific conventions:
- **Innovation numbers** are central to the algorithm and always explicitly tracked
- **Speciation** is described as a distinct phase
- **Structural mutations** (add node, add connection) are described separately from weight mutations
- **Crossover by gene alignment** using innovation numbers

### Genetic Programming (Koza, 1992)

```
Algorithm: Standard Genetic Programming

  Input: Function set F, terminal set T, max depth D, population size N
  Input: Fitness function f, crossover rate p_c, mutation rate p_m
  Output: Best program found

  1:  // Initialization (ramped half-and-half)
  2:  for i = 1 to N do
  3:    if i <= N/2 then
  4:      P[i] <- GenerateFullTree(F, T, depth=random(2..D))
  5:    else
  6:      P[i] <- GenerateGrowTree(F, T, max_depth=random(2..D))
  7:    end if
  8:  end for
  9:  Evaluate f(P[i]) for all i
  10:
  11: while generation < max_generations do
  12:   P' <- empty
  13:   while |P'| < N do
  14:     // Tournament selection
  15:     parent1 <- TournamentSelect(P, tournament_size)
  16:     op <- ChooseOperator(p_c, p_m, p_reproduction)
  17:     if op = crossover then
  18:       parent2 <- TournamentSelect(P, tournament_size)
  19:       // Subtree crossover (90% internal, 10% any node)
  20:       Select random subtree in parent1 and parent2
  21:       Swap subtrees to produce child1, child2
  22:       P' <- P' union {child1, child2}
  23:     else if op = mutation then
  24:       // Subtree mutation: replace random subtree with new random tree
  25:       child <- ReplaceRandomSubtree(parent1, GenerateGrowTree(F, T, max_depth))
  26:       P' <- P' union {child}
  27:     else  // reproduction
  28:       P' <- P' union {Copy(parent1)}
  29:     end if
  30:   end while
  31:   Apply depth/size limits (discard oversized, replace with parents)
  32:   Evaluate f(P'[i]) for all i
  33:   P <- P'
  34: end while
  return BestOf(P)
```

GP-specific conventions:
- **Tree operations** (subtree selection, subtree swap) are first-class operations
- **Initialization method** (ramped half-and-half) is always specified
- **Bloat control** (depth/size limits) is an essential component
- **Function set F and terminal set T** are always listed as inputs

---

## Cross-Domain Patterns

### How Hyperparameters Are Listed

Three common approaches, often mixed within a single paper:

| Approach | Example | Used In |
|---|---|---|
| **Input/Require section** | `Require: alpha, beta_1, beta_2, epsilon` | Adam, SGD (most ML papers) |
| **Algorithm parameters line** | `Algorithm parameters: alpha in (0,1], epsilon > 0` | Sutton-Barto (RL) |
| **Separate table** | Table 1: Hyperparameters | Transformer, PPO (complex setups) |
| **Inline with defaults** | `alpha = 0.001 (default)` in comments | Adam paper |

**Best practice:** List hyperparameters in the Input/Require section with their types/ranges. State recommended defaults either inline or in a separate table. Never bury hyperparameter choices in the loop body.

### How Convergence Criteria Are Expressed

| Style | Example | Domain |
|---|---|---|
| **Abstract** | `while stopping criterion not met do` | Adam, Goodfellow textbook |
| **Concrete** | `for iteration = 1 to max_iterations do` | PPO, many practical papers |
| **Episode-based** | `Loop for each episode:` ... `until S is terminal` | RL (Sutton-Barto) |
| **Generation-based** | `while generation < max_generations do` | EA (standard) |
| **Patience-based** | `if no improvement for p evaluations: break` | Training loops with early stopping |

**Best practice:** Use "while not converged" when the convergence criterion varies by application. Use fixed iterations when the number of iterations is a tunable hyperparameter. Always mention early stopping if applicable.

### How Parallel/Distributed Variants Are Presented

Three approaches:

1. **Implicit parallelism:** "Compute F_i for i = 1, ..., n" (OpenAI ES). The `for i = 1 to n` suggests independence.

2. **Explicit workers:** "for actor = 1 to N do ... end for" (PPO). Each worker collects data independently.

3. **Separate algorithm:** Some papers (A3C, Ape-X) present the distributed version as a distinct algorithm with explicit send/receive primitives.

**Best practice for ML papers:** Keep the core algorithm sequential and note "steps X-Y can be parallelized across K workers" in text, rather than complicating the pseudo-code.

### Common Anti-Patterns to Avoid

1. **Language-specific syntax.** Writing `for x in range(N)` or `loss.backward()`. Use mathematical notation.

2. **Excessive detail.** Including data loading, logging, checkpointing, or device management. These are implementation details, not algorithmic steps.

3. **Ambiguous assignment.** Using `=` for both assignment and equality testing. Use `<-` for assignment.

4. **Missing initialization.** Not showing how variables are initialized before the main loop. Always show initial values.

5. **Unlabeled algorithms.** Not giving the algorithm a number and caption. Always use "Algorithm N: Name."

6. **Inconsistent notation.** Switching between bold vectors and non-bold vectors, or between subscript and superscript for indexing.

7. **Giant monolithic algorithms.** If an algorithm exceeds ~20 lines, decompose into sub-procedures. E.g., separate "ComputeAdvantages" from the main PPO loop.

8. **Hiding the key insight.** The novel contribution should be visually prominent (e.g., highlighted line, comment, or separate block). Don't bury the `clip(r_t, 1-eps, 1+eps)` in a wall of standard code.

---

## Code Examples

### Complete LaTeX: Adam Optimizer (algpseudocode)

```latex
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{amsmath}

\begin{algorithm}[t]
\caption{Adam Optimizer}\label{alg:adam}
\begin{algorithmic}[1]
\Require Step size $\alpha = 0.001$
\Require Exponential decay rates $\beta_1 = 0.9$, $\beta_2 = 0.999$
\Require Small constant $\epsilon = 10^{-8}$
\Require Stochastic objective $f(\boldsymbol{\theta})$
\Require Initial parameters $\boldsymbol{\theta}_0$
\State $\mathbf{m}_0 \gets \mathbf{0}$, $\mathbf{v}_0 \gets \mathbf{0}$, $t \gets 0$
\While{$\boldsymbol{\theta}_t$ not converged}
    \State $t \gets t + 1$
    \State $\mathbf{g}_t \gets \nabla_{\boldsymbol{\theta}} f_t(\boldsymbol{\theta}_{t-1})$
    \State $\mathbf{m}_t \gets \beta_1 \cdot \mathbf{m}_{t-1} + (1 - \beta_1) \cdot \mathbf{g}_t$ \Comment{First moment}
    \State $\mathbf{v}_t \gets \beta_2 \cdot \mathbf{v}_{t-1} + (1 - \beta_2) \cdot \mathbf{g}_t^2$ \Comment{Second moment}
    \State $\hat{\mathbf{m}}_t \gets \mathbf{m}_t / (1 - \beta_1^t)$ \Comment{Bias correction}
    \State $\hat{\mathbf{v}}_t \gets \mathbf{v}_t / (1 - \beta_2^t)$
    \State $\boldsymbol{\theta}_t \gets \boldsymbol{\theta}_{t-1} - \alpha \cdot \hat{\mathbf{m}}_t / (\sqrt{\hat{\mathbf{v}}_t} + \epsilon)$
\EndWhile
\State \textbf{return} $\boldsymbol{\theta}_t$
\end{algorithmic}
\end{algorithm}
```

### Complete LaTeX: PPO-Clip (algpseudocode)

```latex
\begin{algorithm}[t]
\caption{PPO-Clip}\label{alg:ppo}
\begin{algorithmic}[1]
\Require Initial policy parameters $\boldsymbol{\theta}_0$, value parameters $\boldsymbol{\phi}_0$
\Require Clip ratio $\epsilon$, GAE parameter $\lambda$, discount $\gamma$
\Require Number of actors $N$, rollout length $T$, epochs $K$
\For{$k = 0, 1, 2, \ldots$}
    \For{actor $= 1, \ldots, N$} \Comment{Collect experience}
        \State Run $\pi_{\boldsymbol{\theta}_k}$ for $T$ steps; store $(s_t, a_t, r_t, s_{t+1})$
    \EndFor
    \State Compute rewards-to-go $\hat{R}_t$ and GAE advantages $\hat{A}_t$
    \For{epoch $= 1, \ldots, K$} \Comment{Policy optimization}
        \For{each minibatch $\mathcal{B} \subset \mathcal{D}_k$}
            \State $r_t(\boldsymbol{\theta}) \gets \frac{\pi_{\boldsymbol{\theta}}(a_t | s_t)}{\pi_{\boldsymbol{\theta}_k}(a_t | s_t)}$
            \State $L^{\text{CLIP}} \gets \frac{1}{|\mathcal{B}|} \sum_t \min\!\Big(r_t \hat{A}_t,\; \text{clip}(r_t, 1\!-\!\epsilon, 1\!+\!\epsilon)\, \hat{A}_t\Big)$
            \State $\boldsymbol{\theta} \gets \boldsymbol{\theta} + \alpha \nabla_{\boldsymbol{\theta}} L^{\text{CLIP}}$ \Comment{Gradient ascent}
        \EndFor
    \EndFor
    \State Fit $V_{\boldsymbol{\phi}}$ via regression: $\boldsymbol{\phi} \gets \arg\min_{\boldsymbol{\phi}} \frac{1}{|\mathcal{D}_k|} \sum_t (V_{\boldsymbol{\phi}}(s_t) - \hat{R}_t)^2$
\EndFor
\end{algorithmic}
\end{algorithm}
```

### Complete LaTeX: Genetic Algorithm (algpseudocode)

```latex
\begin{algorithm}[t]
\caption{Simple Genetic Algorithm}\label{alg:ga}
\begin{algorithmic}[1]
\Require Population size $N$, crossover rate $p_c$, mutation rate $p_m$
\Require Fitness function $f$, chromosome length $L$
\State $P \gets \textsc{InitializeRandom}(N, L)$
\State Evaluate $f(x)$ for all $x \in P$
\While{termination criterion not met}
    \State $P' \gets \emptyset$
    \For{$i = 1$ \textbf{to} $N/2$}
        \State $p_1, p_2 \gets \Call{TournamentSelect}{P, k}$ \Comment{Select parents}
        \If{$\textsc{Random}() < p_c$}
            \State $c_1, c_2 \gets \Call{Crossover}{p_1, p_2}$ \Comment{Recombination}
        \Else
            \State $c_1, c_2 \gets p_1, p_2$
        \EndIf
        \State $c_1 \gets \Call{Mutate}{c_1, p_m}$; \quad $c_2 \gets \Call{Mutate}{c_2, p_m}$
        \State $P' \gets P' \cup \{c_1, c_2\}$
    \EndFor
    \State Evaluate $f(x)$ for all $x \in P'$
    \State $P \gets P'$
\EndWhile
\State \textbf{return} $\arg\max_{x \in P} f(x)$
\end{algorithmic}
\end{algorithm}
```

### Complete LaTeX: Q-Learning (algorithm2e, Sutton-Barto style)

```latex
\usepackage[ruled,lined,linesnumbered]{algorithm2e}

\begin{algorithm}[t]
\caption{Q-learning (off-policy TD control)}\label{alg:qlearn}
\KwIn{Step size $\alpha \in (0, 1]$, exploration rate $\epsilon > 0$, discount $\gamma$}
\KwOut{Learned action-value function $Q \approx q_*$}
Initialize $Q(s, a)$ arbitrarily for all $s \in \mathcal{S}^+, a \in \mathcal{A}(s)$, except $Q(\text{terminal}, \cdot) = 0$\;
\For{each episode}{
    Initialize $S$\;
    \Repeat{$S$ is terminal}{
        Choose $A$ from $S$ using policy derived from $Q$ (e.g., $\epsilon$-greedy)\;
        Take action $A$, observe $R, S'$\;
        $Q(S, A) \gets Q(S, A) + \alpha \big[ R + \gamma \max_{a} Q(S', a) - Q(S, A) \big]$\;
        $S \gets S'$\;
    }
}
\Return{$Q$}
\end{algorithm}
```

---

## Recommended References

### Textbooks

| Title | Authors | Year | Relevance |
|---|---|---|---|
| *Deep Learning* | Goodfellow, Bengio, Courville | 2016 | Gold standard for NN algorithm presentation (Ch. 8 optimization) |
| *Reinforcement Learning: An Introduction* (2nd ed.) | Sutton, Barto | 2018 | Canonical RL pseudo-code style; freely available online |
| *Introduction to Algorithms* (CLRS) | Cormen, Leiserson, Rivest, Stein | 2009 | Origin of `Require:`/`Ensure:` convention used in ML |
| *Introduction to Evolutionary Computing* (2nd ed.) | Eiben, Smith | 2015 | Standard EA textbook with clear pseudo-code throughout |
| *Genetic Programming* (series) | Koza | 1992-2003 | Original GP algorithms and conventions |
| *Dive into Deep Learning* | Zhang et al. | 2023 | Modern, interactive; good for training loop presentation |

### Seminal Papers with Influential Pseudo-code

| Paper | Algorithm | Style Contribution |
|---|---|---|
| Kingma & Ba (2015), arXiv:1412.6980 | Adam | Template for optimizer pseudo-code: parenthesized line comments, explicit initialization |
| Schulman et al. (2017), arXiv:1707.06347 | PPO | Compact actor-environment-optimizer loop; clipped objective |
| Williams (1992) | REINFORCE | Foundational policy gradient presentation |
| Salimans et al. (2017), arXiv:1703.03864 | OpenAI ES | Minimalist evolutionary pseudo-code; parallelism implicit |
| Stanley & Miikkulainen (2002), EC 10(2) | NEAT | Speciation and innovation-number conventions |
| Hansen & Ostermeier (2001) | CMA-ES | Heavy-math evolutionary pseudo-code with matrix operations |

### LaTeX Resources

- [Overleaf: Algorithms](https://www.overleaf.com/learn/latex/Algorithms) -- comprehensive package comparison with examples
- [LaTeX Cloud Studio: Algorithms and Pseudocode](https://resources.latex-cloud-studio.com/learn/latex/specialized-notation/algorithms) -- algpseudocode examples
- [jdhao: How to Write Algorithm Pseudo Code in LaTeX](https://jdhao.github.io/2019/09/21/latex_algorithm_pseudo_code/) -- algorithm2e tutorial
- [TeX FAQ: Typesetting pseudocode](https://texfaq.org/FAQ-algorithms) -- package overview and compatibility
- [LaTeX Wikibooks: Algorithms](https://en.wikibooks.org/wiki/LaTeX/Algorithms) -- syntax reference for all three package families

### Online Algorithm References

- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html) -- clean RL algorithm descriptions
- [Lilian Weng's Blog: Evolution Strategies](https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/) -- comprehensive ES survey with algorithms
- [Clever Algorithms](https://cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html) -- EA pseudo-code reference
- [The ICLR Blog Track: 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) -- bridging pseudo-code to implementation

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|---|---|---|---|
| `algorithmic` package (UPPERCASE) | `algpseudocode` (Title-case) | ~2010 onwards | More readable, better customization |
| No line numbers | Always line numbers (`[1]`) | Standard practice | Enables paper text to reference "line 5" |
| Programming-style pseudo-code | Math-heavy pseudo-code | ~2015 (with deep learning boom) | Gradient updates, loss functions expressed as equations |
| Single monolithic algorithms | Decomposed sub-procedures | ~2017 (PPO era) | Complex algorithms split into rollout + optimize phases |
| Fixed iteration counts | Abstract convergence | Varies | "while not converged" allows flexible stopping |

### Current Trends (2024-2026)

- **RLHF/RLAIF algorithms** (DPO, GRPO, CISPO) follow the PPO template but simplify the environment interaction phase since the "environment" is a language model
- **Diffusion model** papers present the denoising loop as an algorithm separate from training
- **AutoML/NAS** papers use evolutionary pseudo-code for architecture search
- **Foundation model training** papers often omit pseudo-code entirely, favoring diagrams + hyperparameter tables

---

## Open Questions

1. **No universal standard exists.** Pseudo-code conventions vary by venue, textbook lineage, and subfield. There is no RFC or formal specification -- only community norms.
   - What we know: CLRS style dominates ML; Sutton-Barto style dominates RL; EA has its own conventions
   - Recommendation: Match your venue's conventions. When in doubt, use algpseudocode with CLRS-style notation.

2. **How to present hybrid algorithms** (e.g., RL + evolutionary, like PBT or ERL) remains ad hoc.
   - What we know: Papers typically present the outer loop (evolutionary) and inner loop (RL) as separate algorithms
   - Recommendation: Use Algorithm 1 for the outer loop referencing Algorithm 2 for the inner loop via `\Call{}`.

---

## Sources

### Primary (HIGH confidence)
- [Overleaf: Algorithms](https://www.overleaf.com/learn/latex/Algorithms) -- LaTeX package syntax and examples
- [LaTeX Cloud Studio: Algorithms and Pseudocode](https://resources.latex-cloud-studio.com/learn/latex/specialized-notation/algorithms) -- algpseudocode complete reference
- [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) -- PPO algorithm description
- [Kingma & Ba (2015), arXiv:1412.6980](https://arxiv.org/abs/1412.6980) -- Adam optimizer original paper
- [Schulman et al. (2017), arXiv:1707.06347](https://arxiv.org/abs/1707.06347) -- PPO original paper
- [Salimans et al. (2017), arXiv:1703.03864](https://arxiv.org/abs/1703.03864) -- OpenAI ES paper
- [Stanley & Miikkulainen (2002)](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) -- NEAT original paper
- Sutton & Barto (2018), *Reinforcement Learning: An Introduction* 2nd ed. -- RL algorithm conventions

### Secondary (MEDIUM confidence)
- [Lilian Weng: Evolution Strategies](https://lilianweng.github.io/posts/2019-09-05-evolution-strategies/) -- ES survey with algorithms
- [jdhao: Algorithm Pseudo Code in LaTeX](https://jdhao.github.io/2019/09/21/latex_algorithm_pseudo_code/) -- algorithm2e tutorial
- [Sebastian Raschka: SGD Methods](https://sebastianraschka.com/faq/docs/sgd-methods.html) -- SGD variants comparison
- [Clever Algorithms: Evolution Strategies](https://cleveralgorithms.com/nature-inspired/evolution/evolution_strategies.html) -- EA pseudo-code
- [ICLR Blog Track: PPO Implementation Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) -- PPO implementation
- [Actor-Critic Methods: A2C](https://avandekleut.github.io/a2c/) -- A2C algorithm walkthrough
- [GitHub: Pseudo-code guidelines](https://gist.github.com/camilstaps/59c4574ab8131fb83612a446606cbcba) -- general pseudo-code conventions

### Tertiary (LOW confidence)
- Various Medium articles on PPO, A2C, genetic algorithms -- used for cross-verification only

---

## Metadata

**Confidence breakdown:**
- LaTeX packages and syntax: HIGH -- verified against official documentation
- Neural network algorithm conventions: HIGH -- based on seminal papers and authoritative textbooks
- RL algorithm conventions: HIGH -- based on Sutton-Barto and original papers
- EA algorithm conventions: HIGH -- based on original papers and established surveys
- Cross-domain patterns: MEDIUM -- synthesized from multiple sources, some patterns are subjective

**Research date:** 2026-03-14
**Valid until:** Indefinite -- pseudo-code conventions are stable and change slowly. LaTeX package syntax is stable. Algorithm presentations in seminal papers are permanently fixed.
