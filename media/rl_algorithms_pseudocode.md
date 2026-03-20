# RL Algorithms Pseudocode

---

## Algorithm 1: PPO-Clip — Practical Vectorized Implementation

**Ensure:** Optimized policy $\pi_{\theta}$

**Hyperparameters:**
- $\theta_0$ — init policy params
- $\phi_0$ — init value params
- $\epsilon$ — clip ratio (typ. 0.2)
- $\gamma$ — discount factor
- $\lambda$ — GAE param
- $N$ — parallel environments
- $T$ — rollout length
- $K$ — epochs
- $M$ — minibatch size
- $\alpha_\pi$ — policy learning rate
- $\alpha_V$ — value learning rate
- $c_e$ — entropy coeff.

**Tensors:**
- $\mathbf{S}, \mathbf{A}, \mathbf{R} \in \mathbb{R}^{N \times T}$
- $\mathbf{D} \in \{0,1\}^{N \times T}$ — done flags
- $\boldsymbol{\pi}_{\text{old}} \in \mathbb{R}^{N \times T}$
- $\mathbf{V} \in \mathbb{R}^{N \times (T+1)}$
- $\boldsymbol{\delta} \in \mathbb{R}^{N \times T}$
- $\hat{\mathbf{A}}, \hat{\mathbf{R}} \in \mathbb{R}^{N \times T}$
- $\boldsymbol{\rho} \in \mathbb{R}^{M}$

---

1. **For** $k = 0, 1, 2, \ldots$
2. &emsp;*// Phase 1: Vectorized rollout — $\mathcal{O}(T \cdot N \cdot C_\pi)$*
3. &emsp;**For** $t = 0, \ldots, T{-}1$
4. &emsp;&emsp;$\mathbf{A}_{:,t} \sim \pi_{\theta_k}(\cdot \mid \mathbf{S}_{:,t})$ — $\mathcal{O}(N \cdot C_\pi)$
5. &emsp;&emsp;$\mathbf{R}_{:,t}, \mathbf{S}_{:,t+1}, \mathbf{D}_{:,t} \gets \texttt{envs.step}(\mathbf{A}_{:,t})$ — $\mathcal{O}(N)$
6. &emsp;&emsp;$\boldsymbol{\pi}_{\text{old}_{:,t}} \gets \pi_{\theta_k}(\mathbf{A}_{:,t} \mid \mathbf{S}_{:,t})$
7. &emsp;**End For**
8. &emsp;*// Phase 2: Vectorized GAE — $\mathcal{O}(NT + N \cdot C_V)$*
9. &emsp;$\mathbf{V} \gets V_{\phi_k}([\mathbf{S}, \mathbf{S}_{:,T}])$ — $\mathcal{O}(N \cdot C_V)$; shape $(N, T{+}1)$
10. &emsp;$\boldsymbol{\delta} \gets \mathbf{R} + \gamma(1 - \mathbf{D}) \odot \mathbf{V}_{:,1:} - \mathbf{V}_{:,:-1}$ — $\mathcal{O}(NT)$
11. &emsp;$\hat{\mathbf{A}}_{:,T-1} \gets \boldsymbol{\delta}_{:,T-1}$
12. &emsp;**For** $t = T{-}2, \ldots, 0$
13. &emsp;&emsp;$\hat{\mathbf{A}}_{:,t} \gets \boldsymbol{\delta}_{:,t} + \gamma\lambda(1 - \mathbf{D}_{:,t}) \odot \hat{\mathbf{A}}_{:,t+1}$ — $\mathcal{O}(N)$
14. &emsp;**End For**
15. &emsp;$\hat{\mathbf{R}} \gets \hat{\mathbf{A}} + \mathbf{V}_{:,:-1}$ — $\mathcal{O}(NT)$
16. &emsp;*// Phase 3: Minibatch SGD — $\mathcal{O}(K \cdot NT \cdot (C_\pi + C_V))$*
17. &emsp;Flatten $\mathcal{D}_k \gets$ reshape all tensors to $(N \cdot T, \ldots)$
18. &emsp;**For** epoch $= 1, \ldots, K$
19. &emsp;&emsp;$\mathcal{I} \gets \texttt{randperm}(N \cdot T)$
20. &emsp;&emsp;**For** $b = 0, M, 2M, \ldots$ — $\lceil NT/M \rceil$ minibatches
21. &emsp;&emsp;&emsp;$\mathcal{B} \gets \mathcal{I}[b : b{+}M]$
22. &emsp;&emsp;&emsp;$\hat{\mathbf{A}}_\mathcal{B} \gets (\hat{\mathbf{A}}_\mathcal{B} - \texttt{mean}(\hat{\mathbf{A}}_\mathcal{B})) / (\texttt{std}(\hat{\mathbf{A}}_\mathcal{B}) + \varepsilon)$ — $\mathcal{O}(M)$
23. &emsp;&emsp;&emsp;$\boldsymbol{\rho} \gets \pi_\theta(\mathbf{A}_\mathcal{B} \mid \mathbf{S}_\mathcal{B}) / \boldsymbol{\pi}_{\text{old}_\mathcal{B}}$ — $\mathcal{O}(M \cdot C_\pi)$
24. &emsp;&emsp;&emsp;$L^{\text{CLIP}} \gets \texttt{mean}(\min(\boldsymbol{\rho} \odot \hat{\mathbf{A}}_\mathcal{B},\; \texttt{clip}(\boldsymbol{\rho}, 1{-}\epsilon, 1{+}\epsilon) \odot \hat{\mathbf{A}}_\mathcal{B}))$
25. &emsp;&emsp;&emsp;$L^V \gets \texttt{mean}((V_\phi(\mathbf{S}_\mathcal{B}) - \hat{\mathbf{R}}_\mathcal{B})^2)$ — $\mathcal{O}(M \cdot C_V)$
26. &emsp;&emsp;&emsp;$H \gets \texttt{mean}(\texttt{entropy}(\pi_\theta(\cdot \mid \mathbf{S}_\mathcal{B})))$
27. &emsp;&emsp;&emsp;$\theta \gets \theta + \alpha_\pi \nabla_\theta (L^{\text{CLIP}} + c_e H)$ — $\mathcal{O}(M \cdot C_\pi)$
28. &emsp;&emsp;&emsp;$\phi \gets \phi - \alpha_V \nabla_\phi L^V$ — $\mathcal{O}(M \cdot C_V)$
29. &emsp;&emsp;**End For**
30. &emsp;**End For**
31. **End For**

**Per-iteration complexity:** $\mathcal{O}(T \cdot N \cdot C_\pi + K \cdot N \cdot T \cdot (C_\pi + C_V))$

$C_\pi$: policy forward/backward cost, $C_V$: value forward/backward cost.
Phase 3 dominates ($K$ epochs $\times$ full buffer); Phase 2 is $\mathcal{O}(NT)$ arithmetic.

---

## Algorithm 2: GRPO — Practical Vectorized Implementation

**Ensure:** Optimized policy $\pi_\theta$

**Hyperparameters:**
- $\theta_0$ — init policy params
- $\pi_{\text{ref}}$ — frozen ref. policy
- $\epsilon$ — clip ratio (typ. 0.2)
- $G$ — group size
- $\beta$ — KL coefficient
- $\alpha$ — learning rate
- $K$ — epochs
- $M$ — minibatch size
- $R(\mathbf{q}, \mathbf{o})$ — reward function

**Tensors:**
- $B$ — queries per batch
- $L$ — max completion length
- $\mathbf{Q} \in \mathbb{Z}^{B \times L_q}$ — queries
- $\mathbf{O} \in \mathbb{Z}^{B \times G \times L}$ — tokens
- $\boldsymbol{\pi}_{\text{old}} \in \mathbb{R}^{B \times G \times L}$
- $\mathbf{M} \in \{0,1\}^{B \times G \times L}$ — mask
- $\mathbf{R} \in \mathbb{R}^{B \times G}$ — rewards
- $\hat{\mathbf{A}} \in \mathbb{R}^{B \times G}$ — advantages
- $\boldsymbol{\rho} \in \mathbb{R}^{M \times L}$

---

1. **For** $k = 0, 1, 2, \ldots$
2. &emsp;*// Phase 1: Batch generation — $\mathcal{O}(B \cdot G \cdot L \cdot C_\pi)$*
3. &emsp;**For** each query batch $\mathbf{Q}_{1:B}$
4. &emsp;&emsp;$\mathbf{O}_{:,:,:} \sim \pi_{\theta_k}(\cdot \mid \mathbf{Q})$ — $B \cdot G$ completions in parallel
5. &emsp;&emsp;$\mathbf{R}_{:,:} \gets R(\mathbf{Q}, \mathbf{O})$ — batch reward evaluation
6. &emsp;&emsp;$\boldsymbol{\pi}_{\text{old}} \gets \pi_{\theta_k}(\mathbf{O} \mid \mathbf{Q})$ — $\mathcal{O}(BGL \cdot C_\pi)$
7. &emsp;&emsp;$\mathbf{M} \gets \texttt{token\_mask}(\mathbf{O})$ — 1 where valid, 0 for padding
8. &emsp;**End For**
9. &emsp;*// Phase 2: Group-relative advantage — $\mathcal{O}(BG)$*
10. &emsp;$\boldsymbol{\mu} \gets \texttt{mean}(\mathbf{R}, \text{dim}{=}G)$ — $\in \mathbb{R}^{B}$
11. &emsp;$\boldsymbol{\sigma} \gets \texttt{std}(\mathbf{R}, \text{dim}{=}G)$ — $\in \mathbb{R}^{B}$
12. &emsp;$\hat{\mathbf{A}} \gets (\mathbf{R} - \boldsymbol{\mu}) / \max(\boldsymbol{\sigma}, \varepsilon_{\min})$ — broadcast over $G$
13. &emsp;*// Phase 3: Minibatch SGD — $\mathcal{O}(K \cdot BG \cdot L \cdot C_\pi)$*
14. &emsp;Flatten $\mathcal{D}_k \gets$ reshape to $(B \cdot G, L)$ completions
15. &emsp;**For** epoch $= 1, \ldots, K$
16. &emsp;&emsp;$\mathcal{I} \gets \texttt{randperm}(B \cdot G)$ — shuffle completions
17. &emsp;&emsp;**For** $b = 0, M, 2M, \ldots$ — $\lceil BG/M \rceil$ minibatches
18. &emsp;&emsp;&emsp;$\mathcal{B} \gets \mathcal{I}[b : b{+}M]$
19. &emsp;&emsp;&emsp;$\boldsymbol{\rho} \gets \pi_\theta(\mathbf{O}_\mathcal{B} \mid \mathbf{Q}_\mathcal{B}) / \boldsymbol{\pi}_{\text{old}_\mathcal{B}}$ — $\mathcal{O}(ML \cdot C_\pi)$
20. &emsp;&emsp;&emsp;$\mathbf{L}_{\text{tok}} \gets \min(\boldsymbol{\rho} \odot \hat{\mathbf{A}}_\mathcal{B},\; \texttt{clip}(\boldsymbol{\rho}, 1{-}\epsilon, 1{+}\epsilon) \odot \hat{\mathbf{A}}_\mathcal{B})$
21. &emsp;&emsp;&emsp;$L^{\text{GRPO}} \gets \texttt{sum}(\mathbf{L}_{\text{tok}} \odot \mathbf{M}_\mathcal{B}) / \texttt{sum}(\mathbf{M}_\mathcal{B})$ — masked mean
22. &emsp;&emsp;&emsp;$\mathbf{r}_{\text{ref}} \gets \pi_\theta(\mathbf{O}_\mathcal{B} \mid \cdot) / \pi_{\text{ref}}(\mathbf{O}_\mathcal{B} \mid \cdot)$
23. &emsp;&emsp;&emsp;$D_{\text{KL}} \gets \texttt{mean}((\mathbf{r}_{\text{ref}} - \log \mathbf{r}_{\text{ref}} - 1) \odot \mathbf{M}_\mathcal{B})$
24. &emsp;&emsp;&emsp;$\theta \gets \theta + \alpha \nabla_\theta (L^{\text{GRPO}} - \beta D_{\text{KL}})$ — $\mathcal{O}(ML \cdot C_\pi)$
25. &emsp;&emsp;**End For**
26. &emsp;**End For**
27. **End For**

**Per-iteration complexity:** $\mathcal{O}(B \cdot G \cdot L \cdot C_\pi + K \cdot B \cdot G \cdot L \cdot C_\pi) = \mathcal{O}((1{+}K) \cdot BGL \cdot C_\pi)$

$C_\pi$: policy forward/backward cost. No $C_V$ term (critic-free).
Phase 2 is $\mathcal{O}(BG)$ arithmetic — negligible vs. neural forward passes.

---

## Algorithm 3: CISPO — Practical Vectorized Implementation

**Ensure:** Optimized policy $\pi_\theta$

**Hyperparameters:**
- $\theta_0$ — init policy params
- $\epsilon_{\text{high}}$ — upper clip (typ. 0.3)
- $G$ — group size
- $\alpha$ — learning rate
- $K$ — epochs
- $M$ — minibatch size
- $R(\mathbf{q}, \mathbf{o})$ — reward function

**Tensors:**
- $B$ — queries per batch
- $L$ — max completion length
- $\mathbf{Q} \in \mathbb{Z}^{B \times L_q}$ — queries
- $\mathbf{O} \in \mathbb{Z}^{B \times G \times L}$ — tokens
- $\boldsymbol{\pi}_{\text{old}} \in \mathbb{R}^{B \times G \times L}$
- $\mathbf{M} \in \{0,1\}^{B \times G \times L}$ — mask
- $\mathbf{R} \in \mathbb{R}^{B \times G}$ — rewards
- $\hat{\mathbf{A}} \in \mathbb{R}^{B \times G}$ — advantages
- $\boldsymbol{\rho}, \hat{\boldsymbol{\rho}}, \bar{\boldsymbol{\rho}} \in \mathbb{R}^{M \times L}$

---

1. **For** $k = 0, 1, 2, \ldots$
2. &emsp;*// Phase 1: Batch generation (same as GRPO) — $\mathcal{O}(BGL \cdot C_\pi)$*
3. &emsp;**For** each query batch $\mathbf{Q}_{1:B}$
4. &emsp;&emsp;$\mathbf{O}_{:,:,:} \sim \pi_{\theta_k}(\cdot \mid \mathbf{Q})$ — $B \cdot G$ completions in parallel
5. &emsp;&emsp;$\mathbf{R}_{:,:} \gets R(\mathbf{Q}, \mathbf{O})$ — batch reward evaluation
6. &emsp;&emsp;$\boldsymbol{\pi}_{\text{old}} \gets \pi_{\theta_k}(\mathbf{O} \mid \mathbf{Q})$ — $\mathcal{O}(BGL \cdot C_\pi)$
7. &emsp;&emsp;$\mathbf{M} \gets \texttt{token\_mask}(\mathbf{O})$ — 1 where valid, 0 for padding
8. &emsp;**End For**
9. &emsp;*// Phase 2: Group-relative advantage (same as GRPO) — $\mathcal{O}(BG)$*
10. &emsp;$\boldsymbol{\mu} \gets \texttt{mean}(\mathbf{R}, \text{dim}{=}G)$ — $\in \mathbb{R}^{B}$
11. &emsp;$\boldsymbol{\sigma} \gets \texttt{std}(\mathbf{R}, \text{dim}{=}G)$ — $\in \mathbb{R}^{B}$
12. &emsp;$\hat{\mathbf{A}} \gets (\mathbf{R} - \boldsymbol{\mu}) / \max(\boldsymbol{\sigma}, \varepsilon_{\min})$ — broadcast over $G$
13. &emsp;*// Phase 3: Clipped IS weight SGD — $\mathcal{O}(K \cdot BG \cdot L \cdot C_\pi)$*
14. &emsp;Flatten $\mathcal{D}_k \gets$ reshape to $(B \cdot G, L)$ completions
15. &emsp;**For** epoch $= 1, \ldots, K$
16. &emsp;&emsp;$\mathcal{I} \gets \texttt{randperm}(B \cdot G)$
17. &emsp;&emsp;**For** $b = 0, M, 2M, \ldots$ — $\lceil BG/M \rceil$ minibatches
18. &emsp;&emsp;&emsp;$\mathcal{B} \gets \mathcal{I}[b : b{+}M]$
19. &emsp;&emsp;&emsp;$\boldsymbol{\rho} \gets \pi_\theta(\mathbf{O}_\mathcal{B} \mid \mathbf{Q}_\mathcal{B}) / \boldsymbol{\pi}_{\text{old}_\mathcal{B}}$ — $\mathcal{O}(ML \cdot C_\pi)$
20. &emsp;&emsp;&emsp;$\hat{\boldsymbol{\rho}} \gets \texttt{clamp}(\boldsymbol{\rho}, \max = 1 + \epsilon_{\text{high}})$ — $\mathcal{O}(ML)$
21. &emsp;&emsp;&emsp;$\bar{\boldsymbol{\rho}} \gets \texttt{stop\_grad}(\hat{\boldsymbol{\rho}})$ — detach from graph
22. &emsp;&emsp;&emsp;$\mathbf{L}_{\text{tok}} \gets \bar{\boldsymbol{\rho}} \odot \hat{\mathbf{A}}_\mathcal{B} \odot \log \pi_\theta(\mathbf{O}_\mathcal{B} \mid \mathbf{Q}_\mathcal{B})$
23. &emsp;&emsp;&emsp;$L^{\text{CISPO}} \gets \texttt{sum}(\mathbf{L}_{\text{tok}} \odot \mathbf{M}_\mathcal{B}) / \texttt{sum}(\mathbf{M}_\mathcal{B})$ — masked mean
24. &emsp;&emsp;&emsp;$\theta \gets \theta + \alpha \nabla_\theta L^{\text{CISPO}}$ — $\mathcal{O}(ML \cdot C_\pi)$
25. &emsp;&emsp;**End For**
26. &emsp;**End For**
27. **End For**

**Per-iteration complexity:** $\mathcal{O}((1{+}K) \cdot BGL \cdot C_\pi)$ — same as GRPO

No $\pi_{\text{ref}}$ forward pass needed (no KL term). $\texttt{stop\_grad}$ is $\mathcal{O}(1)$.
All tokens receive gradient (vs. GRPO where clipped tokens are masked out).

---

## Algorithm 4: DDPG — Deep Deterministic Policy Gradient

**Ensure:** Optimized deterministic policy $\mu_\theta$

**Hyperparameters:**
- $\theta_0$ — init actor params
- $\phi_0$ — init critic params
- $\gamma$ — discount factor
- $\tau$ — soft update rate (typ. 0.005)
- $\alpha_\mu$ — actor learning rate
- $\alpha_Q$ — critic learning rate
- $B$ — batch size
- $W$ — warmup steps (random actions)
- $|\mathcal{D}|$ — replay buffer capacity
- OU noise: $\theta_{\text{OU}}, \sigma_{\text{OU}}$

**Tensors:**
- $\mathbf{s}, \mathbf{s}' \in \mathbb{R}^{B \times d_s}$ — states
- $\mathbf{a} \in \mathbb{R}^{B \times d_a}$ — actions
- $\mathbf{r} \in \mathbb{R}^{B}$ — rewards
- $\mathbf{d} \in \{0,1\}^{B}$ — done flags
- $\mathbf{y} \in \mathbb{R}^{B}$ — Bellman targets

---

1. **Initialize** $\theta' \gets \theta,\; \phi' \gets \phi$ — copy to targets
2. **Initialize** $\mathcal{D} \gets \emptyset$ — empty replay buffer
3. **Initialize** OU noise state $\mathbf{n} \gets \mathbf{0}$
4. **For** $t = 0, 1, 2, \ldots$
5. &emsp;*// Phase 1: Environment interaction — $\mathcal{O}(C_\mu)$*
6. &emsp;**If** $t < W$: $\mathbf{a}_t \sim \mathcal{U}[-1, 1]$ — random warmup
7. &emsp;**Else**: $\mathbf{a}_t \gets \tanh(\mu_\theta(\mathbf{s}_t)) + \mathbf{n}_t$ — $\mathcal{O}(C_\mu)$; OU noise
8. &emsp;$\mathbf{r}_t, \mathbf{s}_{t+1}, \mathbf{d}_t \gets \texttt{env.step}(\mathbf{a}_t)$
9. &emsp;$\mathcal{D} \gets \mathcal{D} \cup \{(\mathbf{s}_t, \mathbf{a}_t, \mathbf{r}_t, \mathbf{s}_{t+1}, \mathbf{d}_t)\}$
10. &emsp;**If** $t < W$: **continue**
11. &emsp;*// Phase 2: Critic update — $\mathcal{O}(B \cdot (C_\mu + C_Q))$*
12. &emsp;$\{(\mathbf{s}, \mathbf{a}, \mathbf{r}, \mathbf{s}', \mathbf{d})\} \sim \mathcal{D}$ — sample minibatch of $B$
13. &emsp;$\mathbf{y} \gets \mathbf{r} + \gamma(1 - \mathbf{d}) \odot Q_{\phi'}(\mathbf{s}', \mu_{\theta'}(\mathbf{s}'))$ — $\mathcal{O}(B \cdot (C_\mu + C_Q))$
14. &emsp;$L^Q \gets \texttt{mean}((Q_\phi(\mathbf{s}, \mathbf{a}) - \mathbf{y})^2)$ — $\mathcal{O}(B \cdot C_Q)$
15. &emsp;$\phi \gets \phi - \alpha_Q \nabla_\phi L^Q$
16. &emsp;*// Phase 3: Actor update — $\mathcal{O}(B \cdot (C_\mu + C_Q))$*
17. &emsp;$L^\mu \gets -\texttt{mean}(Q_\phi(\mathbf{s}, \mu_\theta(\mathbf{s})))$ — $\mathcal{O}(B \cdot (C_\mu + C_Q))$
18. &emsp;$\theta \gets \theta - \alpha_\mu \nabla_\theta L^\mu$
19. &emsp;*// Phase 4: Target soft update — $\mathcal{O}(|\theta| + |\phi|)$*
20. &emsp;$\theta' \gets \tau\theta + (1 - \tau)\theta'$
21. &emsp;$\phi' \gets \tau\phi + (1 - \tau)\phi'$
22. **End For**

**Per-step complexity:** $\mathcal{O}(B \cdot (C_\mu + C_Q))$

$C_\mu$: actor forward/backward cost, $C_Q$: critic forward/backward cost.
Deterministic policy — no log-prob computation. OU noise is $\mathcal{O}(d_a)$.

---

## Algorithm 5: SAC — Soft Actor-Critic with Automatic Entropy Tuning

**Ensure:** Optimized stochastic policy $\pi_\theta$

**Hyperparameters:**
- $\theta_0$ — init actor params
- $\phi_{1,0}, \phi_{2,0}$ — init twin critic params
- $\gamma$ — discount factor
- $\tau$ — soft update rate (typ. 0.005)
- $\alpha_\pi$ — actor learning rate
- $\alpha_Q$ — critic learning rate
- $\alpha_\alpha$ — entropy coeff. learning rate
- $B$ — batch size
- $W$ — warmup steps
- $|\mathcal{D}|$ — replay buffer capacity
- $\bar{\mathcal{H}} = -d_a$ — target entropy
- $f_\mu$ — actor update frequency (typ. 2)

**Tensors:**
- $\mathbf{s}, \mathbf{s}' \in \mathbb{R}^{B \times d_s}$
- $\mathbf{a}, \tilde{\mathbf{a}} \in \mathbb{R}^{B \times d_a}$ — buffer / fresh actions
- $\mathbf{r} \in \mathbb{R}^{B}$, $\mathbf{d} \in \{0,1\}^{B}$
- $\mathbf{y} \in \mathbb{R}^{B}$ — Bellman targets

---

1. **Initialize** $\phi'_1 \gets \phi_1,\; \phi'_2 \gets \phi_2$ — copy to targets
2. **Initialize** $\log\alpha \gets \log(\alpha_0)$ — learnable log-entropy coeff.
3. **Initialize** $\mathcal{D} \gets \emptyset$
4. **For** $t = 0, 1, 2, \ldots$
5. &emsp;*// Phase 1: Environment interaction — $\mathcal{O}(C_\pi)$*
6. &emsp;**If** $t < W$: $\mathbf{a}_t \sim \mathcal{U}[-1, 1]$ — random warmup
7. &emsp;**Else**: $\mathbf{a}_t \sim \pi_\theta(\cdot \mid \mathbf{s}_t)$ — TanhNormal; $\mathcal{O}(C_\pi)$
8. &emsp;$\mathbf{r}_t, \mathbf{s}_{t+1}, \mathbf{d}_t \gets \texttt{env.step}(\mathbf{a}_t)$
9. &emsp;$\mathcal{D} \gets \mathcal{D} \cup \{(\mathbf{s}_t, \mathbf{a}_t, \mathbf{r}_t, \mathbf{s}_{t+1}, \mathbf{d}_t)\}$
10. &emsp;**If** $t < W$: **continue**
11. &emsp;*// Phase 2: Twin critic update — $\mathcal{O}(B \cdot (C_\pi + 2C_Q))$*
12. &emsp;$\{(\mathbf{s}, \mathbf{a}, \mathbf{r}, \mathbf{s}', \mathbf{d})\} \sim \mathcal{D}$
13. &emsp;$\tilde{\mathbf{a}}', \log\pi' \gets \pi_\theta(\cdot \mid \mathbf{s}')$ — sample next actions; $\mathcal{O}(B \cdot C_\pi)$
14. &emsp;$\mathbf{y} \gets \mathbf{r} + \gamma(1 - \mathbf{d}) \odot [\min(Q_{\phi'_1}, Q_{\phi'_2})(\mathbf{s}', \tilde{\mathbf{a}}') - \alpha \cdot \log\pi']$
15. &emsp;$L^Q \gets \texttt{mean}((Q_{\phi_1}(\mathbf{s}, \mathbf{a}) - \mathbf{y})^2) + \texttt{mean}((Q_{\phi_2}(\mathbf{s}, \mathbf{a}) - \mathbf{y})^2)$
16. &emsp;$\phi_1, \phi_2 \gets \phi_1, \phi_2 - \alpha_Q \nabla_{\phi_1, \phi_2} L^Q$ — $\mathcal{O}(B \cdot 2C_Q)$
17. &emsp;*// Phase 3: Actor update (every $f_\mu$ critic steps) — $\mathcal{O}(B \cdot (C_\pi + C_Q))$*
18. &emsp;**If** $t \bmod f_\mu = 0$:
19. &emsp;&emsp;$\tilde{\mathbf{a}}, \log\pi \gets \pi_\theta(\cdot \mid \mathbf{s})$ — rsample; $\mathcal{O}(B \cdot C_\pi)$
20. &emsp;&emsp;$L^\pi \gets \texttt{mean}(\alpha \cdot \log\pi - \min(Q_{\phi_1}, Q_{\phi_2})(\mathbf{s}, \tilde{\mathbf{a}}))$
21. &emsp;&emsp;$\theta \gets \theta - \alpha_\pi \nabla_\theta L^\pi$
22. &emsp;&emsp;*// Phase 4: Entropy coefficient update — $\mathcal{O}(B)$*
23. &emsp;&emsp;$L^\alpha \gets -\texttt{mean}(\log\alpha \cdot (\log\pi + \bar{\mathcal{H}}))$ — $\mathcal{O}(B)$
24. &emsp;&emsp;$\log\alpha \gets \log\alpha - \alpha_\alpha \nabla_{\log\alpha} L^\alpha$
25. &emsp;&emsp;$\alpha \gets \exp(\log\alpha)$
26. &emsp;*// Phase 5: Target soft update — $\mathcal{O}(|\phi|)$*
27. &emsp;$\phi'_i \gets \tau\phi_i + (1 - \tau)\phi'_i$ &emsp; for $i = 1, 2$
28. **End For**

**Per-step complexity:** $\mathcal{O}(B \cdot (C_\pi + 2C_Q))$

$C_\pi$: stochastic actor cost (TanhNormal rsample + log-prob), $C_Q$: single Q-network cost.
Twin critics double $C_Q$; actor + entropy updated every $f_\mu$ steps amortizes $C_\pi$.

---

## Algorithm 6: OTPG — Operator-Theoretic Policy Gradient (MM-RKHS)

**Ensure:** Optimized policy $\pi_\theta$

**Hyperparameters:**
- $\theta_0$ — init policy params
- $\phi_0$ — init value params
- $\beta$ — majorization bound coeff. (typ. 1.0)
- $\eta$ — mirror descent step size (typ. 1.0)
- $\sigma$ — RBF kernel bandwidth (typ. 1.0)
- $P$ — MMD samples per state (typ. 16)
- $\gamma$ — discount factor
- $\lambda$ — GAE param
- $N$ — parallel environments
- $T$ — rollout length
- $K$ — epochs
- $M$ — minibatch size
- $\alpha$ — learning rate
- $c_V$ — value coeff. (typ. 0.5)

**Tensors:**
- $\mathbf{S}, \mathbf{A}, \mathbf{R} \in \mathbb{R}^{N \times T}$
- $\mathbf{D} \in \{0,1\}^{N \times T}$ — done flags
- $\boldsymbol{\pi}_{\text{old}} \in \mathbb{R}^{N \times T}$ — old log-probs
- $\boldsymbol{\mu}_{\text{old}}, \boldsymbol{\sigma}_{\text{old}} \in \mathbb{R}^{N \times T \times d_a}$ — old distribution params
- $\mathbf{V} \in \mathbb{R}^{N \times (T+1)}$
- $\hat{\mathbf{A}}, \hat{\mathbf{R}} \in \mathbb{R}^{N \times T}$
- $\boldsymbol{\rho} \in \mathbb{R}^{M}$

---

1. **For** $k = 0, 1, 2, \ldots$
2. &emsp;*// Phase 1: Vectorized rollout (same as PPO) — $\mathcal{O}(TN \cdot C_\pi)$*
3. &emsp;**For** $t = 0, \ldots, T{-}1$
4. &emsp;&emsp;$\mathbf{A}_{:,t} \sim \pi_{\theta_k}(\cdot \mid \mathbf{S}_{:,t})$ — TanhNormal; $\mathcal{O}(N \cdot C_\pi)$
5. &emsp;&emsp;$\mathbf{R}_{:,t}, \mathbf{S}_{:,t+1}, \mathbf{D}_{:,t} \gets \texttt{envs.step}(\mathbf{A}_{:,t})$
6. &emsp;&emsp;Store $\boldsymbol{\pi}_{\text{old}_{:,t}},\; \boldsymbol{\mu}_{\text{old}_{:,t}},\; \boldsymbol{\sigma}_{\text{old}_{:,t}}$ — log-prob + distribution params
7. &emsp;**End For**
8. &emsp;*// Phase 2: Vectorized GAE (same as PPO) — $\mathcal{O}(NT + N \cdot C_V)$*
9. &emsp;$\mathbf{V} \gets V_{\phi_k}([\mathbf{S}, \mathbf{S}_{:,T}])$
10. &emsp;$\boldsymbol{\delta} \gets \mathbf{R} + \gamma(1 - \mathbf{D}) \odot \mathbf{V}_{:,1:} - \mathbf{V}_{:,:-1}$
11. &emsp;$\hat{\mathbf{A}}_{:,T-1} \gets \boldsymbol{\delta}_{:,T-1}$
12. &emsp;**For** $t = T{-}2, \ldots, 0$
13. &emsp;&emsp;$\hat{\mathbf{A}}_{:,t} \gets \boldsymbol{\delta}_{:,t} + \gamma\lambda(1 - \mathbf{D}_{:,t}) \odot \hat{\mathbf{A}}_{:,t+1}$
14. &emsp;**End For**
15. &emsp;$\hat{\mathbf{R}} \gets \hat{\mathbf{A}} + \mathbf{V}_{:,:-1}$
16. &emsp;*// Phase 3: MM-RKHS policy update — $\mathcal{O}(K \cdot NT \cdot (P \cdot C_\pi + C_V))$*
17. &emsp;Flatten $\mathcal{D}_k \gets$ reshape to $(N \cdot T, \ldots)$
18. &emsp;**For** epoch $= 1, \ldots, K$
19. &emsp;&emsp;$\mathcal{I} \gets \texttt{randperm}(N \cdot T)$
20. &emsp;&emsp;**For** $b = 0, M, 2M, \ldots$ — $\lceil NT/M \rceil$ minibatches
21. &emsp;&emsp;&emsp;$\mathcal{B} \gets \mathcal{I}[b : b{+}M]$
22. &emsp;&emsp;&emsp;$\hat{\mathbf{A}}_\mathcal{B} \gets (\hat{\mathbf{A}}_\mathcal{B} - \texttt{mean}) / (\texttt{std} + \varepsilon)$ — normalize
23. &emsp;&emsp;&emsp;*// Surrogate advantage (unclipped)*
24. &emsp;&emsp;&emsp;$\boldsymbol{\rho} \gets \exp(\texttt{clamp}(\log\pi_\theta - \log\boldsymbol{\pi}_{\text{old}_\mathcal{B}},\; -20, 20))$ — $\mathcal{O}(M \cdot C_\pi)$
25. &emsp;&emsp;&emsp;$L^{\text{surr}} \gets \texttt{mean}(\boldsymbol{\rho} \odot \hat{\mathbf{A}}_\mathcal{B})$
26. &emsp;&emsp;&emsp;*// MMD trust region penalty — $\mathcal{O}(M \cdot P \cdot d_a)$*
27. &emsp;&emsp;&emsp;$\{\mathbf{x}_1, \mathbf{x}_2\} \sim \text{TanhNormal}(\boldsymbol{\mu}_{\text{old}_\mathcal{B}}, \boldsymbol{\sigma}_{\text{old}_\mathcal{B}})$ — $P/2$ pairs
28. &emsp;&emsp;&emsp;$\{\mathbf{y}_1, \mathbf{y}_2\} \sim \pi_\theta(\cdot \mid \mathbf{S}_\mathcal{B})$ — $P/2$ pairs
29. &emsp;&emsp;&emsp;$k(\mathbf{a}, \mathbf{b}) = \exp(-\|\mathbf{a} - \mathbf{b}\|^2 / (2\sigma^2))$ — RBF kernel
30. &emsp;&emsp;&emsp;$\widehat{\text{MMD}}^2 \gets \texttt{mean}(k(\mathbf{x}_1, \mathbf{x}_2) + k(\mathbf{y}_1, \mathbf{y}_2) - k(\mathbf{x}_1, \mathbf{y}_2) - k(\mathbf{x}_2, \mathbf{y}_1))$
31. &emsp;&emsp;&emsp;*// Mirror descent KL regularizer — $\mathcal{O}(M)$*
32. &emsp;&emsp;&emsp;$D_{\text{KL}} \gets \texttt{mean}((\boldsymbol{\rho} - 1) - \log\boldsymbol{\rho})$ — approx. $\text{KL}(\pi_{\text{old}} \| \pi_\theta)$
33. &emsp;&emsp;&emsp;*// Value baseline*
34. &emsp;&emsp;&emsp;$L^V \gets \texttt{mean}((V_\phi(\mathbf{S}_\mathcal{B}) - \hat{\mathbf{R}}_\mathcal{B})^2)$
35. &emsp;&emsp;&emsp;*// Combined MM-RKHS objective (ascent on surr, descent on penalties)*
36. &emsp;&emsp;&emsp;$L \gets -L^{\text{surr}} + \beta \cdot \widehat{\text{MMD}}^2 + \tfrac{1}{\eta} D_{\text{KL}} + c_V \cdot L^V$
37. &emsp;&emsp;&emsp;$\theta, \phi \gets \theta, \phi - \alpha \nabla_{\theta, \phi} L$ — $\mathcal{O}(M \cdot (C_\pi + C_V))$
38. &emsp;&emsp;**End For**
39. &emsp;**End For**
40. **End For**

**Per-iteration complexity:** $\mathcal{O}(TN \cdot C_\pi + K \cdot NT \cdot (P \cdot C_\pi + C_V))$

$C_\pi$: policy cost, $C_V$: value cost, $P$: MMD samples.
No clipping — trust region via MMD penalty ($\beta$) + KL regularizer ($1/\eta$).
MMD is linear-time unbiased estimator (Gretton et al., 2012): $\mathcal{O}(P)$ per state.

---

## Key Differences: All Six Algorithms

**DDPG** *(Lillicrap et al., 2016):*
- $L^\mu = -\mathbb{E}[Q_\phi(\mathbf{s}, \mu_\theta(\mathbf{s}))]$ — deterministic policy gradient (ascent on Q)
- *Deterministic* actor — no log-prob, no entropy. Exploration via OU noise.
- Off-policy with *replay buffer* + *single Q-network* + target networks.

**SAC** *(Haarnoja et al., 2018):*
- $L^\pi = \mathbb{E}[\alpha \log\pi - \min(Q_1, Q_2)]$ — maximum entropy RL (descent)
- *Stochastic* actor (TanhNormal). Automatic entropy tuning via dual variable $\alpha$.
- Off-policy with *replay buffer* + *twin Q-networks* + target networks.

**PPO** *(Schulman et al., 2017):*
- $L^{\text{CLIP}} = \min(\rho_t \hat{A}_t,\; \text{clip}(\rho_t, 1{-}\epsilon, 1{+}\epsilon) \hat{A}_t)$ — (ascent)
- Advantage via *learned critic* $V_\phi$. On-policy with *rollout buffer*.
- When $\rho_t > 1{+}\epsilon$ and $\hat{A}_t > 0$: $\nabla_\theta = \mathbf{0}$ *(token masked out).*

**OTPG** *(Gupta & Mahajan, 2026):*
- $L = -\mathbb{E}[\rho \hat{A}] + \beta \cdot \widehat{\text{MMD}}^2 + \tfrac{1}{\eta} D_{\text{KL}}$ — majorization-minimization (descent)
- Advantage via *learned critic* $V_\phi$. On-policy with *rollout buffer*.
- No clipping — trust region via *MMD penalty* (RKHS) + *KL regularizer* (mirror descent).
- For all transitions: $\nabla_\theta \neq \mathbf{0}$ *(no gradient masking).*

**GRPO** *(DeepSeek, 2024):*
- $L = -\min(\rho_{j,t} \hat{A}_j,\; \text{clip}(\rho_{j,t}, 1{-}\epsilon, 1{+}\epsilon) \hat{A}_j)$
- Advantage via *group-relative normalization* (no critic).
- When $\rho_{j,t} > 1{+}\epsilon$ and $\hat{A}_j > 0$: $\nabla_\theta = \mathbf{0}$ *(token masked out).*

**CISPO** *(MiniMax-M1, 2025):*
- $L = -\texttt{sg}(\text{clamp}(\rho_{j,t}, \max{=}1{+}\epsilon_h)) \cdot \hat{A}_j \cdot \log \pi_\theta$
- Advantage via *group-relative normalization* (no critic).
- For all tokens: $\nabla_\theta \neq \mathbf{0}$ *(gradient always flows through $\log \pi$).*
