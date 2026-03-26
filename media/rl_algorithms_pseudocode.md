# RL Algorithms Pseudocode

## Contents

**Part I — Canonical Scalar Formulations**
1. [A2C — Advantage Actor-Critic](#algorithm-1-a2c--advantage-actor-critic)
2. [DDPG — Deep Deterministic Policy Gradient](#algorithm-2-ddpg--deep-deterministic-policy-gradient)
3. [SAC — Soft Actor-Critic](#algorithm-3-sac--soft-actor-critic-with-automatic-entropy-tuning)

**Part II — Practical Vectorized Implementations**
4. [PPO-Clip — Proximal Policy Optimization](#algorithm-4-ppo-clip--practical-vectorized-implementation)
5. [DDPG — Vectorized](#algorithm-5-ddpg--practical-vectorized-implementation)
6. [SAC — Vectorized](#algorithm-6-sac--practical-vectorized-implementation)
7. [MM-RKHS — Gupta & Mahajan](#algorithm-7-mmrkhs--gupta--mahajan)

**Part III — LLM Policy Optimization**
8. [GRPO — Group Relative Policy Optimization](#algorithm-8-grpo--practical-vectorized-implementation)
9. [CISPO — Clipped Importance-Sampled Policy Optimization](#algorithm-9-cispo--practical-vectorized-implementation)

**Appendices**
- [Why CISPO: Fixing the Gradient Dead Zone](#why-cispo-fixing-the-gradient-dead-zone)
- [Key Differences: All Nine Algorithms](#key-differences-all-nine-algorithms)

---

# Part I — Canonical Scalar Formulations

---

## Algorithm 1: A2C — Advantage Actor-Critic

**Ensure:** Optimized policy $\pi_\theta$

**Hyperparameters:**
- $\theta_0$ — init policy params
- $\phi_0$ — init value params
- $\gamma$ — discount factor
- $\alpha_\pi$ — policy learning rate
- $\alpha_V$ — value learning rate
- $c_e$ — entropy coeff.
- $n$ — $n$-step return length

---

1. **For** each episode
2. &emsp;$s_0 \gets \texttt{env.reset}()$
3. &emsp;**For** $t = 0, 1, 2, \ldots$ until done
4. &emsp;&emsp;*// Act*
5. &emsp;&emsp;$a_t \sim \pi_\theta(\cdot \mid s_t)$ — sample from stochastic policy
6. &emsp;&emsp;$r_t, s_{t+1}, d_t \gets \texttt{env.step}(a_t)$ — $d_t = 1$ if terminal
7. &emsp;&emsp;*// Compute $n$-step return*
8. &emsp;&emsp;$R_t^{(n)} \gets \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n (1 - d_{t+n-1}) V_\phi(s_{t+n})$ — bootstrap if not terminal
9. &emsp;&emsp;*// Advantage estimate*
10. &emsp;&emsp;$\hat{A}_t \gets R_t^{(n)} - V_\phi(s_t)$ — how much better than expected
11. &emsp;&emsp;*// Critic update — regress $V_\phi$ toward $n$-step return*
12. &emsp;&emsp;$L^V \gets (V_\phi(s_t) - R_t^{(n)})^2$
13. &emsp;&emsp;$\phi \gets \phi - \alpha_V \nabla_\phi L^V$
14. &emsp;&emsp;*// Actor update — policy gradient with baseline*
15. &emsp;&emsp;$L^\pi \gets -\log \pi_\theta(a_t \mid s_t) \cdot \texttt{stop\_grad}(\hat{A}_t) - c_e \cdot H(\pi_\theta(\cdot \mid s_t))$
16. &emsp;&emsp;$\theta \gets \theta - \alpha_\pi \nabla_\theta L^\pi$
17. &emsp;**End For**
18. **End For**

**Notes:**
- A2C is the synchronous, single-environment form of the actor-critic. A3C adds $N$ asynchronous workers with shared $\theta, \phi$.
- $\hat{A}_t = R_t^{(n)} - V_\phi(s_t)$ is the simplest advantage; PPO upgrades this to GAE($\lambda$) for lower variance.
- The entropy bonus $c_e H$ encourages exploration by penalizing peaky distributions.
- $\texttt{stop\_grad}(\hat{A}_t)$: advantage is a scalar weight on $\log\pi$, not differentiated through $V_\phi$.

---

## Algorithm 2: DDPG — Deep Deterministic Policy Gradient

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
- $\theta_{\text{OU}}$ — OU mean reversion rate (typ. 0.15)
- $\sigma_{\text{OU}}$ — OU volatility (typ. 0.2)

---

1. **Initialize** $\theta' \gets \theta,\; \phi' \gets \phi$ — copy params to target networks
2. **Initialize** $\mathcal{D} \gets \texttt{ReplayBuffer}(|\mathcal{D}|)$, $\;n \gets 0$ — OU noise state
3. **For** each episode
4. &emsp;$s \gets \texttt{env.reset}()$, &emsp; $n \gets 0$ — reset noise at episode start
5. &emsp;**For** $t = 0, 1, 2, \ldots$ until done
6. &emsp;&emsp;*// Act with exploration noise*
7. &emsp;&emsp;**If** total frames $< W$: $a \sim \mathcal{U}[-1, 1]^{d_a}$ — random warmup
8. &emsp;&emsp;**Else**:
9. &emsp;&emsp;&emsp;$n \gets n + \theta_{\text{OU}}(0 - n) + \sigma_{\text{OU}} \epsilon$, &emsp; $\epsilon \sim \mathcal{N}(0, I)^{d_a}$ — OU noise step
10. &emsp;&emsp;&emsp;$a \gets \texttt{clamp}(\mu_\theta(s) + n,\; -1, 1)$ — deterministic action + noise + clip
11. &emsp;&emsp;$r, s', d \gets \texttt{env.step}(a)$ — $d = 1$ if terminal
12. &emsp;&emsp;$\mathcal{D} \gets \mathcal{D} \cup \{(s, a, r, s', d)\}$ — store transition
13. &emsp;&emsp;$s \gets s'$
14. &emsp;&emsp;**If** total frames $< W$: **continue**
15. &emsp;&emsp;*// Sample minibatch and update*
16. &emsp;&emsp;$\{(s_i, a_i, r_i, s'_i, d_i)\}_{i=1}^B \sim \mathcal{D}$ — uniformly sample $B$ transitions
17. &emsp;&emsp;*// Critic update*
18. &emsp;&emsp;**With** $\texttt{no\_grad}$:
19. &emsp;&emsp;&emsp;$a'_i \gets \mu_{\theta'}(s'_i)$ — target actor computes next action
20. &emsp;&emsp;&emsp;$q'_i \gets Q_{\phi'}(s'_i, a'_i)$ — target critic evaluates next $(s', a')$
21. &emsp;&emsp;&emsp;$y_i \gets r_i + \gamma(1 - d_i) q'_i$ — Bellman target; $(1-d)$ zeros bootstrap at terminals
22. &emsp;&emsp;$L^Q \gets \frac{1}{B}\sum_{i=1}^B (Q_\phi(s_i, a_i) - y_i)^2$ — MSE critic loss
23. &emsp;&emsp;$\phi \gets \phi - \alpha_Q \nabla_\phi L^Q$
24. &emsp;&emsp;*// Actor update — deterministic policy gradient*
25. &emsp;&emsp;$\hat{a}_i \gets \mu_\theta(s_i)$ — forward through actor
26. &emsp;&emsp;$L^\mu \gets -\frac{1}{B}\sum_{i=1}^B Q_\phi(s_i, \hat{a}_i)$ — $\nabla_\theta$ flows through $\mu_\theta \to Q_\phi$
27. &emsp;&emsp;$\theta \gets \theta - \alpha_\mu \nabla_\theta L^\mu$
28. &emsp;&emsp;*// Polyak-averaged target update*
29. &emsp;&emsp;$\theta' \gets \tau\theta + (1 - \tau)\theta'$ — slowly track actor
30. &emsp;&emsp;$\phi' \gets \tau\phi + (1 - \tau)\phi'$ — slowly track critic
31. &emsp;**End For**
32. **End For**

---

## Algorithm 3: SAC — Soft Actor-Critic with Automatic Entropy Tuning

**Ensure:** Optimized stochastic policy $\pi_\theta$

**Hyperparameters:**
- $\theta_0$ — init actor params (outputs $\mu, \log\sigma$)
- $\phi_{1,0}, \phi_{2,0}$ — init twin critic params
- $\gamma$ — discount factor
- $\tau$ — soft update rate (typ. 0.005)
- $\alpha_\pi$ — actor learning rate
- $\alpha_Q$ — critic learning rate
- $\alpha_\alpha$ — entropy coeff. learning rate
- $B$ — batch size
- $W$ — warmup steps
- $|\mathcal{D}|$ — replay buffer capacity
- $\bar{\mathcal{H}} = -d_a$ — target entropy (negative action dim)
- $f_\mu$ — actor update frequency (typ. 2)

**Networks:** Actor $\pi_\theta$: MLP $s \to (\mu, \log\sigma)$, produces TanhNormal distribution; Twin critics $Q_{\phi_1}, Q_{\phi_2}$: MLP $[s;a] \to \mathbb{R}$; Target critics $Q_{\phi'_1}, Q_{\phi'_2}$: same architecture, separate params (Polyak-averaged)

---

1. **Initialize** $\phi'_1 \gets \phi_1,\; \phi'_2 \gets \phi_2$ — copy params to target critic networks
2. **Initialize** $\log\alpha \gets \log(\alpha_0)$ — learnable log-entropy coeff., $\alpha = \exp(\log\alpha)$
3. **Initialize** $\mathcal{D} \gets \texttt{ReplayBuffer}(|\mathcal{D}|)$
4. **For** each episode
5. &emsp;$s \gets \texttt{env.reset}()$
6. &emsp;**For** $t = 0, 1, 2, \ldots$ until done
7. &emsp;&emsp;*// Act*
8. &emsp;&emsp;**If** total frames $< W$: $a \sim \mathcal{U}[-1, 1]^{d_a}$ — random warmup
9. &emsp;&emsp;**Else**:
10. &emsp;&emsp;&emsp;$\mu, \log\sigma \gets \pi_\theta(s)$ — actor forward
11. &emsp;&emsp;&emsp;$\epsilon \sim \mathcal{N}(0, I)^{d_a}$ — reparameterization noise
12. &emsp;&emsp;&emsp;$u \gets \mu + \exp(\log\sigma) \odot \epsilon$ — pre-squash action
13. &emsp;&emsp;&emsp;$a \gets \tanh(u)$ — squashed into $[-1, 1]$
14. &emsp;&emsp;$r, s', d \gets \texttt{env.step}(a)$ — $d = 1$ if terminal
15. &emsp;&emsp;$\mathcal{D} \gets \mathcal{D} \cup \{(s, a, r, s', d)\}$ — store transition
16. &emsp;&emsp;$s \gets s'$
17. &emsp;&emsp;**If** total frames $< W$: **continue**
18. &emsp;&emsp;*// Sample minibatch*
19. &emsp;&emsp;$\{(s_i, a_i, r_i, s'_i, d_i)\}_{i=1}^B \sim \mathcal{D}$ — uniformly sample $B$ transitions
20. &emsp;&emsp;*// Twin critic update*
21. &emsp;&emsp;**With** $\texttt{no\_grad}$: — no gradient through $\theta$, $\phi'_1$, $\phi'_2$
22. &emsp;&emsp;&emsp;$\mu'_i, \log\sigma'_i \gets \pi_\theta(s'_i)$ — actor on next state
23. &emsp;&emsp;&emsp;$u'_i \gets \mu'_i + \exp(\log\sigma'_i) \odot \epsilon'_i$, &emsp; $\epsilon'_i \sim \mathcal{N}(0, I)$ — pre-squash
24. &emsp;&emsp;&emsp;$\tilde{a}'_i \gets \tanh(u'_i)$ — squashed next action
25. &emsp;&emsp;&emsp;$\ell'_i \gets \sum_{j=1}^{d_a} [\log\mathcal{N}(u'_{i,j} \mid \mu'_{i,j}, \sigma'_{i,j}) - \log(1 - \tanh^2(u'_{i,j}) + \varepsilon)]$ — log-prob with Jacobian correction
26. &emsp;&emsp;&emsp;$q'_{1,i} \gets Q_{\phi'_1}(s'_i, \tilde{a}'_i)$, &emsp; $q'_{2,i} \gets Q_{\phi'_2}(s'_i, \tilde{a}'_i)$ — target critics evaluate $(s', \tilde{a}')$
27. &emsp;&emsp;&emsp;$y_i \gets r_i + \gamma(1 - d_i)[\min(q'_{1,i}, q'_{2,i}) - \alpha \cdot \ell'_i]$ — soft Bellman; $(1-d)$ zeros bootstrap at terminals
28. &emsp;&emsp;$L^Q \gets \frac{1}{B}\sum_{i=1}^B (Q_{\phi_1}(s_i, a_i) - y_i)^2 + \frac{1}{B}\sum_{i=1}^B (Q_{\phi_2}(s_i, a_i) - y_i)^2$ — sum of MSE
29. &emsp;&emsp;$\phi_1 \gets \phi_1 - \alpha_Q \nabla_{\phi_1} L^Q$, &emsp; $\phi_2 \gets \phi_2 - \alpha_Q \nabla_{\phi_2} L^Q$
30. &emsp;&emsp;*// Actor update (every $f_\mu$ critic steps)*
31. &emsp;&emsp;**If** total steps $\bmod f_\mu = 0$:
32. &emsp;&emsp;&emsp;$\mu_i, \log\sigma_i \gets \pi_\theta(s_i)$ — actor forward on batch states
33. &emsp;&emsp;&emsp;$u_i \gets \mu_i + \exp(\log\sigma_i) \odot \epsilon_i$, &emsp; $\epsilon_i \sim \mathcal{N}(0, I)$
34. &emsp;&emsp;&emsp;$\tilde{a}_i \gets \tanh(u_i)$ — reparameterized action; $\nabla_\theta$ flows through $\mu, \sigma$
35. &emsp;&emsp;&emsp;$\ell_i \gets \sum_{j=1}^{d_a} [\log\mathcal{N}(u_{i,j} \mid \mu_{i,j}, \sigma_{i,j}) - \log(1 - \tanh^2(u_{i,j}) + \varepsilon)]$
36. &emsp;&emsp;&emsp;$L^\pi \gets \frac{1}{B}\sum_{i=1}^B [\alpha \cdot \ell_i - \min(Q_{\phi_1}(s_i, \tilde{a}_i),\; Q_{\phi_2}(s_i, \tilde{a}_i))]$ — maximize Q, maximize entropy
37. &emsp;&emsp;&emsp;$\theta \gets \theta - \alpha_\pi \nabla_\theta L^\pi$ — $\nabla_\theta$ backprops through $\tilde{a}$ into $Q_\phi$ (critics frozen)
38. &emsp;&emsp;&emsp;*// Entropy coefficient update*
39. &emsp;&emsp;&emsp;$L^\alpha \gets -\frac{1}{B}\sum_{i=1}^B \log\alpha \cdot \texttt{stop\_grad}(\ell_i + \bar{\mathcal{H}})$ — dual gradient descent
40. &emsp;&emsp;&emsp;$\log\alpha \gets \log\alpha - \alpha_\alpha \nabla_{\log\alpha} L^\alpha$
41. &emsp;&emsp;&emsp;$\alpha \gets \exp(\log\alpha)$
42. &emsp;&emsp;*// Polyak-averaged target update*
43. &emsp;&emsp;$\phi'_1 \gets \tau\phi_1 + (1 - \tau)\phi'_1$ — slowly track online critic 1
44. &emsp;&emsp;$\phi'_2 \gets \tau\phi_2 + (1 - \tau)\phi'_2$ — slowly track online critic 2
45. &emsp;**End For**
46. **End For**

---

# Part II — Practical Vectorized Implementations

---

## Algorithm 4: PPO-Clip — Practical Vectorized Implementation

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
2. &emsp;*// Phase 1: Vectorized rollout*
3. &emsp;**For** $t = 0, \ldots, T{-}1$
4. &emsp;&emsp;$\mathbf{A}_{:,t} \sim \pi_{\theta_k}(\cdot \mid \mathbf{S}_{:,t})$
5. &emsp;&emsp;$\mathbf{R}_{:,t}, \mathbf{S}_{:,t+1}, \mathbf{D}_{:,t} \gets \texttt{envs.step}(\mathbf{A}_{:,t})$
6. &emsp;&emsp;$\boldsymbol{\pi}_{\text{old}_{:,t}} \gets \pi_{\theta_k}(\mathbf{A}_{:,t} \mid \mathbf{S}_{:,t})$
7. &emsp;**End For**
8. &emsp;*// Phase 2: Vectorized GAE*
9. &emsp;$\mathbf{V} \gets V_{\phi_k}([\mathbf{S}, \mathbf{S}_{:,T}])$ — shape $(N, T{+}1)$
10. &emsp;$\boldsymbol{\delta} \gets \mathbf{R} + \gamma(1 - \mathbf{D}) \odot \mathbf{V}_{:,1:} - \mathbf{V}_{:,:-1}$
11. &emsp;$\hat{\mathbf{A}}_{:,T-1} \gets \boldsymbol{\delta}_{:,T-1}$
12. &emsp;**For** $t = T{-}2, \ldots, 0$
13. &emsp;&emsp;$\hat{\mathbf{A}}_{:,t} \gets \boldsymbol{\delta}_{:,t} + \gamma\lambda(1 - \mathbf{D}_{:,t}) \odot \hat{\mathbf{A}}_{:,t+1}$
14. &emsp;**End For**
15. &emsp;$\hat{\mathbf{R}} \gets \hat{\mathbf{A}} + \mathbf{V}_{:,:-1}$
16. &emsp;*// Phase 3: Minibatch SGD*
17. &emsp;Flatten $\mathcal{D}_k \gets$ reshape all tensors to $(N \cdot T, \ldots)$
18. &emsp;**For** epoch $= 1, \ldots, K$
19. &emsp;&emsp;$\mathcal{I} \gets \texttt{randperm}(N \cdot T)$
20. &emsp;&emsp;**For** $b = 0, M, 2M, \ldots$ — $\lceil NT/M \rceil$ minibatches
21. &emsp;&emsp;&emsp;$\mathcal{B} \gets \mathcal{I}[b : b{+}M]$
22. &emsp;&emsp;&emsp;$\hat{\mathbf{A}}_\mathcal{B} \gets (\hat{\mathbf{A}}_\mathcal{B} - \texttt{mean}(\hat{\mathbf{A}}_\mathcal{B})) / (\texttt{std}(\hat{\mathbf{A}}_\mathcal{B}) + \varepsilon)$
23. &emsp;&emsp;&emsp;$\boldsymbol{\rho} \gets \pi_\theta(\mathbf{A}_\mathcal{B} \mid \mathbf{S}_\mathcal{B}) / \boldsymbol{\pi}_{\text{old}_\mathcal{B}}$
24. &emsp;&emsp;&emsp;$L^{\text{CLIP}} \gets \texttt{mean}(\min(\boldsymbol{\rho} \odot \hat{\mathbf{A}}_\mathcal{B},\; \texttt{clip}(\boldsymbol{\rho}, 1{-}\epsilon, 1{+}\epsilon) \odot \hat{\mathbf{A}}_\mathcal{B}))$
25. &emsp;&emsp;&emsp;$L^V \gets \texttt{mean}((V_\phi(\mathbf{S}_\mathcal{B}) - \hat{\mathbf{R}}_\mathcal{B})^2)$
26. &emsp;&emsp;&emsp;$H \gets \texttt{mean}(\texttt{entropy}(\pi_\theta(\cdot \mid \mathbf{S}_\mathcal{B})))$
27. &emsp;&emsp;&emsp;$\theta \gets \theta + \alpha_\pi \nabla_\theta (L^{\text{CLIP}} + c_e H)$
28. &emsp;&emsp;&emsp;$\phi \gets \phi - \alpha_V \nabla_\phi L^V$
29. &emsp;&emsp;**End For**
30. &emsp;**End For**
31. **End For**

---

## Algorithm 5: DDPG — Practical Vectorized Implementation

**Ensure:** Optimized deterministic policy $\mu_\theta$

**Hyperparameters:**
- $\theta_0$ — init actor params
- $\phi_0$ — init critic params
- $\gamma$ — discount factor
- $\tau$ — soft update rate (typ. 0.005)
- $\alpha_\mu$ — actor learning rate
- $\alpha_Q$ — critic learning rate
- $B$ — batch size
- $W$ — warmup frames
- $|\mathcal{D}|$ — replay buffer capacity
- $N$ — parallel environments
- $\theta_{\text{OU}}$ — OU mean reversion rate (typ. 0.15)
- $\sigma_{\text{OU}}$ — OU volatility (typ. 0.2)

**Tensors:**
- $\mathbf{S} \in \mathbb{R}^{N \times d_s}$ — env states
- $\mathbf{A} \in \mathbb{R}^{N \times d_a}$ — env actions
- $\mathbf{N} \in \mathbb{R}^{N \times d_a}$ — OU noise states
- $\mathbf{S}_\mathcal{B}, \mathbf{S}'_\mathcal{B} \in \mathbb{R}^{B \times d_s}$ — replay batch states
- $\mathbf{A}_\mathcal{B} \in \mathbb{R}^{B \times d_a}$ — replay batch actions
- $\mathbf{R}_\mathcal{B} \in \mathbb{R}^{B}$ — replay batch rewards
- $\mathbf{D}_\mathcal{B} \in \{0,1\}^{B}$ — replay batch terminal flags ($1$ = episode ended)
- $\mathbf{Y} \in \mathbb{R}^{B}$ — Bellman targets
- $\mathbf{Q}_{\text{tgt}} \in \mathbb{R}^{B}$ — target Q-values

**Networks:** $\mu_\theta$: actor MLP $\to$ $\tanh$-bounded actions; $Q_\phi(s,a)$: critic MLP $[s;a] \to \mathbb{R}$; $\mu_{\theta'}, Q_{\phi'}$: target copies (same architecture, separate params)

---

1. **Initialize** $\theta' \gets \theta,\; \phi' \gets \phi$ — copy params to target networks
2. **Initialize** $\mathcal{D} \gets \texttt{ReplayBuffer}(|\mathcal{D}|)$, $\;\mathbf{N} \gets \mathbf{0}^{N \times d_a}$
3. $\mathbf{S} \gets \texttt{envs.reset}()$ — start $N$ parallel envs; shape $(N, d_s)$
4. **For** $t = 0, 1, 2, \ldots$
5. &emsp;*// Phase 1: Vectorized environment interaction*
6. &emsp;**If** $t < W/N$: $\mathbf{A} \sim \mathcal{U}[-1, 1]^{N \times d_a}$ — random warmup
7. &emsp;**Else**:
8. &emsp;&emsp;$\mathbf{N} \gets \mathbf{N} + \theta_{\text{OU}}(0 - \mathbf{N}) + \sigma_{\text{OU}} \boldsymbol{\epsilon}$, &emsp; $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})^{N \times d_a}$ — OU noise step
9. &emsp;&emsp;$\mathbf{A} \gets \texttt{clamp}(\mu_\theta(\mathbf{S}) + \mathbf{N},\; -1, 1)$ — batched forward + noise + clip
10. &emsp;$\mathbf{R}, \mathbf{S}', \mathbf{D} \gets \texttt{envs.step}(\mathbf{A})$ — $\mathbf{D}_{i} = 1$ if env $i$ reached terminal state
11. &emsp;$\mathcal{D}.\texttt{extend}(\mathbf{S}, \mathbf{A}, \mathbf{R}, \mathbf{S}', \mathbf{D})$ — store $N$ tuples $(s, a, r, s', d)$
12. &emsp;$\mathbf{S} \gets \mathbf{S}'$ — auto-reset envs replace terminal $s'$ with new $s_0$
13. &emsp;**If** done envs exist: reset corresponding rows of $\mathbf{N}$ to $\mathbf{0}$
14. &emsp;**If** $|\mathcal{D}| < W$: **continue**
15. &emsp;*// Phase 2: Batched critic update*
16. &emsp;$(\mathbf{S}_\mathcal{B}, \mathbf{A}_\mathcal{B}, \mathbf{R}_\mathcal{B}, \mathbf{S}'_\mathcal{B}, \mathbf{D}_\mathcal{B}) \sim \mathcal{D}$ — uniformly sample $B$ transitions
17. &emsp;**With** $\texttt{no\_grad}$: — target computation, no gradient through $\theta'$ or $\phi'$
18. &emsp;&emsp;$\mathbf{A}_{\text{tgt}} \gets \mu_{\theta'}(\mathbf{S}'_\mathcal{B})$ — target actor: next actions; shape $(B, d_a)$
19. &emsp;&emsp;$\mathbf{Q}_{\text{tgt}} \gets Q_{\phi'}(\mathbf{S}'_\mathcal{B},\, \mathbf{A}_{\text{tgt}})$ — target critic: next Q-values; shape $(B,)$
20. &emsp;&emsp;$\mathbf{Y} \gets \mathbf{R}_\mathcal{B} + \gamma(1 - \mathbf{D}_\mathcal{B}) \odot \mathbf{Q}_{\text{tgt}}$ — Bellman target; $(1 - \mathbf{D})$ zeros out bootstrap at terminal states
21. &emsp;$L^Q \gets \texttt{mean}((Q_\phi(\mathbf{S}_\mathcal{B}, \mathbf{A}_\mathcal{B}) - \mathbf{Y})^2)$ — MSE critic loss
22. &emsp;$\phi \gets \phi - \alpha_Q \nabla_\phi L^Q$
23. &emsp;*// Phase 3: Batched actor update*
24. &emsp;$\hat{\mathbf{A}} \gets \mu_\theta(\mathbf{S}_\mathcal{B})$ — forward through actor; shape $(B, d_a)$
25. &emsp;$L^\mu \gets -\texttt{mean}(Q_\phi(\mathbf{S}_\mathcal{B},\, \hat{\mathbf{A}}))$ — $\nabla_\theta$ flows through $\mu_\theta \to Q_\phi$
26. &emsp;$\theta \gets \theta - \alpha_\mu \nabla_\theta L^\mu$
27. &emsp;*// Phase 4: Polyak-averaged target update*
28. &emsp;$\theta' \gets \tau\theta + (1 - \tau)\theta'$ — slowly track actor
29. &emsp;$\phi' \gets \tau\phi + (1 - \tau)\phi'$ — slowly track critic
30. **End For**

---

## Algorithm 6: SAC — Practical Vectorized Implementation

**Ensure:** Optimized stochastic policy $\pi_\theta$

**Hyperparameters:**
- $\theta_0$ — init actor params (outputs $\boldsymbol{\mu}, \log\boldsymbol{\sigma}$)
- $\phi_{1,0}, \phi_{2,0}$ — init twin critic params
- $\gamma$ — discount factor
- $\tau$ — soft update rate (typ. 0.005)
- $\alpha_\pi$ — actor learning rate
- $\alpha_Q$ — critic learning rate
- $\alpha_\alpha$ — entropy coeff. learning rate
- $B$ — batch size
- $W$ — warmup frames
- $|\mathcal{D}|$ — replay buffer capacity
- $N$ — parallel environments
- $\bar{\mathcal{H}} = -d_a$ — target entropy (negative action dim)
- $f_\mu$ — actor update frequency (typ. 2)

**Tensors:**
- $\mathbf{S} \in \mathbb{R}^{N \times d_s}$ — env states
- $\mathbf{A} \in \mathbb{R}^{N \times d_a}$ — env actions
- $\mathbf{S}_\mathcal{B}, \mathbf{S}'_\mathcal{B} \in \mathbb{R}^{B \times d_s}$ — replay batch states
- $\mathbf{A}_\mathcal{B} \in \mathbb{R}^{B \times d_a}$ — replay batch actions
- $\mathbf{R}_\mathcal{B} \in \mathbb{R}^{B}$ — replay batch rewards
- $\mathbf{D}_\mathcal{B} \in \{0,1\}^{B}$ — replay batch terminal flags ($1$ = episode ended)
- $\tilde{\mathbf{A}}, \tilde{\mathbf{A}}' \in \mathbb{R}^{B \times d_a}$ — freshly sampled actions (reparameterized)
- $\boldsymbol{\ell}, \boldsymbol{\ell}' \in \mathbb{R}^{B}$ — log-probabilities of sampled actions
- $\mathbf{Y} \in \mathbb{R}^{B}$ — soft Bellman targets
- $\mathbf{Q}_1, \mathbf{Q}_2 \in \mathbb{R}^{B}$ — twin Q-values

**Networks:** Actor $\pi_\theta$: MLP $s \to (\boldsymbol{\mu}, \log\boldsymbol{\sigma})$, produces TanhNormal dist.; Twin critics $Q_{\phi_1}, Q_{\phi_2}$: MLP $[s;a] \to \mathbb{R}$; Target critics $Q_{\phi'_1}, Q_{\phi'_2}$: same architecture, separate params (Polyak-averaged)

---

1. **Initialize** $\phi'_1 \gets \phi_1,\; \phi'_2 \gets \phi_2$ — copy params to target critic networks
2. **Initialize** $\log\alpha \gets \log(\alpha_0)$ — learnable log-entropy coeff., $\alpha = \exp(\log\alpha)$
3. **Initialize** $\mathcal{D} \gets \texttt{ReplayBuffer}(|\mathcal{D}|)$
4. $\mathbf{S} \gets \texttt{envs.reset}()$ — start $N$ parallel envs; shape $(N, d_s)$
5. **For** $t = 0, 1, 2, \ldots$
6. &emsp;*// Phase 1: Vectorized environment interaction*
7. &emsp;**If** $t < W/N$: $\mathbf{A} \sim \mathcal{U}[-1, 1]^{N \times d_a}$ — random warmup
8. &emsp;**Else**:
9. &emsp;&emsp;$\boldsymbol{\mu}, \log\boldsymbol{\sigma} \gets \pi_\theta(\mathbf{S})$ — actor forward; shapes $(N, d_a)$
10. &emsp;&emsp;$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})^{N \times d_a}$ — reparameterization noise
11. &emsp;&emsp;$\mathbf{A} \gets \tanh(\boldsymbol{\mu} + \exp(\log\boldsymbol{\sigma}) \odot \boldsymbol{\epsilon})$ — TanhNormal rsample
12. &emsp;$\mathbf{R}, \mathbf{S}', \mathbf{D} \gets \texttt{envs.step}(\mathbf{A})$ — $\mathbf{D}_{i} = 1$ if env $i$ reached terminal state
13. &emsp;$\mathcal{D}.\texttt{extend}(\mathbf{S}, \mathbf{A}, \mathbf{R}, \mathbf{S}', \mathbf{D})$ — store $N$ tuples $(s, a, r, s', d)$
14. &emsp;$\mathbf{S} \gets \mathbf{S}'$ — auto-reset envs replace terminal $s'$ with new $s_0$
15. &emsp;**If** $|\mathcal{D}| < W$: **continue**
16. &emsp;*// Phase 2: Batched twin critic update*
17. &emsp;$(\mathbf{S}_\mathcal{B}, \mathbf{A}_\mathcal{B}, \mathbf{R}_\mathcal{B}, \mathbf{S}'_\mathcal{B}, \mathbf{D}_\mathcal{B}) \sim \mathcal{D}$ — uniformly sample $B$ transitions
18. &emsp;**With** $\texttt{no\_grad}$: — target computation, no gradient through $\theta$, $\phi'_1$, or $\phi'_2$
19. &emsp;&emsp;*// Sample next actions from current policy*
20. &emsp;&emsp;$\boldsymbol{\mu}', \log\boldsymbol{\sigma}' \gets \pi_\theta(\mathbf{S}'_\mathcal{B})$ — actor forward on next states
21. &emsp;&emsp;$\boldsymbol{\epsilon}' \sim \mathcal{N}(\mathbf{0}, \mathbf{I})^{B \times d_a}$
22. &emsp;&emsp;$\mathbf{U}' \gets \boldsymbol{\mu}' + \exp(\log\boldsymbol{\sigma}') \odot \boldsymbol{\epsilon}'$ — pre-squash actions; shape $(B, d_a)$
23. &emsp;&emsp;$\tilde{\mathbf{A}}' \gets \tanh(\mathbf{U}')$ — squashed actions; shape $(B, d_a)$
24. &emsp;&emsp;$\boldsymbol{\ell}' \gets \sum_{j=1}^{d_a} [\log\mathcal{N}(U'_j \mid \mu'_j, \sigma'_j) - \log(1 - \tanh^2(U'_j) + \varepsilon)]$ — log-prob with Jacobian correction; shape $(B,)$
25. &emsp;&emsp;*// Compute target Q-values from target networks*
26. &emsp;&emsp;$\mathbf{Q}'_1 \gets Q_{\phi'_1}(\mathbf{S}'_\mathcal{B}, \tilde{\mathbf{A}}')$ — target critic 1; shape $(B,)$
27. &emsp;&emsp;$\mathbf{Q}'_2 \gets Q_{\phi'_2}(\mathbf{S}'_\mathcal{B}, \tilde{\mathbf{A}}')$ — target critic 2; shape $(B,)$
28. &emsp;&emsp;*// Soft Bellman target — $(1-\mathbf{D})$ zeros bootstrap at terminal states*
29. &emsp;&emsp;$\mathbf{Y} \gets \mathbf{R}_\mathcal{B} + \gamma(1 - \mathbf{D}_\mathcal{B}) \odot [\min(\mathbf{Q}'_1, \mathbf{Q}'_2) - \alpha \cdot \boldsymbol{\ell}']$
30. &emsp;*// Critic loss — both Q-networks regress toward same target $\mathbf{Y}$*
31. &emsp;$\mathbf{Q}_1 \gets Q_{\phi_1}(\mathbf{S}_\mathcal{B}, \mathbf{A}_\mathcal{B})$, &emsp; $\mathbf{Q}_2 \gets Q_{\phi_2}(\mathbf{S}_\mathcal{B}, \mathbf{A}_\mathcal{B})$ — current Q-values; shapes $(B,)$
32. &emsp;$L^Q \gets \texttt{mean}((\mathbf{Q}_1 - \mathbf{Y})^2) + \texttt{mean}((\mathbf{Q}_2 - \mathbf{Y})^2)$ — sum of MSE losses
33. &emsp;$\phi_1 \gets \phi_1 - \alpha_Q \nabla_{\phi_1} L^Q$, &emsp; $\phi_2 \gets \phi_2 - \alpha_Q \nabla_{\phi_2} L^Q$
34. &emsp;*// Phase 3: Batched actor update (every $f_\mu$ critic steps)*
35. &emsp;**If** $t \bmod f_\mu = 0$:
36. &emsp;&emsp;*// Fresh actions from current policy (with gradient through $\theta$)*
37. &emsp;&emsp;$\boldsymbol{\mu}, \log\boldsymbol{\sigma} \gets \pi_\theta(\mathbf{S}_\mathcal{B})$
38. &emsp;&emsp;$\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})^{B \times d_a}$
39. &emsp;&emsp;$\mathbf{U} \gets \boldsymbol{\mu} + \exp(\log\boldsymbol{\sigma}) \odot \boldsymbol{\epsilon}$
40. &emsp;&emsp;$\tilde{\mathbf{A}} \gets \tanh(\mathbf{U})$ — reparameterized actions; $\nabla_\theta$ flows through $\boldsymbol{\mu}, \boldsymbol{\sigma}$
41. &emsp;&emsp;$\boldsymbol{\ell} \gets \sum_{j=1}^{d_a} [\log\mathcal{N}(U_j \mid \mu_j, \sigma_j) - \log(1 - \tanh^2(U_j) + \varepsilon)]$ — log-prob; shape $(B,)$
42. &emsp;&emsp;*// Actor loss — maximize Q, maximize entropy*
43. &emsp;&emsp;$L^\pi \gets \texttt{mean}(\alpha \cdot \boldsymbol{\ell} - \min(Q_{\phi_1}(\mathbf{S}_\mathcal{B}, \tilde{\mathbf{A}}),\; Q_{\phi_2}(\mathbf{S}_\mathcal{B}, \tilde{\mathbf{A}})))$
44. &emsp;&emsp;$\theta \gets \theta - \alpha_\pi \nabla_\theta L^\pi$ — $\nabla_\theta$ backprops through $\tilde{\mathbf{A}}$ into $Q_\phi$ (critics frozen)
45. &emsp;&emsp;*// Phase 4: Entropy coefficient update*
46. &emsp;&emsp;$L^\alpha \gets -\texttt{mean}(\log\alpha \cdot \texttt{stop\_grad}(\boldsymbol{\ell} + \bar{\mathcal{H}}))$ — dual gradient descent
47. &emsp;&emsp;$\log\alpha \gets \log\alpha - \alpha_\alpha \nabla_{\log\alpha} L^\alpha$
48. &emsp;&emsp;$\alpha \gets \exp(\log\alpha)$
49. &emsp;*// Phase 5: Polyak-averaged target update — slowly track online critics*
50. &emsp;$\phi'_1 \gets \tau\phi_1 + (1 - \tau)\phi'_1$
51. &emsp;$\phi'_2 \gets \tau\phi_2 + (1 - \tau)\phi'_2$
52. **End For**

---

## Algorithm 7: MM-RKHS — Gupta & Mahajan (MM-RKHS)

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
2. &emsp;*// Phase 1: Vectorized rollout (same as PPO)*
3. &emsp;**For** $t = 0, \ldots, T{-}1$
4. &emsp;&emsp;$\mathbf{A}_{:,t} \sim \pi_{\theta_k}(\cdot \mid \mathbf{S}_{:,t})$ — TanhNormal
5. &emsp;&emsp;$\mathbf{R}_{:,t}, \mathbf{S}_{:,t+1}, \mathbf{D}_{:,t} \gets \texttt{envs.step}(\mathbf{A}_{:,t})$
6. &emsp;&emsp;Store $\boldsymbol{\pi}_{\text{old}_{:,t}},\; \boldsymbol{\mu}_{\text{old}_{:,t}},\; \boldsymbol{\sigma}_{\text{old}_{:,t}}$ — log-prob + distribution params
7. &emsp;**End For**
8. &emsp;*// Phase 2: Vectorized GAE (same as PPO)*
9. &emsp;$\mathbf{V} \gets V_{\phi_k}([\mathbf{S}, \mathbf{S}_{:,T}])$
10. &emsp;$\boldsymbol{\delta} \gets \mathbf{R} + \gamma(1 - \mathbf{D}) \odot \mathbf{V}_{:,1:} - \mathbf{V}_{:,:-1}$
11. &emsp;$\hat{\mathbf{A}}_{:,T-1} \gets \boldsymbol{\delta}_{:,T-1}$
12. &emsp;**For** $t = T{-}2, \ldots, 0$
13. &emsp;&emsp;$\hat{\mathbf{A}}_{:,t} \gets \boldsymbol{\delta}_{:,t} + \gamma\lambda(1 - \mathbf{D}_{:,t}) \odot \hat{\mathbf{A}}_{:,t+1}$
14. &emsp;**End For**
15. &emsp;$\hat{\mathbf{R}} \gets \hat{\mathbf{A}} + \mathbf{V}_{:,:-1}$
16. &emsp;*// Phase 3: MM-RKHS policy update*
17. &emsp;Flatten $\mathcal{D}_k \gets$ reshape to $(N \cdot T, \ldots)$
18. &emsp;**For** epoch $= 1, \ldots, K$
19. &emsp;&emsp;$\mathcal{I} \gets \texttt{randperm}(N \cdot T)$
20. &emsp;&emsp;**For** $b = 0, M, 2M, \ldots$ — $\lceil NT/M \rceil$ minibatches
21. &emsp;&emsp;&emsp;$\mathcal{B} \gets \mathcal{I}[b : b{+}M]$
22. &emsp;&emsp;&emsp;$\hat{\mathbf{A}}_\mathcal{B} \gets (\hat{\mathbf{A}}_\mathcal{B} - \texttt{mean}) / (\texttt{std} + \varepsilon)$ — normalize
23. &emsp;&emsp;&emsp;*// Surrogate advantage (unclipped)*
24. &emsp;&emsp;&emsp;$\boldsymbol{\rho} \gets \exp(\texttt{clamp}(\log\pi_\theta - \log\boldsymbol{\pi}_{\text{old}_\mathcal{B}},\; -20, 20))$
25. &emsp;&emsp;&emsp;$L^{\text{surr}} \gets \texttt{mean}(\boldsymbol{\rho} \odot \hat{\mathbf{A}}_\mathcal{B})$
26. &emsp;&emsp;&emsp;*// MMD trust region penalty*
27. &emsp;&emsp;&emsp;$\{\mathbf{x}_1, \mathbf{x}_2\} \sim \text{TanhNormal}(\boldsymbol{\mu}_{\text{old}_\mathcal{B}}, \boldsymbol{\sigma}_{\text{old}_\mathcal{B}})$ — $P/2$ pairs
28. &emsp;&emsp;&emsp;$\{\mathbf{y}_1, \mathbf{y}_2\} \sim \pi_\theta(\cdot \mid \mathbf{S}_\mathcal{B})$ — $P/2$ pairs
29. &emsp;&emsp;&emsp;$k(\mathbf{a}, \mathbf{b}) = \exp(-\|\mathbf{a} - \mathbf{b}\|^2 / (2\sigma^2))$ — RBF kernel
30. &emsp;&emsp;&emsp;$\widehat{\text{MMD}}^2 \gets \texttt{mean}(k(\mathbf{x}_1, \mathbf{x}_2) + k(\mathbf{y}_1, \mathbf{y}_2) - k(\mathbf{x}_1, \mathbf{y}_2) - k(\mathbf{x}_2, \mathbf{y}_1))$
31. &emsp;&emsp;&emsp;*// Mirror descent KL regularizer*
32. &emsp;&emsp;&emsp;$D_{\text{KL}} \gets \texttt{mean}((\boldsymbol{\rho} - 1) - \log\boldsymbol{\rho})$ — approx. $\text{KL}(\pi_{\text{old}} \| \pi_\theta)$
33. &emsp;&emsp;&emsp;*// Value baseline*
34. &emsp;&emsp;&emsp;$L^V \gets \texttt{mean}((V_\phi(\mathbf{S}_\mathcal{B}) - \hat{\mathbf{R}}_\mathcal{B})^2)$
35. &emsp;&emsp;&emsp;*// Combined MM-RKHS objective (ascent on surr, descent on penalties)*
36. &emsp;&emsp;&emsp;$L \gets -L^{\text{surr}} + \beta \cdot \widehat{\text{MMD}}^2 + \tfrac{1}{\eta} D_{\text{KL}} + c_V \cdot L^V$
37. &emsp;&emsp;&emsp;$\theta, \phi \gets \theta, \phi - \alpha \nabla_{\theta, \phi} L$
38. &emsp;&emsp;**End For**
39. &emsp;**End For**
40. **End For**

---

# Part III — LLM Policy Optimization

---

## Algorithm 8: GRPO — Practical Vectorized Implementation

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
2. &emsp;*// Phase 1: Batch generation*
3. &emsp;**For** each query batch $\mathbf{Q}_{1:B}$
4. &emsp;&emsp;$\mathbf{O}_{:,:,:} \sim \pi_{\theta_k}(\cdot \mid \mathbf{Q})$ — $B \cdot G$ completions in parallel
5. &emsp;&emsp;$\mathbf{R}_{:,:} \gets R(\mathbf{Q}, \mathbf{O})$ — batch reward evaluation
6. &emsp;&emsp;$\boldsymbol{\pi}_{\text{old}} \gets \pi_{\theta_k}(\mathbf{O} \mid \mathbf{Q})$
7. &emsp;&emsp;$\mathbf{M} \gets \texttt{token\_mask}(\mathbf{O})$ — 1 where valid, 0 for padding
8. &emsp;**End For**
9. &emsp;*// Phase 2: Group-relative advantage*
10. &emsp;$\boldsymbol{\mu} \gets \texttt{mean}(\mathbf{R}, \text{dim}{=}G)$ — $\in \mathbb{R}^{B}$
11. &emsp;$\boldsymbol{\sigma} \gets \texttt{std}(\mathbf{R}, \text{dim}{=}G)$ — $\in \mathbb{R}^{B}$
12. &emsp;$\hat{\mathbf{A}} \gets (\mathbf{R} - \boldsymbol{\mu}) / \max(\boldsymbol{\sigma}, \varepsilon_{\min})$ — broadcast over $G$
13. &emsp;*// Phase 3: Minibatch SGD*
14. &emsp;Flatten $\mathcal{D}_k \gets$ reshape to $(B \cdot G, L)$ completions
15. &emsp;**For** epoch $= 1, \ldots, K$
16. &emsp;&emsp;$\mathcal{I} \gets \texttt{randperm}(B \cdot G)$ — shuffle completions
17. &emsp;&emsp;**For** $b = 0, M, 2M, \ldots$ — $\lceil BG/M \rceil$ minibatches
18. &emsp;&emsp;&emsp;$\mathcal{B} \gets \mathcal{I}[b : b{+}M]$
19. &emsp;&emsp;&emsp;$\boldsymbol{\rho} \gets \pi_\theta(\mathbf{O}_\mathcal{B} \mid \mathbf{Q}_\mathcal{B}) / \boldsymbol{\pi}_{\text{old}_\mathcal{B}}$
20. &emsp;&emsp;&emsp;$\mathbf{L}_{\text{tok}} \gets \min(\boldsymbol{\rho} \odot \hat{\mathbf{A}}_\mathcal{B},\; \texttt{clip}(\boldsymbol{\rho}, 1{-}\epsilon, 1{+}\epsilon) \odot \hat{\mathbf{A}}_\mathcal{B})$
21. &emsp;&emsp;&emsp;$L^{\text{GRPO}} \gets \texttt{sum}(\mathbf{L}_{\text{tok}} \odot \mathbf{M}_\mathcal{B}) / \texttt{sum}(\mathbf{M}_\mathcal{B})$ — masked mean
22. &emsp;&emsp;&emsp;$\mathbf{r}_{\text{ref}} \gets \pi_\theta(\mathbf{O}_\mathcal{B} \mid \cdot) / \pi_{\text{ref}}(\mathbf{O}_\mathcal{B} \mid \cdot)$
23. &emsp;&emsp;&emsp;$D_{\text{KL}} \gets \texttt{mean}((\mathbf{r}_{\text{ref}} - \log \mathbf{r}_{\text{ref}} - 1) \odot \mathbf{M}_\mathcal{B})$
24. &emsp;&emsp;&emsp;$\theta \gets \theta + \alpha \nabla_\theta (L^{\text{GRPO}} - \beta D_{\text{KL}})$
25. &emsp;&emsp;**End For**
26. &emsp;**End For**
27. **End For**

---

## Algorithm 9: CISPO — Practical Vectorized Implementation

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
2. &emsp;*// Phase 1: Batch generation (same as GRPO)*
3. &emsp;**For** each query batch $\mathbf{Q}_{1:B}$
4. &emsp;&emsp;$\mathbf{O}_{:,:,:} \sim \pi_{\theta_k}(\cdot \mid \mathbf{Q})$ — $B \cdot G$ completions in parallel
5. &emsp;&emsp;$\mathbf{R}_{:,:} \gets R(\mathbf{Q}, \mathbf{O})$ — batch reward evaluation
6. &emsp;&emsp;$\boldsymbol{\pi}_{\text{old}} \gets \pi_{\theta_k}(\mathbf{O} \mid \mathbf{Q})$
7. &emsp;&emsp;$\mathbf{M} \gets \texttt{token\_mask}(\mathbf{O})$ — 1 where valid, 0 for padding
8. &emsp;**End For**
9. &emsp;*// Phase 2: Group-relative advantage (same as GRPO)*
10. &emsp;$\boldsymbol{\mu} \gets \texttt{mean}(\mathbf{R}, \text{dim}{=}G)$ — $\in \mathbb{R}^{B}$
11. &emsp;$\boldsymbol{\sigma} \gets \texttt{std}(\mathbf{R}, \text{dim}{=}G)$ — $\in \mathbb{R}^{B}$
12. &emsp;$\hat{\mathbf{A}} \gets (\mathbf{R} - \boldsymbol{\mu}) / \max(\boldsymbol{\sigma}, \varepsilon_{\min})$ — broadcast over $G$
13. &emsp;*// Phase 3: Clipped IS weight SGD*
14. &emsp;Flatten $\mathcal{D}_k \gets$ reshape to $(B \cdot G, L)$ completions
15. &emsp;**For** epoch $= 1, \ldots, K$
16. &emsp;&emsp;$\mathcal{I} \gets \texttt{randperm}(B \cdot G)$
17. &emsp;&emsp;**For** $b = 0, M, 2M, \ldots$ — $\lceil BG/M \rceil$ minibatches
18. &emsp;&emsp;&emsp;$\mathcal{B} \gets \mathcal{I}[b : b{+}M]$
19. &emsp;&emsp;&emsp;$\boldsymbol{\rho} \gets \pi_\theta(\mathbf{O}_\mathcal{B} \mid \mathbf{Q}_\mathcal{B}) / \boldsymbol{\pi}_{\text{old}_\mathcal{B}}$
20. &emsp;&emsp;&emsp;$\hat{\boldsymbol{\rho}} \gets \texttt{clamp}(\boldsymbol{\rho}, \max = 1 + \epsilon_{\text{high}})$
21. &emsp;&emsp;&emsp;$\bar{\boldsymbol{\rho}} \gets \texttt{stop\_grad}(\hat{\boldsymbol{\rho}})$ — detach from graph
22. &emsp;&emsp;&emsp;$\mathbf{L}_{\text{tok}} \gets \bar{\boldsymbol{\rho}} \odot \hat{\mathbf{A}}_\mathcal{B} \odot \log \pi_\theta(\mathbf{O}_\mathcal{B} \mid \mathbf{Q}_\mathcal{B})$
23. &emsp;&emsp;&emsp;$L^{\text{CISPO}} \gets \texttt{sum}(\mathbf{L}_{\text{tok}} \odot \mathbf{M}_\mathcal{B}) / \texttt{sum}(\mathbf{M}_\mathcal{B})$ — masked mean
24. &emsp;&emsp;&emsp;$\theta \gets \theta + \alpha \nabla_\theta L^{\text{CISPO}}$
25. &emsp;&emsp;**End For**
26. &emsp;**End For**
27. **End For**

### Why CISPO: Fixing the Gradient Dead Zone

**The PPO/GRPO problem — clipped tokens receive zero gradient:**
In PPO and GRPO, the clipped surrogate $\min(\rho_t \hat{A}_t,\; \text{clip}(\rho_t, 1{-}\epsilon, 1{+}\epsilon) \hat{A}_t)$ kills the gradient for any token where $\rho_t > 1{+}\epsilon$ and $\hat{A}_t > 0$. The policy improved on that token — but once the ratio exceeds the clip boundary, learning stops entirely. Over multiple epochs, more and more tokens fall into this dead zone, wasting compute on samples that produce $\nabla_\theta = \mathbf{0}$.

**CISPO's fix — decouple the weight from the gradient path:**
1. **Clamp** the ratio $\hat{\rho} = \text{clamp}(\rho, \max = 1{+}\epsilon_h)$ — bounding the magnitude of the update, same safety goal as PPO's clip.
2. **Detach** the clamped weight: $\bar{\rho} = \texttt{stop\_grad}(\hat{\rho})$ — the weight scales the loss but does not appear in the gradient graph.
3. **Multiply through $\log\pi$**: the loss becomes $\bar{\rho} \cdot \hat{A} \cdot \log\pi_\theta$, so the gradient always flows through $\log\pi_\theta$ regardless of how large $\rho$ is.

The result: every token contributes a nonzero gradient at every epoch. The detached weight $\bar{\rho}$ still down-weights tokens that have moved far from the old policy (safety), but unlike PPO/GRPO it never completely shuts off learning. This eliminates the "wasted epochs" problem and allows CISPO to extract more signal from each batch.

---

# Appendices

---

## Key Differences: All Nine Algorithms

**A2C** *(Mnih et al., 2016):*
- $L^\pi = -\log\pi(a|s) \cdot \hat{A}$ — vanilla policy gradient with baseline (descent)
- Advantage via *learned critic* $V_\phi$ ($n$-step returns). On-policy, single environment.
- Simplest actor-critic; no trust region, no replay buffer.

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
- Advantage via *learned critic* $V_\phi$ (GAE). On-policy with *rollout buffer*.
- When $\rho_t > 1{+}\epsilon$ and $\hat{A}_t > 0$: $\nabla_\theta = \mathbf{0}$ *(gradient masked out).*

**MM-RKHS** *(Gupta & Mahajan, 2026):*
- $L = -\mathbb{E}[\rho \hat{A}] + \beta \cdot \widehat{\text{MMD}}^2 + \tfrac{1}{\eta} D_{\text{KL}}$ — majorization-minimization (descent)
- Advantage via *learned critic* $V_\phi$ (GAE). On-policy with *rollout buffer*.
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
