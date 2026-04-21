---
name: Periodic Pattern Learning for Surrogate Models
description: Research on how neural network surrogates learn periodic/oscillatory patterns — Fourier features, periodic activations (SIREN, Snake), spectral bias mitigation, and recommendations for the Cosserat rod CPG surrogate
type: knowledge
created: 2026-03-16
updated: 2026-03-16
tags:
  - surrogate-model
  - periodic-patterns
  - fourier-features
  - spectral-bias
  - siren
  - snake-activation
  - cpg
  - neural-network
  - architecture
aliases:
  - periodic surrogate research
  - fourier feature encoding research
  - spectral bias mitigation
---

# Periodic Pattern Learning for Surrogate Models

Research on how neural network surrogates can effectively learn periodic/oscillatory dynamics, specifically for replacing a Cosserat rod physics simulation driven by a CPG (Central Pattern Generator) traveling wave.

---

## Table of Contents

1. [Executive Summary and Recommendations](#1-executive-summary-and-recommendations)
2. [The Core Challenge: Periodic Forcing in the Surrogate](#2-the-core-challenge-periodic-forcing-in-the-surrogate)
3. [The Spectral Bias Problem](#3-the-spectral-bias-problem)
4. [Fourier Feature Encoding](#4-fourier-feature-encoding)
5. [Periodic Activation Functions](#5-periodic-activation-functions)
6. [Does the Surrogate Need Explicit Time?](#6-does-the-surrogate-need-explicit-time)
7. [Analysis of the Existing Per-Element Phase Encoding](#7-analysis-of-the-existing-per-element-phase-encoding)
8. [Neural ODEs and FNOs: Relevance Assessment](#8-neural-odes-and-fnos-relevance-assessment)
9. [Frequency Generalization and Extrapolation](#9-frequency-generalization-and-extrapolation)
10. [Concrete Recommendations for This Surrogate](#10-concrete-recommendations-for-this-surrogate)
11. [Implementation Guidance](#11-implementation-guidance)
12. [Experiment Priority Order](#12-experiment-priority-order)
13. [Sources](#13-sources)

---

## 1. Executive Summary and Recommendations

### The Problem

The surrogate network maps (s_t, a_t) in R^129 to s_{t+1} in R^124, replacing 500 Verlet substeps of Cosserat rod simulation. During those 500 substeps, the CPG produces a *time-varying* rest curvature:

```
kappa_v^rest(t) = A * sin(2*pi*k*s_v + 2*pi*f*t + phi_0) + b
```

where (A, f, k, phi_0, b) are the RL action parameters. The frequency f ranges from 0.5 to 3.0 Hz, and the control interval is 0.5s, meaning the CPG completes between 0.25 and 1.5 full oscillation cycles per RL step. The surrogate must learn the *integrated effect* of this oscillatory forcing across qualitatively different frequency regimes.

### Primary Finding

**The existing per-element phase encoding (60-dim) already provides the critical periodic information the network needs.** It encodes sin(k*s_j + omega*t + phi), cos(k*s_j + omega*t + phi), and A*sin(k*s_j + omega*t + phi) for each of 20 elements. This is structurally identical to a hand-crafted Fourier feature encoding of the CPG phase at each spatial location.

### Recommendations (Ordered by Expected Impact)

1. **Keep the existing 60-dim per-element phase encoding.** It is already a well-designed Fourier feature encoding of the CPG state. Do not replace it.

2. **Add Fourier features of the raw action parameters (f, k).** The frequency f and wavenumber k are the parameters that most strongly control the qualitative behavior of the integrated dynamics. Apply random Fourier features (RFF) to these two dimensions specifically to help the network learn the frequency-dependent integrated response. This is the highest-impact change. See Section 11 for implementation.

3. **Do NOT switch to SIREN/periodic activations.** The periodic information is already in the input encoding. SIREN's advantage is for learning unknown periodic structure from raw coordinates; here, the periodicity is *known* and explicitly encoded. SiLU is a good activation for this regression task. The risk-reward ratio of switching is unfavorable.

4. **Do NOT pursue FNO or Neural ODE for this task.** The surrogate is a fixed-step, finite-dimensional map (R^129 -> R^124). Neither FNO (designed for function-to-function maps on spatial grids) nor Neural ODE (designed for continuous-time evolution) provides a meaningful advantage over a well-designed MLP for this specific problem.

5. **Consider multi-frequency Fourier features of the CPG phase** as an enhancement to recommendation 2. Instead of single-frequency sin/cos at each element, add harmonics (2x, 3x the fundamental) to help the network represent the nonlinear response to the forcing.

**Confidence: HIGH** for recommendations 1-4. MEDIUM for recommendation 5 (needs experimental validation).

---

## 2. The Core Challenge: Periodic Forcing in the Surrogate

### What the Network Must Learn

The surrogate sees the action a = (A, f, k, phi_0, b) as a fixed input and must predict the *result* of 500 substeps where the rest curvature changes at every substep due to the 2*pi*f*t term. This is equivalent to learning:

```
s_{t+1} = T(s_t, A, f, k, phi_0, b)
```

where T is the composition of 500 distinct Verlet substeps, each with a different rest curvature configuration. The key challenge is that the behavior of T depends *qualitatively* on f and k:

| Frequency f | Cycles in dt=0.5s | Dynamics Character |
|-------------|--------------------|--------------------|
| 0.5 Hz | 0.25 | Slow sweep — rod barely completes a quarter-wave |
| 1.0 Hz | 0.5 | Half-cycle — rod swings one direction |
| 2.0 Hz | 1.0 | Full cycle — rod returns near initial config |
| 3.0 Hz | 1.5 | One-and-a-half cycles — rod reverses |

At f=2.0 Hz, the omega_z (angular velocity) nearly returns to its initial value because the forcing completes a full cycle. At f=3.0 Hz (1.5 cycles), the omega_z is at the *opposite* sign from its initial value. This was identified in `issues/surrogate-omega-z-poor-prediction.md` as the root cause of poor omega_z prediction (R^2=0.23).

### Why This Is Fundamentally a Frequency-Domain Problem

The mapping from f to Delta_omega_z is itself periodic/oscillatory:
- Delta_omega_z ~ 0 when f*dt is near an integer (full cycles)
- Delta_omega_z is maximal when f*dt is near a half-integer

An MLP with standard activations (ReLU, SiLU) learning this relationship must approximate a sinusoidal function of f. This is where spectral bias becomes relevant.

---

## 3. The Spectral Bias Problem

### Definition

**Spectral bias** (also called the "frequency principle") is the well-documented tendency of neural networks to learn low-frequency components of a target function before high-frequency components. First characterized by Rahaman et al. (ICML 2019), it means:

- Networks with standard activations (ReLU, tanh, SiLU) preferentially fit smooth, slowly-varying functions
- High-frequency patterns in the target require more training time, more parameters, or may never be learned
- The root cause is that the Neural Tangent Kernel (NTK) of standard MLPs has a rapid frequency falloff, effectively filtering out high-frequency information

**Confidence: HIGH** -- This is one of the most well-established results in deep learning theory, confirmed by multiple independent groups.

### Relevance to This Surrogate

The surrogate's target function has frequency-dependent behavior: the mapping from action parameters to state delta contains oscillatory dependence on f and k. Specifically:

1. **Delta_omega_z as a function of f** is approximately sinusoidal (see Section 2). An MLP with SiLU activation can represent this, but may learn it slowly or imprecisely.

2. **The per-element phase encoding already lifts the CPG phase into sinusoidal features.** This is the equivalent of Fourier feature encoding for the spatial-temporal phase. The spectral bias problem for the phase dimension is already addressed.

3. **The remaining spectral bias risk is in the f and k dimensions of the action space.** The network receives f and k as raw scalars (normalized to [-1, 1]). It must learn that the integrated response is a periodic function of f*dt. This is where Fourier features would help most.

### Severity Assessment

**MEDIUM.** The existing per-element phase encoding already solves the dominant source of spectral bias (the CPG phase). The residual spectral bias in the frequency/wavenumber dimensions is a secondary effect that may explain the poor omega_z R^2 but can be addressed with targeted Fourier features on the action parameters.

---

## 4. Fourier Feature Encoding

### Theory (Tancik et al., NeurIPS 2020)

The core idea: map low-dimensional input coordinates through sinusoidal functions before feeding to the MLP:

```
gamma(v) = [cos(2*pi*B*v), sin(2*pi*B*v)]
```

where B in R^{m x d} is a matrix of frequency coefficients, v in R^d is the input, and the output is 2m-dimensional. Each row of B defines a frequency direction in input space.

**Two variants:**

| Variant | B matrix | Properties |
|---------|----------|------------|
| **Gaussian RFF** | B_ij ~ N(0, sigma^2) | Isotropic, tunable bandwidth via sigma |
| **Deterministic** | B_j = sigma^{j/m} for j=0..m-1 | Log-linear spacing, covers wide frequency range |

The bandwidth parameter sigma controls the frequency range: small sigma yields low-frequency features (smooth, but may underfit), large sigma yields high-frequency features (expressive, but may overfit). The optimal sigma depends on the target function's frequency content.

**Confidence: HIGH** -- This is a widely-replicated result with clear theoretical backing from NTK analysis.

### Practical Guidance

| Parameter | Recommendation | Rationale |
|-----------|----------------|-----------|
| Number of features m | 32-128 per input dimension | Diminishing returns beyond ~128; 64 is a good default |
| Bandwidth sigma | Hyperparameter sweep over [1, 3, 10, 30] | Depends on target function's frequency content |
| Distribution | Gaussian RFF | More robust than deterministic for unknown frequency spectra |
| Trainable | Start with fixed, try learnable later | Fixed is simpler and often sufficient |

### Application to THIS Surrogate

The existing per-element phase encoding is already a form of *deterministic* Fourier feature encoding:

```python
# Existing: per-element phase at spatial location s_j
phase_j = k * s_j + omega * t + phi  # scalar phase angle
features = [sin(phase_j), cos(phase_j), A * sin(phase_j)]  # 3 features per element
```

This encodes the CPG phase at each element. But it encodes only the *fundamental frequency* of the CPG. The integrated effect of 500 substeps may depend on *harmonics* and *cross-frequency terms* that a single sin/cos pair cannot capture.

**What is NOT Fourier-encoded in the current system:**
1. The raw frequency f (received as a normalized scalar)
2. The raw wavenumber k (received as a normalized scalar)
3. The relationship f * dt_ctrl (how many oscillation cycles fit in the control interval)
4. Higher harmonics of the CPG phase

---

## 5. Periodic Activation Functions

### SIREN (Sitzmann et al., NeurIPS 2020)

SIREN replaces all activation functions with sin(omega_0 * x) where omega_0 is a hyperparameter controlling the frequency of the activation. Key properties:

- The derivative of a SIREN layer is another SIREN layer (cosine is shifted sine)
- Requires special initialization (omega_0 typically set to 30, first-layer weights uniform in [-1/n, 1/n], deeper layers uniform in [-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0])
- Excels at representing signals where derivatives must be accurate (PDEs, signed distance functions)
- Converges faster than ReLU for smooth, multi-scale signals

**Confidence: HIGH** -- Well-established in neural implicit representations.

### Snake Activation (Ziyin et al., NeurIPS 2020)

Defined as: Snake_a(x) = x + sin^2(a*x) / a

Key properties:
- Combines ReLU-like monotonicity (the x term) with periodic inductive bias (the sin^2 term)
- Trainable parameter a controls the frequency; a in [5, 50] for known-periodic data, a ~ 0.2 for non-periodic
- Maintains favorable optimization properties of ReLU-based networks
- The sin^2 term has a non-vanishing second-order expansion, giving better approximation than x + sin(x)
- Designed for learning *periodic functions* that should extrapolate beyond the training domain

**Confidence: HIGH** -- Directly addresses the "neural networks fail to learn periodic functions" problem.

### Why Periodic Activations Are NOT Recommended Here

1. **The periodicity is already encoded in the input.** The 60-dim per-element phase encoding explicitly provides sin and cos features. The network does not need to *discover* periodicity from raw inputs -- it is given periodic features directly.

2. **The network's primary task is regression on state deltas.** The mapping from (body shape + CPG phase features) to (body shape change) is not itself a periodic function. It is a complex, nonlinear mapping where the periodic structure has already been factored out via the phase encoding.

3. **SIREN has known issues with non-periodic targets.** SIREN can produce ringing artifacts when the target contains both periodic and non-periodic components. The state delta prediction has many non-periodic components (position drift, velocity changes from friction, etc.).

4. **Snake's advantage is extrapolation.** Snake excels when the network must extrapolate periodic behavior beyond the training domain. But the surrogate operates within a bounded action space and state distribution -- extrapolation is not the primary concern (the body-frame representation work addresses generalization more fundamentally).

5. **Switching activations introduces risk with minimal expected gain.** SiLU is well-understood, stable, and works with the existing LayerNorm + residual block architecture. SIREN and Snake require different initialization schemes, different learning rates, and may interact poorly with LayerNorm.

**Recommendation: Keep SiLU.** The investment in periodic activations has poor risk-reward for this specific problem because the periodicity is in the *input encoding*, not the *activation function*.

---

## 6. Does the Surrogate Need Explicit Time?

### The Question

The CPG rest curvature depends on time t via the 2*pi*f*t term. Does the surrogate need to know what time it is, or is it sufficient to provide just (s_t, a_t)?

### Analysis

The transition operator T(s_t, a_t) integrates the rod dynamics from t_step to t_step + dt_ctrl. The CPG phase at the *start* of the step is:

```
phi_start = 2*pi*f*t_step + phi_0
```

And the phase *within* the step sweeps from phi_start to phi_start + 2*pi*f*dt_ctrl. The integrated effect depends on:

1. **phi_start mod 2*pi** -- where in the oscillation cycle the step begins
2. **2*pi*f*dt_ctrl** -- how many cycles are traversed during the step
3. **The rod state s_t** -- how the rod's current configuration responds to the forcing

Item (2) is determined entirely by f, which is in the action. Item (3) is the state. Item (1) is the *only* quantity that requires knowing time.

The existing per-element phase encoding provides phi_start implicitly:

```python
phase_j = k * s_j + omega * t_step + phi_0  # this IS the CPG phase at step start
```

So **the per-element phase encoding already provides the time information the network needs.** The raw time t is not needed as a separate input because its only role is determining the CPG phase at the step start, which is already encoded.

### Verification

The model input is (state_124, action_5, per_element_phase_60) = 189 dimensions. The per-element phase at element j encodes sin(k*s_j + omega*t + phi) and cos(k*s_j + omega*t + phi). From these 40 sin/cos features across 20 elements, the network can reconstruct omega*t + phi (modulo the spatial variation k*s_j, which is deterministic given k and the fixed element positions). It therefore has full information about the CPG phase at the step start.

**Conclusion: The surrogate does NOT need explicit time as a separate input.** The per-element phase encoding is sufficient. The old 2-dim [sin(omega*t), cos(omega*t)] encoding was a less informative predecessor of the current 60-dim per-element encoding.

**Confidence: HIGH** -- Follows directly from the CPG equation and the per-element phase encoding formula.

---

## 7. Analysis of the Existing Per-Element Phase Encoding

### What It Provides

The current 60-dim per-element phase encoding (from `state.py`) computes for each of 20 elements:

```python
phase_j = 2*pi*k * s_j + 2*pi*f * t + phi_0   # CPG phase at element j
sin(phase_j)       # 20 features: oscillation phase
cos(phase_j)       # 20 features: oscillation phase (quadrature)
A * sin(phase_j)   # 20 features: commanded rest curvature kappa_j^rest
```

### Strengths

1. **Spatially resolved.** Each element gets its own phase, capturing the traveling wave structure. This was identified as essential in `issues/surrogate-omega-z-poor-prediction.md`.

2. **Sin/cos pair avoids phase wrapping.** The (sin, cos) representation is continuous and avoids the discontinuity at 2*pi.

3. **Commanded curvature is explicit.** The A * sin(phase_j) features directly provide kappa_rest at each element, which is the actual forcing term in the constitutive law: tau = B * (kappa - kappa_rest).

4. **Structurally equivalent to Fourier features.** This encoding IS a hand-crafted Fourier feature encoding of the CPG state at each spatial location, with exactly the right frequency (the CPG fundamental).

### Limitations

1. **Only the fundamental frequency.** The encoding provides sin and cos at the CPG fundamental frequency only. The *integrated response* of 500 nonlinear substeps generates harmonics -- the rod's actual curvature at step end is not a pure sinusoid even if the forcing is. Cross-frequency terms (e.g., sin(2*phase_j), sin(phase_j)*sin(phase_k)) are absent.

2. **No explicit representation of f*dt_ctrl.** The number of oscillation cycles in the control interval (f*dt_ctrl) is a critical quantity that determines whether omega_z increases, decreases, or stays the same (see Section 2). This quantity is implicitly available from f (in the action) and dt_ctrl (constant = 0.5s), but not explicitly encoded.

3. **Phase at step START only.** The encoding provides the CPG phase at the *beginning* of the RL step. But the integrated response depends on the phase *trajectory* over 500 substeps. The phase sweeps from phase_j(0) to phase_j(0) + 2*pi*f*dt_ctrl. This sweep information must be inferred from f.

### Verdict

The existing encoding is good but has a specific gap: the network must learn the relationship between f and the integrated oscillatory response purely from the raw frequency scalar. Fourier-encoding f would directly address this.

---

## 8. Neural ODEs and FNOs: Relevance Assessment

### Neural ODEs

**Relevance: LOW for this specific surrogate task.**

Neural ODEs learn continuous-time dynamics dx/dt = f_theta(x, t) and integrate using ODE solvers. They are appropriate when:
- The time horizon varies (ours is fixed at 0.5s)
- You want adaptive step sizes (our substeps are fixed at 0.001s)
- You need O(1) memory training (our MLP is already memory-efficient)

For our fixed-step, fixed-horizon surrogate mapping, a Neural ODE adds computational overhead (ODE solver in the forward pass) without corresponding benefit. The surrogate explicitly avoids stepwise integration -- it replaces 500 steps with a single forward pass.

Where Neural ODE IS relevant: the KNODE-Cosserat hybrid approach described in `knowledge/knode-cosserat-hybrid-surrogate-report.md`, which replaces *individual substeps* rather than the full 500-step composition. That is a different design choice with different tradeoffs.

### Fourier Neural Operators

**Relevance: LOW for this specific surrogate task.**

FNO learns mappings between function spaces, operating on spatial grids via spectral convolutions. It excels for parametric PDEs where:
- The input/output are functions on a spatial domain
- Resolution invariance is needed
- The PDE is defined on a structured grid

Our surrogate maps between *finite-dimensional vectors* (R^129 -> R^124), not functions. The 20-element rod is already discretized at a fixed resolution. FNO would be overkill and architecturally mismatched.

Where FNO IS relevant: if we wanted to learn the *spatial* relationship between CPG parameters and the curvature field kappa(s) as a function of arc length s, and needed resolution invariance. This is not our use case.

**Confidence: HIGH** for both assessments. These are architectural mismatches, not judgment calls.

---

## 9. Frequency Generalization and Extrapolation

### The Problem

The RL policy will explore the full frequency range [0.5, 3.0] Hz during training. The surrogate must generalize within this range. Key concern: does the surrogate's accuracy degrade at certain frequencies?

### Known Failure Modes

1. **Neural networks with standard activations fail to extrapolate periodic functions** (Ziyin et al., NeurIPS 2020). Outside the training distribution, ReLU extrapolates linearly, tanh extrapolates to a constant, and SiLU extrapolates approximately linearly. None produce periodic extrapolation.

2. **Within the training range, interpolation works.** Standard MLPs can interpolate well between training frequencies. The risk is not generalization to *unseen* frequencies (the range is fixed) but rather learning the *correct functional relationship* between frequency and integrated response.

3. **Aliasing at high frequencies.** At f=3.0 Hz, the CPG completes 1.5 cycles per step. The omega_z signal reverses sign 2-3 times. The network must predict the *net* effect of these reversals. This is fundamentally a resolution/bandwidth issue, not an architecture issue.

### Mitigation Strategy

For this bounded, in-distribution task, the key is **giving the network the right features**, not changing the architecture:

1. Fourier-encode f and k so the network can easily learn the oscillatory dependence on these parameters
2. Ensure training data covers the full frequency range densely (already done)
3. Accept that omega_z prediction at high frequencies will have inherently higher uncertainty due to the substepping aliasing

**Confidence: HIGH** that within-distribution generalization is sufficient for RL training; MEDIUM that omega_z R^2 can be significantly improved beyond the 0.23 baseline, because the aliasing problem (1.5 cycles per step) is fundamental.

---

## 10. Concrete Recommendations for This Surrogate

### Recommendation 1: Add Fourier Features of (f, k) -- HIGH priority

The frequency f and wavenumber k are the action parameters that most strongly control the periodic dynamics. Add random Fourier features of these two parameters:

```python
# For f (frequency) and k (wavenumber), both in [-1, 1] normalized
v = [f_norm, k_norm]  # (2,)
gamma(v) = [sin(2*pi*B*v), cos(2*pi*B*v)]  # (2*m,) where m=32
```

with B in R^{32 x 2} sampled from N(0, sigma^2). Sweep sigma in {1, 3, 10, 30}.

Expected input dimension change: 189 + 64 = 253 (adding 64 Fourier features). This is a modest increase that targets the specific weakness identified in the omega_z analysis.

### Recommendation 2: Add Harmonic Phase Features -- MEDIUM priority

Extend the per-element phase encoding with second and third harmonics:

```python
# Current: sin(phase_j), cos(phase_j), A*sin(phase_j)  -> 60 features
# Proposed: add sin(2*phase_j), cos(2*phase_j), sin(3*phase_j), cos(3*phase_j) -> +80 features
```

This helps the network represent nonlinear responses to the sinusoidal forcing. Total per-element phase features: 60 + 80 = 140. Total input: 189 + 80 = 269.

Run this as a separate experiment from Recommendation 1 to isolate the effect.

### Recommendation 3: Encode f*dt_ctrl Explicitly -- LOW-MEDIUM priority

Add a derived feature: n_cycles = f * dt_ctrl (number of oscillation cycles in the control interval). Encode as sin(2*pi*n_cycles) and cos(2*pi*n_cycles). This directly provides the "how many cycles" information that determines whether omega_z reverses.

This costs only 2 additional input dimensions and directly targets the omega_z prediction problem.

### What NOT to Do

- Do NOT replace SiLU with sin activations (SIREN) or Snake activations
- Do NOT switch to FNO or Neural ODE architecture
- Do NOT remove the existing per-element phase encoding
- Do NOT add Fourier features to ALL input dimensions (124-dim state) -- this would massively expand the input dimension without targeting the actual bottleneck

---

## 11. Implementation Guidance

### Fourier Feature Module (PyTorch)

```python
import math
import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    """Random Fourier feature encoding for low-dimensional inputs.

    Maps input v in R^d to gamma(v) in R^{2m} via:
        gamma(v) = [sin(2*pi*B*v), cos(2*pi*B*v)]

    where B in R^{m x d} ~ N(0, sigma^2).

    Args:
        input_dim: Dimension of input vector d.
        num_features: Number of random frequencies m (output is 2*m).
        sigma: Bandwidth of the Gaussian frequency distribution.
        trainable: If True, B is a trainable parameter (learnable frequencies).
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int = 32,
        sigma: float = 10.0,
        trainable: bool = False,
    ):
        super().__init__()
        B = torch.randn(num_features, input_dim) * sigma
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Encode input with random Fourier features.

        Args:
            v: (..., d) input tensor.

        Returns:
            (..., 2*m) Fourier feature tensor.
        """
        projection = 2 * math.pi * v @ self.B.T  # (..., m)
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)

    @property
    def output_dim(self) -> int:
        return 2 * self.B.shape[0]
```

### Integration with Existing Model

```python
class EnhancedSurrogateModel(nn.Module):
    """Surrogate model with Fourier feature encoding of frequency/wavenumber."""

    def __init__(self, config, ff_num_features=32, ff_sigma=10.0):
        super().__init__()

        # Fourier features for (f_norm, k_norm) from action
        self.ff_encoder = FourierFeatures(
            input_dim=2,              # f and k
            num_features=ff_num_features,
            sigma=ff_sigma,
        )

        # Adjust input dimension: original 189 + 2*ff_num_features
        extended_input_dim = config.input_dim + self.ff_encoder.output_dim

        # Build MLP with extended input dim
        # ... (same as SurrogateModel but with extended_input_dim)

    def forward(self, state, action, per_element_phase):
        # Extract f and k from action (indices 1 and 2)
        fk = action[..., 1:3]  # (B, 2) — normalized f and k
        fk_features = self.ff_encoder(fk)  # (B, 2*m)

        x = torch.cat([state, action, per_element_phase, fk_features], dim=-1)
        return self.mlp(x)
```

### Harmonic Phase Extension

```python
def encode_per_element_phase_with_harmonics(
    actions: torch.Tensor,
    t: torch.Tensor,
    n_harmonics: int = 3,
    freq_range: tuple = (0.5, 3.0),
) -> torch.Tensor:
    """Extended per-element phase encoding with harmonics.

    Returns (N, 20 * (2 * n_harmonics + 1)) tensor:
        For each element j, for each harmonic h=1..n_harmonics:
            sin(h * phase_j), cos(h * phase_j)
        Plus: A * sin(phase_j) (the commanded curvature, fundamental only)
    """
    # ... denormalize action components (same as existing) ...

    # Phase angles: (N, 20)
    phase_angles = k[:, None] * s[None, :] + (omega * t + phi)[:, None]

    features = []
    for h in range(1, n_harmonics + 1):
        features.append(torch.sin(h * phase_angles))  # (N, 20)
        features.append(torch.cos(h * phase_angles))  # (N, 20)

    # Commanded curvature (fundamental only)
    features.append(A[:, None] * torch.sin(phase_angles))  # (N, 20)

    return torch.cat(features, dim=-1).float()  # (N, 20 * (2*n_harmonics + 1))
```

### Sigma Sweep Protocol

For the Fourier feature bandwidth sigma, run a sweep:

```
sigma values: [1.0, 3.0, 10.0, 30.0]
num_features: 32 (fixed)
metric: val R^2 on omega_z components (the weakest current prediction)
secondary: val MSE on all components
```

Expected outcome: sigma=10 or sigma=30 will likely work best because the target function (integrated oscillatory response) varies on the scale of the frequency range [0.5, 3.0] Hz, mapped to [-1, 1], so the characteristic frequency in normalized space is O(1-10).

---

## 12. Experiment Priority Order

| Priority | Experiment | Expected Gain | Effort | Risk |
|----------|-----------|---------------|--------|------|
| 1 | Fourier features on (f, k), sigma sweep | HIGH -- directly targets omega_z weakness | LOW -- add module, ~50 lines | LOW |
| 2 | Explicit n_cycles = f*dt feature, sin/cos encoded | MEDIUM -- 2 extra dims, targets aliasing | VERY LOW -- 2 lines | NONE |
| 3 | Harmonic phase features (2nd + 3rd harmonic) | MEDIUM -- captures nonlinear response | LOW -- extend existing function | LOW |
| 4 | Combined: FF on (f,k) + harmonics + n_cycles | HIGH if 1-3 individually help | LOW -- combine above | LOW |
| 5 | Body-frame representation + Fourier features | VERY HIGH (from separate research) | MEDIUM | MEDIUM |

Do NOT pursue (low priority, high cost):
- SIREN/Snake activation swap (LOW expected gain, HIGH risk)
- FNO architecture (WRONG abstraction for this problem)
- Neural ODE for full-step surrogate (WRONG paradigm)
- Fourier features on all 124 state dimensions (WRONG target, high cost)

---

## 13. Sources

### Primary (HIGH confidence)

- **Tancik, Srinivasan, Mildenhall et al. (NeurIPS 2020)** "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains." Establishes the theoretical basis for Fourier feature encoding and its connection to NTK bandwidth.
  - [arXiv:2006.10739](https://arxiv.org/abs/2006.10739)
  - [Project page](https://bmild.github.io/fourfeat/)
  - [GitHub](https://github.com/tancik/fourier-feature-networks)

- **Sitzmann, Martel, Bergman, Lindell, Wetzstein (NeurIPS 2020)** "Implicit Neural Representations with Periodic Activation Functions." Defines SIREN architecture with sin activations.
  - [arXiv:2006.09661](https://arxiv.org/abs/2006.09661)
  - [Project page](https://www.vincentsitzmann.com/siren/)
  - [GitHub](https://github.com/vsitzmann/siren)

- **Ziyin, Hartwig, Ueda (NeurIPS 2020)** "Neural Networks Fail to Learn Periodic Functions and How to Fix It." Defines Snake activation x + sin^2(ax)/a.
  - [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/file/1160453108d3e537255e9f7b931f4e90-Paper.pdf)
  - [arXiv:2006.08195](https://arxiv.org/abs/2006.08195)

- **Rahaman et al. (ICML 2019)** "On the Spectral Bias of Neural Networks." First characterization of spectral bias.
  - [arXiv:1806.08734](https://arxiv.org/abs/1806.08734)
  - [ICML proceedings](https://proceedings.mlr.press/v97/rahaman19a.html)

### Secondary (MEDIUM confidence)

- **Spectral bias in physics-informed and operator learning** (arXiv 2025) -- Analysis and mitigation guidelines.
  - [arXiv:2602.19265](https://arxiv.org/html/2602.19265v1)

- **Neural Functions for Learning Periodic Signals** (arXiv 2025) -- NeRT architecture with learnable Fourier features.
  - [arXiv:2506.09526](https://arxiv.org/html/2506.09526v1)

- **Multi-Grade Deep Learning for Spectral Bias** (NeurIPS 2024) -- Composition of low-frequency functions to approximate high-frequency ones.
  - [OpenReview](https://openreview.net/forum?id=IoRT7EhFap)

- **Extrapolation of Periodic Functions Using Binary Encoding** (arXiv 2024) -- NB2E approach for periodic extrapolation.
  - [arXiv:2512.10817](https://arxiv.org/html/2512.10817)

### Project-Internal (HIGH confidence)

- `issues/surrogate-omega-z-poor-prediction.md` -- Documents R^2=0.23 on omega_z, identifies per-element phase as the fix (now implemented)
- `papers/aprx_model_elastica/state.py` -- Implementation of per-element phase encoding (60-dim)
- `papers/aprx_model_elastica/model.py` -- Current SurrogateModel and ResidualSurrogateModel architectures
- `report/system-formulation.tex` -- Full CPG equation and RL step dataflow
- `knowledge/surrogate-input-representation-research.md` -- Body-frame representation research
- `knowledge/neural-ode-pde-approximation-survey.md` -- Survey of Neural ODE, FNO, PINN approaches

---

## Confidence Assessment

| Finding | Confidence | Basis |
|---------|------------|-------|
| Spectral bias is real and affects MLPs on periodic targets | HIGH | Multiple papers, widely replicated |
| Existing per-element phase encoding addresses the dominant periodic input | HIGH | Direct analysis of the encoding vs CPG equation |
| Fourier features on (f, k) would help omega_z prediction | MEDIUM-HIGH | Theory strongly supports; needs experimental validation |
| SIREN/Snake not needed given existing phase encoding | HIGH | Follows from: periodicity is in input, not in target function |
| FNO/Neural ODE wrong for this problem | HIGH | Architectural mismatch analysis |
| Harmonic phase features would help | MEDIUM | Theoretical argument; benefit depends on nonlinearity strength |
| Sigma sweep [1, 3, 10, 30] covers the right range | MEDIUM | Estimated from characteristic frequency of f in normalized action space |
