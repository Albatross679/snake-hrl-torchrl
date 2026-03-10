---
name: Surrogate Model Mathematical Formulation
description: Formal mathematical definition of the neural surrogate approximating the PyElastica Cosserat rod transition operator
type: knowledge
created: 2026-03-10
updated: 2026-03-10
tags: [surrogate, mathematics, cosserat-rod, ode, formulation]
aliases: [surrogate-formulation, surrogate-math]
---

## Overview

The surrogate model $f_\theta$ approximates one control-step of the symplectic integration of the discretized 2D Cosserat rod PDE, including RFT friction contact forces and CPG-driven muscle moments.

## Ground-Truth Transition Operator

PyElastica integrates the 2D-projected Cosserat rod ODEs. Each RL control step (duration $\Delta t_\text{ctrl}$) runs $n_\text{sub}$ internal physics substeps. The resulting transition operator is:

$$T: (\mathbf{s}_t, \mathbf{a}_t) \mapsto \mathbf{s}_{t+1}$$

The underlying continuous PDEs (per unit arc length $s$) are:

**Linear momentum:**
$$\rho A \frac{\partial^2 \mathbf{x}}{\partial t^2} = \frac{\partial \mathbf{F}}{\partial s} + \mathbf{f}_\text{ext}$$

**Angular momentum:**
$$\rho \mathbf{I} \frac{\partial^2 \mathbf{d}}{\partial t^2} = \frac{\partial \mathbf{M}}{\partial s} + \mathbf{m}_\text{ext} + \mathbf{x}' \times \mathbf{F}$$

where:
- $\mathbf{x}(s,t)$ — centerline position along arc length $s$
- $\mathbf{d}(s,t)$ — director frame (orientation)
- $\mathbf{F}$ — internal force (shear + tension, from constitutive law)
- $\mathbf{M}$ — internal moment (bending + torsion, from constitutive law)
- $\mathbf{f}_\text{ext}$ — RFT anisotropic friction contact forces
- $\mathbf{m}_\text{ext}$ — CPG-driven muscle moments proportional to $(\kappa^\text{target} - \kappa^\text{actual})$

After spatial discretization into $N=21$ nodes ($N_e=20$ elements), this becomes a first-order ODE system integrated by a symplectic Euler stepper.

## State Space

$$\mathbf{s}_t \in \mathbb{R}^{124}$$

| Slice | Variable | Count | Physical meaning |
|---|---|---|---|
| `[0:21]` | $x_i$ | 21 | Node x-positions (m) |
| `[21:42]` | $y_i$ | 21 | Node y-positions (m) |
| `[42:63]` | $\dot{x}_i$ | 21 | Node x-velocities (m/s) |
| `[63:84]` | $\dot{y}_i$ | 21 | Node y-velocities (m/s) |
| `[84:104]` | $\psi_e$ | 20 | Element yaw: $\arctan2(\text{tang}_y, \text{tang}_x)$ (rad) |
| `[104:124]` | $\omega_{z,e}$ | 20 | Element angular velocity about z-axis (rad/s) |

Code: [aprx_model_elastica/state.py](../aprx_model_elastica/state.py) — `RodState2D`, slices at lines 26–40.

## Action Space

$$\mathbf{a}_t \in [-1, 1]^5$$

The raw action is mapped to physical CPG parameters that define the target serpenoid curvature wave:

$$\kappa^\text{target}(s, t) = A \sin\!\left(2\pi k \cdot s + 2\pi f \cdot t + \phi\right) + b$$

| Index | Parameter | Physical range | Role |
|---|---|---|---|
| 0 | Amplitude $A$ | $[0, 5]$ rad/m | Body bending magnitude |
| 1 | Frequency $f$ | $[0.5, 3.0]$ Hz | Oscillation rate |
| 2 | Wave number $k$ | $[0.5, 3.5]$ | Wavelengths per body length |
| 3 | Phase $\phi$ | $[0, 2\pi]$ rad | Initial wave offset |
| 4 | Turn bias $b$ | $[-2, 2]$ rad/m | Constant curvature for steering |

Code: [src/physics/cpg/action_wrapper.py](../src/physics/cpg/action_wrapper.py)

## Surrogate Model Definition

### Input

$$\mathbf{z}_t = \left[\bar{\mathbf{s}}_t \;\|\; \mathbf{a}_t \;\|\; \sin(\omega t),\, \cos(\omega t)\right] \in \mathbb{R}^{131}$$

where:
- $\bar{\mathbf{s}}_t = (\mathbf{s}_t - \mu_s) / \sigma_s$ — per-feature z-score normalized state
- $\omega = 2\pi f$ — angular frequency from action component 1
- $[\sin(\omega t), \cos(\omega t)]$ — oscillation phase encoding (not raw time), since the curvature wave is periodic in $\omega t$

### Output

The model predicts a **normalized state delta**:

$$f_\theta(\mathbf{z}_t) = \overline{\Delta \mathbf{s}} \in \mathbb{R}^{124}$$

Next-state reconstruction:

$$\hat{\mathbf{s}}_{t+1} = \mathbf{s}_t + \sigma_\Delta \odot \overline{\Delta \mathbf{s}} + \mu_\Delta$$

where $\mu_\Delta, \sigma_\Delta$ are the training-set mean and std of $(\mathbf{s}_{t+1} - \mathbf{s}_t)$.

### Architecture (baseline)

$$\text{Linear}(131) \to [\text{Linear}(512) + \text{LayerNorm} + \text{SiLU}]^{\times 3} \to \text{Linear}(124)$$

Output layer is zero-initialized so the model starts by predicting zero delta.

Code: [aprx_model_elastica/model.py](../aprx_model_elastica/model.py)

## Training Objective

### Phase 1 — Single-Step MSE (epochs 1–20)

$$\mathcal{L}_\text{single} = \frac{1}{N} \sum_{i=1}^{N} \left\| f_\theta(\mathbf{z}_t^{(i)}) - \overline{\Delta \mathbf{s}}^{(i)}_\text{true} \right\|^2$$

### Phase 2 — Combined Loss with Rollout (epochs 20+)

$$\mathcal{L} = \mathcal{L}_\text{single} + \lambda_r \cdot \mathcal{L}_\text{rollout}$$

**Rollout loss** (8-step autoregressive BPTT, $\lambda_r = 0.1$):

$$\mathcal{L}_\text{rollout} = \frac{1}{L} \sum_{t=0}^{L-1} \gamma^t \left\| \hat{\mathbf{s}}_{t+1} - \mathbf{s}_{t+1}^\text{true} \right\|^2$$

where $\hat{\mathbf{s}}_{t+1}$ is computed autoregressively using the model's own predictions as input.

Code: [aprx_model_elastica/train_surrogate.py](../aprx_model_elastica/train_surrogate.py)

## Sample Weighting

Rare state configurations are upweighted via inverse density:

$$w^{(i)} = \frac{1}{\hat{p}(\mathbf{c}^{(i)})}$$

where $\mathbf{c}^{(i)} = [\bar{x}_\text{CoM},\, \bar{y}_\text{CoM},\, \|\dot{\mathbf{x}}\|,\, \overline{|\omega_z|}]$ is a 4D summary feature vector and $\hat{p}$ is estimated via a joint histogram with 20 bins per dimension. Weights are clipped at 10× mean.

Code: [aprx_model_elastica/dataset.py](../aprx_model_elastica/dataset.py)

## Summary

| Aspect | Value |
|---|---|
| Approximated operator | Symplectic Euler integration of discretized 2D Cosserat rod PDE |
| State dim | 124 (21 nodes × {x,y,ẋ,ẏ} + 20 elements × {ψ, ω_z}) |
| Action dim | 5 (A, f, k, φ, b) |
| Time encoding | 2 (sin(ωt), cos(ωt)) |
| Total input dim | 131 |
| Output | 124-dim normalized state delta |
| Prediction type | Residual (next = current + delta) |
| Loss | MSE single-step + 0.1 × MSE rollout (8-step BPTT, from epoch 20) |
| Normalization | Per-feature z-score for state and delta separately |
| Sample weighting | Inverse density on 4D summary features |
