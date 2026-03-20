---
name: Surrogate model spatial structure analysis
description: >
  Investigation into whether the surrogate MLP should exploit spatial
  relationships between consecutive nodes along the Cosserat rod body.
type: issue
status: resolved
severity: low
subtype: physics
created: 2026-03-09
updated: 2026-03-09
tags: [surrogate, architecture, cosserat-rod, spatial-locality, cnn]
aliases: [surrogate-node-independence, surrogate-cnn-vs-mlp]
---

# Surrogate Model: Spatial Structure Analysis

## Problem Statement

The surrogate MLP in `aprx_model_elastica/model.py` treats the 124-dim rod
state as a flat feature vector, ignoring the spatial ordering of measurements
along the snake's body. Consecutive nodes on the Cosserat rod are physically
connected — should the architecture exploit this structure (e.g., 1D CNN)?

## State Layout (from `aprx_model_elastica/state.py`)

| Group   | Slice       | Indices   | Grid         | Count | Units   | Typical Scale |
|---------|-------------|-----------|--------------|-------|---------|---------------|
| pos_x   | `[0:21]`    | 0–20      | 21 nodes     | 21    | meters  | 0–1 m         |
| pos_y   | `[21:42]`   | 21–41     | 21 nodes     | 21    | meters  | 0–1 m         |
| vel_x   | `[42:63]`   | 42–62     | 21 nodes     | 21    | m/s     | ±0.1 m/s      |
| vel_y   | `[63:84]`   | 63–83     | 21 nodes     | 21    | m/s     | ±0.1 m/s      |
| yaw     | `[84:104]`  | 84–103    | 20 elements  | 20    | radians | ±π            |
| omega_z | `[104:124]` | 104–123   | 20 elements  | 20    | rad/s   | ±10 rad/s     |

Named slices defined in `aprx_model_elastica/state.py` as `POS_X`, `POS_Y`,
`VEL_X`, `VEL_Y`, `YAW`, `OMEGA_Z`.

Nodes and elements live on a staggered grid (21 vs 20 points), standard in
Cosserat rod discretizations.

## Analysis: Per-Substep Locality

Curvature computation in PyElastica is strictly local:

- **Curvature at Voronoi point k** depends only on director frames at nodes
  k and k+1 (2-node stencil), computed via relative rotation in
  `_rotations.py`.
- **Bending torque on node k** depends on couples at Voronoi points k-1 and
  k, computed via `difference_kernel_for_block_structure` in `_calculus.py`:
  `torque[k] = couple[k] - couple[k-1]`. This gives a **3-node stencil**
  (nodes k-1, k, k+1).
- **No global coupling**: PyElastica uses explicit symplectic integrators
  (PositionVerlet/PEFRL) — no tridiagonal solves or implicit matrix
  inversions. All force computations are purely local per substep.

The project's own `get_curvatures()` in `src/physics/elastica_snake_robot.py`
confirms the 3-node stencil:
```python
for i in range(1, self.num_nodes - 1):
    v1 = self.positions[i] - self.positions[i - 1]
    v2 = self.positions[i + 1] - self.positions[i]
```

## Key Finding: Global Coupling Over One RL Step

The surrogate does not predict a single substep — it predicts the result of
**500 substeps** integrated together (`locomotion_elastica/config.py:127`,
`substeps_per_action = 500`).

Information propagation:
- Per substep: ±1 node (3-node stencil)
- Rod length: 21 nodes
- After 20 substeps: perturbation at node 0 reaches node 20
- Over 500 substeps: signal traverses the full rod **~25 times**

**Every node influences every other node within a single RL step.** The
spatial locality that holds per-substep is completely diffused over the 500
substeps the surrogate must predict.

## Conclusion

A narrow-kernel 1D CNN (kernel_size=3) would impose an **incorrect** inductive
bias — it would restrict early layers to local interactions even though the
target output reflects fully global coupling. A deep CNN stack could eventually
reach full receptive field, but at 21 nodes this offers no advantage over a
flat MLP.

| Architecture               | Verdict       | Reason                                    |
|----------------------------|---------------|-------------------------------------------|
| Narrow CNN (kernel=3)      | Wrong bias    | Too local; needs ~10 layers for 21 nodes  |
| Wide CNN (kernel=21)       | Degenerates   | Equivalent to fully connected at this size |
| Positional embedding + attn| Overkill      | 21 tokens is tiny; adds complexity         |
| Flat MLP (current)         | **Adequate**  | Global coupling justifies flat structure   |

The current flat MLP architecture is not fundamentally mismatched for this
problem. If the surrogate is underperforming, the bottleneck is more likely
data coverage, normalization, or training dynamics rather than missing spatial
inductive bias.

## Normalization

The `StateNormalizer` in `aprx_model_elastica/state.py` applies **per-element
z-score normalization**: each of the 124 dimensions gets its own mean and std,
computed from training data via `normalizer.fit(states, deltas)`.

```python
# Per-element: state_mean.shape = (124,), state_std.shape = (124,)
normalized = (state - state_mean) / (state_std + eps)
```

This is sufficient because:
- Each physical quantity (pos_x[0], vel_y[15], yaw[7], etc.) gets its own
  statistics, so different scales across feature groups are handled.
- Per-element is strictly more flexible than per-group normalization (which
  would force all pos_x values to share statistics).
- The 6 feature groups span very different scales (meters, m/s, radians,
  rad/s), but per-element normalization absorbs this automatically.

**No per-feature-group normalization is needed** beyond what already exists.

## Reshape Analysis

**Question**: Should the flat (124,) input be reshaped to (21, 6) — treating
each node as a 6-feature token — before feeding to the network?

**Answer**: No. For an MLP, input ordering is irrelevant — the weight matrix
`Linear(124, 512)` learns arbitrary connections regardless of feature
arrangement. Reshaping to (21, 6) and flattening back is a permutation of
indices, which the first linear layer absorbs trivially. Reshaping only matters
if a spatial operation (convolution, attention) is applied in between, but we
have established those are the wrong inductive bias for this problem.

## Hidden Layer Normalization (LayerNorm)

The MLP uses **LayerNorm** after each hidden linear layer, before the activation
(`model.py`):

```
Linear(in, 512) → LayerNorm(512) → SiLU → ... → Linear(512, 124)
```

LayerNorm normalizes across the feature dimension of each sample independently:
`y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta`, where mean/var are
computed over the 512 hidden features for each sample in the batch.

**Why LayerNorm (not BatchNorm):**
- **Batch-size independent**: LayerNorm computes statistics per-sample, so it
  works identically regardless of batch size. BatchNorm requires sufficiently
  large batches for stable running statistics.
- **No train/eval discrepancy**: LayerNorm has no running mean/var buffers, so
  behavior is identical in training and inference. BatchNorm's running stats can
  drift from actual data distribution.
- **Standard for regression MLPs**: BatchNorm is common in vision CNNs;
  LayerNorm is preferred for transformers and regression MLPs where batch
  statistics are less meaningful.

Configured via `use_layer_norm: bool = True` in `SurrogateModelConfig`.

## Dropout

Dropout is **supported but disabled** (`dropout: float = 0.0` in
`SurrogateModelConfig`). When enabled, it would be applied after each activation:

```
Linear → LayerNorm → SiLU → Dropout(p) → ...
```

**Why dropout is disabled:**
- Physics surrogates typically **underfit**, not overfit — the challenge is
  accurately capturing complex nonlinear dynamics, not memorizing training data.
- The model has ~400k parameters predicting 124-dim output from 131-dim input
  over a highly nonlinear 500-substep integration. This is a hard regression
  problem where capacity is needed, not regularized away.
- Weight decay (`weight_decay: float = 1e-5`) provides mild regularization
  without reducing effective capacity during forward passes.
- If overfitting is observed (val loss diverges from train loss), dropout can be
  enabled by setting `dropout > 0` in the config without code changes.

## Full Normalization Pipeline

The three-stage normalization pipeline:

1. **Input z-score** (`StateNormalizer`): Per-element normalization of the 124-dim
   state before entering the network. Handles different physical scales across
   the 6 feature groups (meters, m/s, radians, rad/s).
2. **Hidden LayerNorm** (`nn.LayerNorm`): Per-sample normalization of 512-dim
   hidden activations between layers. Stabilizes training dynamics and gradient
   flow through the 3-layer MLP.
3. **Raw output**: No normalization or activation on the output layer — deltas
   can be positive or negative. Denormalized back to physical units by
   `StateNormalizer.denormalize_delta()`.

## Resolution

No architecture change needed. The flat MLP is appropriate given that 500
substeps of explicit integration fully mix information across the 21-node rod.
Per-element z-score normalization already handles the different physical scales
across the 6 feature groups. No reshape is beneficial for a flat MLP.
