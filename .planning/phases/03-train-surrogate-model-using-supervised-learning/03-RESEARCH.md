# Phase 3: Surrogate Architecture Experiments — Research

**Researched:** 2026-03-10
**Domain:** Neural surrogate model architecture — rollout loss, residual connections, history window
**Confidence:** HIGH (codebase analysis) / MEDIUM (external literature)

---

## Summary

Phase 3.1 inserts three targeted architectural experiments between the Phase 3 hyperparameter sweep and Phase 4 trajectory validation. The baseline is a 512x3 MLP with SiLU + LayerNorm, lr=1e-3, val_loss=0.2161, R²=0.784. The primary weakness is omega_z prediction (R²=0.23), caused by the model having no memory of recent angular velocity history.

The three experiments each address a different hypothesis:

1. **Rollout loss tuning**: The training script already has `compute_rollout_loss()` and `TrajectoryDataset` implemented and wired. Phase 3 never varied `rollout_loss_weight` or `rollout_steps` — it used defaults (weight=0.1, steps=8, start_epoch=20). A targeted sweep of these two hyperparameters is the lowest-risk, highest-reward experiment because all infrastructure is already in place.

2. **Residual connections**: Adding skip connections every 2 hidden layers is a straightforward architectural change to `SurrogateModel.__init__()`. The model currently uses `nn.Sequential`, which must be replaced with explicit `forward()` layer management. Residual connections primarily help gradient flow in depth; with only 3 hidden layers, the benefit is marginal but non-zero.

3. **History window**: Concatenating K prior `(state, action)` pairs to the current input. The 124D full state is Markov by construction (verified in `knowledge/surrogate-architecture-comparison.md`), but omega_z poor prediction suggests that the second-derivative nature of angular velocity creates near-noise deltas for a single-step model. A K=2 or K=3 window may give the model access to finite differences of omega_z, improving its effective signal-to-noise. **Key risk**: increases dataset complexity (requires contiguous K-step windows from TrajectoryDataset), input dimension (131 + K×129 per additional step), and inference state management during RL rollout.

**Primary recommendation:** Run rollout loss sweep first (already instrumented, ~2 hours), then residual variant (1 code change, 1 run, ~45 min), then history window only if rollout loss + residual fail to improve omega_z R² above 0.5.

---

## Standard Stack

### Core (already installed)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| PyTorch | 2.x (system) | Model, training, autograd | Project-wide DL stack |
| torch.utils.data | — | DataLoader, WeightedRandomSampler | Already used in train_surrogate.py |
| wandb | 0.25.0 | Sweep logging | Already integrated |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | system | Sweep summary, metrics | Summary JSON writing |
| json / argparse | stdlib | CLI args, config files | Existing pattern |

No new dependencies are needed for any of the three experiments.

**Installation:** None required.

---

## Architecture Patterns

### Recommended Project Structure

No new files are strictly required. Changes are localized to:

```
aprx_model_elastica/
├── model.py             # Add ResidualSurrogateModel + HistorySurrogateModel variants
├── train_config.py      # Add rollout_loss_weight, rollout_steps, use_residual, history_k params
├── train_surrogate.py   # Add --rollout-weight, --rollout-steps, --use-residual, --history-k CLI args
└── sweep.py             # Add arch_sweep.py (new file for 3.1 sweep runner)
```

### Pattern 1: Rollout Loss Tuning (Experiment A)

**What:** Vary `rollout_loss_weight` in {0.0, 0.1, 0.3, 0.5} and `rollout_steps` in {4, 8, 16}. Keep architecture fixed at 512x3.

**When to use:** Already instrumented — lowest friction path. `compute_rollout_loss()` in `train_surrogate.py` uses autoregressive unrolling with per-step discount 0.95.

**Key insight from existing code:** The rollout loss is currently added to the single-step loss but the gradient flows through autoregressive state predictions. Higher weight forces the model to care about multi-step consistency. The discount `0.95^t` means step 8 contributes ~66% of step 1's weight — a reasonable decay.

**Sweep design (5 runs, ~3.5 hours):**

| Run | rollout_loss_weight | rollout_steps | Expected effect |
|-----|---------------------|---------------|-----------------|
| A1  | 0.0                 | —             | Single-step only baseline |
| A2  | 0.1 (current)       | 8             | Current default |
| A3  | 0.3                 | 8             | Stronger rollout signal |
| A4  | 0.5                 | 8             | Aggressive rollout |
| A5  | 0.3                 | 16            | Longer horizon |

**Expected outcome:** Higher rollout weight should reduce multi-step drift (Phase 4 concern). May slightly increase single-step val_loss. If best rollout config beats 0.2161, use it. If not, take the config with best val_loss AND lowest rollout_loss for Phase 4 input.

**Known tradeoff (MBPO evidence):** MBPO uses k=1–5 steps; beyond 5 steps, model error accumulates O(k·ε). Our 8-step default is already on the edge — 16 steps may not help and could destabilize training. This is why A5 is the only long-horizon variant.

**Code change needed:** Add `--rollout-weight` and `--rollout-steps` CLI args to `train_surrogate.py`. Currently `rollout_loss_weight` and `rollout_steps` are only in `SurrogateTrainConfig` defaults.

### Pattern 2: Residual MLP (Experiment B)

**What:** Wrap every pair of hidden layers in a residual block. Skip connection adds input to output element-wise.

**When to use:** Only valid when all hidden dims are equal (they are: 512x3). If dims differ, need a projection layer.

**Implementation sketch:**

```python
# In model.py — replaces nn.Sequential with explicit forward

class ResidualBlock(nn.Module):
    """Two hidden layers with a skip connection."""

    def __init__(self, dim: int, activation: nn.Module, use_layer_norm: bool):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.act = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.linear1(x)))
        out = self.norm2(self.linear2(out))
        return self.act(out + residual)  # skip connection before final activation


class ResidualSurrogateModel(nn.Module):
    """MLP surrogate with residual blocks every 2 hidden layers."""

    def __init__(self, config: SurrogateModelConfig):
        super().__init__()
        # Project input to hidden dim
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dims[0])
        # Build residual blocks (pairs of layers)
        n_hidden = len(config.hidden_dims)
        n_blocks = n_hidden // 2
        act = nn.SiLU()
        self.blocks = nn.ModuleList([
            ResidualBlock(config.hidden_dims[0], act, config.use_layer_norm)
            for _ in range(n_blocks)
        ])
        # Handle odd number of layers: add final plain layer if needed
        self.extra_layer = None
        if n_hidden % 2 == 1:
            self.extra_layer = nn.Linear(config.hidden_dims[0], config.hidden_dims[0])
        self.output = nn.Linear(config.hidden_dims[-1], config.output_dim)
        # Zero-init output layer (same as base model)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
```

**With 512x3:** 3 hidden layers → 1 full residual block (layers 1+2) + 1 plain layer (layer 3). The skip connection gives a shortcut over 2 of the 3 hidden layers.

**Expected benefit:** MEDIUM. With only 3 layers, residual connections solve a problem (vanishing gradients) that barely exists. The real benefit of ResNets is in 20+ layer networks. However, for delta prediction (predicting near-zero outputs), the identity shortcut can help the model learn "mostly zero" deltas more easily — this is the mechanism that matters here.

**Parameter count:** Same as base model (~2.6M). No capacity increase.

**Code change needed:** Add `ResidualSurrogateModel` to `model.py`. Add `--use-residual` flag to `train_surrogate.py`. Architecture chosen by flag.

### Pattern 3: History Window (Experiment C)

**What:** Concatenate K previous (state, action) pairs to current input. Input becomes `[state_t(124) | action_t(5) | time_t(2) | state_{t-1}(124) | action_{t-1}(5) | ... ] = 131 + K*129`.

**When to use:** Only if Experiments A and B fail to improve omega_z above R²~0.5. High complexity, high development cost, moderate expected gain.

**Why omega_z might benefit from history:**

The state is Markov at the 124D full-state level for continuous dynamics. However:
- `Δomega_z` is the difference of angular velocities — effectively a second derivative of position
- In the dataset, `omega_z` exhibits rapid sign-changing patterns tied to the CPG half-cycle
- A single-step model fitting `Δomega_z` from `(state, action)` is trying to predict an acceleration from a velocity snapshot
- With K=2, the model can compute approximate angular acceleration as `(omega_z_t - omega_z_{t-1}) / dt`, which the MLP can learn to use as an input feature
- This is the classic "velocity estimation from position history" pattern used in physics-based RL (e.g., MuJoCo obs include both pos and vel; without vel, history is needed)

**Dataset requirement:** `TrajectoryDataset` already builds contiguous windows. For K=2, need windows of length K+1 = 3. Feed `states[0]...states[K-1]` and `actions[0]...actions[K-1]` as input, predict `delta[K]`. This is a small extension to the existing `__getitem__`.

**Input dimension for K=2:** 131 + 1×129 = 260 (doubles input size)
**Input dimension for K=3:** 131 + 2×129 = 389

**Inference complexity:** During RL rollout, must maintain a rolling buffer of K prior states/actions. This adds state to the surrogate environment — manageable but not trivial.

**Code change needed:**
1. `HistorySurrogateModel` in `model.py` — input_dim = 131 + K*129
2. `HistoryDataset` in `dataset.py` — extend `TrajectoryDataset` to return K-step history context
3. `--history-k` CLI arg in `train_surrogate.py`
4. `SurrogateEnv` in `env.py` must maintain state buffer during RL rollout

**Recommendation: K=2 only.** K=3+ provides diminishing returns and triples the inference state size.

### Anti-Patterns to Avoid

- **Running all 3 experiments in parallel from the start**: Experiment A (rollout loss) is low-risk and fully instrumented. Run it first. If it resolves omega_z, experiments B and C may be unnecessary.
- **Using rollout_steps > 16**: Error accumulates O(k·ε). At step 16, the surrogate's own predicted state is far from the data manifold; the loss signal becomes noise.
- **History window K > 2 without verifying K=2 first**: Each K doubles the dataset indexing complexity and doubles inference state size.
- **Changing the base 512x3 architecture in the residual variant**: Residual blocks only work without projection layers when `hidden_dims` are all equal. Keep 512x3.
- **Evaluating by val_loss alone**: val_loss is single-step MSE. The actual goal is multi-step rollout quality (Phase 4). An experiment that slightly increases val_loss but reduces rollout drift may still be the correct choice. Track rollout_loss as a secondary metric.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Trajectory window indexing | Custom episode-aware sampler | Existing `TrajectoryDataset._build_trajectory_index()` | Already handles contiguous episode windows, train/val split |
| Sweep execution | Manual subprocess loop | Extend existing `sweep.py` pattern | subprocess + metrics.json pattern is proven |
| Gradient clipping | Custom clipper | `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` | Already used; BPTT through rollout needs this especially |
| LR scheduling | Custom scheduler | Existing `build_lr_scheduler()` with cosine+warmup | Already matches LambdaLR pattern |
| Normalization | Custom normalizer | Existing `StateNormalizer` | Fitted on training data, handles state and delta normalization separately |

**Key insight:** The rollout loss infrastructure (`compute_rollout_loss`, `TrajectoryDataset`, lazy `traj_loader` initialization) is already complete and wired. Experiment A requires only CLI arg additions, not architectural changes.

---

## Common Pitfalls

### Pitfall 1: Gradient Explosion in Long Rollouts

**What goes wrong:** BPTT through 16 autoregressive steps can produce large gradients, especially early in training when the model is far from optimal.

**Why it happens:** Errors compound multiplicatively: `d(state_t)/d(params)` involves products of Jacobians across all `t` steps.

**How to avoid:** The existing `clip_grad_norm_(model.parameters(), 1.0)` in `train_surrogate.py` handles this. Keep it in place. Also, `rollout_start_epoch=20` (warmup phase) ensures the model has learned reasonable single-step predictions before rollout loss is added. Do not reduce this warmup.

**Warning signs:** Rollout loss spikes after epoch 20, val_loss diverges, NaN weights.

### Pitfall 2: Residual Block with Unequal Layer Dims

**What goes wrong:** Skip connection `out + residual` fails if `linear1` and `linear2` have different input/output dims (shape mismatch).

**Why it happens:** For a 512x3 model, all dims match (512→512). But if someone tries residual with a 512x4 (4-layer) model with a projection at layer 3, dims may not align.

**How to avoid:** The residual model should hard-code `hidden_dims[0]` as the residual block dim. Add an assertion: `assert len(set(config.hidden_dims)) == 1, "Residual model requires uniform hidden dims"`.

**Warning signs:** `RuntimeError: The size of tensor a (512) must match the size of tensor b (256) at non-singleton dimension 1`.

### Pitfall 3: TrajectoryDataset Missing Windows for Short Episodes

**What goes wrong:** If episodes are shorter than `rollout_length + 1` (or `history_k + 1`), no windows are extracted from those episodes. With `rollout_length=16`, episodes shorter than 17 steps are silently skipped.

**Why it happens:** `TrajectoryDataset._build_trajectory_index()` requires `len(indices) - rollout_length > 0`.

**How to avoid:** Print window count after `TrajectoryDataset` construction. Verify `len(traj_dataset) > 0`. For `rollout_steps=16`, check window count vs `rollout_steps=8`.

**Warning signs:** `traj_dataset` has 0 windows → `DataLoader` produces no batches → `next(traj_iter)` raises `StopIteration` immediately.

### Pitfall 4: History Window Breaks Standard SurrogateDataset Items

**What goes wrong:** History window model requires K-step context, but `SurrogateDataset.__getitem__` returns single transitions. Training with `train_loader` (which uses `SurrogateDataset`) will fail if the model expects a history stack.

**Why it happens:** Architecturally, history models need a different dataset that returns (state_t-K, ..., state_t, action_t-K, ..., action_t, delta_t). The existing single-step and trajectory datasets don't return this shape.

**How to avoid:** History training must use a dedicated `HistoryDataset` that extends `TrajectoryDataset`. The single-step loss is computed on the last step only; history context is the K prior steps.

**Warning signs:** Shape mismatch in `model.forward()`: expected `(B, 260)`, got `(B, 131)`.

### Pitfall 5: val_loss Not Comparable Across Experiments

**What goes wrong:** Experiment C (history window) may show lower val_loss than A or B, but only because the larger input (K prior states) gives the model "easier" access to the answer. The gain may not generalize to RL rollout where prior states are model-predicted.

**Why it happens:** Validation data provides ground-truth prior states from the dataset. During RL rollout, prior states are predicted — and prediction errors accumulate. A model trained with ground-truth history may perform worse in rollout than a model trained without history.

**How to avoid:** For experiment C, report rollout loss (during training) as the primary metric in addition to val_loss. Compare rollout_loss across all experiments. Phase 4 trajectory validation is the final arbiter.

---

## Code Examples

### Adding CLI Args for Rollout Tuning (Experiment A)

```python
# In train_surrogate.py parse_args():
parser.add_argument(
    "--rollout-weight", type=float, default=None,
    help="Rollout loss weight (default: from config)",
)
parser.add_argument(
    "--rollout-steps", type=int, default=None,
    help="Number of rollout steps (default: from config)",
)

# In main() after config setup:
if args.rollout_weight is not None:
    config.rollout_loss_weight = args.rollout_weight
if args.rollout_steps is not None:
    config.rollout_steps = args.rollout_steps
```

### Residual Block PyTorch Pattern

```python
# In model.py

class ResidualBlock(nn.Module):
    """Two linear layers with LayerNorm, SiLU, and skip connection.

    Skip connection: output = activation(norm(linear2(activation(norm(linear1(x))))) + x)
    Requires input_dim == output_dim (no projection).
    """

    def __init__(self, dim: int, use_layer_norm: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(dim) if use_layer_norm else nn.Identity()
        self.act = nn.SiLU()

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.linear1(x)))
        out = self.norm2(self.linear2(out))
        return self.act(out + x)  # skip before final activation
```

### History Window Input Construction

```python
# In HistoryDataset.__getitem__ (extends TrajectoryDataset):
def __getitem__(self, idx: int) -> dict:
    """Return history context for step K, predicting delta at step K."""
    indices = self.windows[idx]  # length = history_k + 1 + 1 (need K+1 states)
    # states[0..K-1]: history context; states[K]: current; states[K+1]: target
    history_states = self.states[indices[:self.history_k]]   # (K, 124)
    history_actions = self.actions[indices[:self.history_k]] # (K, 5)
    current_state = self.states[indices[self.history_k]]     # (124,)
    current_action = self.actions[indices[self.history_k]]   # (5,)
    current_time = self.serpenoid_times[indices[self.history_k]]  # scalar
    next_state = self.states[indices[self.history_k + 1]]    # (124,)
    delta = next_state - current_state                        # (124,)
    return {
        "history_states": history_states,    # (K, 124)
        "history_actions": history_actions,  # (K, 5)
        "state": current_state,
        "action": current_action,
        "serpenoid_time": current_time,
        "delta": delta,
    }

# In HistorySurrogateModel.forward():
def forward(self, state, action, time_enc, history_states, history_actions):
    # Flatten history into input
    history_flat = torch.cat([
        history_states.flatten(-2, -1),   # (B, K*124)
        history_actions.flatten(-2, -1),  # (B, K*5)
    ], dim=-1)                             # (B, K*129)
    x = torch.cat([state, action, time_enc, history_flat], dim=-1)  # (B, 131 + K*129)
    return self.mlp(x)
```

---

## Sweep Design for Phase 3.1

### Recommended 3-experiment sweep structure

**Total estimated time: ~5–7 hours on V100 (3.5h Exp A + 45min Exp B + 1.5h Exp C)**

| Exp | Variant | Config | Architecture | Est. Time |
|-----|---------|--------|--------------|-----------|
| A1 | rollout_w=0.0 | 512x3, lr=1e-3 | base MLP | ~45 min |
| A2 | rollout_w=0.1 (baseline) | 512x3, lr=1e-3 | base MLP | skip (already done) |
| A3 | rollout_w=0.3 | 512x3, lr=1e-3 | base MLP | ~45 min |
| A4 | rollout_w=0.5 | 512x3, lr=1e-3 | base MLP | ~45 min |
| A5 | rollout_w=0.3, steps=16 | 512x3, lr=1e-3 | base MLP | ~45 min |
| B1 | residual, rollout_w=0.1 | 512x3, lr=1e-3 | residual MLP | ~45 min |
| B2 | residual, best rollout_w | 512x3, lr=1e-3 | residual MLP | ~45 min |
| C1 | history K=2, rollout_w=0.1 | 512x3, lr=1e-3 | history MLP | ~90 min |

**Decision gate between experiments:**
- After Exp A: if best variant achieves val_loss < 0.21 AND rollout_loss substantially lower than A2 → use as B baseline. If A2 (current defaults) still wins → run B anyway, skip C.
- After Exp B: if B + best rollout achieves R²(omega_z) > 0.4 → skip C. Otherwise run C.
- The winner is the single run with lowest val_loss AND lowest rollout_loss (may differ — document the tradeoff for Phase 4).

### Sweep runner extension

Add `aprx_model_elastica/arch_sweep.py` following the pattern of `sweep.py`. It launches configs sequentially with `subprocess.run()`, reads `metrics.json`, prints ranked table, saves `arch_sweep_summary.json`. Accept `--output-base output/surrogate/arch_sweep`.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single-step MSE only | Single-step + multi-step rollout loss | Dreamer/MBPO era (2019–2020) | Reduces rollout drift |
| Vanilla MLP | Residual MLP (ResNet-style) | He et al. 2016, widely adopted 2018+ | Better gradient flow, identity shortcut |
| Markov input only | History-conditioned MLP / RNN | RSSM (Dreamer 2020), S4/Mamba (2023) | Required for partial observability |
| Flat global MLP | GNN or 1D-CNN with spatial structure | Active research 2022–2024 | Better inductive bias for rod dynamics |

**Note:** For our specific task (full 124D Markov state, 3-layer MLP at 512 width), we are already using current standard practice for flat surrogate models. The three experiments are low-risk incremental improvements within the MLP family, not architecture overhauls.

**Deprecated/outdated:**
- Using `sin(t) / cos(t)` raw time encoding: fixed to `sin(omega*t) / cos(omega*t)` after Phase 3 analysis
- `rollout_loss_weight=0.1` was a default, never tuned — treat as a prior to validate

---

## Open Questions

1. **Will rollout loss help omega_z specifically?**
   - What we know: omega_z R²=0.23 under single-step loss. Rollout loss forces multi-step angular consistency.
   - What's unclear: Whether 8 rollout steps is long enough to "see" the CPG half-cycle periodicity of omega_z (CPG period ~0.5–2s at 0.5–3 Hz; 8 RL steps = 8 seconds wall clock? No — need to clarify RL step duration).
   - Recommendation: Verify RL step duration (seconds per step) to ensure rollout_steps covers at least 1 full CPG period.

2. **Does omega_z need its own loss component?**
   - What we know: omega_z occupies 20/124 output dims. MSE loss treats all dims equally by count. But omega_z deltas have higher variance than position deltas (phase 3 showed large component loss for omega_z).
   - What's unclear: Whether upweighting omega_z in the MSE (e.g., 5× the weight for omega_z dims) would help more than rollout loss.
   - Recommendation: Add as Experiment A6 if time permits — pure component-weighting baseline.

3. **RL rollout compatibility for history window**
   - What we know: `aprx_model_elastica/env.py` exists but its internal state management isn't shown here.
   - What's unclear: Whether the surrogate env's `step()` method can maintain a rolling state buffer without architectural rework.
   - Recommendation: Read `env.py` at plan time before committing to Experiment C.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 8.x |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `python3 -m pytest tests/ -x -q` |
| Full suite command | `python3 -m pytest tests/ -v` |

### Phase Requirements → Test Map

| ID | Behavior | Test Type | Automated Command | File Exists? |
|----|----------|-----------|-------------------|-------------|
| ARCH-01 | ResidualSurrogateModel forward pass matches input/output dims | unit | `pytest tests/test_surrogate_arch.py::test_residual_model_forward -x` | ❌ Wave 0 |
| ARCH-02 | HistorySurrogateModel forward pass with K=2 | unit | `pytest tests/test_surrogate_arch.py::test_history_model_forward -x` | ❌ Wave 0 |
| ARCH-03 | TrajectoryDataset window count > 0 for rollout_length=16 | unit | `pytest tests/test_surrogate_arch.py::test_trajectory_dataset_windows -x` | ❌ Wave 0 |
| ARCH-04 | CLI args --rollout-weight and --rollout-steps pass through to config | unit | `pytest tests/test_surrogate_arch.py::test_train_cli_args -x` | ❌ Wave 0 |
| ARCH-05 | arch_sweep runner completes dry-run (1 epoch) without error | smoke | `python3 -m aprx_model_elastica.arch_sweep --epochs 1 --dry-run` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `python3 -m pytest tests/test_surrogate_arch.py -x -q`
- **Per wave merge:** `python3 -m pytest tests/ -v`
- **Phase gate:** Full suite green before moving to Phase 4

### Wave 0 Gaps

- [ ] `tests/test_surrogate_arch.py` — covers ARCH-01 through ARCH-04
- [ ] Architecture dry-run via `--epochs 1` in sweep runner — covers ARCH-05

---

## Sources

### Primary (HIGH confidence)

- Codebase: `aprx_model_elastica/model.py` — current SurrogateModel architecture (131→512x3→124)
- Codebase: `aprx_model_elastica/train_surrogate.py` — existing `compute_rollout_loss()`, `TrajectoryDataset` integration
- Codebase: `aprx_model_elastica/train_config.py` — current defaults: rollout_steps=8, rollout_loss_weight=0.1, rollout_start_epoch=20
- Codebase: `aprx_model_elastica/dataset.py` — `TrajectoryDataset._build_trajectory_index()` window logic
- Codebase: `knowledge/surrogate-architecture-comparison.md` — confirms full 124D state is Markov (elastic propagation fully settles in 500 substeps)
- Codebase: `knowledge/surrogate-time-encoding-and-elastic-propagation.md` — confirms `omega*t` phase encoding is correct
- Codebase: `issues/surrogate-omega-z-poor-prediction.md` — omega_z R²=0.23 root cause analysis
- Codebase: `.planning/phases/03-surrogate-training/03-01-SUMMARY.md` — sweep results: best 512x3 at lr=1e-3, val_loss=0.2161

### Secondary (MEDIUM confidence)

- MBPO (Janner et al., 2019): k=1–5 rollout steps prevent error accumulation O(k·ε) — supports keeping rollout_steps ≤ 16
- He et al. (2016) ResNet: residual blocks help gradient flow; benefit is largest in deep networks (20+ layers); 3-layer MLP gains are marginal but nonzero
- Dreamer/RSSM (Hafner et al., 2020): history conditioning helps for partially observable dynamics — relevant if 124D state has non-Markov artifacts

### Tertiary (LOW confidence, needs validation)

- WebSearch: MBPO optimal horizon k≤5 — our rollout_steps=8 is at the edge; 16 may be too long (needs empirical confirmation for this specific task)
- WebSearch: Residual connections improve surrogate accuracy ~6.6% in thermal design context — domain-specific, may not transfer to rod dynamics

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in use, zero new dependencies
- Architecture patterns: HIGH (rollout loss) / MEDIUM (residual) / MEDIUM-LOW (history window) — rollout loss is already implemented; residual and history are architectural changes requiring validation
- Pitfalls: HIGH — all pitfalls derived from direct code analysis of existing `model.py`, `dataset.py`, `train_surrogate.py`
- Sweep design: HIGH — based on Phase 3 timing data (45 min per 512x3 run at 200 epochs on V100)

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (stable domain; PyTorch API stable)
