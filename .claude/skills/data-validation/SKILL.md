---
name: data-validation
description: >
  Validate data quality for machine learning and reinforcement learning tasks.
  Covers schema validation, label quality, train/test split integrity, drift detection,
  reward signal validation, and environment checks.
  Use when: (1) Setting up data validation for an ML pipeline,
  (2) Checking data quality before training (SL or RL),
  (3) Detecting data drift or label noise,
  (4) Validating RL environments or reward signals,
  (5) Choosing data validation tools (Pandera, Deepchecks, Cleanlab, Evidently, Gymnasium),
  (6) Implementing data quality gates or monitoring,
  (7) User asks about "data quality", "data validation", "label noise", "drift detection",
  "data leakage", "reward hacking", or "environment validation".
---

# Data Validation

## Core Principle: Layered Validation

Always validate in layers, cheapest first. Never skip to expensive checks without passing cheap ones.

```
Layer 1: Schema (ms)     → Types, nulls, ranges, format         → Pandera / Pydantic
Layer 2: Distribution (s) → Drift, anomalies, statistical tests  → Evidently / whylogs
Layer 3: Domain (s-min)   → Business rules, cross-field, SQL exec → Deepchecks / custom
Layer 4: Model-aware (min)→ Label noise, data influence           → Cleanlab
```

## Tool Selection

| Need | Tool | Install |
|------|------|---------|
| Schema validation (research) | **Pandera** | `pip install pandera` |
| Schema validation (production) | **Great Expectations** | `pip install great_expectations` |
| ML-specific validation suite | **Deepchecks** | `pip install deepchecks` |
| Label error detection | **Cleanlab** | `pip install cleanlab` |
| Drift detection (batch) | **Evidently AI** | `pip install evidently` |
| Data profiling (lightweight) | **whylogs** | `pip install whylogs` |
| RL environment validation | **Gymnasium** | `pip install gymnasium` |

## Quick Start Patterns

### Schema Validation (Pandera)

```python
import pandera as pa
from pandera import Column, Check, DataFrameSchema

schema = DataFrameSchema({
    "input_text": Column(str, [
        Check(lambda s: s.str.len() > 0, error="Empty input"),
        Check(lambda s: s.str.len() < 2048, error="Input too long"),
    ]),
    "target": Column(str, Check(lambda s: s.str.len() > 0, error="Empty target")),
    "split": Column(str, Check.isin(["train", "dev", "test"])),
})
validated_df = schema.validate(df)
```

### Drift Detection (Evidently)

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

report = Report(metrics=[DataDriftPreset(stattest="wasserstein")])
report.run(reference_data=train_df, current_data=new_df)
report.save_html("drift_report.html")
```

### Label Quality (Cleanlab)

```python
from cleanlab import Datalab

lab = Datalab(data={"text": texts, "label": labels}, label_name="label")
lab.find_issues(pred_probs=model_pred_probs, features=embeddings)
lab.report()
```

### RL Environment (Gymnasium)

```python
from gymnasium.utils.env_checker import check_env
env = CustomEnv()
check_env(env)  # Validates obs/action spaces, step(), reset()
```

## Drift Metric Selection

Use **Wasserstein Distance** as default. KS test is too sensitive on large datasets; PSI too insensitive.

| Metric | Best For | Sensitivity |
|--------|----------|-------------|
| Wasserstein | General-purpose (default) | Balanced |
| KS test | Small datasets (<1K), critical accuracy | Very high |
| PSI | Finance, major-change-only monitoring | Low |
| Jensen-Shannon | Bounded, symmetric comparison | Medium |
| Chi-Square | Categorical features | Medium |

## Key Don'ts

- **Don't hand-roll** schema validation, drift stats, dedup, or label noise detection — use the tools above
- **Don't use KS test** on large datasets (>10K) — it flags meaningless 0.5% shifts
- **Don't trust benchmark labels** — ImageNet has 100K+ errors; Spider/BIRD >30% incorrect
- **Don't validate once and forget** — data quality must be monitored continuously
- **Don't skip train-test leakage checks** — near-duplicates inflate metrics by 4%+

## Detailed References

- **Supervised learning validation** (schema, labels, splits, drift, NLP-specific): See [references/sl_validation.md](references/sl_validation.md)
- **Reinforcement learning validation** (rewards, environments, trajectories, replay buffers): See [references/rl_validation.md](references/rl_validation.md)
- **Data quality metrics** (statistical tests, label quality, feature quality, thresholds): See [references/metrics.md](references/metrics.md)
