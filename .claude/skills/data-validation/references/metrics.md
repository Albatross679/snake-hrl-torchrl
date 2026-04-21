# Data Quality Metrics Reference

## Table of Contents
1. [Core Quality Dimensions](#core-quality-dimensions)
2. [Statistical Drift Detection Metrics](#statistical-drift-detection-metrics)
3. [Label Quality Metrics](#label-quality-metrics)
4. [Feature Quality Metrics](#feature-quality-metrics)
5. [Metric Selection Guide](#metric-selection-guide)

---

## Core Quality Dimensions

Based on ISO/IEC 25012 adapted for ML:

| Dimension | Definition | ML Impact | How to Measure |
|-----------|-----------|-----------|----------------|
| **Accuracy** | Data matches real-world truth | Wrong labels = wrong learning signal | Label audit, execution testing, Cleanlab |
| **Completeness** | All required data present | Missing features = model bias | Null ratio per feature, coverage metrics |
| **Consistency** | Same data looks same everywhere | Feature skew = train-serve mismatch | Cross-source comparison, schema checks |
| **Uniqueness** | No unintended duplicates | Duplicates inflate metrics, bias learning | Exact/near-duplicate detection |
| **Timeliness** | Data is current | Stale data = concept drift | Timestamp analysis, freshness monitoring |
| **Representativeness** | Data reflects target population | Selection bias = poor generalization | Subgroup analysis, slice-based evaluation |

---

## Statistical Drift Detection Metrics

| Metric | Intuition | Range | Threshold | Best For |
|--------|-----------|-------|-----------|----------|
| **Wasserstein** | Earth mover's distance | [0, inf) | ~0.1 std dev | **General-purpose (recommended default)** |
| **KS Statistic** | Max CDF difference | [0, 1] | p < 0.05 | Small datasets (<1K), critical accuracy |
| **PSI** | Symmetric KL with binning | [0, inf) | <0.1 stable; 0.1-0.2 moderate; >0.2 significant | Production monitoring, finance |
| **Jensen-Shannon** | Symmetric bounded KL | [0, 1] | Domain-dependent | Bounded symmetric comparison |
| **Chi-Square** | Observed vs expected frequencies | [0, inf) | p < 0.05 | Categorical features |
| **KL Divergence** | Information-theoretic | [0, inf) | Domain-dependent | Directional difference, new values |

**Selection rationale:**
- **Wasserstein** provides balanced sensitivity with interpretable units (same unit as features)
- **KS test** is too sensitive on large datasets — flags meaningless 0.5% shifts
- **PSI** requires >10% drift to trigger — may miss gradual degradation
- Pair any metric with **effect-size thresholds** to avoid false alarms

---

## Label Quality Metrics

| Metric | What It Measures | When to Use |
|--------|-----------------|-------------|
| **Cohen's Kappa** | Inter-annotator agreement (2 annotators) | Paired annotation quality |
| **Krippendorff's Alpha** | Inter-annotator agreement (N annotators) | Multi-annotator quality |
| **Confident Learning joint** | p(noisy_label, true_label) jointly | Finding specific mislabeled examples |
| **Cross-validation loss residuals** | Per-example difficulty/correctness | Identifying problematic examples |
| **Execution Accuracy (EX)** | SQL execution result match | NL-to-SQL label validation |

**Interpretation:**
- Cohen's Kappa: <0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, >0.8 almost perfect
- Krippendorff's Alpha: <0.667 unreliable, 0.667-0.8 acceptable, >0.8 good
- Cleanlab noise rate: >5% warrants investigation, >15% requires serious label cleanup

---

## Feature Quality Metrics

| Metric | What It Measures | Threshold |
|--------|-----------------|-----------|
| **Null ratio** | Missing values per feature | Domain-specific; flag >5% |
| **Cardinality ratio** | Unique values / total values | Flag sudden changes |
| **Feature correlation stability** | Correlation between features over time | Track pairwise correlation matrix drift |
| **Feature importance stability** | Rank stability across folds | Kendall's tau > 0.7 between folds |
| **Outlier ratio** | Values beyond 3-sigma or IQR bounds | Flag if significantly different from training |

---

## Metric Selection Guide

| Scenario | Primary Metric | Secondary |
|----------|---------------|-----------|
| General drift monitoring | Wasserstein Distance | Jensen-Shannon |
| Small dataset (<1K), high stakes | KS test | Chi-Square (categoricals) |
| Production monitoring (finance) | PSI | Wasserstein |
| Categorical features | Chi-Square | Jensen-Shannon |
| Label quality (classification) | Cleanlab confident learning | Cross-val residuals |
| Label quality (NL-to-SQL) | Execution Accuracy | SQL syntax validation |
| Feature stability over time | Kendall's tau (importance rank) | Correlation matrix drift |
| Text data distribution | Vocabulary overlap, token length | Wasserstein on embeddings |
