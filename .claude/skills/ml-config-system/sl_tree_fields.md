# SL Tree Fields Reference

Field specifications for SL Tree (Level 1b), SL Tree Regression (Level 2b), and project task-specific configs (Level 3). All inherit Base fields and built-in infrastructure (output, console, checkpointing, metricslog, mlflow).

---

## Level 1b: SL Tree

For tree ensemble models (XGBoost, LightGBM, CatBoost, etc.) that train via boosting rounds.

### Ensemble fields

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| n_estimators | integer | `1000` | Maximum number of boosting rounds |
| learning_rate | float | `0.1` | Shrinkage rate (step size for each tree added) |
| max_depth | integer | `6` | Maximum tree depth |
| min_child_weight | float | `1.0` | Minimum sum of instance weight in a child node |

### Sampling and regularization

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| subsample | float | `1.0` | Row subsampling ratio per boosting round (0.0–1.0) |
| colsample_bytree | float | `1.0` | Column subsampling ratio per tree (0.0–1.0) |
| reg_alpha | float | `0.0` | L1 regularization on leaf weights |
| reg_lambda | float | `1.0` | L2 regularization on leaf weights |

### Early stopping

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| early_stopping_rounds | integer | `50` | Stop after N rounds without improvement on validation metric; 0 disables |

### Tree-specific settings

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| tree_method | string | `"auto"` | Tree construction method: `"auto"`, `"hist"`, `"gpu_hist"`, `"exact"` |
| verbosity | integer | `1` | Logging verbosity (0=silent, 1=warning, 2=info, 3=debug) |

### Feature importance

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| log_feature_importance | boolean | `true` | Compute and save feature importance rankings |
| importance_type | string | `"gain"` | Importance metric: `"gain"`, `"weight"`, `"cover"` |
| top_n_features | integer | `20` | Number of top features to display in plots |

### SHAP explainability

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| compute_shap | boolean | `true` | Compute SHAP values for model explanations |
| shap_max_samples | integer | `1000` | Maximum samples for SHAP computation (controls speed) |

---

## Level 2b: SL Tree Regression

Inherits all SL Tree fields. Adds regression-specific configuration.

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| objective | string | `"reg:squarederror"` | Training objective (framework-native string). XGBoost: `"reg:squarederror"`, `"reg:squaredlogerror"`, `"reg:absoluteerror"`. LightGBM: `"regression"`, `"regression_l1"`, `"huber"`. CatBoost: `"RMSE"`, `"MAE"` |
| eval_metric | string | `"rmse"` | Validation metric for early stopping (framework-native string). XGBoost: `"rmse"`, `"mae"`, `"mape"`. LightGBM: `"rmse"`, `"mae"`. CatBoost: `"RMSE"`, `"MAE"` |
| eval_metrics | list of strings | `["rmse", "mae", "r2", "mape"]` | Metrics to compute and save in results (framework-agnostic names) |
| figures | list of strings | `["pred_vs_actual", "residual_dist", "feature_importance", "shap_importance", "shap_summary"]` | Figures to generate at the end of training |

---

## Level 3: Task-Specific Tree Regression Configs

These inherit all SL Tree Regression fields and add framework-specific or task-specific parameters.

### XGBoost Regression

Inherits: SL Tree Regression. Standard gradient boosted trees via XGBoost.

**Recommended parent overrides:**

| Field | Override | Reason |
|-------|----------|--------|
| n_estimators | `1000` | Default is fine |
| max_depth | `7` | Slightly deeper trees for complex interactions |
| learning_rate | `0.05` | Lower shrinkage for more trees |
| subsample | `0.8` | Row subsampling for regularization |
| colsample_bytree | `0.8` | Column subsampling for regularization |
| min_child_weight | `5` | Prevent overfitting on small leaf nodes |
| reg_alpha | `0.1` | Light L1 regularization |
| reg_lambda | `1.0` | Standard L2 regularization |
| tree_method | `"hist"` | Fast histogram-based training |

**Data config** (feature engineering for tabular models):

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| lag_hours | list of integers | `[1, 6, 24, 168]` | Hours to look back (converted to 15-min intervals for lag features) |
| rolling_windows | list of integers | `[24, 168]` | Hours for rolling mean/std features |
| add_interactions | boolean | `true` | Add interaction features (temp×area, humidity×area) |

**Checkpoint format:** `.json` (XGBoost native format)

---

### XGBoost Two-Stage Regression (Sparse Targets)

Inherits: SL Tree Regression. Two-stage pipeline for utilities with many zero readings (e.g., GAS). Stage 1 classifies on/off, Stage 2 predicts magnitude for non-zero only.

**Stage 1: Classifier params** (on/off detection):

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| classifier.n_estimators | integer | `500` | Boosting rounds for classifier |
| classifier.max_depth | integer | `6` | Tree depth |
| classifier.learning_rate | float | `0.05` | Shrinkage rate |
| classifier.subsample | float | `0.8` | Row subsampling |
| classifier.colsample_bytree | float | `0.8` | Column subsampling |
| classifier.min_child_weight | integer | `5` | Min child weight |
| classifier.scale_pos_weight | float | `1.0` | Class imbalance weight (adjusted at runtime) |
| classifier.tree_method | string | `"hist"` | Tree method |
| classifier.early_stopping_rounds | integer | `50` | Early stopping |
| classifier.eval_metric | string | `"logloss"` | Classification metric |

**Stage 2: Regressor params** (magnitude prediction):

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| regressor.n_estimators | integer | `1000` | Boosting rounds for regressor |
| regressor.max_depth | integer | `7` | Tree depth |
| regressor.learning_rate | float | `0.05` | Shrinkage rate |
| regressor.subsample | float | `0.8` | Row subsampling |
| regressor.colsample_bytree | float | `0.8` | Column subsampling |
| regressor.min_child_weight | integer | `5` | Min child weight |
| regressor.reg_alpha | float | `0.1` | L1 regularization |
| regressor.reg_lambda | float | `1.0` | L2 regularization |
| regressor.tree_method | string | `"hist"` | Tree method |
| regressor.early_stopping_rounds | integer | `50` | Early stopping |
| regressor.eval_metric | string | `"rmse"` | Regression metric |

**Task-specific fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| zero_threshold | float | `1e-5` | Values below this are treated as zero (off) |

**Data config:** Same as XGBoost Regression.

**Checkpoint format:** `.json` (two separate model files: classifier + regressor)

---

### LightGBM Regression

Inherits: SL Tree Regression. Gradient boosted trees via LightGBM (leaf-wise growth).

**Recommended parent overrides:**

| Field | Override | Reason |
|-------|----------|--------|
| n_estimators | `1000` | Default is fine |
| max_depth | `-1` | Unlimited depth; use `num_leaves` instead |
| learning_rate | `0.05` | Lower shrinkage |
| subsample | `0.8` | Row subsampling |
| colsample_bytree | `0.8` | Maps to `feature_fraction` |
| reg_alpha | `0.1` | Maps to `lambda_l1` |
| reg_lambda | `1.0` | Maps to `lambda_l2` |

**LightGBM-specific fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| num_leaves | integer | `63` | Maximum number of leaves per tree (LightGBM uses leaf-wise growth, not depth-wise) |
| min_child_samples | integer | `20` | Minimum samples in a leaf |
| verbose | integer | `50` | Log every N iterations |

**Data config:** Same as XGBoost Regression.

**Checkpoint format:** `.txt` (LightGBM native format)

---

### CatBoost Regression

Inherits: SL Tree Regression. Gradient boosted trees via CatBoost (ordered boosting).

**Recommended parent overrides:**

| Field | Override | Reason |
|-------|----------|--------|
| n_estimators | `1000` | Maps to `iterations` |
| max_depth | `7` | Maps to `depth` |
| learning_rate | `0.05` | Standard |
| reg_lambda | `3.0` | Maps to `l2_leaf_reg` (CatBoost's primary regularizer) |

**CatBoost-specific fields:**

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| random_strength | float | `1.0` | Amount of randomness for scoring splits |
| bagging_temperature | float | `1.0` | Controls Bayesian bootstrap intensity |
| border_count | integer | `254` | Number of splits for numerical features |
| verbose | integer | `50` | Log every N iterations |

**Data config:** Same as XGBoost Regression.

**Checkpoint format:** `.cbm` (CatBoost native format)

---

## Metrics Contract

### SL Tree Metrics

Tree models log metrics reflecting boosting-round-based training. All metrics are dual-logged to both `metrics.jsonl` (local) and MLflow (SQLite via `log_epoch_metrics`):

| Metric Tag | Frequency | Controlled By |
|------------|-----------|---------------|
| `eval/{eval_metric}` | per boosting round (logged by framework callback) | `eval_metric` field |
| `feature_importance/{importance_type}` | end of training | `log_feature_importance` |
| `shap/summary` | end of training | `compute_shap` |

### SL Tree Regression Metrics

Final results saved to `metrics.json`, containing:

| Metric | Source | Controlled By |
|--------|--------|---------------|
| Each name in `eval_metrics` (rmse, mae, r2, mape) | computed on test set | `eval_metrics` list |
| `n_trees_used` | actual trees used (may differ from `n_estimators` due to early stopping) | always |
| `top_features` | top N features by importance | `top_n_features` |

Figures generated per `figures` list and saved to `plots/` subdirectory.

---

## Framework Param Mapping

Tree frameworks use different parameter names for the same concepts. When generating code, map SL Tree fields to the target framework:

| SL Tree Field | XGBoost | LightGBM | CatBoost |
|---------------|---------|----------|----------|
| `n_estimators` | `n_estimators` | `n_estimators` | `iterations` |
| `learning_rate` | `learning_rate` / `eta` | `learning_rate` | `learning_rate` |
| `max_depth` | `max_depth` | `max_depth` | `depth` |
| `min_child_weight` | `min_child_weight` | `min_child_weight` | `min_data_in_leaf` (different semantics) |
| `subsample` | `subsample` | `bagging_fraction` | `subsample` |
| `colsample_bytree` | `colsample_bytree` | `feature_fraction` | `rsm` |
| `reg_alpha` | `reg_alpha` / `alpha` | `lambda_l1` | `l2_leaf_reg` (L2 only) |
| `reg_lambda` | `reg_lambda` / `lambda` | `lambda_l2` | `l2_leaf_reg` |
| `early_stopping_rounds` | `early_stopping_rounds` | `early_stopping_rounds` (via callback) | `od_wait` |
| `tree_method` | `tree_method` | N/A | N/A |
