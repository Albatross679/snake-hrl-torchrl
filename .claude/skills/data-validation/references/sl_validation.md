# Supervised Learning Data Validation

## Table of Contents
1. [Input Data Validation](#input-data-validation)
2. [Label Quality Validation](#label-quality-validation)
3. [Train/Val/Test Split Validation](#trainvaltest-split-validation)
4. [Feature Drift Detection](#feature-drift-detection)
5. [NLP/Text-Specific Checks](#nlptext-specific-checks)
6. [NL-to-SQL Specific Validation](#nl-to-sql-specific-validation)
7. [Deepchecks Full Suite Example](#deepchecks-full-suite-example)

---

## Input Data Validation

**What to validate:**
- **Data types:** Columns have expected types (string, int, float, categorical)
- **Completeness:** Missing values, null ratios, required fields
- **Value ranges:** Numerical features within expected min/max bounds
- **Cardinality:** Categorical features have expected number of unique values
- **Format compliance:** Strings match expected patterns (dates, IDs, SQL queries)
- **Uniqueness:** No unexpected duplicates in ID columns or primary keys

**Schema generation (Google TFDV approach):**
1. Auto-infer schema from training data statistics
2. Manually review and lock it
3. Version schema alongside data and model
4. Flag deviations (new categories, out-of-range, type mismatches)

**Pandera example:**
```python
import pandera as pa
from pandera import Column, Check, DataFrameSchema

schema = DataFrameSchema({
    "input_text": Column(str, [
        Check(lambda s: s.str.len() > 0, error="Empty input"),
        Check(lambda s: s.str.len() < 2048, error="Input too long"),
    ]),
    "target_sql": Column(str, [
        Check(lambda s: s.str.contains(
            "SELECT|INSERT|UPDATE|DELETE", regex=True, case=False
        ), error="Target must contain SQL keywords"),
    ]),
    "split": Column(str, Check.isin(["train", "dev", "test"])),
})
validated_df = schema.validate(df)
```

---

## Label Quality Validation

Label errors are pervasive. Cleanlab found 100K+ errors in ImageNet. For NL-to-SQL, label quality means SQL correctness.

| Method | What It Detects | Tool |
|--------|----------------|------|
| Confident Learning | Mislabeled examples via joint noise estimation | Cleanlab |
| Inter-annotator Agreement | Inconsistent labeling across annotators | Cohen's Kappa, Krippendorff's Alpha |
| Execution Consistency | SQL labels that produce wrong results | Custom (execute and compare) |
| Cross-validation residuals | Examples the model consistently gets wrong | Any ML framework |
| Data Shapley | Per-example contribution to model performance | OpenDataVal |

**Cleanlab example:**
```python
from cleanlab import Datalab

lab = Datalab(data=dataset, label_name="label")
lab.find_issues(pred_probs=model_pred_probs, features=embeddings)
lab.report()  # Shows label issues, outliers, near-duplicates

label_issues = lab.get_issues("label")
problematic = label_issues[label_issues["is_label_issue"]].index
print(f"Found {len(problematic)} potential label errors")
```

---

## Train/Val/Test Split Validation

| Check | Why | Method |
|-------|-----|--------|
| Duplicate detection | Duplicates across splits inflate metrics | MinHash + LSH (near-dedup); exact hash (exact dedup) |
| Feature leakage | Features encoding target info | Correlation analysis, feature importance on random labels |
| Temporal leakage | Future data in training set | Timestamp-based split verification |
| Distribution alignment | Train/test mismatch | KS test, PSI, Wasserstein on key features |
| Stratification | Class balance across splits | Chi-square on label distributions |
| Group leakage | Same entity in multiple splits | Group-aware splitting |

**Near-dedup for text (MinHash-LSH):**
```python
from datasketch import MinHash, MinHashLSH

def get_minhash(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    for word in text.split():
        m.update(word.encode('utf8'))
    return m

# Build LSH index from training set
lsh = MinHashLSH(threshold=0.8, num_perm=128)
for idx, text in enumerate(train_texts):
    lsh.insert(f"train_{idx}", get_minhash(text))

# Query test texts for leakage
leaks = []
for idx, text in enumerate(test_texts):
    matches = lsh.query(get_minhash(text))
    if matches:
        leaks.append((idx, matches))
print(f"Found {len(leaks)} near-duplicate leaks across splits")
```

---

## Feature Drift Detection

**Types of drift:**

| Type | What Changes | Impact | Detection |
|------|-------------|--------|-----------|
| Data drift (covariate shift) | Input distributions | Unfamiliar inputs | Statistical tests on features |
| Concept drift | Input-target relationship | Learned patterns invalid | Monitor accuracy over time |
| Schema drift | Data format/structure | Pipeline breaks | Schema validation |
| Label drift (prior shift) | Target distribution | Calibration wrong | Monitor prediction distribution |

**Evidently drift detection:**
```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

report = Report(metrics=[
    DataDriftPreset(stattest="wasserstein"),
    DataQualityPreset(),
])
report.run(reference_data=train_df, current_data=new_df)
report.save_html("drift_report.html")

# Programmatic access
results = report.as_dict()
for col, col_result in results["metrics"][0]["result"]["drift_by_columns"].items():
    if col_result["drift_detected"]:
        print(f"DRIFT in {col}: score={col_result['drift_score']:.4f}")
```

**whylogs profiling:**
```python
import whylogs as why

# Training time: save reference profile
train_profile = why.log(training_df).profile()
train_profile.save("reference_profile.bin")

# Serving time: compare
serving_profile = why.log(serving_batch).profile()
from whylogs.viz import NotebookProfileVisualizer
viz = NotebookProfileVisualizer()
viz.set_profiles(target_profile=serving_profile, reference_profile=train_profile)
viz.summary_drift_report()
```

---

## NLP/Text-Specific Checks

- **Token length distributions:** min, max, mean, percentiles between train/test
- **Vocabulary coverage:** OOV rate between splits
- **Character encoding consistency:** UTF-8 validation
- **Empty/whitespace-only strings:** Filter before training
- **Language detection consistency:** Ensure monolingual or expected multilingual mix

---

## NL-to-SQL Specific Validation

```python
import sqlite3

def validate_nl_to_sql_dataset(data, db_path):
    """Validate NL-to-SQL dataset quality."""
    conn = sqlite3.connect(db_path)
    issues = []

    for idx, (nl, sql) in enumerate(data):
        # 1. SQL syntax check
        try:
            conn.execute(f"EXPLAIN QUERY PLAN {sql}")
        except sqlite3.OperationalError as e:
            issues.append({"idx": idx, "type": "syntax_error", "detail": str(e)})
            continue

        # 2. SQL execution check
        try:
            result = conn.execute(sql).fetchall()
        except Exception as e:
            issues.append({"idx": idx, "type": "execution_error", "detail": str(e)})
            continue

        # 3. Empty result check
        if len(result) == 0:
            issues.append({"idx": idx, "type": "empty_result", "detail": "SQL returns no rows"})

        # 4. NL length check
        if len(nl.split()) < 3:
            issues.append({"idx": idx, "type": "short_input", "detail": f"Only {len(nl.split())} words"})

    conn.close()
    return issues
```

---

## Deepchecks Full Suite Example

```python
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation
from deepchecks.tabular.checks import (
    DataDuplicates, FeatureFeatureCorrelation,
    TrainTestFeatureDrift, TrainTestLabelDrift
)

train_ds = Dataset(train_df, label="target", features=feature_cols)
test_ds = Dataset(test_df, label="target", features=feature_cols)

# Full integrity suite
integrity_result = data_integrity().run(train_ds)
integrity_result.save_as_html("data_integrity_report.html")

# Train-test validation suite
tt_result = train_test_validation().run(train_ds, test_ds)
tt_result.save_as_html("train_test_report.html")

# Individual checks
DataDuplicates().run(train_ds)
TrainTestFeatureDrift().run(train_ds, test_ds)
```
