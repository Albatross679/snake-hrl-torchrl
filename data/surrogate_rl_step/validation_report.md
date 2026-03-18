# Surrogate Data Validation Report

**Generated:** 2026-03-11 13:20:44 UTC
**Dataset:** 4,336,600 transitions, 43366 batch files, 4336600 episodes

## Pass/Fail Rubric

**Summary:** 7 PASS, 0 WARN, 1 FAIL

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| NaN/Inf rate | 0.0000% | PASS: 0% | WARN: <0.1% | FAIL: >=0.1% | **PASS** |
| Duplicate rate | 3.8398% | PASS: <0.1% | WARN: <1% | FAIL: >=1% | **FAIL** |
| Constant features | 0 | PASS: 0 | WARN: 1-3 | FAIL: >3 | **PASS** |
| Outlier rate (>5 sigma) | 0.0959% | PASS: <0.5% | WARN: <2% | FAIL: >=2% | **PASS** |
| Episode length CV | 0.0000 | PASS: <0.5 | WARN: <1.0 | FAIL: >=1.0 | **PASS** |
| Step index bias (Q1/Q4) | 0.9808 | PASS: 0.7-1.3 | WARN: 0.5-1.5 | FAIL: outside | **PASS** |
| Action 5D fill fraction | 74.47% | PASS: >5% | WARN: >1% | FAIL: <=1% | **PASS** |
| Per-dim action fill | min=100% (amplitude=100%, frequency=100%, wave_number=100%, phase_offset=100%, direction_bias=100%) | PASS: all >90% | WARN: all >70% | FAIL: any <=70% | **PASS** |

## Distribution Analysis (DVAL-01)

### Summary Features

| Feature | Min | Max | Mean | Std | Skewness | Kurtosis |
|---------|-----|-----|------|-----|----------|----------|
| CoM_x | -0.0022 | 0.0021 | -0.0000 | 0.0002 | 0.0069 | 7.0276 |
| CoM_y | -0.0023 | 0.0021 | -0.0000 | 0.0002 | -0.0053 | 7.0064 |
| vel_mag | 0.0000 | 0.0208 | 0.0038 | 0.0058 | 0.9340 | -1.0414 |
| mean_omega | 0.0000 | 2.4512 | 0.3587 | 0.5594 | 1.0029 | -0.8084 |

### Action Dimensions

| Dimension | Min | Max | Mean | Std | Skewness | Kurtosis |
|-----------|-----|-----|------|-----|----------|----------|
| amplitude | -1.0000 | 1.0000 | 0.0000 | 0.5774 | -0.0000 | -1.2000 |
| frequency | -1.0000 | 1.0000 | -0.0000 | 0.5774 | -0.0000 | -1.2000 |
| wave_number | -1.0000 | 1.0000 | 0.0000 | 0.5774 | 0.0000 | -1.2000 |
| phase_offset | -1.0000 | 1.0000 | 0.0000 | 0.5774 | 0.0000 | -1.2000 |
| direction_bias | -1.0000 | 1.0000 | -0.0000 | 0.5773 | -0.0000 | -1.2000 |

See: `figures/data_validation/summary_feature_histograms.png`, `figures/data_validation/action_histograms.png`

## Data Quality (DVAL-02)

### NaN/Inf Values
- States: 0
- Next states: 0
- Actions: 0
- **Total: 0 (0.0000%)**

### Duplicate Transitions
- Duplicates detected: 166,518 (3.8398%)
- Unique transitions: 4,170,082

### Constant/Near-Constant Features (std < 1e-6)
- Count: 0
- None detected

### Outlier Detection (>5 sigma)
- Total outliers: 515,833 (0.0959%)
- By state group:
  - pos_x: 8,176
  - pos_y: 8,041
  - vel_x: 169,721
  - vel_y: 169,275
  - yaw: 0
  - omega_z: 160,620

See: `figures/data_validation/outlier_counts.png`

## Temporal Analysis (DVAL-03)

### Episode Lengths
- Episodes: 4336600
- Min: 1, Max: 1
- Mean: 1.0, Median: 1.0
- Std: 0.0
- CV (std/mean): 0.0000

### Step Index Bias
- Step range: 0 to 150060999
- First quartile transitions: 1,081,500
- Last quartile transitions: 1,102,700
- **Bias ratio (Q1/Q4): 0.9808**

See: `figures/data_validation/episode_length_distribution.png`, `figures/data_validation/step_index_distribution.png`

## Action Coverage (DVAL-04)

### Per-Dimension Fill
- Bins per dimension: 20
- amplitude: 100%
- frequency: 100%
- wave_number: 100%
- phase_offset: 100%
- direction_bias: 100%

### 5D Joint Coverage
- Occupied bins: 2,382,957 / 3,200,000
- **Fill fraction: 74.47%**

### Under-Sampled Regions
- None detected (all bins above threshold)

See: `figures/data_validation/action_coverage_per_dim.png`, `figures/data_validation/action_coverage_heatmap.png`

## Recommendations

- Duplicate rate is 3.84%. Consider deduplicating before training or verifying collection pipeline.

## Overall Assessment

**NO-GO** -- Dataset has failing metrics that should be addressed before surrogate training.
