# Surrogate Data Validation Report

**Generated:** 2026-03-10 02:35:47 UTC
**Dataset:** 814,000 transitions, 27 batch files, 1628 episodes

## Pass/Fail Rubric

**Summary:** 7 PASS, 1 WARN, 0 FAIL

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| NaN/Inf rate | 0.0000% | PASS: 0% | WARN: <0.1% | FAIL: >=0.1% | **PASS** |
| Duplicate rate | 0.7439% | PASS: <0.1% | WARN: <1% | FAIL: >=1% | **WARN** |
| Constant features | 0 | PASS: 0 | WARN: 1-3 | FAIL: >3 | **PASS** |
| Outlier rate (>5 sigma) | 0.0309% | PASS: <0.5% | WARN: <2% | FAIL: >=2% | **PASS** |
| Episode length CV | 0.0000 | PASS: <0.5 | WARN: <1.0 | FAIL: >=1.0 | **PASS** |
| Step index bias (Q1/Q4) | 1.0000 | PASS: 0.7-1.3 | WARN: 0.5-1.5 | FAIL: outside | **PASS** |
| Action 5D fill fraction | 22.59% | PASS: >5% | WARN: >1% | FAIL: <=1% | **PASS** |
| Per-dim action fill | min=100% (amplitude=100%, frequency=100%, wave_number=100%, phase_offset=100%, direction_bias=100%) | PASS: all >90% | WARN: all >70% | FAIL: any <=70% | **PASS** |

## Distribution Analysis (DVAL-01)

### Summary Features

| Feature | Min | Max | Mean | Std | Skewness | Kurtosis |
|---------|-----|-----|------|-----|----------|----------|
| CoM_x | -1.2195 | 1.2866 | -0.0037 | 0.3489 | -0.0279 | 0.2745 |
| CoM_y | -1.2882 | 1.2490 | -0.0063 | 0.3502 | -0.0326 | 0.2033 |
| vel_mag | 0.0000 | 1.0722 | 0.2226 | 0.1034 | 1.6978 | 4.0294 |
| mean_omega | 0.0000 | 3.2197 | 0.0001 | 0.0046 | 494.7762 | 311333.4062 |

### Action Dimensions

| Dimension | Min | Max | Mean | Std | Skewness | Kurtosis |
|-----------|-----|-----|------|-----|----------|----------|
| amplitude | -1.0000 | 1.0000 | -0.0000 | 0.5774 | -0.0000 | -1.2000 |
| frequency | -1.0000 | 1.0000 | 0.0000 | 0.5773 | -0.0000 | -1.2000 |
| wave_number | -1.0000 | 1.0000 | -0.0000 | 0.5773 | 0.0000 | -1.2000 |
| phase_offset | -1.0000 | 1.0000 | 0.0000 | 0.5774 | -0.0000 | -1.2000 |
| direction_bias | -1.0000 | 1.0000 | -0.0000 | 0.5773 | 0.0000 | -1.2000 |

See: `figures/data_validation/summary_feature_histograms.png`, `figures/data_validation/action_histograms.png`

## Data Quality (DVAL-02)

### NaN/Inf Values
- States: 0
- Next states: 0
- Actions: 0
- **Total: 0 (0.0000%)**

### Duplicate Transitions
- Duplicates detected: 6,055 (0.7439%)
- Unique transitions: 807,945

### Constant/Near-Constant Features (std < 1e-6)
- Count: 0
- None detected

### Outlier Detection (>5 sigma)
- Total outliers: 31,195 (0.0309%)
- By state group:
  - pos_x: 0
  - pos_y: 0
  - vel_x: 11,449
  - vel_y: 11,550
  - yaw: 0
  - omega_z: 8,196

See: `figures/data_validation/outlier_counts.png`

## Temporal Analysis (DVAL-03)

### Episode Lengths
- Episodes: 1628
- Min: 500, Max: 500
- Mean: 500.0, Median: 500.0
- Std: 0.0
- CV (std/mean): 0.0000

### Step Index Bias
- Step range: 0 to 499
- First quartile transitions: 203,500
- Last quartile transitions: 203,500
- **Bias ratio (Q1/Q4): 1.0000**

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
- Occupied bins: 722,826 / 3,200,000
- **Fill fraction: 22.59%**

### Under-Sampled Regions
- None detected (all bins above threshold)

See: `figures/data_validation/action_coverage_per_dim.png`, `figures/data_validation/action_coverage_heatmap.png`

## Recommendations

- No significant issues found. Dataset appears well-suited for surrogate training.

## Overall Assessment

**GO** -- Dataset is ready for surrogate model training.
