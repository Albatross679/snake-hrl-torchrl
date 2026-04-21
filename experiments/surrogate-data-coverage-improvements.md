---
name: Surrogate Data Coverage Improvements
description: Sobol actions, state perturbation, curvature perturbation, and inverse density weighting for surrogate training data
type: experiment
status: complete
created: 2026-03-09
updated: 2026-03-09
tags: [surrogate, data-collection, exploration, coverage]
---

# Surrogate Data Coverage Improvements

## Objective

Address three coverage gaps in the surrogate training data pipeline to improve model accuracy across the full state-action space.

## Problem Analysis

The original data collection had three systematic biases:

1. **Action space coverage**: Uniform random sampling in 5D leaves large gaps — random points cluster in the center of the hypercube, leaving corners and edges undersampled.

2. **Initial state bias**: The rod always starts straight (zero curvature). During actual locomotion, the rod is continuously curved by the serpenoid controller. The surrogate never sees transitions starting from bent configurations.

3. **State distribution skew**: Most transitions come from common states (straight rod near origin, low velocity). Rare but important states (high curvature, fast movement, extreme angles) are underrepresented in training.

## Solutions Implemented

### 1. Sobol Quasi-Random Actions

**What**: Replace uniform random action sampling with scrambled Sobol sequences (`torch.quasirandom.SobolEngine`).

**Why**: Sobol sequences are quasi-random — they fill the 5D action hypercube more evenly than pseudorandom sampling. For the same number of samples, Sobol covers ~4x more unique regions of action space.

**Config**: `use_sobol_actions: bool = True` (default on), `--sobol` / `--no-sobol` CLI flags.

**Implementation**: `SobolActionSampler` class maps Sobol points from [0,1]^5 to [-1,1]^5. Each worker uses a different Sobol seed (`config.seed + worker_id * 1000`) for complementary coverage.

### 2. Initial State Perturbation

**What**: After env reset, add Gaussian noise to rod positions, velocities, and angular velocities, plus set a randomized sinusoidal rest curvature.

**Why**: The rod always resets to a straight configuration. Adding perturbation means the surrogate trains on transitions starting from diverse states, including mid-locomotion-like bent configurations.

**Config**:
- `perturbation_fraction: float = 0.3` — 30% of episodes get perturbed initial states
- `perturb_position_std: float = 0.002` — position noise (meters, rod is 0.5m long)
- `perturb_velocity_std: float = 0.01` — velocity noise (m/s)
- `perturb_omega_std: float = 0.05` — angular velocity noise (rad/s)
- `perturb_curvature_max: float = 3.0` — max curvature amplitude (rad/m, serpenoid range 0-5)

**Curvature perturbation details**: Sets `rod.rest_kappa[0, :]` to a randomized sinusoidal pattern:
```
curvature(s) = A * sin(2π * k * s + φ) + b
```
where A ~ U(0, 3.0), k ~ U(0.5, 3.5), φ ~ U(0, 2π), b ~ U(-1, 1). This matches the range of curvatures the serpenoid controller produces during locomotion.

### 3. Inverse Density Weighting (Training)

**What**: During surrogate model training, weight samples inversely proportional to their density in state space. Rare states get upweighted, common states get downweighted.

**Why**: Even with perturbation and Sobol actions, the collected data distribution is non-uniform. Density weighting ensures the model allocates capacity to rare but important state regions.

**Implementation**:
1. Project 124D states to 4 summary features: CoM_x, CoM_y, velocity_magnitude, mean|omega_z|
2. Bin each feature into 20 histogram bins → joint bin index
3. Weight = 1 / bin_count, normalized to mean=1, clipped at 10x
4. Feed weights to `torch.utils.data.WeightedRandomSampler`

**Config** (in `SurrogateTrainConfig`):
- `use_density_weighting: bool = True`
- `density_bins: int = 20`
- `density_clip_max: float = 10.0`

## Files Changed

- `aprx_model_elastica/collect_config.py` — added Sobol, perturbation, curvature config fields
- `aprx_model_elastica/collect_data.py` — added `SobolActionSampler`, `perturb_rod_state()`, CLI args, wiring
- `aprx_model_elastica/train_config.py` — added density weighting config fields
- `aprx_model_elastica/dataset.py` — added `compute_density_weights()`, `get_sample_weights()`, parquet loading
- `aprx_model_elastica/train_surrogate.py` — conditional `WeightedRandomSampler` in DataLoader

## All features are configurable

Every improvement can be toggled on/off via config dataclass defaults or CLI arguments. This allows ablation studies to measure the marginal value of each technique.
