---
name: Non-Uniform Data Coverage in Supervised Learning
description: Strategies for handling imbalanced or non-uniform feature space coverage in supervised learning, with application to physics surrogate models
type: knowledge
created: 2026-03-09
updated: 2026-03-09
tags: [supervised-learning, data-imbalance, surrogate-model, sampling, loss-weighting]
aliases: [data imbalance, non-uniform coverage, underrepresented data]
---

# Non-Uniform Data Coverage in Supervised Learning

When training data is highly concentrated in some regions of the feature space and sparse in others, the model learns a biased approximation that performs well only in dense regions.

## 1. Data-Level Approaches

- **Oversampling sparse regions** — SMOTE, ADASYN, or simple duplication of underrepresented samples
- **Undersampling dense regions** — random or informed (e.g., Tomek links, cluster centroids)
- **Active data collection** — deliberately sample states from underrepresented regions using uniform grid, Latin hypercube sampling (LHS), or Sobol sequences
- **Exploration noise** — add noise during collection to push into underrepresented regions

## 2. Loss-Level Approaches

- **Inverse frequency/density weighting** — weight each sample by `1/density` so sparse regions contribute more to the loss; estimate density via KDE or histogram binning
- **Focal loss** — upweight hard/rare examples automatically
- **Per-region loss balancing** — bin the feature space and normalize loss contributions per bin

## 3. Architecture / Training Approaches

- **Mixture of experts** — different sub-networks specialize in different regions
- **Curriculum learning** — progressively shift training focus toward underrepresented regions
- **Ensemble with stratified bagging** — each model sees a balanced subset

## 4. Application to Physics Surrogate Models

For neural surrogate models trained on simulation data (e.g., Elastica Cosserat rod dynamics):

1. **Design the collection policy** to sweep the state-action space uniformly (grid/LHS/Sobol) rather than relying on a trained policy's trajectory distribution, which clusters around a few modes
2. **Add exploration noise** during collection to push into underrepresented regions
3. **Use sample weighting** during training — estimate density and weight inversely

The combination of **uniform/space-filling data collection + inverse-density weighting** tends to give the best results for physics surrogate models.
