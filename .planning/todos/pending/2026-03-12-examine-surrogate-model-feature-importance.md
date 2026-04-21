---
created: 2026-03-12T02:36:55.876Z
title: Examine surrogate model feature importance
area: general
files:
  - aprx_model_elastica/model.py
  - aprx_model_elastica/train_surrogate.py
---

## Problem

The surrogate model predicts state transitions but we lack insight into which input features most strongly influence the predictions. Understanding feature importance would reveal:
- Which state variables (positions, velocities, curvatures, etc.) drive the dynamics most
- Whether certain features can be dropped to simplify the model without losing accuracy
- Physical intuition about what the model has learned about snake locomotion dynamics

## Solution

Approaches to investigate:
- **Permutation importance**: Shuffle each input feature and measure prediction degradation
- **Gradient-based saliency**: Compute ∂output/∂input for each feature across the validation set
- **SHAP values**: Use SHAP library for model-agnostic feature attribution
- **Ablation study**: Retrain with subsets of features and compare validation loss
- Visualize importance rankings and compare against physics intuition (e.g., do curvature features dominate?)
