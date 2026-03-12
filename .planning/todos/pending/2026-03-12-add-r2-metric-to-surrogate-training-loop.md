---
created: 2026-03-12T16:57:48.633Z
title: Add R2 metric to surrogate training loop
area: general
files:
  - aprx_model_elastica/train_surrogate.py
  - aprx_model_elastica/model.py
---

## Problem

The surrogate model training currently only logs per-component MSE losses (pos_x, pos_y, vel_x, vel_y, yaw, omega_z) and aggregate train/val loss to W&B. There is no R2 (coefficient of determination) metric, which makes it harder to assess how well the model explains variance in each output component. MSE alone doesn't convey whether the model is meaningfully better than predicting the mean.

## Solution

Compute R2 = 1 - MSE/Var per component on the validation set each epoch. Log per-component R2 and an aggregate R2 to W&B alongside existing metrics. This is a lightweight addition — just needs variance computation from the validation targets.
