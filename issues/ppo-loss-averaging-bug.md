---
name: PPO loss metrics averaged over theoretical max instead of actual updates
description: KL early stopping causes premature epoch termination but metrics divided by full epoch count, producing near-zero displayed values
type: issue
status: resolved
severity: high
subtype: training
created: 2026-03-19
updated: 2026-03-19
tags: [ppo, metrics, kl-early-stopping, logging]
aliases: [ppo-zero-loss-bug]
---

## Symptom

All PPO loss metrics (actor_loss, critic_loss, entropy, kl) displayed as 0.0000 in training logs despite rewards indicating active learning.

## Root Cause

In `src/trainers/ppo.py` `_update()`, metrics were accumulated across mini-batch updates then divided by `num_updates = num_epochs * num_batches` (the theoretical maximum). When KL early stopping triggered early (e.g., after 1-2 updates out of 40), the accumulated values were divided by 40x their actual count, producing values <0.00005 that rounded to 0.0000.

## Fix Applied

Replaced `num_updates = self.config.num_epochs * num_batches` with an `actual_updates` counter that increments only when `optimizer.step()` executes. Metrics are now divided by `actual_updates` instead of the theoretical max.

## Files Modified

- `src/trainers/ppo.py`: Track `actual_updates` counter, use it for metric averaging
