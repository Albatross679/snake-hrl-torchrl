---
name: MM-RKHS 100K Validation - follow_target
description: Quick 100K-frame MM-RKHS training validation on follow_target task to verify learning signal
type: experiment
status: complete
created: 2026-03-19
updated: 2026-03-19
tags: [mmrkhs, validation, follow_target, choi2025, mm-rkhs]
aliases: []
---

# MM-RKHS 100K Validation - follow_target

## Setup

- **Algorithm:** MM-RKHS (Operator-Theoretic Policy Gradient with MM-RKHS loss)
- **Task:** follow_target (soft manipulator tracks moving target)
- **Frames:** 100K (106496 actual due to batch rounding)
- **Environments:** 32 parallel (ParallelEnv, CPU workers)
- **Network:** 3x256 ReLU MLP (paper architecture)
- **Device:** CUDA (RTX A4000)
- **Seed:** 42
- **W&B Run:** https://wandb.ai/qifan_wen-ohio-state-university/choi2025-replication/runs/bswz0spf

## Hyperparameters

| Parameter | Value |
|---|---|
| beta (MMD) | 1.0 |
| eta (KL) | 1.0 |
| learning_rate | 3e-4 |
| frames_per_batch | 8192 |
| mini_batch_size | 1024 |
| num_epochs | 10 |
| gae_lambda | 0.95 |
| mmd_bandwidth | 1.0 |
| mmd_num_samples | 16 |

## Results

- **Duration:** ~2 min 21s
- **Episodes:** 512
- **Best reward:** 21.18
- **Final rolling-100 reward:** 16.92
- **Reward progression:** 9.38 -> 10.79 -> 17.74 -> 16.92 (non-monotonic but trending up)

## Key Metrics

| Metric | First Batch | Last Batch |
|---|---|---|
| policy_loss | -0.0000 | -0.0000 |
| critic_loss | 1.9030 | 3.4453 |
| mmd_penalty | 0.0015 | 0.0011 |
| kl_divergence | 8.0396 | 5.8474 |
| mean_reward | 9.38 | 20.52 |

## Observations

1. Learning signal confirmed: reward improved from ~9 to ~17-21 over 100K frames
2. MMD penalty stays small (0.0008-0.0015), indicating policy doesn't drift far from old policy
3. KL divergence decreases over training (8.0 -> 5.8), suggesting policy converges
4. Policy loss near zero throughout -- expected since MM-RKHS uses surrogate MMD loss not direct policy gradient
5. Critic loss increases slightly (1.9 -> 3.4) as value targets become more varied with improving policy
6. No NaN or training instability

## Conclusion

MM-RKHS produces a viable learning signal on follow_target. The 100K validation confirms the trainer is wired correctly for benchmark comparison with PPO and SAC.
