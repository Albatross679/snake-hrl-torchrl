---
phase: quick-260319-stp
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - papers/choi2025/config.py
autonomous: true
requirements: []
---

<objective>
Align choi2025 PPO config to match the original paper's Table A.1 as closely as possible.

SAC config is already well-aligned (verified against Table A.1). The PPO config uses 1024×3 networks instead of the paper's 256×3.

Output: PPO config uses paper-matching 256×3 network, matching SAC's network config.
</objective>

<execution_context>
@/home/user/snake-hrl-torchrl/.claude/get-shit-done/workflows/execute-plan.md
@/home/user/snake-hrl-torchrl/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@papers/choi2025/config.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Align PPO network to paper's 256×3 and document SAC verification</name>
  <files>papers/choi2025/config.py</files>
  <action>
In `Choi2025PPOConfig`:
1. Change `network` field from `Choi2025NetworkConfig` (1024×3) to `Choi2025PaperNetworkConfig` (256×3) to match Table A.1
2. Update docstring/comments to document alignment with paper

Paper Table A.1 parameters (SAC, verified matching):
- Number of envs: 500 ✓
- UTD ratio: 4 ✓
- Hidden dim: (256, 256, 256) ✓
- τ: 0.005 ✓ (base SACConfig default)
- Soft update period: 8 ✓
- γ: 0.99 ✓ (base RLConfig default)
- Entropy: None ✓ (alpha=0.0, auto_alpha=False)
- Optimizer: Adam ✓
- lr: 0.001 ✓
- Batch size: 2048 ✓
- Buffer: 2,000,000 ✓
  </action>
  <verify>
    <automated>cd /home/user/snake-hrl-torchrl && python -c "from choi2025.config import Choi2025PPOConfig; c = Choi2025PPOConfig(); print('PPO network:', c.network.actor.hidden_dims); assert c.network.actor.hidden_dims == [256, 256, 256], f'Expected [256,256,256] got {c.network.actor.hidden_dims}'; print('OK')"</automated>
  </verify>
  <done>PPO config uses paper-matching 256×3 network</done>
</task>

</tasks>

<verification>
- `python -c "from choi2025.config import Choi2025PPOConfig; c = Choi2025PPOConfig(); assert c.network.actor.hidden_dims == [256,256,256]"`
- `python -c "from choi2025.config import Choi2025Config; c = Choi2025Config(); assert c.network.actor.hidden_dims == [256,256,256]"`
</verification>

<success_criteria>
Both SAC and PPO configs use the paper's 256×3 network. All paper Table A.1 parameters verified matching.
</success_criteria>

<output>
After completion, create `.planning/quick/260319-stp-align-choi2025-sac-and-ppo-configs-to-ma/260319-stp-SUMMARY.md`
</output>
