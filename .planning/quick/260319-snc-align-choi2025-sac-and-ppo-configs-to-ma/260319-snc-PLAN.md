---
phase: quick-260319-snc
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - papers/choi2025/config.py
  - papers/choi2025/env.py
  - src/configs/training.py
  - src/trainers/sac.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "SAC config matches paper: auto_alpha=False, soft_update_period=8, num_envs=500"
    - "PPO config matches paper: same env settings (dt, gravity, substeps), num_envs=500"
    - "Physics config matches paper: dt=0.05, enable_gravity=True, max_newton_iter=2"
    - "Environment steps physics multiple substeps per action (control_period=2 for 10 Hz control)"
    - "SAC trainer respects soft_update_period from config (every N updates, not every update)"
  artifacts:
    - path: "papers/choi2025/config.py"
      provides: "Aligned SAC and PPO configs"
    - path: "papers/choi2025/env.py"
      provides: "Multi-substep physics stepping"
    - path: "src/configs/training.py"
      provides: "soft_update_period field on SACConfig"
    - path: "src/trainers/sac.py"
      provides: "soft_update_period gating logic"
  key_links:
    - from: "papers/choi2025/config.py"
      to: "src/configs/training.py"
      via: "SACConfig.soft_update_period inherited by Choi2025Config"
    - from: "src/trainers/sac.py"
      to: "src/configs/training.py"
      via: "self.config.soft_update_period controls _soft_update frequency"
    - from: "papers/choi2025/env.py"
      to: "papers/choi2025/config.py"
      via: "control_period drives substep loop in _step()"
---

<objective>
Align choi2025 SAC and PPO configs to match the original paper's hyperparameters. Fix 7 confirmed mismatches: dt/control period, entropy auto-tuning, gravity, max Newton iterations, soft update period, num_envs, and temporal smoothing (substep interpolation).

Purpose: Without these fixes, training results are not comparable to the paper. The dt/control period mismatch alone changes the fundamental control frequency from the paper's 10 Hz to our incorrect 100 Hz.

Output: Corrected config files and trainer logic that faithfully reproduce the paper's setup.
</objective>

<context>
@papers/choi2025/config.py
@papers/choi2025/env.py
@src/configs/training.py
@src/trainers/sac.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Add soft_update_period to SACConfig and implement in trainer</name>
  <files>src/configs/training.py, src/trainers/sac.py</files>
  <action>
1. In `src/configs/training.py` SACConfig, add field:
   - `soft_update_period: int = 1` (default 1 preserves backward compat — update target every critic update)

2. In `src/trainers/sac.py` `_update()` method, gate `self._soft_update()` call (currently at line 543) so it only fires every `self.config.soft_update_period` updates. Use `self._update_count` which is already tracked:
   ```python
   # Update target networks (every soft_update_period updates)
   if self._update_count % self.config.soft_update_period == 0:
       self._soft_update()
   ```
   Move the `self._soft_update()` call inside this conditional. The `_update_count` increment (line 508) already happens before this point, so the gating works correctly.

No other files import or depend on these fields, so backward compatibility is preserved.
  </action>
  <verify>
    <automated>cd /home/user/snake-hrl-torchrl && python -c "from src.configs.training import SACConfig; c = SACConfig(); assert hasattr(c, 'soft_update_period'); assert c.soft_update_period == 1; print('OK: soft_update_period field exists with default 1')"</automated>
  </verify>
  <done>SACConfig has soft_update_period field defaulting to 1. SACTrainer._update() only calls _soft_update() every soft_update_period updates.</done>
</task>

<task type="auto">
  <name>Task 2: Fix all choi2025 config mismatches (SAC, PPO, physics, env)</name>
  <files>papers/choi2025/config.py, papers/choi2025/env.py</files>
  <action>
**A. Fix Choi2025PhysicsConfig in config.py:**
- `dt: float = 0.05` (was 0.01 — paper uses 50ms substep)
- `enable_gravity: bool = True` (was False — paper explicitly enables gravity)
- `max_newton_iter_noncontact: int = 2` (was 15 — paper uses 2, saves compute)
- `max_newton_iter_contact: int = 5` (was 25 — paper uses 5 for contact tasks)

**B. Add control_period to Choi2025EnvConfig in config.py:**
- Add field `control_period: int = 2` with docstring: "Number of physics substeps per RL action. Paper uses 2 substeps at dt=0.05 = 0.1s per action = 10 Hz control."

**C. Fix Choi2025Config (SAC) in config.py:**
- `auto_alpha: bool = False` — override SACConfig default. Paper explicitly excludes entropy tuning (citing Yu et al. 2022). Set `alpha: float = 0.0` to disable entropy bonus entirely.
- `soft_update_period: int = 8` — paper updates target network every 8 critic updates
- `num_envs: int = 500` (was 32 — paper uses 500)
- Update `__post_init__` name template to reflect 500 envs

**D. Fix Choi2025PPOConfig in config.py:**
- `num_envs: int = 500` (was 1 — match SAC parallelism for fair comparison)
- `total_frames: int = 5_000_000` (was 1_000_000 — match SAC training budget)

**E. Implement multi-substep physics in env.py `_step()`:**
In `SoftManipulatorEnv._step()`, after computing `curvature_state` from the action, loop over `self.config.control_period` substeps instead of doing a single physics step:

```python
# Apply delta curvature control (once per RL step)
curvature_state = self.controller.apply_delta(action, two_d_sim=self._two_d_sim)

# Step physics for control_period substeps (temporal smoothing)
num_substeps = getattr(self.config, 'control_period', 1)
for _ in range(num_substeps):
    self._apply_curvature(curvature_state)
    if self._use_dismech:
        try:
            self._dismech_robot, _ = self._time_stepper.step(
                self._dismech_robot, debug=False
            )
        except ValueError:
            pass
```

Remove the existing single-step physics block (lines 462-468) and replace with the loop above. The mock physics path already steps in `_apply_curvature`, so for mock, only call `_apply_curvature` once (on the first substep) to avoid double-stepping:

```python
for substep_i in range(num_substeps):
    if substep_i == 0 or self._use_dismech:
        self._apply_curvature(curvature_state)
    if self._use_dismech:
        try:
            self._dismech_robot, _ = self._time_stepper.step(
                self._dismech_robot, debug=False
            )
        except ValueError:
            pass
```

Also update the target movement to account for total time per RL step:
```python
if self.config.task == TaskType.FOLLOW_TARGET:
    self._target.step(self.config.physics.dt * num_substeps)
```

**F. Update docstrings** in Choi2025PhysicsConfig to reflect the corrected values. Update the class docstring bullet "dt=0.01s" to "dt=0.05s (50ms substep, 10 Hz control with period=2)".
  </action>
  <verify>
    <automated>cd /home/user/snake-hrl-torchrl && python -c "
from choi2025.config import Choi2025Config, Choi2025PPOConfig, Choi2025PhysicsConfig, Choi2025EnvConfig
# Physics
p = Choi2025PhysicsConfig()
assert p.dt == 0.05, f'dt={p.dt}'
assert p.enable_gravity == True, f'gravity={p.enable_gravity}'
assert p.max_newton_iter_noncontact == 2, f'newton={p.max_newton_iter_noncontact}'
# SAC
c = Choi2025Config()
assert c.auto_alpha == False, f'auto_alpha={c.auto_alpha}'
assert c.alpha == 0.0, f'alpha={c.alpha}'
assert c.soft_update_period == 8, f'sup={c.soft_update_period}'
assert c.num_envs == 500, f'num_envs={c.num_envs}'
# PPO
pp = Choi2025PPOConfig()
assert pp.num_envs == 500, f'ppo_envs={pp.num_envs}'
assert pp.total_frames == 5_000_000, f'ppo_frames={pp.total_frames}'
# Env control period
e = Choi2025EnvConfig()
assert e.control_period == 2, f'control_period={e.control_period}'
print('ALL CHECKS PASSED')
"</automated>
  </verify>
  <done>All 7 paper mismatches fixed: dt=0.05, gravity ON, newton_iter=2, auto_alpha=False with alpha=0.0, soft_update_period=8, num_envs=500 (both SAC and PPO), control_period=2 with multi-substep physics loop in env.</done>
</task>

</tasks>

<verification>
1. Config values match paper Table A.1:
   - dt=0.05, control_period=2 (10 Hz control)
   - No entropy auto-tuning (auto_alpha=False, alpha=0.0)
   - Gravity enabled
   - Newton iterations: 2 (non-contact), 5 (contact)
   - Soft update period: 8
   - 500 parallel environments
2. SACTrainer only updates target network every soft_update_period steps
3. Environment steps physics twice per action (substep loop)
4. PPO config aligned (same env, 500 envs, 5M frames)
</verification>

<success_criteria>
- `python -c "from choi2025.config import Choi2025Config; ..."` verifies all hyperparameters
- SAC trainer compiles and soft_update gating works
- Environment substep loop executes without errors on mock backend
</success_criteria>

<output>
After completion, create `.planning/quick/260319-snc-align-choi2025-sac-and-ppo-configs-to-ma/260319-snc-SUMMARY.md`
</output>
