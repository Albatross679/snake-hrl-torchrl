# Snake-HRL Project Todo List

A hierarchical checklist of tasks to diagnose and fix the snake robot RL training pipeline.

---

## Phase 0: Verification & Diagnostics (Do First!)

### 0.1 Physics Simulation Verification
- [ ] **Test basic undulation**
  - [ ] Apply constant sinusoidal curvature wave to all joints
  - [ ] Verify snake body visually undulates
  - [ ] Record: Does the snake deform correctly? (Y/N)
- [ ] **Test forward locomotion**
  - [ ] Run serpenoid controller with default parameters (A=1.0, f=1.0, k=1.5)
  - [ ] Measure displacement after 1000 timesteps
  - [ ] Expected: 0.1-0.5 m forward movement
  - [ ] Record actual displacement: ______ m
- [ ] **Test physics stability**
  - [ ] Run simulation for 10,000 steps
  - [ ] Check for NaN values in positions/velocities
  - [ ] Check for energy explosion (velocities > 100 m/s)
  - [ ] Record: Simulation stable? (Y/N)

### 0.2 Ground Contact & Friction Verification
- [ ] **Verify ground contact model**
  - [ ] Check that snake nodes stay above ground (z >= 0)
  - [ ] Verify normal forces prevent penetration
- [ ] **Test friction anisotropy**
  - [ ] Current friction coefficient: μ = 0.5
  - [ ] Test with μ = 0.3, 0.5, 0.7, 1.0
  - [ ] Record which produces best forward motion: μ = ______
- [ ] **Consider anisotropic friction**
  - [ ] Snake locomotion requires more friction perpendicular to body
  - [ ] Check if DisMech supports directional friction
  - [ ] If not, consider implementing or approximating

### 0.3 Manual Task Achievement Test
- [ ] **Write scripted approach controller**
  - [ ] Implement: Orient toward prey, then undulate forward
  - [ ] Test: Can it reach within 0.15m of prey?
  - [ ] Record success rate: ______%
- [ ] **Write scripted coil controller**
  - [ ] Implement: Position near prey, apply coiling curvature
  - [ ] Test: Can it achieve contact_fraction > 0.6?
  - [ ] Test: Can it achieve wrap_count > 1.5?
  - [ ] Record success rate: ______%
- [ ] **If manual control fails → Fix physics before RL**

### 0.4 Reward Signal Verification
- [ ] **Log raw rewards every step**
  - [ ] Add logging to environment step function
  - [ ] Run random policy for 1000 steps
  - [ ] Record: Mean reward = ______, Std = ______
- [ ] **Verify reward components activate**
  - [ ] Distance reward changing? (Y/N)
  - [ ] Velocity reward non-zero? (Y/N)
  - [ ] Energy penalty applied? (Y/N)
- [ ] **Check for reward scale issues**
  - [ ] Are rewards in reasonable range [-10, 10]?
  - [ ] Is success bonus (1.0) large enough vs step penalties?

---

## Phase 1: Pre-Training Foundation

### 1.1 Observation Normalization
- [ ] **Audit observation scales**
  - [ ] Print min/max of each observation component
  - [ ] Document current scales:
    - [ ] Positions: [______, ______]
    - [ ] Velocities: [______, ______]
    - [ ] Curvatures: [______, ______]
    - [ ] Prey position: [______, ______]
- [ ] **Implement normalization**
  - [ ] Option A: Running mean/std normalization (VecNormalize)
  - [ ] Option B: Fixed normalization based on expected ranges
  - [ ] Target: All features in [-1, 1] or [-5, 5]
- [ ] **Test normalized observations**
  - [ ] Verify no features dominate others
  - [ ] Check gradients flow equally to all inputs

### 1.2 Action Space Tuning
- [ ] **Review current action bounds**
  - [ ] Current: tanh output scaled by action_scale
  - [ ] Curvature clipping: [-10, 10]
- [ ] **Reduce action magnitude**
  - [ ] Try action_scale = 0.5 (effective range [-2.5, 2.5])
  - [ ] Try clipping to [-5, 5]
  - [ ] Record: Which produces stable physics?
- [ ] **Test action smoothing**
  - [ ] Add low-pass filter on actions
  - [ ] Prevents jerky motions that destabilize simulation

### 1.3 Serpenoid Controller Baseline
- [ ] **Establish baseline performance**
  - [ ] Run serpenoid controller with varied parameters
  - [ ] Grid search: A ∈ [0.5, 1.0, 1.5, 2.0], f ∈ [0.5, 1.0, 1.5, 2.0]
  - [ ] Record best parameters: A=______, f=______, k=______
  - [ ] Record best displacement: ______ m in 5 seconds
- [ ] **Save successful trajectories**
  - [ ] Store state-action pairs from best controllers
  - [ ] Use for behavioral cloning dataset
- [ ] **Identify steering parameters**
  - [ ] Test turn_bias values for SERPENOID_STEERING
  - [ ] Verify snake can turn left/right on command

---

## Phase 2: Approach Skill Training

### 2.1 Curriculum Learning Setup
- [ ] **Define curriculum stages**
  ```
  Stage 1: d_thresh=0.5m, init_distance=0.3m, max_steps=200
  Stage 2: d_thresh=0.3m, init_distance=0.5m, max_steps=500
  Stage 3: d_thresh=0.15m, init_distance=1.0m, max_steps=1000
  ```
- [ ] **Implement stage advancement**
  - [ ] Track success rate over last 100 episodes
  - [ ] Advance when success_rate > 80%
  - [ ] Log stage transitions
- [ ] **Add intermediate rewards for early stages**
  - [ ] Reward any distance reduction
  - [ ] Decay shaping reward as stage advances

### 2.2 Behavioral Cloning Pre-Training
- [ ] **Generate diverse demonstrations**
  - [ ] Run `scripts/generate_approach_experiences.py`
  - [ ] Ensure direction diversity (8 directions)
  - [ ] Minimum 1000 trajectories
  - [ ] Filter for min_displacement > 0.1m
- [ ] **Train BC policy**
  - [ ] Run `scripts/pretrain_approach_policy.py`
  - [ ] Monitor validation loss
  - [ ] Use early stopping (patience=10)
  - [ ] Save best checkpoint
- [ ] **Evaluate BC policy**
  - [ ] Test on 100 random initial conditions
  - [ ] Record success rate: ______%
  - [ ] Record mean distance achieved: ______ m
- [ ] **Consider Posterior BC or Residual approach**
  - [ ] If BC success < 30%, try wider action distribution
  - [ ] Implement residual policy wrapper

### 2.3 RL Fine-Tuning for Approach
- [ ] **Initialize from BC checkpoint**
  - [ ] Load pre-trained weights
  - [ ] Use reduced learning rate (1e-4 instead of 3e-4)
- [ ] **Configure PPO for fine-tuning**
  - [ ] Increase entropy coefficient initially (0.05)
  - [ ] Use larger batch size (8192)
  - [ ] frames_per_batch = 4096
- [ ] **Train with curriculum**
  - [ ] Start at Stage 1
  - [ ] Train until convergence or 500K frames
  - [ ] Advance stages based on success rate
- [ ] **Monitor training metrics**
  - [ ] Episode reward (should increase)
  - [ ] Episode length (should stabilize)
  - [ ] Policy entropy (should decrease slowly)
  - [ ] Value loss (should decrease)
- [ ] **Save best approach policy**
  - [ ] Checkpoint at each stage advancement
  - [ ] Save final policy when Stage 3 success > 80%

---

## Phase 3: Coil Skill Training

### 3.1 Staged Curriculum for Coiling
- [ ] **Define coil curriculum stages**
  ```
  Stage 1: Any contact (contact_fraction > 0.1), max_steps=200
  Stage 2: Sustained contact (contact_fraction > 0.3 for 10 steps)
  Stage 3: Partial wrap (contact_fraction > 0.5, wrap_count > 0.5)
  Stage 4: Full coil (contact_fraction > 0.6, wrap_count > 1.5)
  ```
- [ ] **Implement stage tracking**
  - [ ] Log contact_fraction and wrap_count each episode
  - [ ] Advance when success_rate > 70%

### 3.2 Coil-Specific Observations
- [ ] **Use REDUCED_COIL representation (22-dim)**
  - [ ] Includes contact features critical for coiling
  - [ ] Verify contact_fraction, wrap_count in observations
- [ ] **Consider adding more contact info**
  - [ ] Which segments are in contact?
  - [ ] Distance of each segment to prey surface
  - [ ] Normal direction at contact points

### 3.3 Coil Reward Tuning
- [ ] **Review coil potential weights**
  - [ ] contact_reward_weight = 1.0
  - [ ] wrap_reward_weight = 2.0
  - [ ] constriction_reward_weight = 1.5
- [ ] **Test different weight combinations**
  - [ ] Emphasize contact first: contact=2.0, wrap=1.0
  - [ ] Then emphasize wrap: contact=1.0, wrap=3.0
  - [ ] Record which produces faster learning
- [ ] **Add progress-based shaping**
  - [ ] Reward for increasing contact_fraction
  - [ ] Reward for wrap_angle increase (even before full wrap)

### 3.4 Initial State for Coil Training
- [ ] **Start coil training from pre-positioned state**
  - [ ] Initialize snake head within 0.2m of prey
  - [ ] Orient snake toward prey
  - [ ] This isolates coiling from approach learning
- [ ] **Gradually increase starting distance**
  - [ ] Once coiling works at 0.2m, try 0.3m, 0.5m
  - [ ] Eventually train end-to-end

---

## Phase 4: Hierarchical Integration

### 4.1 Manager Training Setup
- [ ] **Freeze worker policies initially**
  - [ ] Load trained approach policy (frozen)
  - [ ] Load trained coil policy (frozen)
- [ ] **Configure manager observations**
  - [ ] Base observation + hierarchical state (3-dim)
  - [ ] current_skill, task_progress[2]
- [ ] **Set skill_duration appropriately**
  - [ ] Current: 50 steps
  - [ ] Test: 25, 50, 100 steps
  - [ ] Record which allows smoothest transitions

### 4.2 Manager Reward Tuning
- [ ] **Review manager reward structure**
  - [ ] approach_completion_bonus = 10.0
  - [ ] task_completion_bonus = 100.0
  - [ ] skill_switch_penalty = 0.1
- [ ] **Test skill switch penalty values**
  - [ ] Too high: manager never switches
  - [ ] Too low: manager switches erratically
  - [ ] Find balance: ______

### 4.3 Joint Fine-Tuning
- [ ] **Unfreeze workers after manager converges**
  - [ ] Use very low learning rate for workers (1e-5)
  - [ ] Higher learning rate for manager (1e-4)
- [ ] **Train end-to-end**
  - [ ] 500K-1M additional frames
  - [ ] Monitor for skill degradation
- [ ] **Evaluate full pipeline**
  - [ ] Test on 100 random scenarios
  - [ ] Record overall task success rate: ______%

---

## Phase 5: Debugging & Monitoring

### 5.1 Metrics to Track During Training
- [ ] **Episode-level metrics**
  - [ ] Episode reward (mean, std)
  - [ ] Episode length
  - [ ] Success rate (per skill)
- [ ] **Policy metrics**
  - [ ] Policy entropy
  - [ ] KL divergence from previous policy
  - [ ] Action mean and std
- [ ] **Value function metrics**
  - [ ] Value loss
  - [ ] Explained variance
- [ ] **Gradient metrics**
  - [ ] Gradient norm
  - [ ] Check for NaN/Inf gradients
- [ ] **Physics metrics**
  - [ ] Max velocity in episode
  - [ ] Max curvature achieved
  - [ ] Contact events

### 5.2 Sanity Checks
- [ ] **Random policy baseline**
  - [ ] Run random policy for 100 episodes
  - [ ] Record mean reward: ______
  - [ ] This is the "floor" - trained policy must beat this
- [ ] **Expert policy ceiling**
  - [ ] Run scripted/serpenoid controller
  - [ ] Record mean reward: ______
  - [ ] This is the "ceiling" - target for trained policy
- [ ] **Memorization test**
  - [ ] Train on single fixed initial state
  - [ ] Policy should converge quickly
  - [ ] If not, model/training issue

### 5.3 Visualization
- [ ] **Render training episodes periodically**
  - [ ] Every 10K frames, save video of episode
  - [ ] Visually verify snake behavior makes sense
- [ ] **Plot learning curves**
  - [ ] Reward vs frames
  - [ ] Success rate vs frames
  - [ ] Compare across hyperparameter settings

---

## Phase 6: Alternative Approaches (If Needed)

### 6.1 Try Existing Environments
- [ ] **Test with snakebot-gym**
  - [ ] Clone: https://github.com/williamcorsel/snakebot-gym
  - [ ] Train PPO on their environment
  - [ ] Verify RL training works in simpler setting
- [ ] **Transfer insights back**
  - [ ] What hyperparameters worked?
  - [ ] What reward structure worked?

### 6.2 Simplify Environment
- [ ] **2D version**
  - [ ] Disable z-axis movement
  - [ ] Simpler physics, faster iteration
- [ ] **Fewer segments**
  - [ ] Try N=5 segments instead of N=20
  - [ ] Faster simulation, easier control
- [ ] **No prey (locomotion only)**
  - [ ] Reward x-displacement only
  - [ ] Verify basic locomotion learning works

### 6.3 Alternative Simulators
- [ ] **Consider MuJoCo**
  - [ ] Better documentation and RL support
  - [ ] Can approximate elastic rod with joints
- [ ] **Consider Isaac Gym**
  - [ ] GPU-accelerated, massively parallel
  - [ ] Could train 1000x faster

---

## Progress Tracking

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Verification | Not Started | |
| Phase 1: Foundation | Not Started | |
| Phase 2: Approach | Not Started | |
| Phase 3: Coil | Not Started | |
| Phase 4: HRL | Not Started | |
| Phase 5: Debug | Ongoing | |
| Phase 6: Alternatives | Not Started | |

---

## Key Success Criteria

- [ ] **Milestone 1**: Serpenoid controller moves snake > 0.3m in 5 seconds
- [ ] **Milestone 2**: Approach policy reaches prey (d < 0.15m) in > 50% of episodes
- [ ] **Milestone 3**: Coil policy achieves wrap_count > 1.5 in > 30% of episodes
- [ ] **Milestone 4**: Full HRL pipeline completes predation task in > 20% of episodes

---

*Last Updated: 2026-01-27*
