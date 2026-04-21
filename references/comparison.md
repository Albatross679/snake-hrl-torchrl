---
type: reference
created: 2026-02-16T00:00:00
updated: 2026-02-16T00:00:00
tags: [literature-review, reinforcement-learning, snake-robot, comparison]
---

# Snake Robot RL Literature Comparison

Comparison of reinforcement learning approaches for snake robot locomotion and navigation across 7 papers.

## Summary Table

| Paper | Year | RL Algorithm | Physics Framework | Robot Type | State Dim | Action Dim / Type | Sparse Reward Handling | Open Source |
|---|---|---|---|---|---|---|---|---|
| Bing et al. | 2019 | PPO | MuJoCo | 8-joint rigid, passive wheels | 26 | 8 / continuous | None (dense reward) | [Yes](https://github.com/zhenshan-bing/RL_Snake) |
| Shi et al. | 2020 | DQN | Geometric mechanics (kinematic) | 3-link wheeled / swimming | 3 | 8 / discrete | None (dense reward) | No |
| Liu et al. | 2021 | PPOC + PPO | Custom FEM (GPU) | 4-link soft pneumatic | 26 | 4 + option / cont. + disc. | Curriculum rings | No |
| Liu et al. | 2022 | PPO | MuJoCo | 8-joint rigid, passive wheels | 10 | 1 / continuous | None (dense reward) | No |
| Liu et al. | 2023 | PPOC | Custom FEM (GPU) | 4-link soft pneumatic | 8 | 4 + option / cont. + disc. | Curriculum learning (12 levels) | No |
| Jiang et al. | 2024 | DDPG | Euler-Lagrange + Coulomb friction | 11-joint COBRA | 21 | 7 CPG params / continuous | None (dense reward) | No |
| Choi & Tong | 2025 | SAC | DisMech / Elastica (DER) | Soft arm (continuum) | Variable | 5-15 / continuous | None (dense reward) | [Yes](https://github.com/QuantuMope/dismech-rl) |

---

## Detailed Comparison

### 1. Bing et al. (2019) — Energy-Efficient Slithering Gait Exploration

**Goals.** Explore energy-efficient slithering gaits for a snake-like robot using RL. Optimize both velocity tracking and power efficiency simultaneously, eliminating manually designed gait parameters.

**Achievements.**
- Discovered novel energy-efficient gaits that outperform traditional serpentine gaits
- Asymmetric joint angle patterns emerge that differ from classical sinusoidal motion
- Trained in ~2 hours wall-clock time

**Discovered gaits.** Two gaits emerge: a concertina-like gait at low speed (~0.05 m/s) and a slithering gait at higher speed (~0.25 m/s). The learned gaits save 35-65% energy vs. the parametric baseline at 0.15 m/s. **No closed-form mathematical representation** is provided — the gaits exist only as neural network weights. The paper compares against the parametric gait equation $\phi(n,t) = (\frac{n}{N}x + y) \times A \times \sin(\omega t + \lambda n)$ but notes the learned gaits explicitly do not follow this sinusoidal form (they are asymmetric across joints).

**Reward function.**

$$r = r_v \cdot r_P$$

- Velocity reward: $r_v = \left(1 - \frac{|v_t - v_1|}{a_1}\right)^{1/a_2}$
- Power efficiency reward: $r_P = r_{\max} \cdot |1 - \hat{P}|^{b_1 - 2}$
- Normalized power: $\hat{P} = \frac{1}{N} \sum_{j=1}^{N} \frac{|f_j \cdot h_j \cdot \dot{\phi}_j|}{f_{\max} \cdot h_j \cdot \dot{\phi}_{\max}}$

Where:
- $v_t$ — target (desired) velocity
- $v_1$ — current forward velocity of the head link
- $a_1 = 0.2$ — spread parameter controlling the x-axis intersections of the velocity reward curve ($v_t \pm a_1$)
- $a_2 = 0.2$ — gradient parameter controlling how sharply the velocity reward falls off
- $r_{\max}$ — maximum power efficiency reward (scales with $r_v$ in the combined reward)
- $b_1 = 0.6$ — slope parameter for the power efficiency curve
- $\hat{P}$ — normalized total power consumption across all joints, in $[0, 1]$
- $N$ — number of joints (8)
- $f_j$ — friction force at the $j$-th joint's contact point
- $h_j$ — gear length (distance from contact point to joint axis, = 0.175 m)
- $\dot{\phi}_j$ — angular velocity of the $j$-th joint
- $f_{\max}$ — maximum actuator force (20 N)
- $\dot{\phi}_{\max}$ — maximum angular velocity

**State space** (26D): 8 joint angles, 8 joint angular velocities, forward velocity, 8 joint torques, target velocity.

**Action space** (8D continuous): Desired joint angles $a \in [-1.5, 1.5]^8$, mapped to $[-90°, 90°]$.

**Sparse reward handling.** None — reward is fully dense at every timestep.

**Physics.** MuJoCo. 9 rigid modules, 8 yaw joints, passive wheels providing anisotropic friction.

**Open source.** Yes — [zhenshan-bing/RL_Snake](https://github.com/zhenshan-bing/RL_Snake)

---

### 2. Shi et al. (2020) — Deep RL for Snake Robot Locomotion

**Goals.** Apply DQN to learn locomotion gaits for 3-link snake robots using geometric mechanics. Exploit SE(2) symmetry to reduce state space from 5D to 3D.

**Achievements.**
- Learned gaits closely match optimal gaits predicted by geometric mechanics theory
- Learned gaits trace regions of high connection exterior derivative in shape space
- Converges in ~5000 iterations (~10 episodes)

**Discovered gaits.** Forward and rotational gaits for both wheeled and swimming 3-link robots. The gaits are visualized as closed-loop trajectories in shape space $(\alpha_1, \alpha_2)$ overlaid on connection exterior derivative surfaces, confirming they trace high-$|dA|$ regions (geometrically optimal). Described qualitatively as "sinusoidal with various differences in phase." **No closed-form mathematical representation** is provided — the gaits exist only as DQN policies. However, the geometric mechanics framework (kinematic reconstruction $\xi = -\mathbf{A}(\alpha)\dot{\alpha}$ and exterior derivative $d\mathbf{A}_i = \frac{\partial \mathbf{A}_{i,2}}{\partial \alpha_1} - \frac{\partial \mathbf{A}_{i,1}}{\partial \alpha_2}$) provides a theoretical tool to analyze optimality of the learned trajectories.

**Reward function.**

- Forward: $R = c_1 \Delta x - c_2 P_0 + c_3 R_\theta$
- Rotation: $R = c_1 \Delta \theta - c_2 P_0$
- Orientation term: $R_\theta = 1$ if $|\theta_{\text{new}}| \leq \pi/4$, else $\pi/4 - |\theta_{\text{new}}|$

Where:
- $\Delta x$ — forward displacement in the world-frame x-direction over one action interval (positive = forward, negative = backward)
- $\Delta \theta$ — change in body orientation angle over one action interval
- $P_0 = 1$ — constant penalty applied when the robot makes zero displacement (prevents idle/do-nothing policies)
- $R_\theta$ — orientation maintenance reward that encourages the robot to face forward ($|\theta| \leq \pi/4$) and penalizes large deviations
- $\theta_{\text{new}}$ — body orientation angle in the inertial frame after executing the action
- $c_1 = 10$ — weight on displacement (or rotation) reward
- $c_2 = 10$ — weight on zero-displacement penalty
- $c_3 = 1$ — weight on orientation maintenance (forward task only)

**State space** (3D): Two joint angles $(\alpha_1, \alpha_2)$ and body orientation $\theta$ (position removed via SE(2) symmetry).

**Action space** (discrete, 8 actions): Joint velocities $(\dot{\alpha}_1, \dot{\alpha}_2) \in \{-\pi/8, 0, \pi/8\}^2 \setminus \{(0,0)\}$.

**Sparse reward handling.** None — dense displacement-based reward. $P_0$ penalty prevents degenerate do-nothing policies.

**Physics.** Custom kinematic simulation based on geometric mechanics. Wheeled robot uses nonholonomic constraints; swimming robot uses resistive force theory. No dynamic simulation (purely first-order kinematics).

**Open source.** No

---

### 3. Liu et al. (2021) — Contact-Aware CPG-Based Locomotion in a Soft Snake Robot

**Goals.** Develop contact-aware CPG-based locomotion for a soft snake that exploits environmental contacts (obstacle-aided locomotion). Train two cooperative controllers via fictitious play.

**Achievements.**
- 91% goal-reaching success rate in obstacle-rich environments
- 0.165 jam ratio, 0.139 m/s average velocity
- Contact-aware pushing-off behavior emerges naturally
- Two-controller PPOC+PPO architecture outperforms single-controller baselines

**Reward function.**

$$R = w_1 R_{\text{goal}} + w_2 R_{\text{att}} + w_3 R_{\text{rep}}$$

- Goal proximity: $R_{\text{goal}} = \cos(\theta_g) \sum_k \frac{1}{l_k} \cdot \mathbf{1}(\rho_g < l_k)$
- Attractive potential: $R_{\text{att}} = \mathbf{v} \cdot F_{\text{att}}$, where $F_{\text{att}} = -k_{\text{att}}(\mathbf{p} - \mathbf{p}_g)$
- Repulsive potential: $R_{\text{rep}} = \mathbf{v} \cdot F_{\text{rep}}$, where $F_{\text{rep}} = \sum_i k_{\text{rep}} (\mathbf{p} - \mathbf{p}_{o_i}) \frac{(1/\rho_i - 1/\rho_0)}{\rho_i^3}$ if $\rho_i \leq \rho_0$, else 0

Where:
- $w_1, w_2, w_3$ — weighting coefficients for the three reward components
- $\theta_g$ — angle between the robot's heading direction and the direction to the goal
- $\rho_g$ — Euclidean distance from robot center to the goal
- $l_k$ — distance threshold for the $k$-th curriculum ring (expanding radii)
- $\mathbf{1}(\cdot)$ — indicator function (1 if condition true, 0 otherwise)
- $\mathbf{v}$ — robot's velocity vector
- $F_{\text{att}}$ — attractive force field pointing toward the goal
- $k_{\text{att}}$ — attractive potential gain
- $\mathbf{p}$ — robot's current position
- $\mathbf{p}_g$ — goal position
- $F_{\text{rep}}$ — repulsive force field pushing away from obstacles
- $k_{\text{rep}}$ — repulsive potential gain
- $\rho_i$ — distance from the robot to obstacle $i$
- $\rho_0$ — influence radius beyond which repulsive force is zero
- $\mathbf{p}_{o_i}$ — position of obstacle $i$

**State space** (26D): Dynamic state (4), body curvatures (4), previous actions/option/termination (6), contact forces (10), distance and angle to nearest obstacle (2).

**Action space.**
- Controller C1 (PPOC): 4 continuous tonic inputs to Matsuoka CPG oscillators + discrete frequency option
- Controller R2 (PPO): 4 continuous tonic inputs (event-triggered on contact)
- Linear composition: $u = w_1 u_1 + w_2 u_2$

**Sparse reward handling.** Curriculum-based distance thresholds $l_k$ act as expanding rings that bridge sparse-to-dense signals. Potential field rewards provide fully dense gradients throughout.

**Physics.** Custom FEM-based GPU-accelerated simulator. 4-link pneumatic soft robot (Ecoflex 00-30). Captures soft body deformation, frictional ground contact, and obstacle contact. Matsuoka CPG oscillator network.

**Open source.** No

---

### 4. Liu et al. (2022) — RL-Based Path Following with Onboard Camera

**Goals.** Develop RL-based path-following for a wheeled snake robot using onboard camera self-localization. Demonstrate sim-to-real transfer.

**Achievements.**
- Successful sim-to-real transfer on a physical 9-module snake robot
- RL learns a single 1D action (joint offset) that modulates serpentine gait for steering
- Robust performance on circular and S-shaped paths
- Hierarchical decomposition (RL high-level + sinusoidal gait low-level) simplifies learning

**Reward function.**

$$r_t = r_p + r_e - p_h$$

- Path proximity: $r_p = c_p$ if $|d^p_{t+1}| < d_1$; $c_p \exp(d_1 - |d^p_{t+1}|)$ if $d_1 \leq |d^p_{t+1}| \leq d_2$; 0 otherwise
- Endpoint approach: $r_e = c_e (d^e_t - d^e_{t+1})$
- Head swing penalty: $p_h = c_h$ if $|\phi^1_{t+1} - \phi^1_t| \geq \phi_*$, else 0

Where:
- $r_p$ — path proximity reward (how close the robot is to the desired path)
- $r_e$ — endpoint approach reward (progress toward the target point on the path)
- $p_h$ — head swing penalty (penalizes large oscillations of the head joint)
- $d^p_{t+1}$ — perpendicular distance from the robot's head to the desired path at timestep $t+1$
- $d_1$ — inner distance threshold (full reward within this band)
- $d_2$ — outer distance threshold (zero reward beyond this)
- $c_p$ — constant reward magnitude for path proximity
- $d^e_t, d^e_{t+1}$ — distance from robot to the target endpoint at timesteps $t$ and $t+1$
- $c_e$ — constant weight for endpoint progress reward
- $\phi^1_t, \phi^1_{t+1}$ — head joint angle at timesteps $t$ and $t+1$
- $\phi_*$ — angular change threshold beyond which the head swing penalty activates
- $c_h$ — constant penalty magnitude for excessive head swing

**State space** (10D): Perpendicular distance to path, distance to endpoint, 8 previous joint angles.

**Action space** (1D continuous): Joint offset $\phi_o$ applied to serpentine gait $\phi^i(t) = \alpha \sin(\omega t + (i-1)\delta) + \phi_o$.

**Sparse reward handling.** None — inherently dense via exponential decay in $r_p$ and progress-based $r_e$.

**Physics.** MuJoCo for training. 9-module snake with 8 yaw joints and passive wheels. Sim-to-real transfer via matched model parameters and onboard camera state estimation.

**Open source.** No

---

### 5. Liu et al. (2023) — CPG-Regulated Locomotion Controller for a Soft Snake Robot

**Goals.** Develop a CPG-regulated RL controller with Free-response Oscillation Constraint (FOC) that establishes analytical relationships between CPG parameters and locomotion behavior. Demonstrate robust sim-to-real transfer.

**Achievements.**
- 0.135 m/s in simulation, 0.121 m/s on real hardware (11% drop)
- 0.9 real-world success rate (8.1% drop from simulation)
- FOC provides interpretable steering-velocity relationships
- Dense reward converges to level 12 (hardest) vs. sparse reward only to level 8
- Successful transfer to real WPI-SRS pneumatic soft snake with domain randomization

**Reward function.**

$$R = c_v v_g + c_g U + c_g \cos(\theta_g) \sum_k \frac{1}{r_k} \mathbf{I}(\|\rho_g\| < r_k)$$

- Potential field: $U = \frac{\mathbf{v}_s \cdot \mathbf{f}_g}{\|e_g\|}$, where $\mathbf{f}_g = \frac{\mathbf{e}_g}{\|\mathbf{e}_g\|}$
- 12 curriculum levels with progressively increasing goal distances

Where:
- $c_v$ — weight for the goal-directed velocity term
- $v_g$ — component of the robot's velocity along the goal direction (scalar)
- $c_g$ — weight for potential field and curriculum ring terms
- $U$ — normalized potential field reward (dot product of velocity with unit goal direction)
- $\mathbf{v}_s$ — robot's velocity vector
- $\mathbf{f}_g$ — unit vector pointing from robot to goal
- $\mathbf{e}_g$ — vector from robot position to goal position
- $\theta_g$ — angle between the robot's heading direction and the direction to the goal
- $r_k$ — radius of the $k$-th curriculum ring (distance threshold)
- $\|\rho_g\|$ — Euclidean distance from the robot to the goal
- $\mathbf{I}(\cdot)$ — indicator function (1 if condition true, 0 otherwise)

**State space** (8D): Distance to goal, goal-directed velocity, heading-to-goal angle and its rate, 4 body curvatures.

**Action space** (4 continuous + discrete option): Tonic inputs to Matsuoka CPG oscillators mapped through sigmoid; discrete frequency ratio $K_f$ via PPOC option-critic.

**Sparse reward handling.** Curriculum learning with 12 progressive task levels. The paper explicitly shows dense reward (converges to level 12) outperforms sparse reward (converges to level 8). Curriculum rings $r_k$ provide intermediate dense signals.

**Physics.** Custom FEM-based GPU-accelerated simulator (same as Liu et al. 2021). Domain randomization (friction, stiffness, actuator response) for sim-to-real. Validated on real WPI-SRS platform with OptiTrack.

**Open source.** No

---

### 6. Jiang et al. (2024) — Hierarchical RL-Guided Large-Scale Navigation

**Goals.** Develop a hierarchical control framework for large-scale autonomous navigation (>10m) of the COBRA snake robot in complex environments with obstacle avoidance.

**Achievements.**
- Navigation over >10m distances with obstacle avoidance
- 4-layer hierarchy: global planning (A*) → local navigation (DDPG) → gait generation (CPG) → gait control (PID)
- RL learns to select CPG parameters for effective waypoint tracking
- Validated on 11-joint COBRA platform

**Reward function.**

$$r = r_1 + r_2 + r_3$$

- Proximity: $r_1 = \frac{1}{0.1 + d_t}$
- Progress: $r_2 = d_{t-1} - d_t$
- Smoothness penalty: $r_3 = -\|a_t - a_{t-1}\|_2$

Where:
- $r_1$ — proximity reward (inversely proportional to distance; saturates at 10 when at the waypoint)
- $d_t$ — Euclidean distance between the robot body frame and the current waypoint at timestep $t$
- $r_2$ — progress reward (positive when the robot moves closer to the waypoint)
- $d_{t-1}$ — distance to waypoint at the previous timestep
- $r_3$ — smoothness penalty (penalizes large changes in action between consecutive timesteps)
- $a_t, a_{t-1}$ — action vectors (7D CPG parameters) at timesteps $t$ and $t-1$
- $\|\cdot\|_2$ — L2 (Euclidean) norm

**State space** (21D): 11 joint positions, 3 IMU readings, 3 spatial displacement to waypoint, 4 relative rotation to waypoint.

**Action space** (7D continuous CPG parameters): Amplitudes $R_1, R_2 \in [0, 1.5]$, angular frequency $\omega \in [-0.1, 0.1]$, phase offsets $\theta_1, \theta_2 \in [-\pi, \pi]$, joint angle offsets $\delta_1, \delta_2 \in [-0.1, 0.1]$. RL outputs at 0.5 Hz; CPG generates joint commands at 50 Hz.

**Sparse reward handling.** None — all three reward terms are fully dense.

**Physics.** Custom Euler-Lagrange dynamics with spring-damper ground contact and Coulomb+Stribeck friction model. Rigid-link chain, 11 revolute joints. CPG coupled oscillator network.

**Open source.** No

---

### 7. Choi & Tong (2025) — Rapidly Learning Soft Robot Control via Implicit Time-Stepping

**Goals.** Demonstrate that implicit time-stepping (DisMech) enables dramatically faster RL training for soft robot control compared to explicit time-stepping (Elastica/PyElastica).

**Achievements.**
- Up to 6x speedup for non-contact tasks, up to 40x speedup for contact-rich tasks vs. Elastica
- Trained SAC policies for 4 tasks: Follow Target, 4D Inverse Kinematics, 2D Tight Obstacles, 3D Random Obstacles
- Implicit method allows $\Delta t = 0.05$s vs. $0.0002$s for explicit Elastica
- Contact-rich scenarios particularly benefit from implicit methods

**Reward function.** Reused directly from Naughton et al. (2021); dense distance-to-target formulations. Not restated in the paper.

**State space.** Discretized centerline positions and curvatures. Dimensionality varies by task (5-15 DOF depending on number of control points and curvature components).

**Action space** (5-15D continuous): Delta natural curvature commands at 5 equidistant control points. 2D tasks: 5D; 3D tasks: 10-15D. Control frequency: 10 Hz (non-contact) or 2 Hz (contact-rich).

**Sparse reward handling.** None — dense distance-based rewards throughout.

**Physics.** Two frameworks compared:
1. **DisMech** — Discrete Elastic Rods, implicit (backward Euler), $\Delta t = 0.05$s, IPC contact
2. **Elastica/PyElastica** — Cosserat rod theory, explicit (Verlet), $\Delta t = 0.0002$s, penalty-based contact

**Open source.** Yes — [QuantuMope/dismech-rl](https://github.com/QuantuMope/dismech-rl)

---

## Cross-Cutting Themes

### Action Space Design
Papers split into two paradigms:
- **Direct joint control**: Bing 2019 (8 joint angles), Shi 2020 (joint velocities), Choi 2025 (curvature commands)
- **CPG parameter modulation**: Liu 2021/2023 (tonic inputs + frequency option), Liu 2022 (gait offset), Jiang 2024 (7 CPG parameters)

CPG-based approaches reduce action dimensionality and produce smoother, more biologically plausible gaits. Direct control offers more expressiveness but requires more training.

### Physics Fidelity
- **Rigid-body**: MuJoCo (Bing 2019, Liu 2022), Euler-Lagrange (Jiang 2024)
- **Soft-body**: Custom FEM (Liu 2021, Liu 2023), DisMech/Elastica DER (Choi 2025)
- **Kinematic only**: Geometric mechanics (Shi 2020)

Soft-body simulators capture deformation but are computationally expensive. Choi 2025 shows implicit time-stepping (DisMech) can close this gap significantly.

### Reward Function Comparison

All 7 papers use dense rewards. The reward designs fall into three distinct families based on the task objective.

#### Taxonomy

| Paper | Task | Reward Family | Components | Multiplicative or Additive |
|---|---|---|---|---|
| Bing 2019 | Velocity tracking + efficiency | Velocity × Efficiency | $r_v \cdot r_P$ | Multiplicative |
| Shi 2020 | Displacement maximization | Displacement + Penalty | $c_1 \Delta x - c_2 P_0 + c_3 R_\theta$ | Additive |
| Liu 2021 | Goal reaching (cluttered) | Potential field + Curriculum | $w_1 R_{\text{goal}} + w_2 R_{\text{att}} + w_3 R_{\text{rep}}$ | Additive (weighted) |
| Liu 2022 | Path following | Proximity + Progress + Penalty | $r_p + r_e - p_h$ | Additive |
| Liu 2023 | Goal tracking | Potential field + Curriculum | $c_v v_g + c_g U + c_g \cos\theta_g \sum_k \frac{1}{r_k}\mathbf{I}(\rho < r_k)$ | Additive |
| Jiang 2024 | Waypoint navigation | Proximity + Progress + Smoothness | $\frac{1}{0.1+d} + (d_{t-1}-d_t) - \|a_t - a_{t-1}\|$ | Additive |
| Choi 2025 | Soft manipulation | (from Naughton et al.) | Dense distance-to-target | — |

#### Family 1: Velocity/Efficiency Tracking (Bing 2019)

Unique among the papers for using a **multiplicative** reward $r = r_v \cdot r_P$. This means power efficiency only matters when velocity tracking is good, and vice versa — the agent cannot get high reward by optimizing one at the expense of the other. The velocity term $r_v$ uses a peaked function $(1 - |v_t - v_1|/a_1)^{1/a_2}$ centered on the target velocity $v_t$ that falls off sharply (controlled by $a_1, a_2$), while the power term $r_P = r_{\max}|1 - \hat{P}|^{b_1-2}$ rewards low normalized power consumption $\hat{P} \in [0,1]$ (sum of per-joint $|f_j h_j \dot{\phi}_j|$ normalized by max force/velocity). This is the only paper where the reward explicitly optimizes energy efficiency.

#### Family 2: Displacement/Progress Maximization (Shi 2020, Jiang 2024)

Both reward forward progress per timestep:
- **Shi 2020**: $R = c_1 \Delta x - c_2 P_0 + c_3 R_\theta$ — $\Delta x$ is raw forward displacement per action, $P_0 = 1$ is a constant penalty for zero displacement (prevents idle policies), and $R_\theta$ is an orientation maintenance term (= 1 when heading $|\theta| \leq \pi/4$, linearly decaying otherwise). Weights: $c_1 = 10$, $c_2 = 10$, $c_3 = 1$. Minimal shaping; relies on geometric mechanics to simplify the problem.
- **Jiang 2024**: $r = \frac{1}{0.1 + d_t} + (d_{t-1} - d_t) - \|a_t - a_{t-1}\|_2$ — $d_t$ is distance to waypoint at timestep $t$, $a_t$ is the 7D CPG action vector. The inverse-distance proximity term is bounded (saturates at $1/0.1 = 10$ at the waypoint), the progress term $d_{t-1} - d_t$ rewards distance reduction, and the smoothness penalty $-\|a_t - a_{t-1}\|_2$ (L2 norm of action change) is unique among these papers and prevents jerky CPG parameter changes.

**Key difference:** Shi 2020 rewards raw displacement (good for gait discovery), while Jiang 2024 rewards distance reduction to a specific target (good for navigation).

#### Family 3: Potential-Field + Curriculum (Liu 2021, Liu 2023)

Both Liu papers use artificial potential field (APF) theory as the reward backbone:
- **Attractive potential**: $R_{\text{att}} = \mathbf{v} \cdot F_{\text{att}}$ — the dot product of the robot's velocity $\mathbf{v}$ with the attractive force field $F_{\text{att}} = -k_{\text{att}}(\mathbf{p} - \mathbf{p}_g)$ pointing toward the goal position $\mathbf{p}_g$ (gain $k_{\text{att}}$). This rewards motion *along* the force field gradient.
- **Repulsive potential** (Liu 2021 only): $R_{\text{rep}} = \mathbf{v} \cdot F_{\text{rep}}$ — pushes the robot away from obstacles. $F_{\text{rep}}$ depends on distance $\rho_i$ to each obstacle $i$, with influence radius $\rho_0$ (zero beyond that) and gain $k_{\text{rep}}$.
- **Curriculum rings**: Both use $\cos(\theta_g) \sum_k \frac{1}{r_k} \mathbf{I}(\rho_g < r_k)$ — $\theta_g$ is the heading-to-goal angle, $\rho_g$ is the distance to the goal, $r_k$ is the radius of the $k$-th ring, and $\mathbf{I}(\cdot)$ is the indicator function. Inside each ring, the robot gets a bonus inversely proportional to the ring radius. The $\cos(\theta_g)$ factor ensures the robot is rewarded only when facing the goal. This is the only reward component that bridges sparse-to-dense.

Liu 2023 simplifies Liu 2021 by dropping the repulsive potential (no obstacles in their task) and adding a direct velocity term $c_v v_g$ that rewards goal-directed speed ($v_g$ = velocity component along the goal direction).

**Design insight:** The potential-field formulation $R = \mathbf{v} \cdot \nabla U$ ensures the reward is maximized when the robot moves down the potential gradient, directly connecting physics-based planning (APF) with RL reward shaping.

#### Family 4: Path Following (Liu 2022)

The most structured reward with three distinct zones:
- **Path proximity** $r_p$: Full reward $c_p$ when perpendicular distance $|d^p|$ to the desired path is within $d_1$ (inner band), exponential decay $c_p \exp(d_1 - |d^p|)$ in the transition zone $[d_1, d_2]$, and zero beyond the outer threshold $d_2$. The exponential bridge is a smooth version of a hard boundary.
- **Endpoint approach** $r_e = c_e(d^e_t - d^e_{t+1})$: Progress reward proportional to distance reduction toward the target endpoint ($d^e_t$ = distance to endpoint at timestep $t$, $c_e$ = weight).
- **Head swing penalty** $p_h = c_h$: Discrete penalty (magnitude $c_h$) triggered when the head joint angle change $|\phi^1_{t+1} - \phi^1_t|$ exceeds threshold $\phi_*$. Prevents oscillatory steering that would be impractical on real hardware.

This is the only paper with an explicit **behavior penalty** (head swing) rather than just task performance terms.

#### Comparative Observations

1. **Multiplicative vs. additive**: Only Bing 2019 uses multiplicative composition. All others are additive with weights. Multiplicative coupling enforces that *all* objectives must be satisfied simultaneously; additive allows trade-offs between terms.

2. **Smoothness regularization**: Only Jiang 2024 (action change penalty) and Liu 2022 (head swing penalty) explicitly penalize non-smooth behavior. This matters for sim-to-real transfer — jerky policies fail on real hardware.

3. **Progress signals**: Most papers use some form of $d_{t-1} - d_t$ (distance reduction). Bing 2019 is the exception, using absolute velocity matching instead of goal-relative progress.

4. **Curriculum/shaping**: Only Liu 2021 and Liu 2023 use curriculum-based reward shaping via indicator-function rings. Liu 2023 explicitly shows this outperforms sparse reward (converges to difficulty level 12 vs. 8 for sparse).

5. **Physics-informed rewards**: Bing 2019 incorporates normalized power consumption $\hat{P}$ directly from simulation forces and torques. Liu 2021/2023 use potential-field theory from robotics planning. These physics-informed terms provide richer gradient signals than pure distance-based rewards.

6. **Reward complexity vs. action space**: Papers with simpler action spaces (Shi 2020: 8 discrete; Liu 2022: 1D continuous) use simpler rewards. Papers with richer action spaces (Liu 2021: 4+option; Jiang 2024: 7D) tend to need more reward components to shape behavior.

### Gait Discovery and Mathematical Representation
Two papers explicitly claim to have discovered novel gaits:

| Paper | Gaits Discovered | Closed-Form Math | Representation |
|---|---|---|---|
| Bing et al. 2019 | Concertina-like (low speed), slithering (high speed) | No | Neural network weights only. Compared against parametric gait equation $\phi(n,t) = (\frac{n}{N}x + y) \times A \times \sin(\omega t + \lambda n)$ but learned gaits are asymmetric and do not follow this form. |
| Shi et al. 2020 | Forward and rotational gaits for wheeled and swimming robots | No | DQN policy only. Gaits visualized as shape-space trajectories on connection exterior derivative surfaces $d\mathbf{A}_i$, confirming geometric optimality, but no formula is fitted to the learned trajectories. |

Neither paper provides a closed-form mathematical expression for the discovered gaits. Both represent gaits implicitly through neural network policies. Shi et al. provide the strongest analytical framework (geometric mechanics) for *understanding* the gaits, but do not extract a parametric fit.

The remaining papers (Liu 2021/2022/2023, Jiang 2024, Choi 2025) focus on goal-directed control rather than gait discovery — their locomotion emerges from CPG oscillators or curvature commands, not as standalone gait patterns.

### Sim-to-Real Transfer
- Liu 2022: MuJoCo → real 9-module snake (parameter matching + camera)
- Liu 2023: FEM sim → real WPI-SRS soft snake (domain randomization, 11% velocity drop)
- Other papers remain simulation-only

---

## RL Training Details

### Algorithm and Architecture Comparison

| Paper | Algorithm | Library | Hidden Layers | Activation | Learning Rate | Discount γ |
|---|---|---|---|---|---|---|
| Bing 2019 | PPO | Custom | 2 × 200 | ReLU | Not reported | Not reported |
| Shi 2020 | DQN | Custom | 50 + 10 | ReLU | 0.0002 (RMSProp) | 0.99 |
| Liu 2021 | PPOC + PPO | Custom | 2 × 128 | Not reported | Not reported | Not reported |
| Liu 2022 | PPO | Custom | 64, 32, 32 | Tanh | Not reported | Not reported |
| Liu 2023 | PPOC | OpenAI Baselines | 2 × 128 | Not reported | 5e-4 (Adam) | Not reported |
| Jiang 2024 | DDPG | Custom | 512, 256, 128 (actor); 512, 256 (critic) | ReLU + Tanh (output) | Not reported | Not reported |
| Choi 2025 | SAC | Custom (ALF) | 3 × 256 | Not reported | 0.001 (Adam) | 0.99 |

### Training Efficiency

| Paper | Total Steps/Episodes | Wall-Clock Time | Hardware | Parallel Envs |
|---|---|---|---|---|
| Bing 2019 | 3M steps (~1400 updates) | ~2 hours | i7-7700 + GTX 1080 | 1 |
| Shi 2020 | ~5000 iterations (10 episodes × 500 steps) | Not reported | Not reported | 1 |
| Liu 2021 | ~500+ episodes (fictitious play) | Not reported | Not reported | Not reported |
| Liu 2022 | ~1M steps (converged) | Not reported | Not reported | Not reported |
| Liu 2023 | 12,500 episodes | Not reported | i7-9700K + RTX 2080 Super | 4 |
| Jiang 2024 | 3M steps (40k episodes) | ~12.25 hours (RL-CPG) | Not reported | Not reported |
| Choi 2025 | 5M steps | 0.25–5 hours (DisMech) | 128-core Xeon + RTX 3090 | 500 |

Key observations:
- **Choi 2025** is by far the most sample-efficient in wall-clock time thanks to 500 parallel environments and implicit time-stepping
- **Jiang 2024** shows CPG-based action spaces yield 43× faster training than joint-space DDPG (12.25 hrs vs. 526 hrs)
- **Shi 2020** converges in only ~5000 iterations due to the 3D state space (SE(2) symmetry reduction)
- **Liu 2022** shows hierarchical decomposition (1D action) converges 2× faster than end-to-end PPO and reaches 3× higher reward

---

## Robot Morphology and Simulation

### Robot Specifications

| Paper | Robot Type | Links/Joints | Total DOF | Actuated DOF | RL Action DOF | Material |
|---|---|---|---|---|---|---|
| Bing 2019 | Rigid, passive wheels | 9 modules / 8 yaw joints | 8 | 8 | 8 | Rigid (600 kg/m³) |
| Shi 2020 | Rigid, passive wheels (wheeled) / slender bodies (swimming) | 3 links / 2 joints | 2 | 2 | 2 (discrete) | Rigid |
| Liu 2021 | Soft pneumatic | 4 soft links / 5 rigid bodies | 4 | 4 | 4 + option | Ecoflex 00-30 silicone |
| Liu 2022 | Rigid, passive wheels | 9 modules / 8 yaw joints | 8 | 8 | 1 | 3D-printed (0.416 g/module) |
| Liu 2023 | Soft pneumatic | 4 soft links / 5 rigid bodies | 4 | 4 | 4 + option | Ecoflex 00-30 silicone |
| Jiang 2024 | Rigid (COBRA) | 11 alternating pitch/yaw joints | 11 | 11 | 7 (CPG params) | Not reported |
| Choi 2025 | Soft continuum arm | 21 nodes / continuous | 5–15 | 5–15 | 5–15 | E=10 MPa, ρ=1000 kg/m³ |

### Simulation Parameters

| Paper | Simulator | Timestep (dt) | Control Freq | Episode Length | Friction Model |
|---|---|---|---|---|---|
| Bing 2019 | MuJoCo | 50 ms | 20 Hz | 1000 steps | Passive wheels (anisotropic damping) |
| Shi 2020 | Custom kinematic | N/A (kinematic) | 0.25 Hz (wheeled) / 0.5 Hz (swimming) | 500 iterations | Nonholonomic wheel constraints / viscous drag |
| Liu 2021 | Custom FEM (GPU) | Not reported | CPG-driven | Variable (until goal or failure) | One-direction wheels + contact forces |
| Liu 2022 | MuJoCo | Not reported | Per-timestep | Not reported | Passive wheels (anisotropic) |
| Liu 2023 | Custom FEM (GPU) | 60 ms (data collection) | Per-timestep | Variable (until goal or failure) | One-direction wheels; domain randomized [0.1, 1.5] |
| Jiang 2024 | Custom Euler-Lagrange | Not reported | 0.5 Hz (RL) / 50 Hz (CPG) | 160 seconds | Coulomb + Stribeck: $\mu_c$, $\mu_s$, $\mu_v$, $v_s$ |
| Choi 2025 | DisMech / Elastica | 0.05 s / 0.0002 s | 10 Hz / 2 Hz | Not reported | IPC barrier (DisMech) / spring-damper penalty (Elastica) |

---

## Key Dynamics and CPG Equations

### CPG Models

Three papers use Central Pattern Generators. All three use the **Matsuoka oscillator**:

$$K_f \tau_r \dot{x}_i^e = -x_i^e - a z_i^f - b y_i^e - \sum_j w_{ji} y_j^e + u_i^e$$
$$K_f \tau_a \dot{z}_i^e = x_i^e - z_i^e$$
$$y_i^e = \max(0, x_i^e)$$

| Paper | CPG Type | Oscillators | Coupling | RL Controls | Frequency Control |
|---|---|---|---|---|---|
| Liu 2021 | Matsuoka (Network VIII) | 4 (one per link) | Linear coupling $w_{ji}$ | Tonic inputs $u_i^e$ via sigmoid | Discrete option $K_f$ |
| Liu 2023 | Matsuoka + FOC | 4 (one per link) | Linear coupling $w_{ji}$ | Tonic inputs $u_i^e$ via sigmoid | Discrete option $K_f$; FOC constraint $c = 0.75$ |
| Jiang 2024 | Amplitude-controlled | 2 (pitch + yaw) | Tridiagonal matrix $A$, $B$ | 7 params: amplitudes, frequency, phases, offsets | Continuous $\omega \in [-0.1, 0.1]$ |

**Liu 2023's FOC contribution**: The free-response oscillation constraint adds a baseline tonic input $c > 0$ that establishes a minimum oscillation amplitude $A_0(c, \omega)$. This prevents the RL policy from learning vanishingly small oscillations that work in simulation but fail on real hardware (where pneumatic actuators have minimum activation thresholds).

### Friction and Contact Models

| Paper | Model | Key Parameters |
|---|---|---|
| Bing 2019 | MuJoCo soft contact (convex optimization) | Constant damping coefficient per wheel joint |
| Shi 2020 (wheeled) | Nonholonomic constraints | $-\dot{x}_i \sin\theta_i + \dot{y}_i \cos\theta_i = 0$ (no lateral slip) |
| Shi 2020 (swimming) | Resistive force theory | Low-Reynolds viscous drag, coefficient $k$ |
| Liu 2021/2023 | One-direction wheels (FEM contact) | Anisotropic friction via passive wheels on rigid bodies |
| Liu 2022 | MuJoCo soft contact (convex optimization) | Passive wheels (anisotropic damping) |
| Jiang 2024 | Coulomb + Stribeck | $s_i = \mu_c - (\mu_c - \mu_s)\exp(-|\dot{p}|^2/v_s^2)$; spring-damper normal: $F_z = -k_1 p_z - k_2 \dot{p}_z$ |
| Choi 2025 (DisMech) | Incremental Potential Contact (IPC) | $C^1$-continuous barrier: $F = -k\nabla_q(\frac{1}{K}\log(1+e^{K\epsilon}))^2$, $K = 15/\delta$ |
| Choi 2025 (Elastica) | Spring-damper penalty | $F = H(\epsilon)(-F_\perp + k\epsilon + d)\hat{u}$ |

#### How the Models Relate: A Hierarchy of Abstractions

These friction/contact models are **not mutually exclusive** — they form a hierarchy from most idealized to most general. Each level relaxes assumptions from the level above, and the simpler models can be understood as special cases or approximations of the more general ones.

```
Level 0: No friction at all (frictionless)
  │
Level 1: Kinematic constraints (Shi 2020 — wheeled)
  │   Idealized: perfect no-slip, no forces computed
  │
Level 2: Viscous drag (Shi 2020 — swimming)
  │   Force ∝ velocity, no stick-slip transition
  │
Level 3: Coulomb friction (dry friction)
  │   Constant kinetic friction μ_k, discontinuous at v=0
  │
Level 4: Coulomb + Stribeck (Jiang 2024)
  │   Smooth transition from static μ_s to kinetic μ_c
  │
Level 5: Soft contact with friction cones (MuJoCo — Bing 2019, Liu 2022)
  │   Convex optimization, compliant contact, elliptic cones
  │
Level 6: Implicit barrier contact (DisMech IPC — Choi 2025)
      Smooth C¹ barriers, fully implicit, unconditionally stable
```

#### Level 1: Nonholonomic Constraints (Shi 2020, Wheeled)

$$-\dot{x}_i \sin\theta_i + \dot{y}_i \cos\theta_i = 0$$

The simplest model. Assumes wheels enforce **perfect no-lateral-slip** at each link. No friction forces are computed — the constraint is algebraic and baked into the kinematic reconstruction equation $\xi = -\mathbf{A}(\alpha)\dot{\alpha}$. This is valid when wheel contact is ideal and the robot never slides sideways.

- **Assumptions**: Rigid ground, perfect wheel contact, no slip, no dynamics
- **What it misses**: Wheel slip, friction forces, deformation, normal forces, energy dissipation
- **Relationship to others**: This is the $\mu \to \infty$ limit of Coulomb friction — infinite lateral friction means zero lateral velocity, which is exactly the nonholonomic constraint

#### Level 2: Resistive Force Theory (Shi 2020, Swimming)

$$F = (F_x, F_y, F_\theta)^T = \omega_1(\alpha)\xi + \omega_2(\alpha)\dot{\alpha} = 0$$

In a low-Reynolds-number fluid, drag force is **proportional to velocity** (no inertia). Each slender link experiences tangential drag $c_t$ and normal drag $c_n$ (with $c_n > c_t$, providing the anisotropy needed for swimming). The force balance $F = 0$ yields a kinematic connection form identical in structure to the wheeled case.

- **Assumptions**: Low Reynolds number (viscous-dominated), slender body, no inertia
- **What it misses**: Inertial effects, turbulence, vortex shedding, ground contact
- **Relationship to others**: The anisotropy $c_n > c_t$ plays the same role as anisotropic friction in the wheeled models (high lateral resistance, low forward resistance). This is the fluid analogue of passive wheels.

#### Level 3–4: Coulomb and Coulomb + Stribeck (Jiang 2024)

**Coulomb friction** (the foundation for Levels 3+):

$$F_{\text{friction}} = \begin{cases} -\mu_k F_N \cdot \text{sgn}(\dot{p}) & \text{if } |\dot{p}| > 0 \text{ (sliding)} \\ \text{whatever prevents motion} & \text{if } |\dot{p}| = 0 \text{ (sticking)} \end{cases}$$

**Stribeck extension** (Jiang 2024 adds the static-to-kinetic transition):

$$s_i = \mu_c - (\mu_c - \mu_s)\exp\left(-\frac{|\dot{p}_{C,i}|^2}{v_s^2}\right)$$

$$F_{\text{GRF},i} = -s_i F_{\text{GRF},z} \cdot \text{sgn}(\dot{p}_{C,i}) - \mu_v \dot{p}_{C,i}$$

$$F_{\text{GRF},z} = -k_1 p_{C,z} - k_2 \dot{p}_{C,z}$$

Where:
- $\mu_c$ — Coulomb (kinetic) friction coefficient
- $\mu_s$ — static friction coefficient ($\mu_s > \mu_c$)
- $v_s$ — Stribeck velocity (controls the width of the transition region)
- $\mu_v$ — viscous friction coefficient (linear damping at high velocities)
- $k_1, k_2$ — normal contact spring stiffness and damping
- $p_{C,z}$ — penetration depth (normal direction)
- $\dot{p}_{C,i}$ — contact point velocity (tangential direction $i$)

This is the most physically detailed friction model among the rigid-body papers. The Stribeck curve captures the real phenomenon where **static friction is higher than kinetic friction**, with a smooth exponential transition controlled by $v_s$. The viscous term $\mu_v \dot{p}$ adds velocity-proportional damping at high speeds.

- **Assumptions**: Rigid bodies, known contact geometry, point contacts
- **What it misses**: Compliant contact surfaces, distributed contact patches, contact dynamics
- **Relationship to others**: Pure Coulomb is the $v_s \to 0$, $\mu_v \to 0$ limit. The passive wheel models (Bing 2019, Liu 2021/2022/2023) implicitly encode anisotropic Coulomb friction through the mechanical design of one-direction wheels.

#### Level 5: MuJoCo Soft Contact (Bing 2019, Liu 2022)

MuJoCo uses a fundamentally different approach from all the custom simulators above. Instead of the classical Linear Complementarity Problem (LCP) for contact, MuJoCo uses a **convex optimization** formulation (Todorov, 2014):

$$f^* = \arg\min_{\lambda \in \Omega} \frac{1}{2}\lambda^T(A + R)\lambda + \lambda^T(a_u - a_r)$$

Where:
- $\lambda$ — contact force vector
- $A = JM^{-1}J^T$ — inverse inertia matrix projected into constraint space
- $J$ — constraint Jacobian
- $M$ — mass matrix
- $R$ — diagonal regularizer (controls contact softness)
- $a_u$ — unconstrained acceleration in constraint space
- $a_r$ — reference acceleration (spring-damper stabilization)
- $\Omega$ — feasible set (friction cone constraints)

**Friction cone**: MuJoCo supports both elliptic and pyramidal approximations of the Coulomb friction cone:

$$K_{\text{elliptic}} = \{f \in \mathbb{R}^n : f_1 \geq 0,\; f_1^2 \geq \sum_{i>1} f_i^2/\mu_{i-1}^2\}$$

The key innovation is **relaxing the complementarity condition**. Classical LCP enforces that contact force and penetration velocity cannot both be positive (hard contact). MuJoCo allows both, modeling contacts as **soft** (compliant). This makes the problem convex (polynomial-time solvable) rather than NP-hard, and better reflects real materials that deform on contact.

Contact softness is controlled by `solimp` (impedance) and `solref` (reference stabilization) parameters. Constraint stabilization uses a spring-damper:

$$a_r = -b \cdot (J\dot{q})_i - k \cdot r_i$$

Where $b$ is damping, $k$ is stiffness, and $r_i$ is the position residual (penetration depth).

- **Assumptions**: Convex contact geometries (or mesh decomposition), known collision pairs
- **What it misses**: Distributed soft-body contact, mesh-level deformation (contacts are between rigid geoms)
- **Relationship to others**: MuJoCo's friction cone subsumes Coulomb friction — Coulomb is the hard-contact limit ($R \to 0$). The Stribeck effect is not explicitly modeled but can be approximated through solver parameters. The passive wheels in Bing 2019 and Liu 2022 are modeled as MuJoCo joints with anisotropic damping, so MuJoCo handles their contact implicitly.

#### Level 6: Incremental Potential Contact / IPC (Choi 2025, DisMech)

The most sophisticated contact model in these papers. IPC uses a **smooth barrier function** that prevents interpenetration through a potential energy term:

$$F_{\text{contact}} = -k \nabla_q \left(\frac{1}{K}\log(1 + e^{K\epsilon})\right)^2, \quad K = \frac{15}{\delta}$$

Where:
- $k$ — contact stiffness ($10^6$ for DisMech)
- $\epsilon$ — signed distance (gap) between surfaces (negative = penetrating)
- $K$ — barrier sharpness parameter
- $\delta$ — contact tolerance ($0.005$ m)
- $q$ — generalized coordinates

The $\frac{1}{K}\log(1+e^{K\epsilon})$ is a **softplus function** — a smooth ($C^1$-continuous) approximation of $\max(0, \epsilon)$. Squaring it and taking the gradient yields a repulsive force that:
- Grows smoothly as surfaces approach (no discontinuous activation)
- Is exactly zero when surfaces are far apart ($\epsilon \gg \delta$)
- Becomes very stiff near contact ($\epsilon \approx 0$)

Because IPC is fully implicit (solved together with the time-stepping), it is **unconditionally stable** — there is no timestep restriction from contact stiffness, which is why DisMech can use $\Delta t = 0.05$s while Elastica needs $\Delta t = 0.0002$s.

**Elastica's penalty contact** (for comparison):

$$F_{\text{contact}} = H(\epsilon)(-F_\perp + k\epsilon + d)\hat{u}$$

Where $H(\epsilon)$ is the Heaviside step function (discontinuous activation) and $F_\perp$, $d$ are damping terms. This is a standard spring-damper penalty that activates abruptly on contact — the discontinuity requires very small timesteps for stability.

- **Assumptions**: Smooth geometries, known gap function
- **What it misses**: Coulomb-style friction directionality (IPC handles normal forces; tangential friction requires additional modeling)
- **Relationship to others**: IPC generalizes the spring-damper penalty (Elastica) by making the barrier smooth. Both are penalty methods, but IPC's smoothness + implicit solving enables the 40× speedup. MuJoCo's soft contact is conceptually similar (both allow penetration with penalty), but MuJoCo solves a QP in constraint space while IPC adds a potential energy to the time-stepping objective.

#### Overlaps and Relationships

The models share a common theoretical backbone — **Coulomb friction** — but differ in how they implement it:

| Concept | Idealized (Shi) | Explicit (Jiang, Elastica) | Implicit (MuJoCo, IPC) |
|---|---|---|---|
| Contact detection | Algebraic constraint | Penetration check + Heaviside | Smooth barrier / soft constraint |
| Friction force | Infinite (no-slip) or drag | $\mu F_N \cdot \text{sgn}(\dot{p})$ | Friction cone optimization |
| Normal force | Not computed | Spring-damper: $F = kx + c\dot{x}$ | Convex QP (MuJoCo) / barrier potential (IPC) |
| Sticking → sliding | Instantaneous | Stribeck exponential | Cone constraint boundary |
| Differentiability | N/A (kinematic) | Discontinuous ($\text{sgn}$, $H$) | Smooth ($C^1$+) |
| Stability | Unconditional | Timestep-limited | Unconditional (implicit) |

**Key overlaps:**
- All tangential friction models are rooted in Coulomb's law: $|F_t| \leq \mu |F_n|$
- Passive wheels (Bing 2019, Liu 2021/2022/2023) are a **mechanical implementation** of anisotropic Coulomb friction — the wheel geometry enforces high lateral / low forward friction without computing friction forces explicitly
- MuJoCo's elliptic friction cone is a generalization of Coulomb's friction cone to 3D (with tangential + torsional + rolling components)
- Resistive force theory (swimming) and passive wheel damping are both **velocity-proportional** force models, just in different media (fluid vs. ground)

#### Which Is Best?

There is no single "best" — it depends on the application:

| Criterion | Best Model | Why |
|---|---|---|
| **Computational speed** | Nonholonomic / kinematic (Shi 2020) | No forces computed at all; trivial to solve |
| **Speed + moderate fidelity** | MuJoCo soft contact (Bing 2019, Liu 2022) | Convex QP is polynomial; soft contact avoids stiff ODE; mature solver |
| **Physical realism (rigid)** | Coulomb + Stribeck (Jiang 2024) | Captures static-to-kinetic transition, viscous damping |
| **Physical realism (soft)** | IPC (Choi 2025, DisMech) | Smooth barriers, implicit solve, unconditionally stable |
| **Sim-to-real transfer** | IPC or MuJoCo + domain randomization | Smooth gradients help; Liu 2023 shows domain randomization is critical |
| **RL training efficiency** | IPC (DisMech) | 40× speedup over penalty methods due to large stable timesteps |

**For RL specifically**, the critical properties are:
1. **Smoothness** — discontinuous contact activation (Heaviside, sgn) creates non-smooth dynamics that make policy gradients noisy. IPC and MuJoCo both provide smooth contact forces.
2. **Stability at large timesteps** — faster simulation = more training data per wall-clock hour. IPC (implicit) allows $\Delta t = 0.05$s; Elastica (explicit) requires $\Delta t = 0.0002$s.
3. **Differentiability** — for gradient-based optimization, smooth contact models (IPC, MuJoCo) are preferable. Stribeck friction adds smoothness to Coulomb but still has near-discontinuities at low velocity.

MuJoCo is the most widely used in RL research because it strikes the best balance of speed, fidelity, and ease of use for rigid-body robots. For soft-body robots, Choi 2025 makes a strong case that DisMech (IPC) is now the better choice due to its implicit stability advantage.

---

## Baselines and Quantitative Results

| Paper | Baselines | Key Metric | Result vs. Baseline |
|---|---|---|---|
| Bing 2019 | Grid search gait eq., Bayesian opt. | Energy (APPV) | 35–65% energy savings at 0.15 m/s |
| Shi 2020 | Geometric mechanics analysis | Displacement per cycle | Matches theoretical optimum (high $\|dA\|$ regions) |
| Liu 2021 | Obstacle-free PPOC-CPG, vanilla PPO | Success rate | 0.91 vs. 0.82 (obstacle-free) vs. 0.74 (vanilla PPO) |
| Liu 2022 | End-to-end PPO | Episode reward | ~0.8 at 1M steps vs. ~0.25 at 2M steps (3.2× higher) |
| Liu 2023 | Vanilla PPO, PPOC-CPG (no FOC) | Sim-to-real speed drop | 11% (FOC) vs. 54% (PPOC) vs. 81% (vanilla PPO) |
| Jiang 2024 | Joint-space DDPG | Training time | 12.25 hrs vs. 526 hrs (43× faster) |
| Choi 2025 | Elastica (explicit) | Training speedup | 2.4–22× overall; up to 40× per-step (contact tasks) |

---

## Domain Randomization (Liu 2023)

Only Liu 2023 explicitly uses domain randomization for sim-to-real transfer:

| Parameter | Range |
|---|---|
| Ground friction coefficient | [0.1, 1.5] |
| Wheel friction coefficient | [0.05, 0.10] |
| Rigid body mass | [0.035, 0.075] kg |
| Tail mass | [0.065, 0.085] kg |
| Head mass | [0.075, 0.125] kg |
| Max link pressure | [5, 12] psi |
| Gravity angle | [-0.001, 0.001] rad |

This is the most detailed domain randomization scheme among all 7 papers and directly contributes to the small sim-to-real gap (11% speed drop, 8.1% success rate drop).

---

## Limitations and Future Directions

### Acknowledged Limitations

| Paper | Key Limitations |
|---|---|
| Bing 2019 | Flat ground only; passive wheels required; only slithering gait; difficulty matching exact target velocities at high speeds |
| Shi 2020 | 3-link only (scalability unclear); kinematic model (no dynamics/forces); discrete actions limit expressiveness; singularity at $\alpha_1 = \alpha_2$ |
| Liu 2021 | Only 4 soft links; simulation only; simple cylindrical obstacles; event-trigger threshold requires manual tuning |
| Liu 2022 | Camera faces one direction (quarter-circle limit); monocular only; depends on external ArUco marker |
| Liu 2023 | Pneumatic actuator delays; only 4 links; friction mismatch main source of sim-to-real gap |
| Jiang 2024 | Simulation only; flat ground; CPG constrains gait space to sinusoidal patterns; low RL frequency (0.5 Hz) |
| Choi 2025 | CPU-only DisMech (no GPU); no real robot validation; Kirchhoff rod (no shear); cross-sim contact transfer degrades |

### Proposed Future Work

| Paper | Directions |
|---|---|
| Bing 2019 | Wheel-less robots; complex environments |
| Shi 2020 | Physical robot transfer (prototype built); higher-dimensional robots; combine RL with geometric mechanics |
| Liu 2021 | Physical experiments; longer snake (≥10 links); investigate contact sensor placement |
| Liu 2022 | Multiple visual markers; multi-sensor fusion (GPS, IMU, camera, radar); domain randomization |
| Liu 2023 | Sensory feedback to CPG; obstacle-aided locomotion; distributed control for longer snakes |
| Jiang 2024 | Physical COBRA deployment; active posture control |
| Choi 2025 | GPU DisMech; sim-to-real transfer; actuator-aware modeling (SMA dynamics) |

Common themes across future work: **sim-to-real transfer** (5/7 papers), **scaling to more links** (3/7), and **richer environments** (4/7).
