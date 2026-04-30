# Pitfalls Research

**Domain:** Autonomous overnight data collection monitoring for neural surrogate model training (Cosserat rod physics simulation)
**Researched:** 2026-03-09
**Confidence:** HIGH (pitfalls derived from codebase analysis, known project issues, and domain literature)

## Critical Pitfalls

### Pitfall 1: Silent Worker Death With Incomplete Data

**What goes wrong:**
A multiprocessing worker crashes (segfault in PyElastica, OOM, or unhandled exception) but the main monitoring process does not detect it. The `shared_counter` stops incrementing for that worker, but the monitor loop only checks if *any* workers are alive -- it does not track *which* workers are alive. If 15 of 16 workers are running, the run appears healthy while missing 1/16th of expected throughput. Worse, the dead worker's last batch may be a truncated `.pt` file that causes `torch.load` to fail during training, poisoning the entire dataset.

**Why it happens:**
The current `_multiprocess_collect` checks `alive = [p for p in workers if p.is_alive()]` and stops only when all workers are dead. There is no per-worker health tracking. Python's `multiprocessing.Process` does not raise exceptions in the parent when a child crashes -- you must explicitly check `p.exitcode`. The code currently never checks exit codes during the monitoring loop.

**How to avoid:**
1. Track per-worker status: maintain a dict mapping worker_id to last-seen transition count (via per-worker shared counters or a shared array).
2. In the monitoring loop, check `p.exitcode` for each worker. If a worker has died with a non-zero exit code, log the failure, optionally restart it, or alert.
3. Add a `_save_batch` atomic write pattern: write to a temp file first, then rename. This prevents truncated batch files from partially-written saves.
4. After collection completes, verify all batch files load successfully before declaring the run healthy.

**Warning signs:**
- FPS drops by exactly 1/N (where N = num_workers) mid-run
- A worker's batch file count stops increasing while others continue
- `p.exitcode != 0` for any worker (currently unchecked)
- Truncated `.pt` files (file size significantly smaller than siblings)

**Phase to address:**
Worker health monitoring phase -- this must be in the first implementation phase because it is the foundation for all other monitoring.

---

### Pitfall 2: Physics Instability Producing NaN/Inf Transitions That Poison Training

**What goes wrong:**
PyElastica's explicit integrator (PositionVerlet) can produce NaN or Inf values when perturbations push the rod into unstable configurations. The current `perturb_rod_state` function adds Gaussian noise to positions, velocities, and angular velocities, then sets sinusoidal rest curvature. A bad combination (high curvature + high velocity perturbation) can cause the Cosserat rod solver to diverge within a single RL step (500 substeps). The resulting NaN states are dutifully saved to batch files. When `SurrogateDataset` loads this data, a single NaN transition infects the entire training batch through gradient computation, causing the surrogate model's loss to become NaN and training to silently produce a useless model.

**Why it happens:**
The perturbation parameters (`position_std=0.002`, `velocity_std=0.01`, `omega_std=0.05`, `curvature_max=3.0`) are individually reasonable, but their joint distribution can produce extreme configurations. PyElastica's explicit integrator has no built-in stability guard -- it will happily compute NaN forces from NaN positions and propagate them. The data collection loop does not validate the physics output of each step. The existing `validate_surrogate_data.py` catches NaN *after* collection, but by then you have wasted hours of compute.

**How to avoid:**
1. Add per-transition NaN/Inf validation inside `collect_episode()`: check `next_state` after each step, and if invalid, discard the entire episode and reset.
2. Add a `state_magnitude_check`: if any state component exceeds a physically plausible range (e.g., position > 10m from origin, velocity > 10 m/s, omega > 100 rad/s), treat as a diverged simulation and discard.
3. Track and log the NaN/divergence rate per worker. If it exceeds a threshold (e.g., >5% of episodes), reduce perturbation parameters automatically.
4. Log NaN episodes to W&B as a counter metric so the monitoring agent can detect rising instability trends.

**Warning signs:**
- `validate_surrogate_data.py` reports NaN/Inf values in post-collection check
- Surrogate training loss suddenly jumps to NaN after loading new data
- Episode lengths suspiciously short (diverged simulation triggers `done` early)
- State magnitudes grow over time within an episode (energy injection from perturbation)

**Phase to address:**
Data quality monitoring phase -- runtime validation must be implemented before any overnight collection run.

---

### Pitfall 3: Coverage Measurement Giving False Confidence

**What goes wrong:**
The state-action coverage grid appears well-filled because the monitoring metric counts *how many bins have at least one sample*, but ignores that the joint distribution matters, not the marginal distribution. With Sobol sampling of 5D actions and perturbation of 124D states, the marginal coverage (each dimension independently) looks excellent. But the 129D joint state-action space is exponentially large -- even 10M transitions cover a negligible fraction of it. The monitoring agent reports "95% coverage" based on marginal bin filling, creating false confidence that the dataset is sufficient. The surrogate model then fails on state-action combinations that are individually well-represented but jointly rare.

**Why it happens:**
The existing `compute_density_weights()` in `dataset.py` projects 124D states to only 4 summary features (CoM_x, CoM_y, velocity_magnitude, mean_omega_z) with 20 bins each. This gives 20^4 = 160K joint bins, but the true state manifold has much higher intrinsic dimensionality. Coverage metrics that report "percentage of bins filled" conflate marginal coverage with joint coverage. There is no existing mechanism to track whether the collected data covers the state-action regions the RL policy will actually visit.

**How to avoid:**
1. Define coverage in terms of *task-relevant* state-action subspaces rather than the full 129D space. For snake locomotion, the relevant subspaces are: (amplitude, frequency, wave_number) x (curvature_mode, heading, speed).
2. Use the same 4D summary feature projection from `compute_density_weights()` but track *minimum bin count* (not just binary occupancy). A bin with 1 sample is effectively uncovered for training.
3. Set coverage targets as minimum samples per bin (e.g., 50 transitions per bin in the 4D summary space) rather than percentage of bins with any sample.
4. Log the coverage heatmap to W&B so the monitoring agent can detect when marginal coverage is good but specific joint regions are empty.
5. Add an "edge case" coverage metric: track how many bins in the 5D action space corners (where all 5 actions are near -1 or +1 simultaneously) have been visited.

**Warning signs:**
- High marginal coverage but poor surrogate accuracy on validation set
- Surrogate model has very different error magnitudes across different initial conditions
- `compute_density_weights()` returns weights with extreme max/min ratio (>100:1)
- Sobol sampler has not been advanced enough to fill the action space corners

**Phase to address:**
Coverage tracking phase -- this belongs in the monitoring agent implementation, running periodically during collection.

---

### Pitfall 4: Disk Space Exhaustion Mid-Run With Partial Write Corruption

**What goes wrong:**
The current disk space pre-check estimates storage as `num_transitions * 1KB * 1.5` but this underestimates actual disk usage. Force/torque data (`collect_forces=True`, which is the default) adds 4 arrays per transition: external_forces (3x21), internal_forces (3x21), external_torques (3x20), internal_torques (3x20) = 244 floats * 4 bytes = ~1 KB per transition of force data alone. With state+action+forces, actual per-transition size is ~2.5 KB. For a 10M transition run, the estimate says 15 GB but actual is ~25 GB. If the disk fills mid-run, `torch.save` writes a partial `.pt` file. This file will fail to load with a cryptic pickle/torch error, and the monitoring loop will not detect it because the shared counter has already incremented.

**Why it happens:**
The disk space estimate in `_multiprocess_collect` uses a fixed 1 KB/transition heuristic that was likely estimated without force data. The `_save_batch` function uses `torch.save(data, path)` directly -- no atomic write, no verification that the file was written completely. Other processes on the system (logs, W&B artifacts, system journals) also consume disk space unpredictably during an 8+ hour run.

**How to avoid:**
1. Fix the disk estimate: compute actual per-transition bytes from the state/action/force dimensions. With forces: ~2.5 KB/transition. Without forces: ~1.1 KB/transition.
2. Add periodic disk space monitoring in the monitoring loop (already partially done -- `disk_free_gb` is computed but not acted upon). Set a hard threshold (e.g., <2 GB free) that triggers early termination with a clean save.
3. Implement atomic batch writes: write to `batch_XXXX.pt.tmp`, verify file integrity, then rename to `batch_XXXX.pt`.
4. Add a post-write verification: after `torch.save`, try `torch.load` on the file to confirm integrity (can be sampled, e.g., every 10th file).

**Warning signs:**
- `disk_free_gb` in W&B metrics trending toward zero
- Write errors in worker stdout (currently only printed, not captured by monitor)
- Batch files with sizes significantly different from their siblings
- System `dmesg` showing filesystem errors

**Phase to address:**
Infrastructure monitoring phase -- disk monitoring must be active from the first overnight run.

---

### Pitfall 5: W&B Connection Failure Blocking Workers or Losing All Metrics

**What goes wrong:**
W&B initialization happens in the main process before spawning workers. If the W&B backend becomes unreachable during an 8-hour run (network flap, W&B server maintenance, DNS timeout), the `wandb.log()` call in the monitoring loop can either: (a) block for its retry timeout (up to minutes), during which the monitor is not checking worker health, or (b) silently fail, losing all metrics for the rest of the run. The current code wraps `_wb.log()` in a bare `except Exception: pass`, which prevents crashes but also hides the problem entirely. After an overnight run with W&B failures, the monitoring agent has no metrics to analyze and cannot determine if the collection was healthy.

**Why it happens:**
The W&B client has an internal retry loop for transient network errors. When it enters this loop, `wandb.log()` blocks the calling thread. The monitoring loop is single-threaded -- while it is stuck in W&B retry, it is not polling worker health, disk space, or progress. Community reports confirm this is a common issue with long-running jobs (GitHub issues #10025, #8825, #5550 on wandb/wandb). The current `except Exception: pass` catches the error but does not log it locally or set a flag indicating degraded monitoring.

**How to avoid:**
1. Set `WANDB_MODE=online` with aggressive timeouts: `wandb.Settings(init_timeout=60, start_method="thread")`.
2. Add a local fallback: if W&B logging fails, write metrics to a local JSON/CSV file so the monitoring agent can still analyze the run post-hoc.
3. Move W&B logging to a separate thread with a timeout: if `wandb.log()` does not return within 5 seconds, skip it and log locally.
4. Track consecutive W&B failures as a metric. If >5 consecutive failures, switch to local-only logging and alert.
5. Use `WANDB_SILENT=true` to prevent W&B's own retry messages from flooding stdout and obscuring worker output.

**Warning signs:**
- Gaps in W&B metrics timeline (missing data points)
- "Network error...entering retry loop" messages in stdout
- Monitoring loop polls become irregular (visible in W&B step intervals)
- Post-run W&B dashboard shows truncated metrics

**Phase to address:**
W&B integration phase -- must be hardened before relying on W&B for autonomous monitoring decisions.

---

### Pitfall 6: Monitoring Agent Treating Stale Metrics as Current

**What goes wrong:**
The autonomous monitoring agent checks collection health periodically. If it reads metrics from W&B or stdout at time T, but the collection process has been frozen (deadlock, I/O wait, swap thrashing) since time T-30min, the agent sees the last-known-good metrics and concludes everything is fine. The collection could be effectively dead for hours while the agent reports "healthy" based on stale data.

**Why it happens:**
The monitoring agent polls W&B dashboards or shared counters but does not verify *when* those values were last updated. The shared transition counter is a snapshot -- it tells you the count but not whether it is still being incremented. If all workers are stuck in an `env.reset()` call that deadlocks PyElastica (known to happen with extreme perturbations), the counter freezes at its last value. The monitoring agent sees "5M transitions collected, target 10M, FPS was 270 last time I checked" and waits patiently.

**How to avoid:**
1. Track delta between polls: if `shared_counter` has not changed between two consecutive polls (e.g., 60 seconds apart), flag the collection as potentially stalled.
2. Have workers write a heartbeat timestamp to a shared file or memory-mapped value. The monitoring agent checks that each worker's heartbeat is recent (within 2x the expected episode duration).
3. Define a minimum FPS threshold (e.g., 50% of baseline FPS from smoke test). If rolling FPS drops below this, raise an alert.
4. Add a "liveness" metric to W&B: `seconds_since_last_transition`. If this exceeds a threshold, the monitoring agent should investigate.

**Warning signs:**
- `fps_current` drops to 0 while `fps_rolling` remains positive (rolling average masks the stall)
- Transition count unchanged across multiple poll intervals
- Worker process CPU usage drops to 0% (visible in `top` or `/proc/[pid]/stat`)
- ETA suddenly jumps to extremely large values

**Phase to address:**
Core monitoring agent phase -- staleness detection is a fundamental capability the agent needs from the start.

---

### Pitfall 7: Episode ID Offset Arithmetic Overflow on Append Runs

**What goes wrong:**
Each worker's episode IDs start at `existing_episode_offset + worker_id * 10_000_000`. If there have been several append runs, `existing_episode_offset` accumulates. With 16 workers and a max worker_id of 15, the maximum episode ID per run is `offset + 15 * 10_000_000 = offset + 150_000_000`. After 14 append runs, `existing_episode_offset` exceeds 2 billion, approaching int32 limits. Episode IDs are stored as int64 in the batch files, but downstream code (density weighting, train/val splitting) may inadvertently use int32 operations, causing silent overflow and episode ID collisions that leak validation data into training.

**Why it happens:**
The 10_000_000 spacing per worker is generous for a single run but accumulates quickly across appends. `_find_max_episode_id` correctly finds the max, but the arithmetic `existing_episode_offset + worker_id * 10_000_000` can produce very large values. The `SurrogateDataset` train/val split relies on `torch.unique(self.episode_ids)`, which works correctly with int64, but if any intermediate conversion uses int32 or float32 (which has only 24 bits of integer precision), IDs above 16M will lose precision and collide.

**How to avoid:**
1. Verify that all episode ID arithmetic stays in int64 throughout the pipeline. Search for any `.int()` or `.to(torch.int32)` calls on episode IDs.
2. Consider a simpler ID scheme: `run_id * 1_000_000 + episode_within_run`. This keeps IDs smaller and more predictable.
3. Add an assertion in the monitoring agent: after each batch save, verify that episode IDs in the new batch do not overlap with any existing batch.
4. In the post-collection validator, add a check that no episode ID appears in both train and val splits.

**Warning signs:**
- Episode IDs exceeding 1 billion in batch files
- `_find_max_episode_id` taking unusually long (scanning many files)
- Unexpected overlap between train and val accuracy (data leakage signal)

**Phase to address:**
Data integrity phase -- should be checked when implementing append-mode monitoring.

---

### Pitfall 8: Perturbation-Induced Distribution Shift Making Surrogate Useless for RL

**What goes wrong:**
State perturbation (`perturbation_fraction=0.3`) generates diverse initial conditions, but the resulting transitions may come from physically implausible states that the RL policy will never encounter. The surrogate model spends capacity learning dynamics of states that do not exist on the natural manifold of snake locomotion. When the RL policy runs inside the surrogate, it only visits states reachable from a straight-rod reset through sequences of valid actions -- states that were underrepresented in the perturbed training data. The surrogate is accurate for perturbation-initialized states but inaccurate for the states the RL agent actually needs.

**Why it happens:**
The perturbation adds Gaussian noise to positions, velocities, and angular velocities independently. But Cosserat rod states have intrinsic constraints: positions must be consistent with element lengths (the rod is inextensible), velocities must be consistent with the rod's current shape, and angular velocities must be consistent with the torques being applied. Random perturbation violates these constraints, producing states that are not on the physical manifold. The resulting transitions show the rod "snapping back" to a valid configuration -- dynamics that never occur during actual locomotion.

**How to avoid:**
1. Track the fraction of training data that comes from perturbed vs. unperturbed episodes separately. Ensure the surrogate is evaluated on *unperturbed* episodes to measure accuracy for the RL use case.
2. Consider physics-consistent perturbation: instead of adding noise to raw state variables, perturb the *action sequence* (apply random actions for N steps, then start collecting), which produces perturbed-but-physically-valid states.
3. After collection, analyze the delta (next_state - state) distribution for perturbed vs. unperturbed episodes. If the distributions are very different, the perturbation may be counterproductive.
4. If perturbation data hurts surrogate accuracy on the natural manifold, reduce `perturbation_fraction` or switch to action-sequence perturbation.

**Warning signs:**
- Surrogate has low training loss but high validation loss on unperturbed test episodes
- Per-component RMSE is much larger for position deltas than expected (snap-back dynamics)
- The delta distribution for perturbed episodes has much higher variance than unperturbed
- RL policy trained on surrogate performs much worse than expected when transferred to real PyElastica

**Phase to address:**
Coverage analysis phase -- should be assessed during data quality evaluation, before surrogate training.

---

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Bare `except Exception: pass` around W&B logging | Collection never crashes due to W&B | Silently loses all monitoring data; impossible to debug post-hoc | Never -- at minimum log the exception locally |
| Using `shared_counter` as sole progress metric | Simple IPC with minimal overhead | Cannot distinguish healthy collection from stalled workers; no per-worker diagnostics | Only during smoke tests with small transition counts |
| Estimating disk usage with a fixed multiplier | Avoids computing exact byte sizes | Underestimates by 2-3x when forces are enabled; leads to disk exhaustion | Never in production runs -- compute from actual tensor dimensions |
| Printing progress to stdout instead of structured logs | Easy to implement, human-readable | Cannot be parsed by monitoring agent; lost if terminal disconnects | Only for interactive debugging; always duplicate to structured log file |
| Hardcoded episode ID spacing (10M per worker) | Simple arithmetic, no collision within one run | Accumulates across appends; large IDs risk precision loss in float32 operations | Acceptable for a single run; needs revision for multi-run append workflows |

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| W&B | Calling `wandb.init()` in parent process, then `wandb.log()` blocks on network failure | Set `WANDB_MODE=offline` as fallback; use a timeout wrapper around `wandb.log()`; always maintain local metric log |
| W&B + multiprocessing | Initializing W&B in a forkserver child process causes `ManagerConnectionRefusedError` | Only init W&B in the main/monitor process; workers should not touch W&B directly |
| PyElastica + forkserver | Using `fork` start method with PyElastica's global state causes segfaults in children | Always use `forkserver` (already done in current code); set `OMP_NUM_THREADS=1` per worker |
| `torch.save` + concurrent writes | Two workers writing to files with similar names in the same directory can cause filesystem contention | Use worker-specific prefixes (already done: `batch_w{id}`); add atomic write (tmp + rename) |
| `multiprocessing.Value` | Reading shared counter without lock gives torn reads on some architectures | Use `shared_counter.value` within a `with shared_counter.get_lock():` block for reads too, not just writes (current code only locks on writes) |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| `_find_max_episode_id` scans all batch files on every restart | Startup takes seconds for small datasets | Cache the max ID in a metadata file (`meta.json`) alongside batch files | >1000 batch files (typical for 10M+ transitions across multiple append runs) |
| `validate_surrogate_data.py` loads entire dataset into memory | Works for 1M transitions (~500 MB) | Stream validation: check each batch file independently, accumulate statistics | >5M transitions (>2.5 GB) -- OOM on machines with limited RAM |
| Monitor loop calls `_get_dir_size_gb` every 30 seconds | Negligible for 10 files | Only check disk size every 5th poll iteration; use `shutil.disk_usage` (O(1)) instead of `rglob("*")` (O(N files)) | >500 batch files -- directory walk becomes measurable overhead |
| `torch.load` with `weights_only=True` on every batch during validation | Safe and correct | Use memory-mapped loading or stream individual tensors for large datasets | >100 batch files -- loading time dominates validation |
| All workers sharing a single Sobol engine seed offset | Adequate for 1M transitions | Different workers need carefully separated Sobol subsequences; current `seed + worker_id * 1000` may produce overlapping sequences for high worker counts | >32 workers (Sobol sequences with nearby seeds can have correlated early terms) |

## "Looks Done But Isn't" Checklist

Things that appear complete but are missing critical pieces.

- [ ] **Data collection "completed":** Often missing post-collection validation -- verify with `script/validate_surrogate_data.py` that no NaN/Inf exist and episode IDs are unique
- [ ] **Coverage metric "95%":** Often missing joint coverage analysis -- check that coverage is measured in task-relevant subspaces, not just marginal bins
- [ ] **Monitoring "active":** Often missing staleness detection -- verify the agent distinguishes between "last poll was healthy" and "currently healthy"
- [ ] **Worker health "all alive":** Often missing exit code checking -- a worker can be "alive" (zombie process) with a non-zero exit code, meaning it crashed but wasn't reaped
- [ ] **Disk space "sufficient":** Often missing force data accounting -- the pre-check may use 1 KB/transition but actual is 2.5 KB with forces enabled
- [ ] **W&B "logging":** Often missing local fallback -- if W&B silently fails, all metrics for the run are lost with no way to reconstruct
- [ ] **Batch files "saved":** Often missing atomic write verification -- a power loss or OOM during `torch.save` produces a corrupt file that silently poisons the dataset
- [ ] **Stop condition "met":** Often missing quality gate -- reaching the target transition count says nothing about whether the data is *diverse enough* for training

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Silent worker death | LOW | Restart collection with `--save-dir` pointing to same directory (append mode handles deduplication); verify batch file integrity with `validate_surrogate_data.py` |
| NaN transitions in batch files | MEDIUM | Run `validate_surrogate_data.py --data-dir ...`; write a script to filter out NaN-containing transitions from each batch file; re-save clean batches |
| Disk full mid-run | MEDIUM | Delete incomplete/truncated batch files (check file sizes vs median); free disk space; restart with append mode; add `--skip-disk-check` only after verifying free space |
| W&B metrics lost | LOW | Check local stdout logs (if captured via `tee` or `nohup`); manually compute FPS and coverage from batch file timestamps and contents |
| Episode ID collision across appends | HIGH | Must reload all batch files, re-assign globally unique episode IDs, re-save; the train/val split for any models trained on this data is invalid |
| Stale monitoring with stalled workers | MEDIUM | Kill and restart the entire collection; verify last N batch files for integrity; resume with append mode |
| Coverage holes in state-action space | MEDIUM | Run targeted collection: set perturbation parameters to specifically target the uncovered region; append to existing dataset |
| Perturbation distribution shift | HIGH | Must retrain surrogate with adjusted perturbation fraction; may need to discard perturbed data entirely if it hurts accuracy; requires re-collection |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Silent worker death | Phase 1: Worker health monitoring | Each poll iteration logs per-worker status; dead workers trigger alert within 60 seconds |
| NaN/Inf transitions | Phase 1: Data quality validation | Per-episode NaN check in collection loop; NaN rate logged to W&B; rate < 0.1% |
| Coverage false confidence | Phase 2: Coverage tracking | Coverage metric uses minimum samples per bin, not binary occupancy; joint coverage logged |
| Disk exhaustion | Phase 1: Infrastructure monitoring | Disk free checked every poll; early stop if <2 GB free; estimate updated for force data |
| W&B connection failure | Phase 1: Logging resilience | Local JSON fallback always active; W&B timeout wrapper; consecutive failures tracked |
| Stale metric detection | Phase 2: Monitoring agent core logic | Agent computes delta between polls; zero-delta for >120s triggers stall alert |
| Episode ID overflow | Phase 1: Data integrity | Assertion on max episode ID per run; ID scheme reviewed for multi-append safety |
| Perturbation distribution shift | Phase 3: Post-collection analysis | Compare perturbed vs unperturbed delta distributions; surrogate eval stratified by perturbation status |

## Sources

- Codebase analysis of `aprx_model_elastica/collect_data.py`, `collect_config.py`, `dataset.py`, `state.py`, `validate.py`
- Project issue: [surrogate-data-collection-no-append.md](/home/coder/snake-hrl-torchrl/issues/surrogate-data-collection-no-append.md) -- data overwrite on re-run
- Project issue: [surrogate-model-data-imbalance.md](/home/coder/snake-hrl-torchrl/issues/surrogate-model-data-imbalance.md) -- state-action coverage gaps
- Project issue: [parallel-collection-scaling-bottleneck.md](/home/coder/snake-hrl-torchrl/issues/parallel-collection-scaling-bottleneck.md) -- L3 cache contention at high worker counts
- Project issue: [surrogate-spatial-structure-analysis.md](/home/coder/snake-hrl-torchrl/issues/surrogate-spatial-structure-analysis.md) -- architecture decisions
- Project experiment: [surrogate-data-coverage-improvements.md](/home/coder/snake-hrl-torchrl/experiments/surrogate-data-coverage-improvements.md) -- Sobol + perturbation implementation
- Project experiment: [surrogate-parallel-data-collection.md](/home/coder/snake-hrl-torchrl/experiments/surrogate-parallel-data-collection.md) -- multiprocess design
- [Python multiprocessing documentation](https://docs.python.org/3/library/multiprocessing.html) -- worker crash behavior, shared state caveats
- [Python bug #38084](https://bugs.python.org/issue38084) -- multiprocessing cannot recover from crashed worker
- [Stable Baselines3: Dealing with NaNs](https://stable-baselines3.readthedocs.io/en/master/guide/checking_nan.html) -- NaN detection patterns in RL
- [W&B GitHub issue #10025](https://github.com/wandb/wandb/issues/10025) -- ConnectTimeout in long-running jobs
- [W&B Troubleshooting docs](https://docs.wandb.ai/guides/technical-faq/troubleshooting) -- network error handling
- [GROMACS: Managing long simulations](https://manual.gromacs.org/current/user-guide/managing-simulations.html) -- checkpointing best practices for physics simulations
- [Checkpoint-Based Recovery for Long-Running Data Transformations](https://dev3lop.com/checkpoint-based-recovery-for-long-running-data-transformations/) -- checkpoint placement strategy
- [MuJoCo issue #168](https://github.com/google-deepmind/mujoco/issues/168) -- NaN/Inf simulation instability patterns

---
*Pitfalls research for: Autonomous overnight data collection monitoring for neural surrogate models*
*Researched: 2026-03-09*
