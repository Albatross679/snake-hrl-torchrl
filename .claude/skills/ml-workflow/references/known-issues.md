# Known Training Issues

Previously encountered issues in ML training. Check these patterns first when diagnosing problems — many are recurring.

## Table of Contents

- [Lightning-Specific Issues](#lightning-specific-issues)
- [RL / TorchRL Issues](#rl--torchrl-issues)
- [Process & Lifecycle](#process--lifecycle)
- [Memory & OOM](#memory--oom)
- [Model & Checkpoint](#model--checkpoint)
- [Evaluation & Metrics](#evaluation--metrics)
- [Decoding & Generation](#decoding--generation)

---

## Lightning-Specific Issues

### BatchSizeFinder Not Modifying Batch Size

**Symptom:** `BatchSizeFinder` runs but training uses original batch size.

**Cause:** `LightningDataModule` doesn't expose a `batch_size` attribute, or `train_dataloader()` doesn't reference it.

**Fix:** Ensure `self.batch_size = cfg.batch_size` in `__init__` and `DataLoader(..., batch_size=self.batch_size)` in `train_dataloader()`.

### ModelCheckpoint Not Saving Best Model

**Symptom:** Only `last.ckpt` exists, no best checkpoint.

**Cause:** `monitor` metric name doesn't match what `self.log()` uses. E.g., `monitor="val_loss"` but model logs `"validation_loss"`.

**Fix:** Ensure `ModelCheckpoint(monitor=X)` matches exactly the key passed to `self.log(X, ...)` in `validation_step()`.

### EarlyStopping Not Triggering

**Symptom:** Training runs all epochs despite no improvement.

**Cause:** Same metric name mismatch as above, or `patience` set too high.

**Fix:** Verify metric name match. Use `check_on_train_epoch_end=False` (default) to check on validation.

### Lightning + WandbLogger Double Init

**Symptom:** `wandb.errors.UsageError: Call wandb.init() before wandb.log()` or multiple W&B runs created.

**Cause:** Mixing manual `wandb.init()` with `WandbLogger` (which calls `wandb.init()` internally).

**Fix:** Use `WandbLogger` exclusively. Access the run via `logger.experiment` if you need direct W&B API access. Call `wandb.finish()` after `trainer.fit()`.

### Lightning Checkpoint Incompatible with Raw PyTorch

**Symptom:** `KeyError` or shape mismatch when loading `.ckpt` file with `torch.load()`.

**Cause:** Lightning checkpoints store state_dict under `"state_dict"` key with potential `model.` prefix on all keys.

**Fix:** Use `MyModel.load_from_checkpoint("path.ckpt")` for Lightning loading. For raw PyTorch, extract with `checkpoint["state_dict"]` and strip prefix if needed.

---

## RL / TorchRL Issues

### bf16 AMP Corrupts PPO Log-Prob Computation

**Symptom:** Actor loss spikes to 1e11+ intermittently while critic loss stays stable.

**Cause:** bf16 autocast wraps the PPO loss module, including `TanhNormal.log_prob()`. bf16 has ~3 decimal digits of precision — log-probs become garbage, importance ratios exponentiate to inf.

**Fix:** Never wrap loss computation containing log-prob or importance ratio math in bf16 autocast. Only autocast network forward passes. Set `min_std >= 0.1` for TanhNormal distributions.

**Rule:** For any RL algorithm with importance ratios (PPO, GRPO, CISPO), keep loss computation in f32 even when using mixed precision for forward passes.

### PPO NaN Cascade from Unguarded Loss

**Symptom:** All losses become NaN at step N after progressive KL divergence spikes.

**Cause:** Extreme actor loss produces gradients that corrupt weights with NaN even after clipping. Once NaN enters weights, all subsequent computations are NaN.

**Fix:** Three-layer NaN guard:
1. `torch.isfinite(loss)` before `loss.backward()` — skip if not finite
2. `torch.isfinite(grad_norm)` after clipping — skip `optimizer.step()` if not finite
3. Per-batch KL early stopping at 1.5x `target_kl`

### PPO Loss Metrics Averaged Over Theoretical Max

**Symptom:** All PPO metrics (actor_loss, critic_loss, entropy, kl) display as 0.0000 despite active learning.

**Cause:** Metrics accumulated then divided by `num_epochs * num_batches` (theoretical max updates). When KL early stopping triggers early, actual updates << theoretical max.

**Fix:** Track `actual_updates` counter that increments only on `optimizer.step()`. Divide metrics by `actual_updates`.

### Missing RewardSum Transform

**Symptom:** `best_reward: -Infinity` and `total_episodes: 0` across entire training.

**Cause:** TorchRL collectors only populate `episode_reward` at done boundaries if `RewardSum()` transform is applied to the environment.

**Fix:** `env.append_transform(RewardSum())` after env creation. Without it, training works but episode-level monitoring is blind.

### SAC Vectorized Env Never Auto-Resets

**Symptom:** First batch of episodes completes correctly, then all subsequent episodes have length=1 with near-zero rewards.

**Cause:** In TorchRL 0.11.x, `ParallelEnv.step()` does NOT auto-reset done environments. `_step_count` stays at max, causing immediate `done=True`.

**Fix:** Replace `env.step()` + `step_mdp()` with `env.step_and_maybe_reset()` for vectorized paths. Also ensure env `self._device` is `torch.device`, not a string (TorchRL's `_reset()` calls `.type` on it).

### SAC Critic Divergence from Missing Gradient Clipping

**Symptom:** Critic loss rises from normal to 10K+ over ~100K steps. Q-values swing wildly. Alpha collapses to near-zero.

**Cause:** `max_grad_norm` defined in config but `_update()` never calls `clip_grad_norm_()`. With lr=0.001 and UTD>=4, a single large gradient destabilizes the critic.

**Fix:** Always verify `nn.utils.clip_grad_norm_()` is actually called in the update loop for both critic and actor.

### TorchRL v0.11 Collector Drops Custom Keys

**Symptom:** Custom observation/diagnostic keys silently missing from collected batches.

**Cause:** `SyncDataCollector` only preserves keys declared in env specs. Undeclared keys are silently dropped.

**Fix:** Add all custom keys to `observation_spec` as `Unbounded` specs, even diagnostic-only fields.

### TorchRL v0.11 Batch Structure Change

**Symptom:** Episode metrics never registered. Done/reward fields not found at batch root.

**Cause:** v0.11 moved `done`, `reward`, `episode_reward` under `batch["next"]` instead of batch root.

**Fix:** Access via `batch.get("next", batch)` for backward compatibility.

### TorchRL v0.11 Import Renames

**Symptom:** `ImportError` at runtime for spec classes.

**Cause:** `BoundedTensorSpec` → `Bounded`, `CompositeSpec` → `Composite`, `UnboundedContinuousTensorSpec` → `Unbounded`.

**Fix:** Use try/except import fallback for version compatibility.

### TorchRL v0.11 Parameter Renames

**Symptom:** `TypeError` on `ClipPPOLoss` constructor.

**Cause:** `critic_coef` → `critic_coeff`, `entropy_coef` → `entropy_coeff` (note double f).

**Fix:** Update parameter names. Check TorchRL changelog for the full rename list.

### Thread Exhaustion with Many Parallel Environments

**Symptom:** Hundreds of `pthread_create failed`, `BrokenPipeError` during `ParallelEnv` init. Works with 32 envs, fails with 64+.

**Cause:** Each subprocess spawns OpenBLAS thread pool (default 32 threads). 256 workers × 32 = 8,192 threads exceeding limits.

**Fix:** Set at the TOP of the script, before any imports:
```python
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
```

### Velocity-Based Reward Pathology

**Symptom:** Mean reward always negative, PPO cannot learn.

**Cause:** Velocity-based reward (v_g) is noisy, tiny magnitude, and negative whenever snake drifts. No positive signal to bootstrap from.

**Fix:** Replace with distance-based potential reward: `reward = gamma * phi(s') - phi(s)` where `phi(s) = -distance_to_target`.

### Physics Parameter Mismatch with Paper

**Symptom:** Training plateaus well below paper-reported performance.

**Cause:** Config values (rod radius, Young's modulus, damping, friction ratios) don't match the paper's Table of parameters. Common off-by-orders-of-magnitude errors.

**Fix:** Always compare every physics parameter in config against the paper's tables before training. Rod radius and Young's modulus are the most common culprits (r enters I as r^4).

---

## Process & Lifecycle

### Hung Process After Completion (futex deadlock)

**Symptom:** Process alive but no log output. W&B shows FINISHED. `strace` shows `futex_wait_queue_me`.

**Cause:** Race between `ThreadPoolExecutor` background threads and `gc.collect()` / `torch.cuda.empty_cache()` during post-training cleanup.

**Fix:** Drain all async futures in `finally` block before `_pool.shutdown(wait=True)`. Remove `torch.cuda.synchronize()` from `cleanup_vram()`.

**Detection:** Process alive + no log activity + W&B FINISHED → outputs are saved. Safe to `kill <PID>`.

### Buffered stdout with nohup

**Symptom:** Log file shows no output for long periods, then dumps many lines at once.

**Cause:** Python buffers stdout when piped (nohup redirects stdout to file).

**Fix:** Always launch with `PYTHONUNBUFFERED=1 nohup <command> ...`. Cannot fix without restart if already running.

### GPU Lock Contention

**Symptom:** Training command hangs at startup with no output (before any model loading).

**Cause:** Another GPU process holds `/tmp/gpu-task.lock` via `flock`. Expected behavior — the new process queues.

**Detection:** `script/gpu-lock.sh status`. If holder is hung/dead, kill it to release the lock.

### Python Bytecode Cache After Hotfix

**Symptom:** Code fix has no effect when process is restarted.

**Cause:** `__pycache__/*.pyc` files serve stale bytecode. `forkserver` start method compounds by caching imports early.

**Fix:** Kill ALL processes (including forkserver children), clear `__pycache__/` recursively, restart. Or use `PYTHONDONTWRITEBYTECODE=1`.

### Stall Detection False Positives During Init

**Symptom:** All workers killed within 60s of startup in infinite kill-respawn cycle.

**Cause:** Physics init (PyElastica, DisMech) takes >60s with many workers. Stall detection triggers before first output.

**Fix:** Only count zero-progress as stall AFTER first transition produced.

### PyTorch 2.6 weights_only Blocks Checkpoint Resume

**Symptom:** `UnpicklingError: Weights only load failed` when loading checkpoints with custom config objects.

**Cause:** PyTorch 2.6 changed `torch.load` default to `weights_only=True`. Checkpoints storing config dataclasses require unpickling custom classes.

**Fix:** Add `weights_only=False` to `torch.load()` for own checkpoints. These are trusted files.

### LR Scheduler Division by Zero

**Symptom:** `ZeroDivisionError` in smoke tests or very short runs.

**Cause:** `total_frames // frames_per_batch = 0` when total frames < batch size.

**Fix:** Use `max(1, total_frames // frames_per_batch)` for scheduler step calculations.

### Parallel Sweep Thread Contention

**Symptom:** Subset of parallel sweep runs crash at the same epoch with thread creation failures.

**Cause:** All parallel processes simultaneously spawn DataLoader workers for a new training phase (e.g., rollout loss starting at epoch 21).

**Fix:** Stagger resource-intensive phase transitions across parallel runs, or reduce DataLoader `num_workers` when running parallel sweeps.

---

## Memory & OOM

### OOM Crash Loop in Sweeps

**Symptom:** After one trial OOMs, all subsequent trials immediately OOM at `model.to(device)`.

**Cause:** `cleanup_vram()` cannot free model tensors still referenced by `main_with_config()` locals in the exception traceback's stack frame.

**Fix:** `sweep_train()` must catch exceptions explicitly (`except Exception as e`) to unwind the stack frame before calling `cleanup_vram()`. Double `gc.collect()` pass catches reference cycles.

### Sweep Inter-Trial OOM Fragmentation

**Symptom:** Later sweep trials OOM at smaller batch sizes than earlier trials succeeded with.

**Cause:** CUDA memory fragmentation accumulates across trials. `empty_cache()` frees cached memory but doesn't defragment.

**Fix:** Restart the sweep process. The sweep agent resumes from where it left off.

### GPU OOM from Concurrent Sessions

**Symptom:** Training OOMs unexpectedly despite batch size fitting previously.

**Cause:** Another process (Jupyter, another training session) is using GPU memory.

**Fix:** Check with `nvidia-smi`. Use GPU lock (`src/utils/gpu_lock.py`) to serialize GPU tasks.

---

## Model & Checkpoint

### LoRA Checkpoint State Dict Mismatch

**Symptom:** `RuntimeError: Error(s) in loading state_dict` when resuming/evaluating a LoRA-trained model.

**Cause:** LoRA wraps parameters with adapter prefixes. If checkpoint was saved without `merge_and_unload()`, state dict keys don't match vanilla model.

**Fix:** Current code uses `merge_and_unload()` before saving. If loading an old checkpoint, apply LoRA first, then load.

### MLP Head Device Mismatch

**Symptom:** `RuntimeError: Expected all tensors to be on the same device` during forward pass.

**Cause:** Custom layers initialized on CPU while base model is on CUDA.

**Fix:** Call `model = model.to(device)` after wrapping with custom head classes.

### Gradient Deadlock from Zero Initialization

**Symptom:** Loss stuck at ~1.0 (predicting the mean) across all epochs. No gradient flow.

**Cause:** Zero-initialized output layer in models with multiplicative ansatz (`alpha * f(...)`) produces zero gradients everywhere.

**Fix:** Use small random initialization (`nn.init.normal_(std=0.01)`) for output layers.

### Checkpoint Shape Mismatch Across Data Versions

**Symptom:** `RuntimeError: size mismatch for mlp.0.weight` when resuming from old checkpoint.

**Cause:** Input/output dimensions changed (e.g., state_dim 128 → 130 from data processing changes).

**Fix:** Cannot fine-tune cross-architecture. Start fresh. For future: `strict=False` + partial weight loading.

---

## Evaluation & Metrics

### W&B Metric Naming Must Match Dashboard

**Symptom:** W&B dashboard panels show "no data".

**Cause:** Code logs metrics with prefix (`train/loss`) but dashboard panels expect unprefixed names, or vice versa.

**Fix:** Establish metric naming convention early. When adding new metrics, verify they appear in the dashboard.

### SQL Exact Match Tokenizer Artifact

**Symptom:** SQL EM is 0% even though generated SQL looks correct.

**Cause:** Tokenizer artifacts (extra spaces around commas, parentheses) cause string mismatch.

**Fix:** Post-processing regex to normalize whitespace around punctuation.

### BOS Token Mismatch

**Symptom:** Generated outputs start with unexpected tokens or are systematically wrong.

**Cause:** `decoder_start_token_id` doesn't match what the model expects.

**Fix:** Set `decoder_start_token_id` explicitly in generation kwargs.

---

## Decoding & Generation

### Repetition Penalty Conflicts with Structured Output

**Symptom:** Generated SQL missing repeated keywords (`SELECT`, `AND`, `FROM` appearing only once).

**Cause:** `repetition_penalty` or `no_repeat_ngram_size` penalizes tokens that legitimately repeat in SQL.

**Fix:** Do not use repetition penalty for structured output generation tasks.

### Max New Tokens Truncation

**Symptom:** Generated queries cut off mid-statement.

**Cause:** `max_new_tokens` set too low for complex queries.

**Fix:** Increase `max_new_tokens`. For NL-to-SQL, 256 default; complex joins need 384+.
