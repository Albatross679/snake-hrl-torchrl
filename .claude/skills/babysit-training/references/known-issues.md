# Known Training Issues — Quick Reference

Action-oriented summaries for rapid diagnosis during monitoring.

## Quick Lookup Table

| Pattern | Symptom | Immediate Action |
|---------|---------|-----------------|
| **Hung process** | Process alive, no log output, W&B shows FINISHED | Outputs saved. `kill <PID>`. Safe — futex deadlock during cleanup. |
| **OOM** | `CUDA out of memory` in log | Kill process, reduce batch size or set `auto_batch_size=False`, restart. |
| **Buffered stdout** | Log file empty for long periods | Must restart with `PYTHONUNBUFFERED=1`. |
| **GPU lock contention** | Command hangs at startup, no output | Check `flock` on `/tmp/gpu-task.lock`. Kill holder if dead. |
| **Dataset key mismatch** | `KeyError` on data field names | Data format evolved. Add fallback key lookup in dataset loader. |
| **W&B metric invisible** | Dashboard panels show "no data" | Check metric name prefix matches dashboard panel names. |
| **Watchdog kill** | Exit code 137/143 after training completes | Normal — watchdog killed hung cleanup. Outputs are saved. |
| **API key vs subscription** | `Credit balance is too low` in tmux Claude | `unset ANTHROPIC_API_KEY` before launching `claude` CLI. |
| **PPO actor loss spike** | Actor loss jumps to 1e11+, critic stable | bf16 AMP wrapping log-prob math. Disable AMP for loss computation. |
| **PPO NaN cascade** | All losses become NaN after KL spikes | Gradient corruption from extreme actor loss. Add NaN guard (see below). |
| **PPO metrics all zero** | Loss metrics display 0.0000 despite learning | KL early stop + dividing by theoretical max updates. Divide by actual updates. |
| **Missing episode rewards** | `best_reward: -Infinity`, `total_episodes: 0` | Missing `RewardSum()` transform on TorchRL env. |
| **SAC episodes stuck at len=1** | First batch OK, then all episodes length=1 | Missing auto-reset. Use `env.step_and_maybe_reset()` for vectorized envs. |
| **SAC critic divergence** | Critic loss > 10K, Q-values exploding | Gradient clipping not applied. Verify `clip_grad_norm_()` in update loop. |
| **ParallelEnv pthread crash** | `pthread_create failed` during init | Too many OpenBLAS threads. Set `OPENBLAS_NUM_THREADS=1` before imports. |
| **Checkpoint resume fails** | `UnpicklingError: Weights only load failed` | PyTorch 2.6 defaults `weights_only=True`. Add `weights_only=False` for own checkpoints. |
| **Stale bytecode after hotfix** | Fix has no effect after restart | Clear `__pycache__/`, kill all processes (including forkserver), restart. |
| **Stall detection kills workers** | All workers killed in infinite restart loop | Init takes longer than stall threshold. Only count stalls after first output. |
| **Physics params wrong** | Training plateaus well below paper results | Compare config values against paper tables. Rod radius and Young's modulus are common culprits. |

## Detailed Patterns

### PPO bf16 AMP + Log-Prob Precision

**Symptom:** Actor loss spikes to 1e11 intermittently while critic loss remains stable.

**Cause:** bf16 autocast wraps the PPO loss module, including TanhNormal `log_prob()`. bf16 has ~3 decimal digits of precision, producing garbage log-probs that exponentiate to inf in the importance ratio.

**Fix:** Never wrap loss computation containing log-prob or importance ratio math in bf16. Only autocast the network forward passes. Set `min_std >= 0.1` for the TanhNormal distribution.

**Detection:** Monitor `train/actor_loss` for intermittent spikes 10+ orders of magnitude above baseline.

### PPO NaN Guard Pattern

**Symptom:** All losses become NaN at step N after progressive KL divergence spikes.

**Cause:** Massive actor loss produces gradients that, even after clipping, corrupt weights with NaN. Once NaN enters the weights, all subsequent computations are NaN.

**Fix:** Three-layer defense:
1. Check `torch.isfinite(loss)` before `loss.backward()` — skip backward if not finite
2. Check `torch.isfinite(grad_norm)` after gradient clipping — skip `optimizer.step()` if not finite
3. Per-batch KL early stopping at 1.5x `target_kl` — break inner loop when KL exceeds threshold

### PPO Loss Metric Averaging Bug

**Symptom:** All PPO metrics (actor_loss, critic_loss, entropy, kl) display as 0.0000 despite active learning.

**Cause:** Metrics accumulated across mini-batch updates then divided by `num_epochs * num_batches` (theoretical max). When KL early stopping triggers early (e.g., 1-2 of 40 updates), values are divided by 40x their count.

**Fix:** Track `actual_updates` counter that increments only when `optimizer.step()` executes. Divide by `actual_updates`.

### SAC Vectorized Environment Auto-Reset

**Symptom:** First batch of episodes completes normally (correct length), then all subsequent episodes have length=1 with near-zero rewards.

**Cause:** In TorchRL 0.11.x, `ParallelEnv.step()` does NOT auto-reset done environments. `_step_count` stays at max in every worker, causing immediate `done=True`.

**Fix:** Replace `env.step()` + `step_mdp()` with `env.step_and_maybe_reset()` for vectorized paths. Also ensure env device is `torch.device`, not a string.

### SAC Critic Divergence from Missing Gradient Clipping

**Symptom:** Critic loss rises from normal range to 10K+ over ~100K steps. Q-values swing wildly. Alpha collapses to near-zero. Rewards drop.

**Cause:** Gradient clipping config field exists but `_update()` doesn't call `clip_grad_norm_()`. With lr=0.001 and UTD>=4, a single large gradient destabilizes the critic, causing cascading divergence.

**Fix:** Always verify that `nn.utils.clip_grad_norm_()` is actually called in the update loop for both critic and actor, using the config's `max_grad_norm` value.

### Thread Exhaustion with Parallel Environments

**Symptom:** Hundreds of `pthread_create failed` errors, `BrokenPipeError` during `ParallelEnv` initialization. Works with 32 envs but fails with 64+.

**Cause:** Each subprocess spawns its own OpenBLAS thread pool (default 32 threads). 256 workers x 32 threads = 8,192 threads, exceeding system limits.

**Fix:** Set thread-limiting env vars at the top of the script, **before any imports**:
```python
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
```

### Stall Detection False Positives During Init

**Symptom:** All workers killed within 60s of startup in infinite kill-respawn cycle.

**Cause:** Physics simulator init (PyElastica, DisMech) takes >60s with many workers competing. Stall detection threshold was 60s with no grace period.

**Fix:** Only count zero-progress as a stall AFTER the first transition is produced. Alternatively, add an `init_grace_period_seconds` config field.

### Python Bytecode Cache After Hotfix

**Symptom:** Code fix appears to have no effect when process is restarted.

**Cause:** `__pycache__/*.pyc` files serve stale bytecode. The `forkserver` start method compounds this by caching imports early.

**Fix:** Kill ALL processes (including forkserver children), clear `__pycache__/` recursively, then restart. Or use `PYTHONDONTWRITEBYTECODE=1` during development.
