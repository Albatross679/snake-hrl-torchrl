# Known Training Issues

Previously encountered issues in ML training. Check these patterns first when diagnosing problems — many are recurring.

## Process & Lifecycle

### Hung Process After Completion (futex deadlock)

**Symptom:** Process alive but no log output. W&B shows FINISHED. `strace` shows `futex_wait_queue_me`.

**Cause:** Race between `ThreadPoolExecutor` background threads (e.g., async SQL evaluation) and `gc.collect()` / `torch.cuda.empty_cache()` during post-training cleanup.

**Fix:** Drain all async futures in the `finally` block before `_pool.shutdown(wait=True)`. Remove `torch.cuda.synchronize()` from `cleanup_vram()` (redundant after `empty_cache()` and the primary contention point).

**Detection:** Process alive + no log activity + W&B FINISHED → outputs are saved. Safe to `kill <PID>`.

### Buffered stdout with nohup

**Symptom:** Log file shows no output for long periods, then dumps many lines at once.

**Cause:** Python buffers stdout when piped (nohup redirects stdout to file).

**Fix:** Always launch with `PYTHONUNBUFFERED=1 nohup <command> ...`. Cannot fix without restart if already running.

### GPU Lock Contention

**Symptom:** Training command hangs at startup with no output (before any model loading).

**Cause:** Another GPU process holds `/tmp/gpu-task.lock` via `flock`. Expected behavior — the new process queues.

**Detection:** `script/gpu-lock.sh status`. If holder is hung/dead, kill it to release the lock.

## Memory & OOM

### OOM Crash Loop in Sweeps

**Symptom:** After one trial OOMs, all subsequent trials immediately OOM at `model.to(device)`.

**Cause:** `cleanup_vram()` cannot free model tensors still referenced by `main_with_config()` locals in the exception traceback's stack frame.

**Fix:** `sweep_train()` must catch exceptions explicitly (`except Exception as e`) to unwind the stack frame before calling `cleanup_vram()`. Double `gc.collect()` pass catches reference cycles a single pass misses.

### Sweep Inter-Trial OOM Fragmentation

**Symptom:** Later sweep trials OOM at smaller batch sizes than earlier trials succeeded with.

**Cause:** CUDA memory fragmentation accumulates across trials. `empty_cache()` frees cached memory but doesn't defragment.

**Fix:** Restart the sweep process. The sweep agent resumes from where it left off.

### GPU OOM from Concurrent Sessions

**Symptom:** Training OOMs unexpectedly despite batch size fitting previously.

**Cause:** Another process (Jupyter, another training session) is using GPU memory.

**Fix:** Check with `nvidia-smi`. Use GPU lock (`src/utils/gpu_lock.py`) to serialize GPU tasks. Kill competing processes.

## Model & Checkpoint

### LoRA Checkpoint State Dict Mismatch

**Symptom:** `RuntimeError: Error(s) in loading state_dict` when resuming/evaluating a LoRA-trained model.

**Cause:** LoRA wraps parameters with adapter prefixes. If checkpoint was saved without `merge_and_unload()`, state dict keys don't match vanilla model.

**Fix:** Current code uses `merge_and_unload()` before saving. If loading an old checkpoint, apply LoRA first, then load.

### MLP Head Device Mismatch

**Symptom:** `RuntimeError: Expected all tensors to be on the same device` during forward pass with custom head wrappers.

**Cause:** Custom layers (MLP projection, vocab projection) initialized on CPU while base model is on CUDA.

**Fix:** Call `model = model.to(device)` after wrapping with custom head classes.

### PEFT Not Installed

**Symptom:** `ModuleNotFoundError: No module named 'peft'` when running a LoRA config.

**Fix:** `pip install peft`. Only needed for LoRA configs.

## Evaluation & Metrics

### SQL Exact Match Tokenizer Artifact

**Symptom:** SQL EM is 0% even though generated SQL looks correct.

**Cause:** Tokenizer artifacts (extra spaces around commas, parentheses) cause string mismatch.

**Fix:** Post-processing regex to normalize whitespace around punctuation. Check if new artifacts need additional normalization rules.

### BOS Token Mismatch

**Symptom:** Generated outputs start with unexpected tokens or are systematically wrong.

**Cause:** `decoder_start_token_id` doesn't match what the model expects. T5 uses `<pad>` (0) by default, but custom setups may use `<extra_id_0>` (32099).

**Fix:** Set `decoder_start_token_id` explicitly in generation kwargs to match training.

## Decoding & Generation

### Repetition Penalty / No-Repeat N-Gram Conflicts with SQL

**Symptom:** Generated SQL is missing repeated keywords (e.g., `SELECT`, `AND`, `FROM` appearing only once).

**Cause:** `repetition_penalty` or `no_repeat_ngram_size` penalizes tokens that legitimately repeat in SQL.

**Fix:** Do not use `repetition_penalty` or `no_repeat_ngram_size` for SQL generation tasks. These are designed for free-text generation and actively harm structured output.

### Max New Tokens Truncation

**Symptom:** Generated SQL queries are cut off mid-statement.

**Cause:** `max_new_tokens` set too low for complex queries.

**Fix:** Increase `max_new_tokens`. For NL-to-SQL, 256 is a safe default; complex joins may need 384+.
