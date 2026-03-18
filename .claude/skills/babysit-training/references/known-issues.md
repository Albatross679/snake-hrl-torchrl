# Known Training Issues — Quick Reference

Action-oriented summaries for rapid diagnosis during monitoring.

| Pattern | Symptom | Immediate Action |
|---------|---------|-----------------|
| **Hung process** | Process alive, no log output, W&B shows FINISHED | Outputs saved. `kill <PID>`. Safe — futex deadlock during cleanup. |
| **OOM** | `CUDA out of memory` in log | Kill process, reduce batch size or set `auto_batch_size=False`, restart. |
| **Buffered stdout** | Log file empty for long periods | Must restart with `PYTHONUNBUFFERED=1`. |
| **GPU lock contention** | Command hangs at startup, no output | Check `flock` on `/tmp/gpu-task.lock`. Kill holder if dead. |
| **Dataset key mismatch** | `KeyError: 'serpenoid_times'` or `'step_indices'` | Data files use `t_start`/`step_ids` instead. Fix in `dataset.py`. |
| **W&B metric invisible** | Dashboard panels show "no data" | Check metric name prefix matches dashboard panel names. |
| **Watchdog kill** | Exit code 137/143 after training completes | Normal — watchdog killed hung cleanup. Outputs are saved. |
| **API key vs subscription** | `Credit balance is too low` in tmux Claude | `unset ANTHROPIC_API_KEY` before launching `claude` CLI. |
