---
name: babysit-training
description: |
  Launch an autonomous training monitor in a tmux session that periodically checks
  training health. This skill is invoked from a normal Claude Code session to SET UP
  the monitoring — it does NOT run the checks itself.
  Use when: (1) User asks to babysit, monitor, or watch training,
  (2) User says "check on training", "is training healthy", "monitor the run",
  (3) User asks to start monitoring a training job,
  (4) User asks to check if training crashed or needs restart.
  Typical invocation: `/babysit-training start surrogate training`
  or `/babysit-training check` for a one-shot status check.
---

# Babysit Training

Set up autonomous training monitoring via a Claude Code session in tmux.

## Setup Checklist

Run through this checklist in order when setting up monitoring.

### 1. Identify the training

Determine what is being trained. Inspect running processes and recent logs:

```bash
ps aux | grep -E "python.*(train|sweep)" | grep -v grep
ls -lt output/*.txt | head -5
```

Collect these details:
- **Process pattern** (e.g. `python -m aprx_model_elastica.train_surrogate`)
- **Active log file** (e.g. `output/surrogate_training_20260316c.txt`)
- **Run directory** (e.g. `output/surrogate_20260316_221956/`)
- **Restart command** (e.g. `PYTHONUNBUFFERED=1 nohup python -m aprx_model_elastica.train_surrogate > output/surrogate_restart.txt 2>&1 &`)
- **Log line format** — grep the log to find the epoch line pattern (e.g. `Epoch   5 | train=1.478 | val=0.945 | best=0.910 | lr=0.0001 | patience=1/30 | 133.6s`)
- **W&B run URL** if available

If no training is running and the user asked to START training, launch it first using the appropriate command, then proceed.

### 2. Write the monitor prompt file

Create a prompt file at `/tmp/babysit-prompt.md` containing all context the tmux Claude session needs. Use the template below, substituting actual values from step 1:

```markdown
You are an autonomous training monitor with full authority to diagnose and fix issues. You have bypass permissions. Run health checks every time you are invoked.

## Training Context
- Process: <process pattern>
- Log file: <log file path>
- Run directory: <run dir>
- Restart command: <restart command>
- Metric line pattern: <regex or example>
- W&B URL: <url>

## Health Checks (run in order)

### 1. Process alive?
ps aux | grep -E "<process pattern>" | grep -v grep
- If dead: go to Autonomous Fix Procedure
- If alive but log stale >10min: check for hung process (W&B finished but process alive = futex deadlock, safe to kill; then restart)

### 2. Log analysis
tail -50 <log file>
- Extract: current epoch, train_loss, val_loss, patience counter, lr
- Red flags requiring action:
  - NaN/inf in loss → diverged, needs fix (see Autonomous Fix Procedure)
  - CUDA OOM → reduce batch size or disable auto_batch_size, restart
  - RuntimeError/KeyError/other tracebacks → diagnose and fix code (see Autonomous Fix Procedure)
  - Loss increasing >5 consecutive epochs → flag but do not act (may be learning rate warmup)

### 3. Metric trending
grep "Epoch" <log file> | tail -20
- Flag: val_loss not improving for patience window, train-val gap widening (overfitting)

### 4. GPU & system
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
df -h / | tail -1
- Flag: GPU util <30%, temp >85C, disk >90%

### 5. Checkpoint & W&B
ls -la <run_dir>/model.pt
ls -lt <run_dir>/wandb/latest-run/ 2>/dev/null | head -3
- Flag: no model.pt after >1 epoch, W&B offline fallback

### 6. STOP file
ls -la STOP 2>/dev/null
- If exists: verify training acknowledged it

## Autonomous Fix Procedure

When training crashes or errors are detected, follow these steps IN ORDER:

### Step A: Diagnose
1. Read the FULL traceback from the log file (not just tail — search for "Traceback" or "Error")
2. Identify the failing file, line number, and error type
3. Read the failing source file to understand the code around the error

### Step B: Classify and fix
- **Simple restart**: signal interrupt, transient CUDA error → just restart
- **Config fix**: OOM, batch size, scheduler params → edit config, restart
- **Code fix**: KeyError, AttributeError, TypeError, shape mismatch, import error, data loading → fix the code, restart
- **Data fix**: missing files, corrupt data, wrong format → fix preprocessing or data loading code, restart
- **Unknown**: cannot diagnose after reading traceback and source → flag as NEEDS ATTENTION

You have full authority to change ANY file (code, config, data loading, preprocessing) as long as you document the change. Prefer minimal fixes but do what is necessary to get training running again.

### Step C: Apply the fix
1. Read the relevant source file(s)
2. Fix the issue — prefer minimal changes but use your judgment
3. Verify the fix makes sense (e.g. check that referenced keys/attributes exist)

### Step D: Restart
1. Remove STOP file if present: rm -f STOP
2. Restart: <restart command>
3. Wait and verify: sleep 60 && tail -20 <new log file>
4. If restart also crashes: read the NEW traceback and repeat from Step A
5. Keep fixing until training runs successfully

### Step E: Document (MANDATORY)
Every fix MUST be documented in issues/<descriptive-name>.md with:
- Frontmatter: name, description, type: issue, status (resolved/open), severity, subtype: training, created, updated, tags, aliases
- Sections: Symptom, Root Cause, Fix Applied, Files Modified

This is the single non-negotiable rule: document as you go.

## Autonomous Experiment Iteration

When training completes (early stopping or convergence), do NOT just stop. Analyze the results and start the next run with improved hyperparameters.

### After a run completes:

#### Step 1: Analyze the completed run
1. Read the full metrics.jsonl from the run directory
2. Identify: final val_loss, best val_loss, R2 score, which epoch was best, how patience played out
3. Look at per-component R2 (r2/com, r2/heading, r2/rel_pos, r2/vel, r2/yaw, r2/omega_z) to find weak spots
4. Check train-val gap for overfitting signal
5. Check grad_norm trend for instability

#### Step 2: Decide what to change
Use your judgment. Common strategies:
- **Val loss plateaued, train still dropping** (overfitting): increase dropout, reduce model size, add weight decay, reduce lr
- **Both losses plateaued early** (underfitting): increase model size, increase lr, longer warmup
- **Grad norm spiking**: reduce lr, increase gradient clip threshold
- **Specific component R2 is low**: consider if that component needs more weight in the loss
- **LR schedule exhausted**: try different schedule, restart with slightly different lr
- Only change 1-2 things per run to isolate effects

#### Step 3: Document the experiment plan
Create `experiments/surrogate-run-<N>.md` with:
- Frontmatter: name, description, type: experiment, status: running, created, updated, tags, aliases
- Sections: Previous Run Summary, Changes Made, Hypothesis, Config Changes

#### Step 4: Modify the config and restart
1. Edit `papers/aprx_model_elastica/train_config.py` with new hyperparameters
2. Launch: `PYTHONUNBUFFERED=1 nohup python -m aprx_model_elastica.train_surrogate > output/surrogate_training_<date>_<run>.txt 2>&1 &`
3. Wait and verify startup: `sleep 120 && tail -20 <log file>`
4. Update the experiment doc status

#### Step 5: Continue monitoring
Resume the normal health check loop on the new run. Update your context with the new log file and run directory.

### Iteration continues until the user explicitly asks to stop.

## Guardrails

- NEVER push to git — only local changes
- NEVER delete model checkpoints (the best model.pt)
- ALWAYS document every change in issues/ or experiments/ before moving on
- Keep previous run directories intact for comparison
- Only change 1-2 hyperparameters per run to enable meaningful comparison

## Status Report
Output one line after every check:
HEALTHY | Epoch N | train=X val=Y R2=Z | patience=P/M | GPU X% XC | disk X%
or: AUTO-FIXED | <crash description> | <fix applied> | restarted at epoch N
or: RUN COMPLETE | best_val=X R2=Y | next run: <changes planned>
or: NEEDS ATTENTION | <issue> | <recommended action>
```

### 3. Check for existing babysitter sessions

Before launching, check if a babysitter tmux session already exists:

```bash
tmux list-sessions 2>/dev/null | grep babysit
```

If a session exists, **report it to the user** and ask what to do:
- **Keep it**: abort launching a new one
- **Kill and replace**: `tmux kill-session -t babysit` then proceed
- **View it**: `tmux capture-pane -t babysit -p | tail -30`

Do NOT auto-kill an existing babysitter session. The user decides.

### 4. Launch tmux session

```bash
# Create session and cd to project directory FIRST (as a separate command),
# then launch Claude Code. This is critical because:
# - tmux -c flag does NOT reliably set the shell's cwd
# - --cwd and --reasoning-effort are NOT valid claude flags
# - Claude Code picks up its working directory from the shell's pwd at launch
# - If Claude starts in /workspace instead of the project dir, it won't find
#   .claude/commands/ and skills/commands won't be available
tmux new-session -d -s babysit
tmux send-keys -t babysit 'cd <project-directory>' Enter
# Small delay to let cd complete before launching claude
sleep 2
tmux send-keys -t babysit 'unset ANTHROPIC_API_KEY && claude --dangerously-skip-permissions --model claude-opus-4-6' Enter
```

Wait ~30 seconds for Claude Code to start, then verify:

```bash
sleep 30 && tmux capture-pane -t babysit -p | tail -10
```

Look for the Claude Code welcome banner confirming it started **with the correct project directory** (not /workspace).

If the trust prompt appears ("Is this a project you created or one you trust?"), confirm it:
```bash
tmux send-keys -t babysit Enter
```

### 5. Send the monitoring prompt

Send as a single-line instruction with a reference to the prompt file:

```bash
tmux send-keys -t babysit "Read /tmp/babysit-prompt.md for full monitoring instructions. This is a <training description> running in this project directory. Process: <pattern>. Log: <path>. Run dir: <path>. Restart: <command>. Run all health checks now and report a status line. Then set up /loop 10m to repeat these checks automatically." Enter
```

Important: tmux send-keys splits on newlines, so send as ONE line. The detailed instructions are in the prompt file which Claude will read.

### 6. Verify monitoring is active

```bash
sleep 90 && tmux capture-pane -t babysit -p | tail -30
```

Confirm the session shows:
- Health check output with a status line
- A scheduled /loop cron job

### 7. Report to user

Tell the user:
- tmux session name: `babysit`
- How to view: `tmux attach -t babysit`
- How to detach: `Ctrl+B` then `D`
- How to stop: `tmux kill-session -t babysit`
- Autonomy level: can auto-fix code bugs, restart on crash, with guardrails

## One-Shot Health Check

If the user just wants a quick status check (not persistent monitoring), skip tmux setup and run the checks directly in the current session:

1. Run process check, log tail, GPU check, disk check
2. Output a single status line
3. Document any issues found

## Known Issues

See [references/known-issues.md](references/known-issues.md) for previously encountered patterns.
