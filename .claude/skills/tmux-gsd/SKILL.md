---
name: tmux-gsd
description: "Manage persistent Tmux sessions for long-running GSD phase executions, data collection, and training runs. Use when: (1) the user wants to run a GSD phase or long command in a persistent session that survives disconnects, (2) the user mentions tmux in the context of GSD, monitoring, or long-running tasks, (3) the user asks to check on, attach to, or tear down a running GSD session, (4) the user says 'keep this running' or 'run overnight' or 'monitor this'. Always use this skill for any tmux+GSD integration — even if the user doesn't explicitly say 'tmux'."
---

# Tmux-GSD: Persistent Sessions for Long-Running GSD Work

Manages Tmux sessions that pair a long-running process (phase execution, data collection, training) with a Claude Code monitoring pane — so work persists across terminal disconnects and gets tracked in GSD state.

## Why this exists

GSD phase executions and data collection runs can take hours or run overnight. Without Tmux, closing a terminal kills the process. This skill creates named, structured Tmux sessions with two panes: one for the work, one for monitoring. When the process completes or fails, STATE.md gets updated so `/gsd:progress` reflects reality.

## Commands

This skill supports four operations. Determine which one the user needs from context:

| Intent | Operation |
|--------|-----------|
| Start a long-running task | **create** |
| See what's running | **list** |
| Reconnect to a session | **attach** |
| Stop and clean up | **teardown** |

---

## Operation: CREATE

### Step 1 — Determine the command and session name

Ask the user what command to run if not already clear. Derive a session name from context:

- GSD phase execution: `gsd-phase-{N}` (e.g., `gsd-phase-1`)
- Data collection: `gsd-collect`
- Training: `gsd-train`
- Custom: `gsd-{slug}` where slug is a short descriptor

### Step 2 — Check for conflicts

```bash
tmux has-session -t {session_name} 2>/dev/null && echo "EXISTS" || echo "FREE"
```

If EXISTS: ask the user whether to attach to the existing session, tear it down and recreate, or pick a different name.

### Step 3 — Create the session with two panes

```bash
# Create detached session with the main command
tmux new-session -d -s {session_name} -x 200 -y 50

# Pane 0 (top): the long-running command
tmux send-keys -t {session_name}:0.0 '{command}' Enter

# Split horizontally — pane 1 (bottom): monitoring
tmux split-window -v -t {session_name}:0 -p 30

# Pane 1: Claude Code with monitoring loop
tmux send-keys -t {session_name}:0.1 'claude --print "Run /loop 5m /gsd:progress"' Enter
```

Layout:
```
┌──────────────────────────────┐
│                              │
│   Pane 0: Main process       │
│   (70% height)               │
│                              │
├──────────────────────────────┤
│   Pane 1: Claude monitor     │
│   (30% height)               │
└──────────────────────────────┘
```

### Step 4 — Write a session tracking file

Create `.planning/tmux-sessions/{session_name}.json`:

```json
{
  "session_name": "{session_name}",
  "command": "{command}",
  "phase": "{phase_number or null}",
  "created_at": "{ISO timestamp}",
  "status": "running",
  "panes": {
    "0": "main process",
    "1": "claude monitor"
  }
}
```

```bash
mkdir -p .planning/tmux-sessions
```

### Step 5 — Update STATE.md

Append to the "Accumulated Context" > "Decisions" section:

```
- [Tmux]: Session `{session_name}` started — {short description of command}
```

### Step 6 — Report to user

Tell the user:
- Session `{session_name}` is running
- To attach: `tmux attach -t {session_name}`
- To detach (without killing): `Ctrl+B` then `D`
- Monitoring pane is polling `/gsd:progress` every 5 minutes

---

## Operation: LIST

Show all active GSD Tmux sessions:

```bash
tmux list-sessions -F '#{session_name} #{session_created} #{session_windows}' 2>/dev/null | grep '^gsd-'
```

Cross-reference with `.planning/tmux-sessions/*.json` for richer context (what command, which phase, when started). Present a table:

| Session | Phase | Command | Started | Status |
|---------|-------|---------|---------|--------|

If a session file exists but the Tmux session is gone, mark it as `dead` and update the JSON status.

---

## Operation: ATTACH

```bash
tmux attach -t {session_name}
```

If running inside an existing Tmux session, use `switch-client` instead:

```bash
tmux switch-client -t {session_name}
```

Note: Claude Code cannot truly "attach" a terminal — instead, tell the user the exact command to run.

---

## Operation: TEARDOWN

### Step 1 — Confirm with the user

Tearing down kills the running process. Always confirm before proceeding unless the user explicitly said to tear it down.

### Step 2 — Kill the session

```bash
tmux kill-session -t {session_name}
```

### Step 3 — Update tracking

Update `.planning/tmux-sessions/{session_name}.json`:
- Set `status` to `"stopped"`
- Add `stopped_at` timestamp

### Step 4 — Update STATE.md

If the session was for a GSD phase, check whether it completed successfully:

1. Read the process output or check for completion markers (e.g., saved model files, completed data files)
2. Update STATE.md session continuity:
   - Success: `Stopped at: Phase {N} completed via tmux session {session_name}`
   - Failure/manual stop: `Stopped at: Phase {N} interrupted — tmux session {session_name} torn down`

---

## Monitoring Pane Behavior

The monitoring pane (pane 1) runs Claude Code with a `/loop` that periodically checks project status. The default is:

```
/loop 5m /gsd:progress
```

For data collection specifically, a more useful monitor command might be:

```bash
# Tail the events log instead
tail -f {save_dir}/events.jsonl | python -m json.tool
```

Adjust the monitoring command based on what the user is running. If they're running data collection and `events.jsonl` exists, suggest tailing that instead of generic progress polling.

---

## Edge Cases

- **No tmux installed**: Error with `tmux is required but not found. Install with: apt install tmux (or brew install tmux on macOS)`
- **Session name collision**: Ask user to pick a new name or teardown existing
- **Nested tmux**: Detect with `$TMUX` env var. If inside tmux, create the session but don't attach — just report the `tmux switch-client` command
- **Process already finished**: If the user asks to create a session for a command that will finish quickly, warn them. Tmux sessions are for long-running tasks (>5 minutes).
- **Stale tracking files**: On LIST, reconcile `.planning/tmux-sessions/` with actual `tmux list-sessions` output. Remove or mark dead sessions.
