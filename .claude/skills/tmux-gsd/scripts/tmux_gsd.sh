#!/usr/bin/env bash
# tmux-gsd session manager
# Usage:
#   tmux_gsd.sh create <session_name> <command> [monitor_cmd]
#   tmux_gsd.sh list
#   tmux_gsd.sh status <session_name>
#   tmux_gsd.sh teardown <session_name>

set -euo pipefail

TRACKING_DIR=".planning/tmux-sessions"

cmd_create() {
    local session_name="$1"
    local main_cmd="$2"
    local monitor_cmd="${3:-claude --print \"Run /loop 5m /gsd:progress\"}"

    # Check tmux exists
    if ! command -v tmux &>/dev/null; then
        echo "ERROR: tmux is not installed" >&2
        exit 1
    fi

    # Check for conflict
    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "CONFLICT: session '$session_name' already exists"
        exit 2
    fi

    # Create detached session with explicit size (needed for split-window in headless envs)
    tmux new-session -d -s "$session_name" -x 200 -y 50 2>/dev/null \
        || tmux new-session -d -s "$session_name"

    # Send main command to pane 0
    tmux send-keys -t "${session_name}:0.0" "$main_cmd" Enter

    # Split for monitor pane (30% bottom)
    # -l sets absolute lines instead of -p percentage (works better headless)
    tmux split-window -v -t "${session_name}:0" -l 15 2>/dev/null \
        || tmux split-window -v -t "${session_name}:0"
    tmux send-keys -t "${session_name}:0.1" "$monitor_cmd" Enter

    # Select main pane
    tmux select-pane -t "${session_name}:0.0"

    # Write tracking file
    mkdir -p "$TRACKING_DIR"
    cat > "${TRACKING_DIR}/${session_name}.json" <<EOF
{
  "session_name": "${session_name}",
  "command": $(printf '%s' "$main_cmd" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))'),
  "monitor_command": $(printf '%s' "$monitor_cmd" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))'),
  "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "running"
}
EOF

    echo "OK: session '$session_name' created"
    echo "ATTACH: tmux attach -t $session_name"
}

cmd_list() {
    if ! command -v tmux &>/dev/null; then
        echo "ERROR: tmux is not installed" >&2
        exit 1
    fi

    echo "ACTIVE_SESSIONS:"
    tmux list-sessions -F '#{session_name}|#{session_created}|#{session_attached}' 2>/dev/null | grep '^gsd-' || echo "(none)"

    echo ""
    echo "TRACKING_FILES:"
    if [ -d "$TRACKING_DIR" ]; then
        for f in "$TRACKING_DIR"/*.json; do
            [ -f "$f" ] || continue
            local name
            name=$(basename "$f" .json)
            local status
            status=$(python3 -c "import json; print(json.load(open('$f'))['status'])" 2>/dev/null || echo "unknown")
            local created
            created=$(python3 -c "import json; print(json.load(open('$f'))['created_at'])" 2>/dev/null || echo "unknown")

            # Check if tmux session still alive
            local alive="dead"
            if tmux has-session -t "$name" 2>/dev/null; then
                alive="alive"
            fi
            echo "${name}|${status}|${created}|${alive}"
        done
    else
        echo "(no tracking directory)"
    fi
}

cmd_status() {
    local session_name="$1"

    if tmux has-session -t "$session_name" 2>/dev/null; then
        echo "ALIVE"
        tmux list-panes -t "$session_name" -F '#{pane_index}|#{pane_pid}|#{pane_current_command}|#{pane_width}x#{pane_height}'
    else
        echo "DEAD"
    fi

    if [ -f "${TRACKING_DIR}/${session_name}.json" ]; then
        echo "TRACKING:"
        cat "${TRACKING_DIR}/${session_name}.json"
    fi
}

cmd_teardown() {
    local session_name="$1"

    if tmux has-session -t "$session_name" 2>/dev/null; then
        tmux kill-session -t "$session_name"
        echo "KILLED: session '$session_name'"
    else
        echo "NOT_FOUND: session '$session_name' does not exist"
    fi

    # Update tracking
    if [ -f "${TRACKING_DIR}/${session_name}.json" ]; then
        python3 -c "
import json
with open('${TRACKING_DIR}/${session_name}.json', 'r') as f:
    data = json.load(f)
data['status'] = 'stopped'
data['stopped_at'] = '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
with open('${TRACKING_DIR}/${session_name}.json', 'w') as f:
    json.dump(data, f, indent=2)
print('TRACKING_UPDATED')
"
    fi
}

# --- Dispatch ---
case "${1:-}" in
    create)
        shift
        cmd_create "$@"
        ;;
    list)
        cmd_list
        ;;
    status)
        shift
        cmd_status "$@"
        ;;
    teardown)
        shift
        cmd_teardown "$@"
        ;;
    *)
        echo "Usage: $0 {create|list|status|teardown} [args...]" >&2
        exit 1
        ;;
esac
