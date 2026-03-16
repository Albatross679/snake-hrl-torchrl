"""External monitor: reads event log and shows per-worker status.

Usage:
    python -m aprx_model_dismech monitor --save-dir data/surrogate_dismech_rl_step

Tails the events.jsonl file written by the collector and displays
a live per-worker status table updated every poll interval.
"""

import argparse
import json
import time
from pathlib import Path


def _print_status_table(worker_status: dict, elapsed: float) -> None:
    """Print per-worker status table to terminal."""
    print("\033[2J\033[H", end="")  # clear screen + cursor home
    print(f"Collection Monitor (DisMech) -- {elapsed:.0f}s elapsed")
    print()
    print(f"{'Worker':>6} | {'Status':>8} | {'Transitions':>12} | {'Delta/poll':>10} | {'Respawns':>8} | {'NaN':>4}")
    print("-" * 70)

    total_transitions = 0
    total_nan = 0
    for wid in sorted(worker_status.keys()):
        ws = worker_status[wid]
        total_transitions += ws.get("transitions", 0)
        total_nan += ws.get("nan_discards", 0)
        print(
            f"{wid:>6} | {ws.get('status', '?'):>8} | "
            f"{ws.get('transitions', 0):>12,} | "
            f"{ws.get('delta', 0):>10,} | "
            f"{ws.get('respawn_count', 0):>8} | "
            f"{ws.get('nan_discards', 0):>4}"
        )

    print("-" * 70)
    print(f"Total: {total_transitions:,} transitions, {total_nan} NaN discards")


def tail_events(save_dir: str, poll_interval: float = 30.0) -> None:
    """Tail events.jsonl and display live per-worker status.

    Args:
        save_dir: Directory containing events.jsonl.
        poll_interval: Seconds between display refreshes.
    """
    event_log = Path(save_dir) / "events.jsonl"
    worker_status: dict[int, dict] = {}
    file_pos = 0
    t_start = time.monotonic()

    print(f"Monitoring {event_log}")
    print(f"Poll interval: {poll_interval}s")

    while True:
        if not event_log.exists():
            print(f"Waiting for {event_log}...")
            time.sleep(poll_interval)
            continue

        # Read new lines
        try:
            with open(event_log, "r") as f:
                f.seek(file_pos)
                new_lines = f.readlines()
                file_pos = f.tell()
        except OSError:
            time.sleep(poll_interval)
            continue

        for line in new_lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # skip corrupt lines

            event_type = event.get("event_type", "")
            wid = event.get("worker_id")
            details = event.get("details", {})

            if wid is not None and wid not in worker_status:
                worker_status[wid] = {
                    "status": "alive", "transitions": 0, "delta": 0,
                    "respawn_count": 0, "nan_discards": 0,
                }

            if event_type == "worker_status" and wid is not None:
                worker_status[wid]["status"] = details.get("status", "alive")
                worker_status[wid]["transitions"] = details.get("transitions", 0)
                worker_status[wid]["delta"] = details.get("delta", 0)
            elif event_type == "worker_died" and wid is not None:
                worker_status[wid]["status"] = "dead"
            elif event_type == "worker_respawned" and wid is not None:
                worker_status[wid]["status"] = "alive"
                worker_status[wid]["respawn_count"] = details.get("respawn_count", 0)
            elif event_type == "worker_stalled" and wid is not None:
                worker_status[wid]["status"] = "stalled"
            elif event_type == "nan_discard" and wid is not None:
                worker_status[wid]["nan_discards"] = worker_status[wid].get("nan_discards", 0) + 1
            elif event_type == "shutdown_requested":
                print("\n*** Shutdown requested ***")
            elif event_type == "shutdown_complete":
                total = details.get("total_transitions", 0)
                print(f"\n*** Collection complete: {total:,} transitions ***")
                return

        if worker_status:
            elapsed = time.monotonic() - t_start
            _print_status_table(worker_status, elapsed)

        time.sleep(poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor running DisMech data collection")
    parser.add_argument("--save-dir", type=str, required=True, help="Data directory to monitor")
    parser.add_argument("--poll-interval", type=float, default=30.0, help="Seconds between refreshes")
    args = parser.parse_args()
    tail_events(args.save_dir, args.poll_interval)


if __name__ == "__main__":
    main()
