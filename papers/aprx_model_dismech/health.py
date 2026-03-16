"""Health monitoring utilities for data collection pipeline.

Provides event logging (JSONL) and NaN/Inf validation for episodes.
"""

import datetime
import json
from pathlib import Path

import numpy as np


def log_event(
    event_log_path: Path,
    event_type: str,
    severity: str,
    worker_id: int | None = None,
    details: dict | None = None,
) -> None:
    """Append a single JSON event line to the event log.

    Args:
        event_log_path: Path to events.jsonl file.
        event_type: Event type string (e.g. "worker_status", "nan_discard").
        severity: One of "info", "warn", "error".
        worker_id: Worker ID or None for global events.
        details: Additional event-specific data.
    """
    event = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event_type": event_type,
        "severity": severity,
        "worker_id": worker_id,
        "details": details or {},
    }
    with open(event_log_path, "a") as f:
        f.write(json.dumps(event) + "\n")


def validate_episode_finite(ep_data: dict) -> bool:
    """Check if all episode arrays are finite (no NaN or Inf).

    Args:
        ep_data: Episode data dict with "states", "next_states", "actions" arrays.

    Returns:
        True if all arrays are finite, False otherwise.
    """
    for key in ("states", "next_states", "actions"):
        if key in ep_data and not np.all(np.isfinite(ep_data[key])):
            return False
    return True
