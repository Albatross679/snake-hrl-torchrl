"""Timestamped run directory setup.

Creates a consolidated output directory per run:

    output/<name>_<YYYYMMDD_HHMMSS>/
    ├── config.json
    ├── console.log
    └── checkpoints/
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_run_dir(
    config,
    *,
    timestamp: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> Path:
    """Create a timestamped run directory and save a config snapshot.

    Args:
        config: Any dataclass config with a ``name`` attribute.
        timestamp: Override timestamp string (``YYYYMMDD_HHMMSS``).
            If *None*, the current time is used.
        base_dir: Override base output directory.  Defaults to
            ``config.output.base_dir`` if the config has an ``output``
            field, otherwise ``"output"``.

    Returns:
        Path to the created run directory.
    """
    # Resolve base directory
    if base_dir is None:
        if hasattr(config, "output") and hasattr(config.output, "base_dir"):
            base_dir = config.output.base_dir
        else:
            base_dir = "output"

    # Resolve name
    name = getattr(config, "name", "run")

    # Resolve timestamp
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path(base_dir) / f"{name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # Save config snapshot
    save_config = True
    if hasattr(config, "output") and hasattr(config.output, "save_config"):
        save_config = config.output.save_config

    if save_config:
        config_path = run_dir / "config.json"
        config_path.write_text(
            json.dumps(asdict(config), indent=2, default=str)
        )

    return run_dir
