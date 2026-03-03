"""Base ML configuration and shared utilities.

Provides:
    MLBaseConfig          -- root config (name, seed, device, output_dir)
    Checkpointing         -- composable: model saving
    TensorBoard           -- composable: metric logging
    save_config()         -- serialize config to JSON
    load_config()         -- deserialize JSON to config
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import torch

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Composable pieces -- attach to any config via fields
# ---------------------------------------------------------------------------


@dataclass
class Checkpointing:
    """Model checkpointing configuration."""

    enabled: bool = True
    save_best: bool = True
    save_last: bool = True
    save_frequency: int = 0  # epochs or steps; 0 = disabled
    metric: str = "loss"  # metric to track for best model
    mode: str = "min"  # "min" or "max"


@dataclass
class MetricGroups:
    """Which metric groups to log to TensorBoard."""

    episode: bool = True  # episode metrics (reward, length, count)
    train: bool = True  # training losses
    q_values: bool = True  # Q-value stats (DDPG/SAC only; no-op for PPO)
    gradients: bool = True  # gradient norms
    system: bool = True  # CPU, RAM, GPU usage
    timing: bool = True  # wall-clock time, episode duration
    system_interval: int = 10  # log system metrics every N logging events


@dataclass
class TensorBoard:
    """TensorBoard logging configuration."""

    enabled: bool = True
    log_dir: str = "tensorboard"
    flush_secs: int = 120
    log_interval: int = 100  # steps between log writes
    metrics: MetricGroups = field(default_factory=MetricGroups)


@dataclass
class Output:
    """Run output directory configuration."""

    base_dir: str = "output"
    save_config: bool = True


@dataclass
class Console:
    """Console logging configuration (tee stdout/stderr to file)."""

    enabled: bool = True
    filename: str = "console.log"
    tee_to_console: bool = True
    line_timestamps: bool = False
    timestamp_format: str = "%H:%M:%S"


# ---------------------------------------------------------------------------
# Base config -- all ML experiments inherit from this
# ---------------------------------------------------------------------------


@dataclass
class MLBaseConfig:
    """Root of the config hierarchy. Every ML config inherits from this."""

    name: str = "experiment"
    seed: int = 42
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0"
    output_dir: str = "output"


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(device: str = "auto") -> str:
    """Resolve a device string to an actual torch device name.

    Args:
        device: One of "auto", "cpu", "cuda", "cuda:0", etc.
            "auto" selects CUDA when available, otherwise CPU.

    Returns:
        Resolved device string (e.g. "cuda" or "cpu").
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def save_config(cfg, path: str | Path) -> None:
    """Save a dataclass config to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(cfg), indent=2, default=str))


def load_config(cls: Type[T], path: str | Path) -> T:
    """Load a JSON file into a dataclass config.

    Nested dataclass fields are reconstructed automatically.
    Unknown keys in the JSON are silently ignored.
    """
    data = json.loads(Path(path).read_text())
    return _from_dict(cls, data)


def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Recursively construct a dataclass from a dict."""
    import dataclasses

    if not dataclasses.is_dataclass(cls):
        return data
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, val in data.items():
        if key not in fields:
            continue
        ft = fields[key].type
        # Resolve Optional[X] -> X
        origin = getattr(ft, "__origin__", None)
        if origin is type(None):
            kwargs[key] = val
        elif dataclasses.is_dataclass(ft):
            kwargs[key] = _from_dict(ft, val) if isinstance(val, dict) else val
        else:
            kwargs[key] = val
    return cls(**kwargs)
