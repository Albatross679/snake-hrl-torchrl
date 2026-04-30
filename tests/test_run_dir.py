"""Tests for timestamped run directory system and console logger."""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.configs.base import Output, Console
from src.configs.run_dir import setup_run_dir
from src.configs.console import ConsoleLogger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class _DummyConfig:
    """Minimal config for testing setup_run_dir."""

    name: str = "test_run"
    output: Output = field(default_factory=Output)
    console: Console = field(default_factory=Console)


# ---------------------------------------------------------------------------
# setup_run_dir tests
# ---------------------------------------------------------------------------


class TestSetupRunDir:
    def test_creates_directory_structure(self, tmp_path):
        cfg = _DummyConfig(output=Output(base_dir=str(tmp_path)))
        run_dir = setup_run_dir(cfg, timestamp="20260101_120000")

        assert run_dir == tmp_path / "test_run_20260101_120000"
        assert run_dir.is_dir()
        assert (run_dir / "checkpoints").is_dir()

    def test_saves_config_json(self, tmp_path):
        cfg = _DummyConfig(output=Output(base_dir=str(tmp_path)))
        run_dir = setup_run_dir(cfg, timestamp="20260101_120000")

        config_path = run_dir / "config.json"
        assert config_path.exists()

        data = json.loads(config_path.read_text())
        assert data["name"] == "test_run"

    def test_no_config_json_when_disabled(self, tmp_path):
        cfg = _DummyConfig(output=Output(base_dir=str(tmp_path), save_config=False))
        run_dir = setup_run_dir(cfg, timestamp="20260101_120000")

        assert not (run_dir / "config.json").exists()

    def test_uses_base_dir_override(self, tmp_path):
        cfg = _DummyConfig()
        custom = tmp_path / "custom"
        run_dir = setup_run_dir(cfg, timestamp="20260101_120000", base_dir=str(custom))

        assert run_dir == custom / "test_run_20260101_120000"
        assert run_dir.is_dir()

    def test_auto_timestamp_creates_unique_dir(self, tmp_path):
        cfg = _DummyConfig(output=Output(base_dir=str(tmp_path)))
        run_dir = setup_run_dir(cfg)

        assert run_dir.is_dir()
        assert run_dir.name.startswith("test_run_")

    def test_works_without_output_field(self, tmp_path):
        """Config without an output field uses 'output' default."""

        @dataclass
        class _BareConfig:
            name: str = "bare"

        cfg = _BareConfig()
        run_dir = setup_run_dir(cfg, timestamp="20260101_120000", base_dir=str(tmp_path))
        assert run_dir.is_dir()
        assert (run_dir / "config.json").exists()


# ---------------------------------------------------------------------------
# ConsoleLogger tests
# ---------------------------------------------------------------------------


class TestConsoleLogger:
    def test_captures_stdout(self, tmp_path):
        console_cfg = Console(tee_to_console=False)
        with ConsoleLogger(tmp_path, console_cfg):
            print("hello world")

        log = (tmp_path / "console.log").read_text()
        assert "hello world" in log

    def test_captures_stderr(self, tmp_path):
        console_cfg = Console(tee_to_console=False)
        with ConsoleLogger(tmp_path, console_cfg):
            print("error message", file=sys.stderr)

        log = (tmp_path / "console.log").read_text()
        assert "error message" in log

    def test_restores_streams(self, tmp_path):
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        console_cfg = Console(tee_to_console=False)
        with ConsoleLogger(tmp_path, console_cfg):
            assert sys.stdout is not original_stdout

        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr

    def test_disabled_does_not_capture(self, tmp_path):
        console_cfg = Console(enabled=False)
        original_stdout = sys.stdout

        with ConsoleLogger(tmp_path, console_cfg):
            assert sys.stdout is original_stdout

        assert not (tmp_path / "console.log").exists()

    def test_custom_filename(self, tmp_path):
        console_cfg = Console(filename="train.log", tee_to_console=False)
        with ConsoleLogger(tmp_path, console_cfg):
            print("test")

        assert (tmp_path / "train.log").exists()
        assert "test" in (tmp_path / "train.log").read_text()

    def test_none_config_uses_defaults(self, tmp_path):
        with ConsoleLogger(tmp_path, None):
            print("default config")

        assert (tmp_path / "console.log").exists()
