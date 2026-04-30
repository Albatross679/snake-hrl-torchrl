"""Tests for arch_sweep.py — ARCH-05 smoke test.

RED phase: Tests written before arch_sweep.py exists.
These tests verify the contract of arch_sweep.py:
  - ARCH_SWEEP_CONFIGS has exactly 5 entries (A1, A3, A4, A5, B1)
  - --dry-run exits 0 and prints config table
  - Required keys present in each config dict
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestArchSweepConfigs:
    """ARCH-05: Verify ARCH_SWEEP_CONFIGS structure."""

    def test_arch_sweep_configs_count(self):
        """ARCH_SWEEP_CONFIGS must have exactly 5 entries (A1, A3, A4, A5, B1)."""
        from aprx_model_elastica.arch_sweep import ARCH_SWEEP_CONFIGS
        assert len(ARCH_SWEEP_CONFIGS) == 5

    def test_arch_sweep_configs_required_keys(self):
        """Each config must have: name, lr, hidden_dims, rollout_weight, rollout_steps."""
        from aprx_model_elastica.arch_sweep import ARCH_SWEEP_CONFIGS
        required_keys = {"name", "lr", "hidden_dims", "rollout_weight", "rollout_steps"}
        for cfg in ARCH_SWEEP_CONFIGS:
            missing = required_keys - set(cfg.keys())
            assert not missing, f"Config {cfg.get('name')} missing keys: {missing}"

    def test_arch_sweep_configs_names(self):
        """Config names must match expected experiment IDs."""
        from aprx_model_elastica.arch_sweep import ARCH_SWEEP_CONFIGS
        names = [cfg["name"] for cfg in ARCH_SWEEP_CONFIGS]
        assert "arch_A1_rw0.0" in names
        assert "arch_A3_rw0.3" in names
        assert "arch_A4_rw0.5" in names
        assert "arch_A5_rw0.3_s16" in names
        assert "arch_B1_residual" in names
        # A2 should NOT be in ARCH_SWEEP_CONFIGS — it's injected as baseline
        for n in names:
            assert "A2" not in n, "A2 should be injected as baseline, not in ARCH_SWEEP_CONFIGS"

    def test_arch_sweep_configs_b1_has_residual(self):
        """B1 config must have use_residual=True."""
        from aprx_model_elastica.arch_sweep import ARCH_SWEEP_CONFIGS
        b1 = next(c for c in ARCH_SWEEP_CONFIGS if c["name"] == "arch_B1_residual")
        assert b1.get("use_residual") is True

    def test_arch_sweep_configs_a1_rollout_weight_zero(self):
        """A1 must have rollout_weight=0.0 (ablation of rollout loss)."""
        from aprx_model_elastica.arch_sweep import ARCH_SWEEP_CONFIGS
        a1 = next(c for c in ARCH_SWEEP_CONFIGS if c["name"] == "arch_A1_rw0.0")
        assert a1["rollout_weight"] == 0.0

    def test_arch_sweep_configs_a5_rollout_steps_16(self):
        """A5 must have rollout_steps=16 (longer horizon)."""
        from aprx_model_elastica.arch_sweep import ARCH_SWEEP_CONFIGS
        a5 = next(c for c in ARCH_SWEEP_CONFIGS if c["name"] == "arch_A5_rw0.3_s16")
        assert a5["rollout_steps"] == 16


class TestArchSweepDryRun:
    """ARCH-05: Smoke test via --dry-run flag."""

    def test_dry_run_exits_zero(self):
        """--dry-run must exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "aprx_model_elastica.arch_sweep", "--dry-run", "--epochs", "1"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"--dry-run failed with rc={result.returncode}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    def test_dry_run_prints_config_table(self):
        """--dry-run must print all config names in the output."""
        result = subprocess.run(
            [sys.executable, "-m", "aprx_model_elastica.arch_sweep", "--dry-run", "--epochs", "1"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        combined = result.stdout + result.stderr
        assert "arch_A1_rw0.0" in combined
        assert "arch_A3_rw0.3" in combined
        assert "arch_A4_rw0.5" in combined
        assert "arch_A5_rw0.3_s16" in combined
        assert "arch_B1_residual" in combined
        # Should also show baseline (A2/BASELINE)
        assert "BASELINE" in combined or "A2" in combined

    def test_dry_run_no_subprocess_launch(self):
        """--dry-run must NOT start any training subprocesses (fast completion)."""
        import time
        start = time.monotonic()
        result = subprocess.run(
            [sys.executable, "-m", "aprx_model_elastica.arch_sweep", "--dry-run", "--epochs", "1"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=30,  # Should complete in <5s; generous 30s timeout
        )
        elapsed = time.monotonic() - start
        assert result.returncode == 0
        # Dry run should be fast (no training), well under 30 seconds
        assert elapsed < 30, f"--dry-run took {elapsed:.1f}s — possible subprocess launch?"

    def test_dry_run_prints_arch_sweep_header(self):
        """--dry-run must print the ARCH SWEEP header."""
        result = subprocess.run(
            [sys.executable, "-m", "aprx_model_elastica.arch_sweep", "--dry-run", "--epochs", "1"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        combined = result.stdout + result.stderr
        assert "ARCH SWEEP" in combined or "arch sweep" in combined.lower()

    def test_help_shows_expected_args(self):
        """--help must show --data-dir, --device, --epochs, --output-base, --dry-run."""
        result = subprocess.run(
            [sys.executable, "-m", "aprx_model_elastica.arch_sweep", "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        combined = result.stdout + result.stderr
        assert "--data-dir" in combined
        assert "--device" in combined
        assert "--epochs" in combined
        assert "--output-base" in combined
        assert "--dry-run" in combined
