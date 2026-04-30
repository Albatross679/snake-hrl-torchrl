"""Tests for OverlappingPairDataset — Phase 2.1 checkpoint format.

Tests cover:
- Dataset length = n_batch_files * n_runs_per_file * steps_per_run (4)
- __getitem__ returns correct shapes: state(124,), per_element_phase(60,), next_state(124,), delta(124,)
- Per-element phase is computed on-the-fly (not a zero tensor)
- Train/val split: no episode_id overlap between splits, counts sum to total
- get_sample_weights returns (N,) positive tensor
- SurrogateDataset emits DeprecationWarning on instantiation
- delta == next_state - state elementwise
"""

import warnings
from pathlib import Path

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fake_batch(tmp_path: Path, n_runs: int = 10, n_checkpoints: int = 5, suffix: str = "w00_0000") -> Path:
    """Create a synthetic Phase 2.1 batch file (substep_states format).

    Args:
        tmp_path: Directory to write the file.
        n_runs: Number of runs per batch file.
        n_checkpoints: K+1 checkpoints per run (default 5 = 4 pairs).
        suffix: Filename suffix after 'batch_'.

    Returns:
        Path to the written .pt file.
    """
    data = {
        "substep_states": torch.randn(n_runs, n_checkpoints, 124),
        "actions": torch.rand(n_runs, 5) * 2 - 1,   # normalized [-1, 1]
        "t_start": torch.rand(n_runs),
        "episode_ids": torch.arange(n_runs, dtype=torch.int64),
        "step_ids": torch.zeros(n_runs, dtype=torch.int64),
    }
    path = tmp_path / f"batch_{suffix}.pt"
    torch.save(data, path)
    return path


# ---------------------------------------------------------------------------
# Tests: OverlappingPairDataset
# ---------------------------------------------------------------------------


class TestOverlappingPairDataset:
    """Tests for aprx_model_elastica.dataset.OverlappingPairDataset."""

    def test_dataset_len(self, tmp_path):
        """Dataset with 2 batch files × 10 runs × 4 pairs = 80 items (train split)."""
        from aprx_model_elastica.dataset import OverlappingPairDataset

        make_fake_batch(tmp_path, n_runs=10, n_checkpoints=5, suffix="w00_0000")
        make_fake_batch(tmp_path, n_runs=10, n_checkpoints=5, suffix="w01_0000")

        # With val_fraction=0.1 and seed=42, 90% of 20 episodes → train
        ds_train = OverlappingPairDataset(str(tmp_path), split="train", val_fraction=0.1, seed=42)
        ds_val = OverlappingPairDataset(str(tmp_path), split="val", val_fraction=0.1, seed=42)

        # Total must sum to 2 * 10 * 4 = 80
        assert len(ds_train) + len(ds_val) == 80, (
            f"Expected train+val=80, got {len(ds_train)}+{len(ds_val)}={len(ds_train)+len(ds_val)}"
        )
        assert len(ds_train) > 0
        assert len(ds_val) > 0

    def test_getitem_shapes(self, tmp_path):
        """__getitem__ returns correct shapes for all keys."""
        from aprx_model_elastica.dataset import OverlappingPairDataset

        make_fake_batch(tmp_path, n_runs=10, suffix="w00_0000")
        ds = OverlappingPairDataset(str(tmp_path), split="train", val_fraction=0.1, seed=42)

        item = ds[0]

        assert "state" in item, "Missing 'state' key"
        assert "per_element_phase" in item, "Missing 'per_element_phase' key"
        assert "next_state" in item, "Missing 'next_state' key"
        assert "delta" in item, "Missing 'delta' key"
        assert "action" in item, "Missing 'action' key"

        assert item["state"].shape == (124,), f"state shape {item['state'].shape} != (124,)"
        assert item["per_element_phase"].shape == (60,), f"per_element_phase shape {item['per_element_phase'].shape} != (60,)"
        assert item["next_state"].shape == (124,), f"next_state shape {item['next_state'].shape} != (124,)"
        assert item["delta"].shape == (124,), f"delta shape {item['delta'].shape} != (124,)"
        assert item["action"].shape == (5,), f"action shape {item['action'].shape} != (5,)"

    def test_phase_encoding_on_the_fly(self, tmp_path):
        """per_element_phase is NOT a zero tensor — actual encoding ran."""
        from aprx_model_elastica.dataset import OverlappingPairDataset

        make_fake_batch(tmp_path, n_runs=10, suffix="w00_0000")
        ds = OverlappingPairDataset(str(tmp_path), split="train", val_fraction=0.1, seed=42)

        item = ds[0]
        phase = item["per_element_phase"]

        assert not torch.all(phase == 0.0), "per_element_phase is all zeros — encoding did not run"
        # Also check it has reasonable magnitude (sin/cos values in [-1, 1] range roughly)
        assert phase.abs().max() <= 10.0, f"per_element_phase has extreme values: max={phase.abs().max()}"

    def test_train_val_split(self, tmp_path):
        """train+val item counts sum to total; no overlap in episode_ids between splits."""
        from aprx_model_elastica.dataset import OverlappingPairDataset

        # Use multiple runs with distinct episode IDs
        make_fake_batch(tmp_path, n_runs=20, suffix="w00_0000")

        ds_train = OverlappingPairDataset(str(tmp_path), split="train", val_fraction=0.2, seed=0)
        ds_val = OverlappingPairDataset(str(tmp_path), split="val", val_fraction=0.2, seed=0)

        # Counts should sum to 20 * 4 = 80
        assert len(ds_train) + len(ds_val) == 20 * 4, (
            f"train({len(ds_train)}) + val({len(ds_val)}) != 80"
        )

        # Episode IDs should not overlap between splits
        train_eids = set(ds_train.episode_ids.tolist())
        val_eids = set(ds_val.episode_ids.tolist())
        overlap = train_eids & val_eids
        assert len(overlap) == 0, f"Episode IDs overlap between splits: {overlap}"

    def test_get_sample_weights(self, tmp_path):
        """get_sample_weights returns tensor of shape (N,) with all positive values."""
        from aprx_model_elastica.dataset import OverlappingPairDataset

        make_fake_batch(tmp_path, n_runs=10, suffix="w00_0000")
        ds = OverlappingPairDataset(str(tmp_path), split="train", val_fraction=0.1, seed=42)

        weights = ds.get_sample_weights()

        assert weights.shape == (len(ds),), f"weights shape {weights.shape} != ({len(ds)},)"
        assert (weights > 0).all(), "Not all sample weights are positive"
        assert weights.dtype == torch.float32, f"weights dtype {weights.dtype} != float32"

    def test_delta_correctness(self, tmp_path):
        """item['delta'] == item['next_state'] - item['state'] elementwise."""
        from aprx_model_elastica.dataset import OverlappingPairDataset

        make_fake_batch(tmp_path, n_runs=10, suffix="w00_0000")
        ds = OverlappingPairDataset(str(tmp_path), split="train", val_fraction=0.1, seed=42)

        for idx in range(min(5, len(ds))):
            item = ds[idx]
            expected_delta = item["next_state"] - item["state"]
            assert torch.allclose(item["delta"], expected_delta, atol=1e-6), (
                f"delta mismatch at idx={idx}: max error {(item['delta'] - expected_delta).abs().max()}"
            )

    def test_missing_substep_states_raises(self, tmp_path):
        """Loading a Phase 1 format file (no 'substep_states' key) raises ValueError."""
        from aprx_model_elastica.dataset import OverlappingPairDataset

        # Simulate a Phase 1 batch file format
        bad_data = {
            "states": torch.randn(10, 124),
            "next_states": torch.randn(10, 124),
            "actions": torch.randn(10, 5),
        }
        bad_path = tmp_path / "batch_w00_0000.pt"
        torch.save(bad_data, bad_path)

        with pytest.raises(ValueError, match="substep_states"):
            OverlappingPairDataset(str(tmp_path))

    def test_empty_dir_raises(self, tmp_path):
        """Empty directory raises FileNotFoundError."""
        from aprx_model_elastica.dataset import OverlappingPairDataset

        with pytest.raises(FileNotFoundError):
            OverlappingPairDataset(str(tmp_path))


# ---------------------------------------------------------------------------
# Tests: SurrogateDataset deprecation
# ---------------------------------------------------------------------------


class TestSurrogateDatasetDeprecation:
    """SurrogateDataset must emit DeprecationWarning on instantiation."""

    def test_deprecation_warning(self, tmp_path):
        """Instantiating SurrogateDataset emits DeprecationWarning."""
        from aprx_model_elastica.dataset import SurrogateDataset

        # Create a Phase 1 format batch file (states/next_states keys)
        phase1_data = {
            "states": torch.randn(10, 124),
            "actions": torch.randn(10, 5),
            "serpenoid_times": torch.rand(10),
            "next_states": torch.randn(10, 124),
            "episode_ids": torch.arange(10, dtype=torch.int64),
            "step_indices": torch.arange(10, dtype=torch.int64),
        }
        torch.save(phase1_data, tmp_path / "batch_0000.pt")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SurrogateDataset(str(tmp_path), split="train")
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1, (
                "SurrogateDataset should emit at least one DeprecationWarning on __init__"
            )
            assert "OverlappingPairDataset" in str(deprecation_warnings[0].message), (
                "DeprecationWarning should mention OverlappingPairDataset"
            )
