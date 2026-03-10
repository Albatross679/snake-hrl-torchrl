"""Tests for FlatStepDataset — Phase 02.2 flat-format .pt batch loading.

8 tests covering:
  1. Batch with forces loads without error, len==5
  2. __getitem__ returns expected keys and shapes
  3. delta == next_state - state
  4. forces dict has correct keys and shapes
  5. get_sample_weights() returns positive-valued tensor shape (5,)
  6. train/val split over 10 episodes: both splits non-empty
  7. episode_ids in train and val splits are disjoint
  8. batch without forces key -> forces is None in __getitem__
"""

import torch
import numpy as np
import pytest

from aprx_model_elastica.dataset import FlatStepDataset


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_batch_with_forces(n: int = 5, episode_ids=None):
    """Return a dict matching the Phase 02.2 flat .pt batch format."""
    if episode_ids is None:
        episode_ids = list(range(n))
    return {
        "states":      torch.randn(n, 124, dtype=torch.float32),
        "next_states": torch.randn(n, 124, dtype=torch.float32),
        "actions":     torch.randn(n, 5, dtype=torch.float32),
        "t_start":     torch.rand(n, dtype=torch.float32),
        "episode_ids": torch.tensor(episode_ids, dtype=torch.int64),
        "step_ids":    torch.zeros(n, dtype=torch.int64),
        "forces": {
            "external_forces":  torch.randn(n, 3, 21, dtype=torch.float32),
            "internal_forces":  torch.randn(n, 3, 21, dtype=torch.float32),
            "external_torques": torch.randn(n, 3, 20, dtype=torch.float32),
            "internal_torques": torch.randn(n, 3, 20, dtype=torch.float32),
        },
    }


def _make_batch_without_forces(n: int = 5, episode_ids=None):
    """Return a Phase 02.2 flat .pt batch without forces key."""
    if episode_ids is None:
        episode_ids = list(range(n))
    return {
        "states":      torch.randn(n, 124, dtype=torch.float32),
        "next_states": torch.randn(n, 124, dtype=torch.float32),
        "actions":     torch.randn(n, 5, dtype=torch.float32),
        "t_start":     torch.rand(n, dtype=torch.float32),
        "episode_ids": torch.tensor(episode_ids, dtype=torch.int64),
        "step_ids":    torch.zeros(n, dtype=torch.int64),
    }


@pytest.fixture
def batch_dir_with_forces(tmp_path):
    """tmp_path with a single batch_0000.pt containing forces."""
    batch = _make_batch_with_forces(n=5, episode_ids=[0, 1, 2, 3, 4])
    torch.save(batch, str(tmp_path / "batch_0000.pt"))
    return tmp_path, batch


@pytest.fixture
def batch_dir_without_forces(tmp_path):
    """tmp_path with a single batch_0000.pt without forces."""
    batch = _make_batch_without_forces(n=5, episode_ids=[0, 1, 2, 3, 4])
    torch.save(batch, str(tmp_path / "batch_0000.pt"))
    return tmp_path, batch


@pytest.fixture
def batch_dir_10_episodes(tmp_path):
    """tmp_path with 2 batch files, 5 items each (10 unique episode_ids)."""
    batch0 = _make_batch_with_forces(n=5, episode_ids=[0, 1, 2, 3, 4])
    batch1 = _make_batch_with_forces(n=5, episode_ids=[5, 6, 7, 8, 9])
    torch.save(batch0, str(tmp_path / "batch_0000.pt"))
    torch.save(batch1, str(tmp_path / "batch_0001.pt"))
    return tmp_path


# ---------------------------------------------------------------------------
# Test 1: batch with forces loads, len == 5
# ---------------------------------------------------------------------------

def test_load_with_forces_and_len(batch_dir_with_forces):
    """Test 1: Batch with forces loads without error; len(dataset)==5."""
    data_dir, _ = batch_dir_with_forces
    ds = FlatStepDataset(str(data_dir), split="train", val_fraction=0.0)
    assert len(ds) == 5, f"Expected 5 items, got {len(ds)}"


# ---------------------------------------------------------------------------
# Test 2: __getitem__ returns expected keys + shapes
# ---------------------------------------------------------------------------

def test_getitem_keys_and_shapes(batch_dir_with_forces):
    """Test 2: __getitem__(0) returns state(124,), action(5,), t_start(scalar),
    next_state(124,), delta(124,), forces dict."""
    data_dir, _ = batch_dir_with_forces
    ds = FlatStepDataset(str(data_dir), split="train", val_fraction=0.0)
    item = ds[0]
    assert "state" in item
    assert "action" in item
    assert "t_start" in item
    assert "next_state" in item
    assert "delta" in item
    assert "forces" in item

    assert item["state"].shape == (124,), f"state shape: {item['state'].shape}"
    assert item["action"].shape == (5,), f"action shape: {item['action'].shape}"
    assert item["next_state"].shape == (124,), f"next_state shape: {item['next_state'].shape}"
    assert item["delta"].shape == (124,), f"delta shape: {item['delta'].shape}"
    # t_start should be a scalar tensor
    assert item["t_start"].ndim == 0, f"t_start should be scalar, got shape {item['t_start'].shape}"


# ---------------------------------------------------------------------------
# Test 3: delta == next_state - state
# ---------------------------------------------------------------------------

def test_delta_equals_next_minus_state(batch_dir_with_forces):
    """Test 3: delta == next_state - state for every item."""
    data_dir, _ = batch_dir_with_forces
    ds = FlatStepDataset(str(data_dir), split="train", val_fraction=0.0)
    for i in range(len(ds)):
        item = ds[i]
        expected_delta = item["next_state"] - item["state"]
        assert torch.allclose(item["delta"], expected_delta), \
            f"delta mismatch at index {i}"


# ---------------------------------------------------------------------------
# Test 4: forces dict has correct keys and shapes
# ---------------------------------------------------------------------------

def test_forces_keys_and_shapes(batch_dir_with_forces):
    """Test 4: forces dict has external_forces(3,21), internal_forces(3,21),
    external_torques(3,20), internal_torques(3,20)."""
    data_dir, _ = batch_dir_with_forces
    ds = FlatStepDataset(str(data_dir), split="train", val_fraction=0.0)
    item = ds[0]
    forces = item["forces"]
    assert forces is not None, "forces should not be None when batch has forces"
    assert "external_forces" in forces
    assert "internal_forces" in forces
    assert "external_torques" in forces
    assert "internal_torques" in forces
    assert forces["external_forces"].shape == (3, 21), \
        f"external_forces shape: {forces['external_forces'].shape}"
    assert forces["internal_forces"].shape == (3, 21), \
        f"internal_forces shape: {forces['internal_forces'].shape}"
    assert forces["external_torques"].shape == (3, 20), \
        f"external_torques shape: {forces['external_torques'].shape}"
    assert forces["internal_torques"].shape == (3, 20), \
        f"internal_torques shape: {forces['internal_torques'].shape}"


# ---------------------------------------------------------------------------
# Test 5: get_sample_weights() returns positive (N,) tensor
# ---------------------------------------------------------------------------

def test_get_sample_weights(batch_dir_with_forces):
    """Test 5: get_sample_weights() returns tensor shape (5,) with all positive values."""
    data_dir, _ = batch_dir_with_forces
    ds = FlatStepDataset(str(data_dir), split="train", val_fraction=0.0)
    weights = ds.get_sample_weights()
    assert weights.shape == (len(ds),), f"weights shape: {weights.shape}"
    assert (weights > 0).all(), "all weights should be positive"


# ---------------------------------------------------------------------------
# Test 6: train/val split over 10 episodes — both non-empty
# ---------------------------------------------------------------------------

def test_train_val_split_nonempty(batch_dir_10_episodes):
    """Test 6: train/val split over 10 episodes: val >= 1, train >= 1."""
    ds_train = FlatStepDataset(str(batch_dir_10_episodes), split="train", val_fraction=0.1)
    ds_val = FlatStepDataset(str(batch_dir_10_episodes), split="val", val_fraction=0.1)
    assert len(ds_train) >= 1, f"train split empty: {len(ds_train)}"
    assert len(ds_val) >= 1, f"val split empty: {len(ds_val)}"
    assert len(ds_train) + len(ds_val) == 10, \
        f"train+val should sum to 10, got {len(ds_train)}+{len(ds_val)}"


# ---------------------------------------------------------------------------
# Test 7: episode_ids in train and val splits are disjoint
# ---------------------------------------------------------------------------

def test_train_val_episode_ids_disjoint(batch_dir_10_episodes):
    """Test 7: episode_ids in train and val are disjoint (no leakage)."""
    ds_train = FlatStepDataset(str(batch_dir_10_episodes), split="train", val_fraction=0.1)
    ds_val = FlatStepDataset(str(batch_dir_10_episodes), split="val", val_fraction=0.1)
    train_eids = set(ds_train.episode_ids.tolist())
    val_eids = set(ds_val.episode_ids.tolist())
    assert train_eids.isdisjoint(val_eids), \
        f"episode_ids overlap between train and val: {train_eids & val_eids}"


# ---------------------------------------------------------------------------
# Test 8: batch without forces key -> forces is None
# ---------------------------------------------------------------------------

def test_no_forces_key_returns_none(batch_dir_without_forces):
    """Test 8: batch without forces key -> forces is None in __getitem__."""
    data_dir, _ = batch_dir_without_forces
    ds = FlatStepDataset(str(data_dir), split="train", val_fraction=0.0)
    item = ds[0]
    assert item["forces"] is None, f"forces should be None, got {item['forces']}"
