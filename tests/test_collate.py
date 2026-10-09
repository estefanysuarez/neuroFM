"""Test collate functions."""

import numpy as np
import torch
from neurofm.data.recording import Window
from neurofm.data.collate import neuralbench_collate, tensor_collate


def test_tensor_collate():
    """Test tensor collate."""
    windows = [
        Window(
            signal=np.random.randn(4, 256).astype(np.float32),
            sfreq=256.0,
            ch_names=["Ch0", "Ch1", "Ch2", "Ch3"],
            target=0,
        ),
        Window(
            signal=np.random.randn(4, 256).astype(np.float32),
            sfreq=256.0,
            ch_names=["Ch0", "Ch1", "Ch2", "Ch3"],
            target=1,
        ),
    ]
    
    batch = tensor_collate(windows)
    
    assert batch["signal"].shape == (2, 4, 256)
    assert batch["target"].shape == (2,)
    assert isinstance(batch["signal"], torch.Tensor)
    assert isinstance(batch["target"], torch.Tensor)
    assert "channel_mask" not in batch
    assert "dataset" not in batch


def test_neuralbench_collate():
    positions = np.random.randn(4, 3).astype(np.float32)
    windows = [
        Window(
            signal=np.random.randn(4, 256).astype(np.float32),
            sfreq=256.0,
            ch_names=["Ch0", "Ch1", "Ch2", "Ch3"],
            ch_positions=positions,
            target=0,
            subject="sub-01",
        ),
        Window(
            signal=np.random.randn(4, 256).astype(np.float32),
            sfreq=256.0,
            ch_names=["Ch0", "Ch1", "Ch2", "Ch3"],
            ch_positions=positions,
            target=1,
            subject="sub-02",
        ),
    ]

    batch = neuralbench_collate(windows)

    assert batch["x"].shape == (2, 4, 256)
    assert batch["y"].shape == (2,)
    assert batch["subject_ids"] == ["sub-01", "sub-02"]
    assert batch["ch_positions"].shape == (4, 3)


def test_tensor_collate_channel_mask_and_dataset():
    windows = [
        Window(
            signal=np.random.randn(3, 16).astype(np.float32),
            sfreq=256.0,
            ch_names=["C3", "Cz", "C4"],
            channel_mask=np.array([True, True, False]),
            dataset="ds_a",
            target=0,
        ),
        Window(
            signal=np.random.randn(3, 16).astype(np.float32),
            sfreq=256.0,
            ch_names=["C3", "Cz", "C4"],
            channel_mask=np.array([False, True, True]),
            dataset="ds_a",
            target=1,
        ),
    ]
    batch = tensor_collate(windows)
    assert batch["channel_mask"].shape == (2, 3)
    assert batch["channel_mask"].dtype == torch.bool
    assert batch["dataset"] == "ds_a"


def test_collate_stacks_positions_when_they_differ():
    pos_a = np.zeros((2, 3), dtype=np.float32)
    pos_b = np.ones((2, 3), dtype=np.float32)
    windows = [
        Window(
            signal=np.random.randn(2, 8).astype(np.float32),
            sfreq=256.0,
            ch_names=["C3", "C4"],
            ch_positions=pos_a,
            target=0,
        ),
        Window(
            signal=np.random.randn(2, 8).astype(np.float32),
            sfreq=256.0,
            ch_names=["C3", "C4"],
            ch_positions=pos_b,
            target=1,
        ),
    ]
    batch = tensor_collate(windows)
    assert batch["ch_positions"].shape == (2, 2, 3)
