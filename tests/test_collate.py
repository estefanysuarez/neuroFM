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


def test_neuralbench_collate():
    windows = [
        Window(
            signal=np.random.randn(4, 256).astype(np.float32),
            sfreq=256.0,
            ch_names=["Ch0", "Ch1", "Ch2", "Ch3"],
            ch_positions=np.random.randn(4, 3).astype(np.float32),
            target=0,
            subject="sub-01",
        ),
        Window(
            signal=np.random.randn(4, 256).astype(np.float32),
            sfreq=256.0,
            ch_names=["Ch0", "Ch1", "Ch2", "Ch3"],
            ch_positions=np.random.randn(4, 3).astype(np.float32),
            target=1,
            subject="sub-02",
        ),
    ]

    batch = neuralbench_collate(windows)

    assert batch["x"].shape == (2, 4, 256)
    assert batch["y"].shape == (2,)
    assert batch["subject_ids"] == ["sub-01", "sub-02"]
    assert batch["ch_positions"].shape == (4, 3)
