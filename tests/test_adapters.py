"""Tests for per-dataset input adapters."""

import torch

from neurofm.models.adapters import InputAdapters
from neurofm.models.custom import TinyEEGNet


def test_input_adapters_project_native_channels():
    model = TinyEEGNet(n_chans=4, n_times=32, n_outputs=2)
    wrapped = InputAdapters(model, {"ds_a": 3, "ds_b": 5})
    batch = {
        "signal": torch.randn(2, 3, 32),
        "dataset": "ds_a",
        "target": torch.tensor([0, 1]),
    }
    logits = wrapped(batch)
    assert logits.shape == (2, 2)
    assert wrapped.n_chans == 4
    assert wrapped.n_outputs == 2
    assert wrapped.input_contract is model.input_contract
