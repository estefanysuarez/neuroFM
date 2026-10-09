"""Test models."""

import pytest
import torch
from neurofm.models.custom import TinyEEGNet
from neurofm.models.external.neuralbench import (
    NeuralBenchWrapper,
    _pretrained_overrides,
    resolve_architecture,
)


def test_tiny_eegnet():
    """Test TinyEEGNet forward pass."""
    model = TinyEEGNet(n_chans=4, n_times=1024, n_outputs=2)
    
    batch = {
        "signal": torch.randn(8, 4, 1024),
    }
    
    logits = model(batch)
    assert logits.shape == (8, 2)


def test_tiny_eegnet_contract():
    """Test TinyEEGNet input contract."""
    model = TinyEEGNet(n_chans=4, n_times=1024, n_outputs=2)
    contract = model.input_contract
    
    assert contract.collate == "tensor"
    assert contract.accepts == "window"


def test_resolve_architecture_aliases():
    assert resolve_architecture("eegnet").class_name == "EEGNet"
    assert resolve_architecture("REVE").class_name == "REVE"
    assert resolve_architecture("biot").needs_positions is False
    assert resolve_architecture("reve").needs_positions is True
    assert resolve_architecture("Deep4Net").class_name == "Deep4Net"


def test_pretrained_overrides_keep_checkpoint_geometry():
    overrides, backbone_only = _pretrained_overrides(
        {"n_chans": None, "n_times": None, "n_outputs": None, "sfreq": 200.0},
        config={"embed_dim": 512},
        weight_names={"cls_query_token", "to_patch_embedding.0.weight"},
    )
    assert overrides == {"sfreq": 200.0, "attention_pooling": True}
    assert backbone_only is True


def test_pretrained_overrides_keep_explicit_head():
    overrides, backbone_only = _pretrained_overrides(
        {"n_chans": None, "n_outputs": 4, "attention_pooling": False},
        config={},
        weight_names={"cls_query_token"},
    )
    assert overrides["n_outputs"] == 4
    assert overrides["attention_pooling"] is False
    assert "n_chans" not in overrides
    assert backbone_only is False


def test_neuralbench_eegnet_forward():
    pytest.importorskip("braindecode")
    model = NeuralBenchWrapper(
        architecture="EEGNet",
        n_chans=4,
        n_times=256,
        n_outputs=2,
        sfreq=120.0,
    )
    logits = model({"signal": torch.randn(2, 4, 256)})
    assert logits.shape == (2, 2)
    assert model.n_chans == 4
    assert model.n_times == 256
    assert model.n_outputs == 2
    assert model.sfreq == 120.0
    assert model.input_shape == (1, 4, 256)
    assert model.chs_info is None
    assert model.input_contract.collate in {"neuralbench", "tensor"}
    assert model.input_contract.accepts == "window"


def test_neuralbench_eegnet_accepts_x_key():
    pytest.importorskip("braindecode")
    model = NeuralBenchWrapper(architecture="EEGNet", n_chans=4, n_times=256, n_outputs=2)
    logits = model({"x": torch.randn(2, 4, 256)})
    assert logits.shape == (2, 2)


def test_neuralbench_reve_forward_with_positions():
    pytest.importorskip("braindecode")
    model = NeuralBenchWrapper(
        architecture="REVE",
        n_chans=4,
        n_times=256,
        n_outputs=2,
        sfreq=200.0,
        kwargs={"embed_dim": 32, "depth": 1, "heads": 2, "head_dim": 16, "patch_size": 64, "patch_overlap": 8},
    )
    batch = {
        "x": torch.randn(2, 4, 256),
        "ch_positions": torch.randn(4, 3),
    }
    logits = model(batch)
    assert logits.shape == (2, 2)


def test_neuralbench_biot_forward():
    pytest.importorskip("braindecode")
    model = NeuralBenchWrapper(
        architecture="BIOT",
        n_chans=4,
        n_times=256,
        n_outputs=2,
        sfreq=200.0,
        kwargs={"embed_dim": 32, "num_heads": 2, "num_layers": 1},
    )
    logits = model({"signal": torch.randn(2, 4, 256)})
    assert logits.shape == (2, 2)

