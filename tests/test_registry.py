"""Test registry functionality."""

import pytest

import neurofm.plugins  # noqa: F401
from neurofm.registry import (
    Registry,
    augmentations_registry,
    collates_registry,
    models_registry,
    sources_registry,
    splits_registry,
    targets_registry,
    tokenizers_registry,
    transforms_registry,
)


def test_registry_register():
    """Test basic registration."""
    reg = Registry("test")
    
    @reg.register("foo")
    class Foo:
        pass
    
    assert reg.get("foo") == Foo


def test_registry_duplicate():
    """Test duplicate registration raises error."""
    reg = Registry("test")
    
    @reg.register("foo")
    class Foo:
        pass
    
    with pytest.raises(ValueError):
        @reg.register("foo")
        class Bar:
            pass


def test_registry_not_found():
    """Test missing component raises error."""
    reg = Registry("test")
    
    with pytest.raises(KeyError):
        reg.get("missing")


def test_plugins_register_builtins():
    """Importing neurofm.plugins registers the built-in components."""
    assert "synthetic" in sources_registry.list()
    assert {"bandpass", "resample", "align_channels"} <= set(transforms_registry.list())
    assert "noise" in augmentations_registry.list()
    assert "patch" in tokenizers_registry.list()
    assert "predefined" in splits_registry.list()
    assert "binary" in targets_registry.list()
    assert {"tiny_eegnet", "neuralbench"} <= set(models_registry.list())
    assert {"tensor", "neuralbench", "braindecode", "eeg_fm_bench"} <= set(collates_registry.list())
