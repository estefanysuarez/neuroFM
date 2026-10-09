"""Test registry functionality."""

import pytest
from neurofm.registry import Registry


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
