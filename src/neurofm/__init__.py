"""neuroFM: a flexible hub for neuroscience foundation-model experiments."""

__version__ = "0.1.0"

from .registry import Registry, register

__all__ = ["Registry", "register"]
