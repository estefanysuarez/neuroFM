"""Augmentations: random transforms applied only on train loader."""

from .noise import GaussianNoise

__all__ = ["GaussianNoise"]
