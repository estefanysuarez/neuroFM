"""Gaussian noise augmentation."""

import numpy as np
from neurofm.registry import register
from neurofm.data.recording import Window


@register("augmentations", "noise")
class GaussianNoise:
    """Add Gaussian noise to a window."""
    
    def __init__(self, std: float = 1e-6):
        """
        Args:
            std: standard deviation of the noise in volts.
        """
        self.std = std
    
    def __call__(self, window: Window) -> Window:
        """Add noise to the window signal."""
        noise = np.random.randn(*window.signal.shape).astype(np.float32) * self.std
        window.signal = window.signal + noise
        return window
