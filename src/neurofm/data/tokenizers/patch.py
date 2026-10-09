"""Patch tokenizer: reshape windows into channel × time patches."""

import numpy as np
from neurofm.registry import register
from neurofm.data.recording import Window


@register("tokenizers", "patch")
class PatchTokenizer:
    """Reshape a window into overlapping or non-overlapping patches."""
    
    output = "(batch, n_tokens, n_chans * patch_size)"
    
    def __init__(self, patch_size: int = 200, stride: int = None):
        """
        Args:
            patch_size: number of time samples per patch.
            stride: stride between patches. If None, stride = patch_size (non-overlapping).
        """
        self.patch_size = patch_size
        self.stride = stride or patch_size
    
    def __call__(self, window: Window) -> np.ndarray:
        """Convert window to patches.
        
        Args:
            window: a Window object.
        
        Returns:
            patches: array of shape (n_patches, n_chans * patch_size).
        """
        signal = window.signal  # (n_chans, n_times)
        n_chans, n_times = signal.shape
        
        patches = []
        for start in range(0, n_times - self.patch_size + 1, self.stride):
            patch = signal[:, start : start + self.patch_size]
            patches.append(patch.reshape(-1))  # Flatten to (n_chans * patch_size,)
        
        if not patches:
            raise ValueError(
                f"Window too short ({n_times}) for patch_size={self.patch_size}"
            )
        
        return np.stack(patches, axis=0)
