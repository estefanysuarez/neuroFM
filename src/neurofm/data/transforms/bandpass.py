"""Bandpass filter transform."""

import numpy as np
from scipy import signal as scipy_signal
from neurofm.registry import register
from neurofm.data.recording import Recording, Window


@register("transforms", "bandpass")
class BandpassFilter:
    """Bandpass filter (causal, Butterworth)."""
    
    level = "temporal"
    scope = "isolated"  # can be overridden in config
    
    def __init__(self, l_freq: float = 0.1, h_freq: float = 40.0, order: int = 4):
        """
        Args:
            l_freq: low cutoff frequency in Hz.
            h_freq: high cutoff frequency in Hz.
            order: filter order.
        """
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.order = order
        self._sos = None
    
    def _design_filter(self, sfreq: float):
        """Design the filter for a given sampling frequency."""
        nyquist = sfreq / 2.0
        low = self.l_freq / nyquist
        high = self.h_freq / nyquist
        low = np.clip(low, 1e-7, 0.9999)
        high = np.clip(high, 1e-7, 0.9999)
        self._sos = scipy_signal.butter(self.order, [low, high], btype="band", output="sos")
    
    def __call__(self, x: Recording | Window) -> Recording | Window:
        """Apply the filter."""
        if self._sos is None or self._sos[0, 0] == 0:
            # Cache miss: design for this sfreq
            self._design_filter(x.sfreq)
        
        # Filter each channel (causal, forward-backward with scipy's sosfilt)
        filtered_signal = np.zeros_like(x.signal)
        for ch in range(x.signal.shape[0]):
            filtered_signal[ch] = scipy_signal.sosfilt(self._sos, x.signal[ch])
        
        # Modify in place and return
        if isinstance(x, Recording):
            x_out = x.copy()
        else:
            x_out = x  # Windows are not copied; caller handles that
        x_out.signal = filtered_signal
        return x_out
