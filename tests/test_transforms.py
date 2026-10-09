"""Test transforms."""

import numpy as np
from neurofm.data.recording import Recording
from neurofm.data.transforms import BandpassFilter


def test_bandpass_filter():
    """Test BandpassFilter."""
    signal = np.random.randn(4, 1024).astype(np.float32)
    rec = Recording(
        signal=signal,
        sfreq=256.0,
        ch_names=["Ch0", "Ch1", "Ch2", "Ch3"],
    )
    
    filt = BandpassFilter(l_freq=0.1, h_freq=40.0)
    rec_filtered = filt(rec)
    
    assert rec_filtered.signal.shape == rec.signal.shape
    assert rec_filtered.sfreq == rec.sfreq
