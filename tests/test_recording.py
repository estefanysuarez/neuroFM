"""Test Recording and Window."""

import numpy as np
from neurofm.data.recording import Recording, Window


def test_recording_creation():
    """Test creating a Recording."""
    signal = np.random.randn(4, 1000).astype(np.float32)
    rec = Recording(
        signal=signal,
        sfreq=256.0,
        ch_names=["Ch0", "Ch1", "Ch2", "Ch3"],
    )
    
    assert rec.signal.shape == (4, 1000)
    assert rec.sfreq == 256.0
    assert len(rec.ch_names) == 4


def test_recording_copy():
    """Test Recording.copy()."""
    signal = np.random.randn(4, 1000).astype(np.float32)
    mask = np.array([True, True, False, True])
    rec = Recording(
        signal=signal,
        sfreq=256.0,
        ch_names=["Ch0", "Ch1", "Ch2", "Ch3"],
        channel_mask=mask,
    )
    rec_copy = rec.copy()
    
    # Modify copy
    rec_copy.signal[0, 0] = 999.0
    rec_copy.channel_mask[2] = True
    
    # Original should not change
    assert rec.signal[0, 0] != 999.0
    assert rec.channel_mask[2] == False


def test_window_creation():
    """Test creating a Window."""
    signal = np.random.randn(4, 256).astype(np.float32)
    window = Window(
        signal=signal,
        sfreq=256.0,
        ch_names=["Ch0", "Ch1", "Ch2", "Ch3"],
        target=1,
    )
    
    assert window.signal.shape == (4, 256)
    assert window.target == 1


def test_window_to_dict():
    """Test Window.to_dict()."""
    signal = np.random.randn(4, 256).astype(np.float32)
    window = Window(
        signal=signal,
        sfreq=256.0,
        ch_names=["Ch0", "Ch1", "Ch2", "Ch3"],
        target=1,
    )
    d = window.to_dict()
    
    assert "signal" in d
    assert "target" in d
    assert "channel_mask" in d
    assert d["target"] == 1


def test_window_copy_is_independent():
    signal = np.random.randn(4, 256).astype(np.float32)
    window = Window(
        signal=signal,
        sfreq=256.0,
        ch_names=["Ch0", "Ch1", "Ch2", "Ch3"],
        channel_mask=np.array([True, True, True, False]),
        target=1,
    )
    copied = window.copy()
    copied.signal[0, 0] = 999.0
    assert window.signal[0, 0] != 999.0
    assert copied.channel_mask[3] == False
