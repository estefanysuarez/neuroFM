"""Test transforms."""

import numpy as np
import pytest
from neurofm.data.recording import Recording
from neurofm.data.transforms import AlignChannels, BandpassFilter, Resample


def _recording(ch_names, n_times=64):
    n_chans = len(ch_names)
    theta = np.linspace(0, 2 * np.pi, n_chans, endpoint=False)
    positions = np.stack(
        [np.cos(theta), np.sin(theta), np.zeros(n_chans)], axis=1
    ).astype(np.float32)
    signal = np.arange(n_chans * n_times, dtype=np.float32).reshape(n_chans, n_times)
    return Recording(
        signal=signal,
        sfreq=256.0,
        ch_names=list(ch_names),
        ch_positions=positions,
    )


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


def test_align_reorders_and_drops_extra_channels():
    rec = _recording(["C4", "C3", "Fz", "Cz"])
    aligned = AlignChannels(["C3", "Cz", "C4"])(rec)
    assert aligned.ch_names == ["C3", "Cz", "C4"]
    assert aligned.signal.shape == (3, 64)
    np.testing.assert_array_equal(aligned.signal[0], rec.signal[1])
    np.testing.assert_array_equal(aligned.signal[1], rec.signal[3])
    np.testing.assert_array_equal(aligned.signal[2], rec.signal[0])
    np.testing.assert_array_equal(aligned.channel_mask, [True, True, True])


def test_align_zero_fills_missing_channel():
    rec = _recording(["C3", "Cz"])
    aligned = AlignChannels(["C3", "Cz", "C4"])(rec)
    np.testing.assert_array_equal(aligned.signal[0], rec.signal[0])
    np.testing.assert_array_equal(aligned.signal[2], np.zeros(64, dtype=np.float32))
    np.testing.assert_array_equal(aligned.channel_mask, [True, True, False])
    assert np.isnan(aligned.ch_positions[2]).all()


def test_align_renames_through_channel_map():
    rec = _recording(["T3", "Cz", "T4"])
    aligned = AlignChannels(
        ["T7", "Cz", "T8"],
        channel_map={"T3": "T7", "T4": "T8"},
    )(rec)
    np.testing.assert_array_equal(aligned.signal[0], rec.signal[0])
    np.testing.assert_array_equal(aligned.signal[2], rec.signal[2])
    np.testing.assert_array_equal(aligned.channel_mask, [True, True, True])


def test_align_rejects_duplicate_names():
    rec = _recording(["Ch0", "Ch1"])
    with pytest.raises(ValueError, match="Duplicate channel names"):
        AlignChannels(["Cz"], channel_map={"Ch0": "Cz", "Ch1": "Cz"})(rec)


def test_resample_changes_length_and_rate():
    rec = _recording(["C3", "C4"], n_times=256)
    rec.events = {"trial": [(0, 128)]}
    resampled = Resample(128.0)(rec)
    assert resampled.sfreq == 128.0
    assert resampled.signal.shape == (2, 128)
    assert resampled.events["trial"] == [(0, 64)]
