"""Test data sources."""

from neurofm.data.sources import SyntheticSource


def test_synthetic_source():
    """Test SyntheticSource."""
    source = SyntheticSource(n_channels=4, duration_seconds=10.0, sfreq=256.0)
    recording = source()
    
    assert recording.signal.shape[0] == 4
    assert recording.sfreq == 256.0
    assert len(recording.ch_names) == 4
    assert recording.modality == "eeg"
    assert recording.events is not None
    assert recording.ch_positions.shape == (4, 3)
