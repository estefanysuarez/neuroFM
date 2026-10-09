"""Synthetic EEG data for testing and development."""

import numpy as np
from neurofm.registry import register
from neurofm.data.recording import Recording


@register("sources", "synthetic")
class SyntheticSource:
    """Generate synthetic EEG data for testing."""
    
    def __init__(
        self,
        n_channels: int = 4,
        duration_seconds: float = 60.0,
        sfreq: float = 256.0,
        n_trials: int = 10,
    ):
        self.n_channels = n_channels
        self.duration_seconds = duration_seconds
        self.sfreq = sfreq
        self.n_trials = n_trials
    
    def __call__(self) -> Recording:
        """Generate a synthetic Recording."""
        n_samples = int(self.duration_seconds * self.sfreq)
        
        # Random signal
        signal = np.random.randn(self.n_channels, n_samples).astype(np.float32) * 10e-6
        
        # Add a slow oscillation
        t = np.arange(n_samples) / self.sfreq
        signal += (5e-6 * np.sin(2 * np.pi * 0.5 * t)).astype(np.float32)
        
        ch_names = [f"Ch{i}" for i in range(self.n_channels)]
        theta = np.linspace(0, 2 * np.pi, self.n_channels, endpoint=False)
        ch_positions = np.stack(
            [np.cos(theta), np.sin(theta), np.zeros(self.n_channels)],
            axis=1,
        ).astype(np.float32)

        # Create synthetic events
        trial_length = n_samples // self.n_trials
        events = {"trial": []}
        for i in range(self.n_trials):
            onset = i * trial_length
            events["trial"].append((onset, trial_length))
        
        return Recording(
            signal=signal,
            sfreq=self.sfreq,
            ch_names=ch_names,
            ch_positions=ch_positions,
            modality="eeg",
            subject="synthetic",
            session="test",
            dataset="synthetic",
            events=events,
        )
