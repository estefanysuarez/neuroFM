"""Recording-level resampling."""

from scipy import signal as scipy_signal

from neurofm.data.recording import Recording
from neurofm.registry import register


@register("transforms", "resample")
class Resample:
    """Resample a continuous recording to a target sampling rate."""

    level = "recording"

    def __init__(self, sfreq: float):
        self.sfreq = float(sfreq)

    def __call__(self, recording: Recording) -> Recording:
        if recording.sfreq == self.sfreq:
            return recording
        n_samples = int(round(recording.signal.shape[1] * self.sfreq / recording.sfreq))
        resampled = scipy_signal.resample(recording.signal, n_samples, axis=1)
        ratio = self.sfreq / recording.sfreq
        out = recording.copy()
        out.signal = resampled.astype(recording.signal.dtype, copy=False)
        out.sfreq = self.sfreq
        if recording.events:
            scaled = {}
            for name, events in recording.events.items():
                scaled[name] = [
                    (int(round(onset * ratio)), int(round(duration * ratio)))
                    for onset, duration in events
                ]
            out.events = scaled
        return out
