"""Recording and Window: canonical data structures."""

from dataclasses import dataclass, field
from typing import Optional, List, Set
import numpy as np


@dataclass
class Recording:
    """Continuous EEG/MEG recording.
    
    Attributes:
        signal: float32 array of shape (channels, time) in volts.
        sfreq: sampling frequency in Hz.
        ch_names: list of channel names.
        modality: "eeg", "meg", or "ecog".
        ch_positions: optional (channels, 3) array of electrode positions.
        subject: subject ID.
        session: session ID.
        dataset: dataset name.
        channel_mask: optional bool array of shape (channels,). True where the
            sample contains a real electrode. None means every channel is real.
        events: optional dict mapping event name to list of (onset_sample, duration_sample).
        applied_steps: set of transform names already applied.
    """
    signal: np.ndarray
    sfreq: float
    ch_names: List[str]
    modality: str = "eeg"
    ch_positions: Optional[np.ndarray] = None
    subject: Optional[str] = None
    session: Optional[str] = None
    dataset: Optional[str] = None
    channel_mask: Optional[np.ndarray] = None
    events: Optional[dict] = None
    applied_steps: Set[str] = field(default_factory=set)

    def __post_init__(self):
        """Validate Recording."""
        if self.signal.ndim != 2:
            raise ValueError(f"signal must be 2D (channels, time), got {self.signal.shape}")
        if len(self.ch_names) != self.signal.shape[0]:
            raise ValueError(
                f"ch_names length {len(self.ch_names)} does not match "
                f"signal channels {self.signal.shape[0]}"
            )
        if self.ch_positions is not None and self.ch_positions.shape != (self.signal.shape[0], 3):
            raise ValueError(
                f"ch_positions must be (channels, 3), got {self.ch_positions.shape}"
            )
        if self.channel_mask is not None and len(self.channel_mask) != self.signal.shape[0]:
            raise ValueError(
                f"channel_mask length {len(self.channel_mask)} does not match "
                f"signal channels {self.signal.shape[0]}"
            )

    def copy(self) -> "Recording":
        """Return a shallow copy with copied arrays."""
        return Recording(
            signal=self.signal.copy(),
            sfreq=self.sfreq,
            ch_names=list(self.ch_names),
            modality=self.modality,
            ch_positions=self.ch_positions.copy() if self.ch_positions is not None else None,
            subject=self.subject,
            session=self.session,
            dataset=self.dataset,
            channel_mask=None if self.channel_mask is None else self.channel_mask.copy(),
            events=dict(self.events) if self.events else None,
            applied_steps=set(self.applied_steps),
        )


@dataclass
class Window:
    """One sample: a windowed EEG/MEG segment with target.
    
    Attributes:
        signal: float32 array of shape (channels, time) in volts.
        sfreq: sampling frequency in Hz.
        ch_names: list of channel names.
        modality: "eeg", "meg", or "ecog".
        ch_positions: optional (channels, 3) array of electrode positions.
        subject: subject ID.
        session: session ID.
        dataset: dataset name.
        channel_mask: optional bool array of shape (channels,). True where the
            sample contains a real electrode. None means every channel is real.
        target: class index (int) or continuous value (float).
        trial_id: optional trial number or event ID.
    """
    signal: np.ndarray
    sfreq: float
    ch_names: List[str]
    modality: str = "eeg"
    ch_positions: Optional[np.ndarray] = None
    subject: Optional[str] = None
    session: Optional[str] = None
    dataset: Optional[str] = None
    channel_mask: Optional[np.ndarray] = None
    target: Optional[float] = None
    trial_id: Optional[int] = None

    def __post_init__(self):
        """Validate Window."""
        if self.signal.ndim != 2:
            raise ValueError(f"signal must be 2D (channels, time), got {self.signal.shape}")
        if len(self.ch_names) != self.signal.shape[0]:
            raise ValueError(
                f"ch_names length {len(self.ch_names)} does not match "
                f"signal channels {self.signal.shape[0]}"
            )
        if self.ch_positions is not None and self.ch_positions.shape != (self.signal.shape[0], 3):
            raise ValueError(
                f"ch_positions must be (channels, 3), got {self.ch_positions.shape}"
            )
        if self.channel_mask is not None and len(self.channel_mask) != self.signal.shape[0]:
            raise ValueError(
                f"channel_mask length {len(self.channel_mask)} does not match "
                f"signal channels {self.signal.shape[0]}"
            )

    def copy(self) -> "Window":
        """Return a copy whose signal array is independent of this window."""
        return Window(
            signal=self.signal.copy(),
            sfreq=self.sfreq,
            ch_names=self.ch_names,
            modality=self.modality,
            ch_positions=self.ch_positions,
            subject=self.subject,
            session=self.session,
            dataset=self.dataset,
            channel_mask=self.channel_mask,
            target=self.target,
            trial_id=self.trial_id,
        )

    def to_dict(self) -> dict:
        """Convert to dict for collate functions."""
        return {
            "signal": self.signal,
            "sfreq": self.sfreq,
            "ch_names": self.ch_names,
            "modality": self.modality,
            "ch_positions": self.ch_positions,
            "subject": self.subject,
            "session": self.session,
            "dataset": self.dataset,
            "channel_mask": self.channel_mask,
            "target": self.target,
            "trial_id": self.trial_id,
        }
