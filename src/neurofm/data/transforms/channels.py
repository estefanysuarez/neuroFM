"""Channel alignment onto a shared montage."""

from typing import Dict, List, Optional

import numpy as np

from neurofm.data.recording import Recording
from neurofm.registry import register


def mapped_channel_names(
    ch_names: List[str], channel_map: Optional[Dict[str, str]] = None
) -> List[str]:
    """Rename channels, then reject duplicates."""
    channel_map = channel_map or {}
    names = [channel_map.get(name, name) for name in ch_names]
    seen = set()
    duplicates = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise ValueError(
            f"Duplicate channel names after mapping: {duplicates}. "
            f"Names: {names}"
        )
    return names


def apply_channel_map(
    recording: Recording, channel_map: Optional[Dict[str, str]] = None
) -> Recording:
    """Return a recording whose channel names have been remapped."""
    names = mapped_channel_names(recording.ch_names, channel_map)
    if names == recording.ch_names:
        return recording
    out = recording.copy()
    out.ch_names = names
    return out


def union_channel_names(recordings: List[Recording]) -> List[str]:
    """Append each new channel name in recording order."""
    union: List[str] = []
    seen = set()
    for recording in recordings:
        for name in recording.ch_names:
            if name not in seen:
                seen.add(name)
                union.append(name)
    return union


@register("transforms", "align_channels")
class AlignChannels:
    """Reorder a recording onto ``ch_names``, zero-filling missing electrodes."""

    level = "recording"

    def __init__(
        self,
        ch_names: List[str],
        channel_map: Optional[Dict[str, str]] = None,
    ):
        self.ch_names = list(ch_names)
        self.channel_map = channel_map or {}

    def __call__(self, recording: Recording) -> Recording:
        names = mapped_channel_names(recording.ch_names, self.channel_map)
        name_to_idx = {name: i for i, name in enumerate(names)}
        n_out = len(self.ch_names)
        n_times = recording.signal.shape[1]
        signal = np.zeros((n_out, n_times), dtype=recording.signal.dtype)
        mask = np.zeros(n_out, dtype=bool)
        positions = None
        if recording.ch_positions is not None:
            positions = np.full((n_out, 3), np.nan, dtype=np.float32)

        for out_i, name in enumerate(self.ch_names):
            in_i = name_to_idx.get(name)
            if in_i is None:
                continue
            signal[out_i] = recording.signal[in_i]
            mask[out_i] = True
            if positions is not None:
                positions[out_i] = recording.ch_positions[in_i]

        out = recording.copy()
        out.signal = signal
        out.ch_names = list(self.ch_names)
        out.channel_mask = mask
        out.ch_positions = positions
        return out
