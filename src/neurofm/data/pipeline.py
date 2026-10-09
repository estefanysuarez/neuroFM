"""Pipeline: orchestrates the data loading and preprocessing stages."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class WindowType(str, Enum):
    """Window extraction strategy."""
    FIXED = "fixed"
    EVENTS = "events"


TransformEntry = Union[str, Dict[str, Any]]


def transform_name(entry: TransformEntry) -> str:
    """Return the registered name of a transform or augmentation entry."""
    if isinstance(entry, str):
        return entry
    return entry["name"]


@dataclass
class Pipeline:
    """Data pipeline configuration.
    
    Attributes:
        source: name of the source (e.g., "synthetic").
        source_params: constructor kwargs for the source (e.g., dataset_name).
        spatial_transforms: channel-layout transforms (recording-level, not time).
        window: "fixed" or "events".
        temporal_transforms: time-axis transforms, including bandpass and resample.
        target: name of the target extractor (e.g., "binary", "event_id").
        split: name of the split strategy (e.g., "predefined", "leave_one_subject_out").
        window_transforms: list of window-level transform names.
        augmentations: list of augmentation names (train only).
        tokenizer: optional tokenizer name; None means model tokenizes itself.
        name: optional dataset label used by adapters and logs.
        channel_map: optional rename of electrodes before channel alignment.
        split_params: constructor kwargs for the split strategy.
        target_params: constructor kwargs for the target extractor.
    """
    source: str
    source_params: Dict[str, Any] = field(default_factory=dict)
    window: str = WindowType.FIXED.value
    target: str = "identity"
    split: str = "predefined"
    spatial_transforms: List[TransformEntry] = field(default_factory=list)
    temporal_transforms: List[TransformEntry] = field(default_factory=list)
    window_transforms: List[TransformEntry] = field(default_factory=list)
    augmentations: List[TransformEntry] = field(default_factory=list)
    tokenizer: Optional[str] = None
    name: Optional[str] = None
    channel_map: Dict[str, str] = field(default_factory=dict)
    split_params: Dict[str, Any] = field(default_factory=dict)
    target_params: Dict[str, Any] = field(default_factory=dict)

    def get_all_transforms(self) -> List[str]:
        """Get all transform names in order."""
        return [
            transform_name(entry)
            for entry in (
                self.spatial_transforms
                + self.temporal_transforms
                + self.window_transforms
            )
        ]
