"""Pipeline: orchestrates the data loading and preprocessing stages."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class WindowType(str, Enum):
    """Window extraction strategy."""
    FIXED = "fixed"
    EVENTS = "events"


@dataclass
class Pipeline:
    """Data pipeline configuration.
    
    Attributes:
        source: name of the source (e.g., "synthetic").
        source_params: constructor kwargs for the source (e.g., dataset_name).
        spatial_transforms: list of transform names (recording-level, non-temporal).
        window: "fixed" or "events".
        temporal_transforms: list of temporal transform names with implicit scope.
        target: name of the target extractor (e.g., "binary", "event_id").
        split: name of the split strategy (e.g., "predefined", "leave_one_subject_out").
        window_transforms: list of window-level transform names.
        augmentations: list of augmentation names (train only).
        tokenizer: optional tokenizer name; None means model tokenizes itself.
    """
    source: str
    source_params: Dict[str, Any] = field(default_factory=dict)
    window: str = WindowType.FIXED.value
    target: str = "identity"
    split: str = "predefined"
    spatial_transforms: List[str] = field(default_factory=list)
    temporal_transforms: List[str] = field(default_factory=list)
    window_transforms: List[str] = field(default_factory=list)
    augmentations: List[str] = field(default_factory=list)
    tokenizer: Optional[str] = None

    def get_all_transforms(self) -> List[str]:
        """Get all transform names in order."""
        return (
            self.spatial_transforms
            + self.temporal_transforms
            + self.window_transforms
        )
