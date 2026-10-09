"""Manifest: log experiment configuration and metadata."""

from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class Manifest:
    """Record of an experiment run."""
    
    package_versions: dict
    pipeline_config: dict
    model_config: dict
    objective: str
    mode: str
    handoff: str
    split: str
    seed: int
    
    def save(self, path: Path):
        """Save manifest to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Load manifest from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)
