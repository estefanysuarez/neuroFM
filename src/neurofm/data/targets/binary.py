"""Binary target: simple class labels."""

import numpy as np
from neurofm.registry import register


@register("targets", "binary")
class BinaryTarget:
    """Assign a binary target based on trial ID."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
    
    def __call__(self, trial_id: int) -> int:
        """Return target (0 or 1) for a trial."""
        rng = np.random.RandomState(self.seed + trial_id)
        return rng.randint(0, 2)
