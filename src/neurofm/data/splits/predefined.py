"""Predefined split: use trial ID modulo to split data."""

from neurofm.registry import register


@register("splits", "predefined")
class PredefinedSplit:
    """Split windows into train/val/test by trial ID modulo."""
    
    def __init__(self, train_ratio: float = 0.6, val_ratio: float = 0.2):
        """
        Args:
            train_ratio: fraction of windows for training.
            val_ratio: fraction of windows for validation.
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
    
    def __call__(self, windows: list, trial_ids: list) -> dict:
        """Partition windows by trial ID.
        
        Args:
            windows: list of Window objects.
            trial_ids: list of trial indices.
        
        Returns:
            splits: dict with keys "train", "val", "test", each a list of indices.
        """
        n_trials = max(trial_ids) + 1 if trial_ids else 1
        
        train_end = int(n_trials * self.train_ratio)
        val_end = train_end + int(n_trials * self.val_ratio)
        
        splits = {"train": [], "val": [], "test": []}
        for idx, trial_id in enumerate(trial_ids):
            if trial_id < train_end:
                splits["train"].append(idx)
            elif trial_id < val_end:
                splits["val"].append(idx)
            else:
                splits["test"].append(idx)
        
        return splits
