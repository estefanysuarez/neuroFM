"""Tensor collate: standard dict format for custom models."""

import torch
from neurofm.registry import register
from neurofm.data.collate.fields import add_channel_fields, add_dataset_field


@register("collates", "tensor")
def tensor_collate(windows: list) -> dict:
    """Convert list of Windows to a batch dict.
    
    Args:
        windows: list of Window objects.
    
    Returns:
        batch: dict with keys "signal", "target", etc.
    """
    signals = []
    targets = []
    
    for w in windows:
        signals.append(torch.from_numpy(w.signal).float())
        targets.append(w.target if w.target is not None else 0)
    
    batch = {
        "signal": torch.stack(signals),  # (batch, n_chans, n_times)
        "target": torch.tensor(targets, dtype=torch.long),
    }
    add_channel_fields(batch, windows)
    add_dataset_field(batch, windows)
    return batch
