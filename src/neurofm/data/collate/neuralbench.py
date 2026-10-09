"""NeuralBench collate: x, y, subject ids, channel positions."""

import torch
from neurofm.registry import register
from neurofm.data.collate.fields import add_channel_fields, add_dataset_field


@register("collates", "neuralbench")
def neuralbench_collate(windows: list) -> dict:
    """NeuralBench format for BrainModule.model_forward.
    
    Matches NeuralBench's batch structure with signal, targets, subject IDs,
    and channel positions that BrainModule expects.
    
    Args:
        windows: list of Window objects.
    
    Returns:
        batch dict with x, y, subject_ids, ch_positions.
    """
    signals = []
    targets = []
    subject_ids = []
    
    for w in windows:
        signals.append(torch.from_numpy(w.signal).float())
        targets.append(w.target if w.target is not None else 0)
        subject_ids.append(w.subject or "unknown")
    
    batch = {
        "x": torch.stack(signals),  # (batch, n_chans, n_times)
        "y": torch.tensor(targets, dtype=torch.long),
        "subject_ids": subject_ids,
    }
    add_channel_fields(batch, windows)
    add_dataset_field(batch, windows)
    return batch
