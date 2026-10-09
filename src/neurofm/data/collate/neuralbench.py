"""NeuralBench collate: x, y, subject ids, channel positions."""

import torch
from neurofm.registry import register


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
    
    # Add channel positions if available
    if windows[0].ch_positions is not None:
        ch_positions = torch.from_numpy(windows[0].ch_positions).float()
        batch["ch_positions"] = ch_positions
    
    return batch
