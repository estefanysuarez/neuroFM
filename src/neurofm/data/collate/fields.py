"""Shared batch fields for collate functions."""

import numpy as np
import torch


def add_channel_fields(batch: dict, windows: list) -> None:
    """Add channel_mask and ch_positions when the windows carry them."""
    masks = []
    has_mask = False
    for window in windows:
        if window.channel_mask is None:
            masks.append(np.ones(window.signal.shape[0], dtype=bool))
        else:
            has_mask = True
            masks.append(np.asarray(window.channel_mask, dtype=bool))
    if has_mask:
        batch["channel_mask"] = torch.from_numpy(np.stack(masks))

    positions = [w.ch_positions for w in windows]
    if all(p is None for p in positions):
        return
    if any(p is None for p in positions):
        return
    first = positions[0]
    same = all(
        p.shape == first.shape and np.allclose(p, first, equal_nan=True)
        for p in positions[1:]
    )
    if same:
        batch["ch_positions"] = torch.from_numpy(np.asarray(first)).float()
    else:
        batch["ch_positions"] = torch.from_numpy(np.stack(positions)).float()


def add_dataset_field(batch: dict, windows: list) -> None:
    """Add a single dataset name when every window in the batch shares it."""
    names = [w.dataset for w in windows]
    if names and names[0] is not None and all(name == names[0] for name in names):
        batch["dataset"] = names[0]
