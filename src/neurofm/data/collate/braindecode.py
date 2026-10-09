"""Braindecode collate: (x, y, index) format."""

from neurofm.registry import register


@register("collates", "braindecode")
def braindecode_collate(windows: list) -> tuple:
    """Braindecode format: (x, y, index).
    
    Stub implementation. Full version would integrate with braindecode's own
    data loading if the braindecode extra is installed.
    
    Args:
        windows: list of Window objects.
    
    Returns:
        (x, y, index) where x is (batch, n_chans, n_times), y is targets, index is window indices.
    """
    raise NotImplementedError(
        "braindecode collate requires the braindecode extra: pip install -e '.[braindecode]'"
    )
