"""EEG-FM-Bench collate: sample dict compatible with AbstractDatasetAdapter."""

from neurofm.registry import register


@register("collates", "eeg_fm_bench")
def eeg_fm_bench_collate(windows: list) -> dict:
    """EEG-FM-Bench format for AbstractDatasetAdapter._process_sample.
    
    Stub implementation. Full version would match EEG-FM-Bench's sample structure.
    
    Args:
        windows: list of Window objects.
    
    Returns:
        batch dict with tensor, ch_names, label, ids, etc.
    """
    raise NotImplementedError(
        "eeg_fm_bench collate requires the eegfm extra: pip install -e '.[eegfm]'"
    )
