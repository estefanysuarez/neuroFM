"""Collate functions: convert Windows to batch dicts for models."""

from .tensor import tensor_collate
from .braindecode import braindecode_collate
from .neuralbench import neuralbench_collate
from .eeg_fm_bench import eeg_fm_bench_collate

__all__ = [
    "tensor_collate",
    "braindecode_collate",
    "neuralbench_collate",
    "eeg_fm_bench_collate",
]
