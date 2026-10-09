"""Data sources: load raw data and return Recording objects."""

from .synthetic import SyntheticSource
from .braindecode import MOABBSource
from .neuralbench import NeuralBenchSource
from .eeg_fm_bench import EEGFMBenchSource

__all__ = ["SyntheticSource", "MOABBSource", "NeuralBenchSource", "EEGFMBenchSource"]
