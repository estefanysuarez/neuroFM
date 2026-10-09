"""Data sources: load raw data and return Recording objects."""

import importlib.util

from .eeg_fm_bench import EEGFMBenchSource
from .neuralbench import NeuralBenchSource
from .synthetic import SyntheticSource

__all__ = ["SyntheticSource", "NeuralBenchSource", "EEGFMBenchSource"]

# braindecode is an optional extra. Importing this package still registers
# the sources that ship with the core dependencies.
if importlib.util.find_spec("braindecode") is not None:
    from .braindecode import MOABBSource

    __all__ += ["MOABBSource"]
