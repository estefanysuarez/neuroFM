"""Base classes and contracts for models."""

from dataclasses import dataclass, field
from typing import Optional, Set, Literal
import torch.nn as nn


@dataclass
class SignalSpec:
    """Specification of the expected signal properties."""
    unit: str = "V"  # "V" for volts, "uV" for microvolts
    sfreq: Optional[float] = None  # Sampling frequency; None = any
    band_hz: Optional[tuple] = None  # (low, high) in Hz; None = any


@dataclass
class InputContract:
    """Contract between the model and the data pipeline.
    
    Attributes:
        expects: SignalSpec of the input signal.
        owned_steps: set of transform step names already applied by the model.
        collate: name of the collate function (e.g., "tensor", "braindecode").
        accepts: "window", "patched", or "tokens".
    """
    expects: SignalSpec = field(default_factory=SignalSpec)
    owned_steps: Set[str] = field(default_factory=set)
    collate: str = "tensor"
    accepts: Literal["window", "patched", "tokens"] = "window"


class NeuroModel(nn.Module):
    """Base class for EEG/MEG models.
    
    All models should define:
        input_contract: InputContract describing the expected input.
        n_outputs: number of outputs (classes or regression targets).
        n_chans: number of input channels.
        n_times: number of time samples (or None if variable).
        sfreq: sampling frequency in Hz (or None if specified in input).
    """
    
    input_contract: InputContract = InputContract()
    n_outputs: int = 2
    n_chans: int = None
    n_times: int = None
    sfreq: float = None
    
    def forward(self, batch: dict):
        """Forward pass. Input is a batch dict from the collate function.
        
        Args:
            batch: dict with keys like "signal", "target", etc.
        
        Returns:
            logits or embeddings, shape (batch, n_outputs) or similar.
        """
        raise NotImplementedError
