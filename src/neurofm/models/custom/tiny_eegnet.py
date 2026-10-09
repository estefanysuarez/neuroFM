"""A tiny EEGNet-style model for testing."""

import torch
import torch.nn as nn
from neurofm.registry import register
from neurofm.models.base import NeuroModel, InputContract, SignalSpec


@register("models", "tiny_eegnet")
class TinyEEGNet(NeuroModel):
    """Tiny EEGNet variant: one temporal conv + one spatial conv + linear head."""
    
    input_contract = InputContract(
        expects=SignalSpec(unit="V", sfreq=None, band_hz=None),
        owned_steps=set(),
        collate="tensor",
        accepts="window",
    )
    
    def __init__(self, n_chans: int = 4, n_times: int = 1024, n_outputs: int = 2):
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.n_outputs = n_outputs
        
        # Temporal conv: (batch, 1, n_chans, n_times) -> (batch, 8, n_chans, n_times)
        self.temporal = nn.Conv2d(1, 8, kernel_size=(1, 5), padding=(0, 2))
        
        # Spatial conv: (batch, 8, n_chans, n_times) -> (batch, 16, 1, n_times)
        self.spatial = nn.Conv2d(8, 16, kernel_size=(n_chans, 1))
        
        # Pooling and flattening
        self.pool = nn.AvgPool2d(kernel_size=(1, 4))
        
        # Compute size after pooling
        pooled_time = n_times // 4
        self.fc_in = 16 * pooled_time
        
        # Classifier
        self.fc = nn.Linear(self.fc_in, n_outputs)
    
    def forward(self, batch: dict) -> torch.Tensor:
        """Forward pass.
        
        Args:
            batch: dict with "signal" of shape (batch, n_chans, n_times).
        
        Returns:
            logits of shape (batch, n_outputs).
        """
        x = batch["signal"]  # (batch, n_chans, n_times)
        
        # Add channel dim for conv2d: (batch, 1, n_chans, n_times)
        x = x.unsqueeze(1)
        
        # Temporal conv
        x = self.temporal(x)
        x = torch.relu(x)
        
        # Spatial conv
        x = self.spatial(x)
        x = torch.relu(x)
        
        # Pool
        x = self.pool(x)
        
        # Flatten
        x = x.flatten(1)
        
        # Classify
        logits = self.fc(x)
        return logits
