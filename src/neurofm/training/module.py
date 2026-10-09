"""Lightning Module for training neuroFM models."""

import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics.classification import Accuracy


class NeuroFMModule(LightningModule):
    """Lightning Module wrapping a NeuroModel with an objective."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        objective: str = "classify",
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.objective = objective
        self.learning_rate = learning_rate
        
        # Metrics
        self.train_acc = Accuracy(task="binary" if model.n_outputs == 2 else "multiclass", num_classes=model.n_outputs)
        self.val_acc = Accuracy(task="binary" if model.n_outputs == 2 else "multiclass", num_classes=model.n_outputs)
    
    def forward(self, batch):
        """Forward pass."""
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        logits = self(batch)
        # Handle both tensor collate ("target") and neuralbench collate ("y")
        targets = batch.get("target", batch.get("y"))
        
        if self.objective == "classify":
            loss = F.cross_entropy(logits, targets)
        else:
            loss = F.mse_loss(logits, targets.unsqueeze(-1).float())
        
        # Log metrics
        if self.objective == "classify":
            acc = self.train_acc(logits.argmax(dim=1), targets)
            self.log("train/acc", acc, on_step=False, on_epoch=True)
        
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        logits = self(batch)
        # Handle both tensor collate ("target") and neuralbench collate ("y")
        targets = batch.get("target", batch.get("y"))
        
        if self.objective == "classify":
            loss = F.cross_entropy(logits, targets)
            acc = self.val_acc(logits.argmax(dim=1), targets)
            self.log("val/acc", acc)
        else:
            loss = F.mse_loss(logits, targets.unsqueeze(-1).float())
        
        self.log("val/loss", loss)
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        logits = self(batch)
        # Handle both tensor collate ("target") and neuralbench collate ("y")
        targets = batch.get("target", batch.get("y"))
        
        if self.objective == "classify":
            loss = F.cross_entropy(logits, targets)
        else:
            loss = F.mse_loss(logits, targets.unsqueeze(-1).float())
        
        self.log("test/loss", loss)
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
