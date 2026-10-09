"""Lightning DataModule for neuroFM experiments."""

import torch
from torch.utils.data import DataLoader, Dataset
from lightning import LightningDataModule
from neurofm.data.pipeline import Pipeline
from neurofm.data.recording import Window
from neurofm.registry import sources_registry, transforms_registry, targets_registry, splits_registry, augmentations_registry, tokenizers_registry, collates_registry


class WindowDataset(Dataset):
    """PyTorch Dataset of Windows."""
    
    def __init__(self, windows: list, augmentations: list = None, tokenizer=None, split: str = "train"):
        self.windows = windows
        self.augmentations = augmentations or []
        self.tokenizer = tokenizer
        self.split = split
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        
        # Apply augmentations only on train split
        if self.split == "train":
            for aug in self.augmentations:
                window = aug(window)
        
        # Tokenize if requested
        if self.tokenizer:
            tokens = self.tokenizer(window)
            window = tokens
        
        return window


class NeuroFMDataModule(LightningDataModule):
    """Lightning DataModule orchestrating the Pipeline."""
    
    def __init__(
        self,
        pipeline: Pipeline,
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_windows = None
        self.val_windows = None
        self.test_windows = None
        self.augmentations = None
        self.tokenizer = None
        self.collate_fn = None
    
    def setup(self, stage: str = None):
        """Load and preprocess data."""
        # Instantiate source
        source_cls = sources_registry.get(self.pipeline.source)
        source = source_cls(**self.pipeline.source_params)
        recording = source()
        
        # Apply spatial transforms
        for t_name in self.pipeline.spatial_transforms:
            t_cls = transforms_registry.get(t_name)
            transform = t_cls()
            recording = transform(recording)
        
        # Create windows
        windows = []
        trial_ids = []
        if self.pipeline.window == "fixed":
            # Extract fixed-size windows
            n_samples = int(4.0 * recording.sfreq)  # 4 second windows
            for start in range(0, recording.signal.shape[1] - n_samples, n_samples):
                signal = recording.signal[:, start : start + n_samples]
                window = Window(
                    signal=signal,
                    sfreq=recording.sfreq,
                    ch_names=recording.ch_names,
                    modality=recording.modality,
                    ch_positions=recording.ch_positions,
                    subject=recording.subject,
                    session=recording.session,
                    dataset=recording.dataset,
                    trial_id=len(windows),
                )
                windows.append(window)
                trial_ids.append(len(windows) - 1)
        
        # Apply temporal transforms
        for t_name in self.pipeline.temporal_transforms:
            t_cls = transforms_registry.get(t_name)
            transform = t_cls()
            windows = [transform(w) for w in windows]
        
        # Assign targets
        target_cls = targets_registry.get(self.pipeline.target)
        target = target_cls()
        for w in windows:
            w.target = target(w.trial_id) if w.trial_id is not None else 0
        
        # Apply split
        split_cls = splits_registry.get(self.pipeline.split)
        split_fn = split_cls()
        split_indices = split_fn(windows, trial_ids)
        
        self.train_windows = [windows[i] for i in split_indices["train"]]
        self.val_windows = [windows[i] for i in split_indices["val"]]
        self.test_windows = [windows[i] for i in split_indices["test"]]
        
        # Instantiate augmentations (train only)
        self.augmentations = []
        for aug_name in self.pipeline.augmentations:
            aug_cls = augmentations_registry.get(aug_name)
            self.augmentations.append(aug_cls())
        
        # Instantiate tokenizer if specified
        if self.pipeline.tokenizer:
            tokenizer_cls = tokenizers_registry.get(self.pipeline.tokenizer)
            self.tokenizer = tokenizer_cls()
        
        # Instantiate collate function
        collate_name = getattr(self, "_collate_name", "tensor")
        collate_fn = collates_registry.get(collate_name)
        self.collate_fn = collate_fn
    
    def train_dataloader(self):
        """Return training DataLoader."""
        dataset = WindowDataset(
            self.train_windows,
            augmentations=self.augmentations,
            tokenizer=self.tokenizer,
            split="train",
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        """Return validation DataLoader."""
        dataset = WindowDataset(
            self.val_windows,
            augmentations=None,
            tokenizer=self.tokenizer,
            split="val",
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def test_dataloader(self):
        """Return test DataLoader."""
        dataset = WindowDataset(
            self.test_windows,
            augmentations=None,
            tokenizer=self.tokenizer,
            split="test",
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
