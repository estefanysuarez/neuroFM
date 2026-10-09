"""Lightning DataModule for neuroFM experiments."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Sampler

from neurofm.data.pipeline import Pipeline
from neurofm.data.recording import Recording, Window
from neurofm.data.transforms.channels import AlignChannels, apply_channel_map, union_channel_names
from neurofm.registry import (
    augmentations_registry,
    collates_registry,
    sources_registry,
    splits_registry,
    targets_registry,
    tokenizers_registry,
    transforms_registry,
)

log = logging.getLogger(__name__)

WINDOW_SECONDS = 4.0


def instantiate_registered(registry, spec, extra_kwargs: Optional[dict] = None):
    """Build a registered component from a name or ``{name, ...kwargs}`` mapping."""
    extra_kwargs = extra_kwargs or {}
    if isinstance(spec, str):
        name, kwargs = spec, {}
    else:
        spec = dict(spec)
        name = spec.pop("name")
        kwargs = spec
    return registry.get(name)(**{**kwargs, **extra_kwargs})


def instantiate_many(registry, specs: Iterable) -> list:
    return [instantiate_registered(registry, spec) for spec in specs]


def _resample_target(pipeline: Pipeline) -> Optional[float]:
    """Sampling rate this pipeline resamples to, if it resamples at all.

    Resample is a temporal transform. A spatial resample is still honored,
    and a later resample replaces an earlier one.
    """
    target = None
    for spec in list(pipeline.spatial_transforms) + list(pipeline.temporal_transforms):
        if isinstance(spec, str) or not isinstance(spec, Mapping):
            continue
        if spec.get("name") != "resample" or spec.get("sfreq") is None:
            continue
        target = float(spec["sfreq"])
    return target


class WindowDataset(Dataset):
    """PyTorch Dataset of Windows.

    Stored windows may share a view into one recording. ``__getitem__`` copies
    the signal so train augmentations cannot mutate that recording.
    """

    def __init__(
        self,
        windows: list,
        augmentations: list = None,
        tokenizer=None,
        split: str = "train",
        window_transforms: list = None,
    ):
        self.windows = windows
        self.augmentations = augmentations or []
        self.window_transforms = window_transforms or []
        self.tokenizer = tokenizer
        self.split = split

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx].copy()

        for transform in self.window_transforms:
            window = transform(window)

        if self.split == "train":
            for aug in self.augmentations:
                window = aug(window)

        if self.tokenizer:
            window = self.tokenizer(window)

        return window


class DatasetPureBatchSampler(Sampler[List[int]]):
    """Yield batches whose indices all belong to one concatenated dataset."""

    def __init__(
        self,
        dataset_sizes: Sequence[int],
        batch_size: int,
        shuffle: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        self.dataset_sizes = list(dataset_sizes)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator

    def __iter__(self):
        offset = 0
        batches: List[List[int]] = []
        for size in self.dataset_sizes:
            indices = list(range(offset, offset + size))
            if self.shuffle:
                perm = torch.randperm(size, generator=self.generator).tolist()
                indices = [indices[i] for i in perm]
            for start in range(0, len(indices), self.batch_size):
                batches.append(indices[start : start + self.batch_size])
            offset += size
        if self.shuffle:
            order = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[i] for i in order]
        yield from batches

    def __len__(self):
        return sum(math.ceil(size / self.batch_size) for size in self.dataset_sizes if size)


class NeuroFMDataModule(LightningDataModule):
    """Lightning DataModule orchestrating one or more Pipelines."""

    def __init__(
        self,
        pipeline: Union[Pipeline, Sequence[Pipeline]],
        batch_size: int = 32,
        num_workers: int = 0,
        channel_layout: str = "union",
        align_ch_names: Optional[List[str]] = None,
        expected_sfreq: Optional[float] = None,
        collate_name: str = "tensor",
    ):
        super().__init__()
        if channel_layout not in {"union", "adapter"}:
            raise ValueError(
                f"channel_layout must be 'union' or 'adapter', got {channel_layout!r}"
            )
        if channel_layout == "adapter" and align_ch_names is not None:
            raise ValueError("align.ch_names is only valid with channel_layout='union'")
        if isinstance(pipeline, Pipeline):
            self.pipelines = [pipeline]
        else:
            self.pipelines = list(pipeline)
        self.pipeline = self.pipelines[0]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.channel_layout = channel_layout
        self.align_ch_names = list(align_ch_names) if align_ch_names is not None else None
        self.expected_sfreq = expected_sfreq
        self._collate_name = collate_name

        self.train_windows = None
        self.val_windows = None
        self.test_windows = None
        self.train_groups: List[list] = []
        self.val_groups: List[list] = []
        self.test_groups: List[list] = []
        self._train_datasets: List[WindowDataset] = []
        self._val_datasets: List[WindowDataset] = []
        self._test_datasets: List[WindowDataset] = []
        self.augmentations = None
        self.tokenizer = None
        self.collate_fn = None
        self.resolved_ch_names: Optional[List[str]] = None
        self.dataset_n_chans: Dict[str, int] = {}
        self._ready = False

    @property
    def n_chans(self) -> Optional[int]:
        """Channel count of the union layout after setup; None in adapter mode."""
        if self.channel_layout != "union" or self.resolved_ch_names is None:
            return None
        return len(self.resolved_ch_names)

    def set_collate(self, name: str) -> None:
        """Bind the collate the model contract selected."""
        self._collate_name = name
        self.collate_fn = collates_registry.get(name)

    def setup(self, stage: str = None):
        """Load and preprocess data."""
        if self._ready:
            return

        prepared = []
        recorded_rates = []
        for pipeline in self.pipelines:
            recording = self._load_recording(pipeline)
            dataset_name = pipeline.name or recording.dataset or pipeline.source
            recording.dataset = dataset_name
            self._check_recorded_sfreq(recording, dataset_name, pipeline)
            recorded_rates.append((dataset_name, float(recording.sfreq)))
            recording = self._apply_transforms(recording, pipeline.spatial_transforms)
            recording = apply_channel_map(recording, pipeline.channel_map)
            prepared.append((pipeline, recording, dataset_name))

        self._check_cross_dataset_sfreq(recorded_rates)

        if self.channel_layout == "union":
            self.resolved_ch_names = self.align_ch_names or union_channel_names(
                [recording for _, recording, _ in prepared]
            )
            log.info("Resolved channel layout: %s", self.resolved_ch_names)
            aligned = []
            for pipeline, recording, dataset_name in prepared:
                recording = AlignChannels(self.resolved_ch_names)(recording)
                aligned.append((pipeline, recording, dataset_name))
            prepared = aligned
        else:
            names = [name for _, _, name in prepared]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(
                    "channel_layout='adapter' requires unique dataset names, "
                    f"got duplicates: {duplicates}"
                )
            self.dataset_n_chans = {
                name: recording.signal.shape[0] for _, recording, name in prepared
            }
            log.info("Dataset channel counts: %s", self.dataset_n_chans)

        self.train_groups = []
        self.val_groups = []
        self.test_groups = []
        self._train_datasets = []
        self._val_datasets = []
        self._test_datasets = []
        window_lengths: Dict[str, int] = {}

        for pipeline, recording, dataset_name in prepared:
            recording = self._apply_transforms(recording, pipeline.temporal_transforms)
            windows, trial_ids = self._fixed_windows(recording, pipeline.window)
            n_times = windows[0].signal.shape[1] if windows else 0
            window_lengths[dataset_name] = n_times

            target = instantiate_registered(
                targets_registry, pipeline.target, pipeline.target_params
            )
            for window in windows:
                window.target = target(window.trial_id) if window.trial_id is not None else 0

            split_fn = instantiate_registered(
                splits_registry, pipeline.split, pipeline.split_params
            )
            split_indices = split_fn(windows, trial_ids)
            train_windows = [windows[i] for i in split_indices["train"]]
            val_windows = [windows[i] for i in split_indices["val"]]
            test_windows = [windows[i] for i in split_indices["test"]]
            self.train_groups.append(train_windows)
            self.val_groups.append(val_windows)
            self.test_groups.append(test_windows)

            augmentations = instantiate_many(augmentations_registry, pipeline.augmentations)
            window_transforms = instantiate_many(
                transforms_registry, pipeline.window_transforms
            )
            tokenizer = None
            if pipeline.tokenizer:
                tokenizer = instantiate_registered(tokenizers_registry, pipeline.tokenizer)
            self._train_datasets.append(
                WindowDataset(
                    train_windows,
                    augmentations=augmentations,
                    tokenizer=tokenizer,
                    split="train",
                    window_transforms=window_transforms,
                )
            )
            self._val_datasets.append(
                WindowDataset(
                    val_windows,
                    augmentations=None,
                    tokenizer=tokenizer,
                    split="val",
                    window_transforms=window_transforms,
                )
            )
            self._test_datasets.append(
                WindowDataset(
                    test_windows,
                    augmentations=None,
                    tokenizer=tokenizer,
                    split="test",
                    window_transforms=window_transforms,
                )
            )

        unique_lengths = set(window_lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(
                "Window time lengths differ across datasets: "
                f"{window_lengths}. Resample each dataset to the same rate."
            )

        self.train_windows = [w for group in self.train_groups for w in group]
        self.val_windows = [w for group in self.val_groups for w in group]
        self.test_windows = [w for group in self.test_groups for w in group]

        self.augmentations = (
            self._train_datasets[0].augmentations if self._train_datasets else []
        )
        self.tokenizer = self._train_datasets[0].tokenizer if self._train_datasets else None

        collate_name = getattr(self, "_collate_name", "tensor")
        self.collate_fn = collates_registry.get(collate_name)
        self._ready = True

    def _load_recording(self, pipeline: Pipeline) -> Recording:
        source_cls = sources_registry.get(pipeline.source)
        source = source_cls(**pipeline.source_params)
        return source()

    def _check_recorded_sfreq(
        self, recording: Recording, dataset_name: str, pipeline: Pipeline
    ) -> None:
        """Compare the recorded rate with the model rate.

        A mismatch is an error unless this dataset is resampled to the model
        rate, in which case it is a warning. The check uses the original
        recording rate, before transforms run.
        """
        if self.expected_sfreq is None:
            return
        if np.isclose(recording.sfreq, self.expected_sfreq):
            return
        target = _resample_target(pipeline)
        if target is not None and np.isclose(target, self.expected_sfreq):
            warnings.warn(
                f"Dataset '{dataset_name}' is recorded at {recording.sfreq} Hz, "
                f"but the model expects {self.expected_sfreq} Hz. "
                f"It will be resampled to {target} Hz.",
                UserWarning,
                stacklevel=2,
            )
            return
        raise ValueError(
            f"Dataset '{dataset_name}' has sampling rate {recording.sfreq} Hz, "
            f"but the model expects {self.expected_sfreq} Hz. "
            "Resample this dataset to the model sampling rate, or record it "
            "at that rate."
        )

    def _check_cross_dataset_sfreq(self, recorded_rates: List[tuple]) -> None:
        if self.expected_sfreq is not None or len(recorded_rates) < 2:
            return
        ref_name, ref_rate = recorded_rates[0]
        for name, rate in recorded_rates[1:]:
            if not np.isclose(rate, ref_rate):
                raise ValueError(
                    f"Dataset '{name}' has sampling rate {rate} Hz, but "
                    f"'{ref_name}' has {ref_rate} Hz. Recorded rates must match "
                    "when the model does not set sfreq."
                )

    def _apply_transforms(self, recording: Recording, specs: Iterable) -> Recording:
        for spec in specs:
            transform = instantiate_registered(transforms_registry, spec)
            recording = transform(recording)
        return recording

    def _fixed_windows(self, recording: Recording, window: str) -> tuple[list, list]:
        windows = []
        trial_ids = []
        if window != "fixed":
            return windows, trial_ids
        n_samples = int(WINDOW_SECONDS * recording.sfreq)
        if n_samples <= 0 or recording.signal.shape[1] < n_samples:
            return windows, trial_ids
        for start in range(0, recording.signal.shape[1] - n_samples, n_samples):
            signal = recording.signal[:, start : start + n_samples]
            item = Window(
                signal=signal,
                sfreq=recording.sfreq,
                ch_names=recording.ch_names,
                modality=recording.modality,
                ch_positions=recording.ch_positions,
                subject=recording.subject,
                session=recording.session,
                dataset=recording.dataset,
                channel_mask=recording.channel_mask,
                trial_id=len(windows),
            )
            windows.append(item)
            trial_ids.append(len(windows) - 1)
        return windows, trial_ids

    def _dataloader_kwargs(self) -> dict:
        kwargs: Dict[str, Any] = {
            "num_workers": self.num_workers,
            "collate_fn": self.collate_fn,
            "pin_memory": torch.cuda.is_available(),
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    def _concat(self, datasets: List[WindowDataset]) -> Dataset:
        if len(datasets) == 1:
            return datasets[0]
        return ConcatDataset(datasets)

    def train_dataloader(self):
        """Return training DataLoader."""
        dataset = self._concat(self._train_datasets)
        kwargs = self._dataloader_kwargs()
        if self.channel_layout == "adapter":
            sizes = [len(ds) for ds in self._train_datasets]
            return DataLoader(
                dataset,
                batch_sampler=DatasetPureBatchSampler(
                    sizes, self.batch_size, shuffle=True
                ),
                **kwargs,
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **kwargs,
        )

    def val_dataloader(self):
        """Return validation DataLoader."""
        dataset = self._concat(self._val_datasets)
        kwargs = self._dataloader_kwargs()
        if self.channel_layout == "adapter":
            sizes = [len(ds) for ds in self._val_datasets]
            return DataLoader(
                dataset,
                batch_sampler=DatasetPureBatchSampler(
                    sizes, self.batch_size, shuffle=False
                ),
                **kwargs,
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **kwargs,
        )

    def test_dataloader(self):
        """Return test DataLoader."""
        dataset = self._concat(self._test_datasets)
        kwargs = self._dataloader_kwargs()
        if self.channel_layout == "adapter":
            sizes = [len(ds) for ds in self._test_datasets]
            return DataLoader(
                dataset,
                batch_sampler=DatasetPureBatchSampler(
                    sizes, self.batch_size, shuffle=False
                ),
                **kwargs,
            )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            **kwargs,
        )
