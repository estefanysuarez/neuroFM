"""Tests for combining datasets with union or adapter layouts."""

import pytest
import torch

import neurofm.plugins  # noqa: F401

from neurofm.data.pipeline import Pipeline
from neurofm.models.adapters import InputAdapters
from neurofm.models.custom import TinyEEGNet
from neurofm.training.datamodule import NeuroFMDataModule


def _pipeline(name, n_channels, channel_map, split_params, sfreq=256.0, **kwargs):
    params = dict(
        source="synthetic",
        source_params={
            "n_channels": n_channels,
            "duration_seconds": 20.0,
            "sfreq": sfreq,
            "n_trials": 4,
        },
        name=name,
        channel_map=channel_map,
        target="binary",
        split="predefined",
        split_params=split_params,
        augmentations=[],
    )
    params.update(kwargs)
    return Pipeline(**params)


PIPE_A = dict(
    name="ds_a",
    n_channels=3,
    channel_map={"Ch0": "C3", "Ch1": "Cz", "Ch2": "C4"},
    split_params={"train_ratio": 0.5, "val_ratio": 0.25},
)
PIPE_B = dict(
    name="ds_b",
    n_channels=3,
    channel_map={"Ch0": "Fz", "Ch1": "Cz", "Ch2": "Pz"},
    split_params={"train_ratio": 0.75, "val_ratio": 0.25},
)


def test_union_layout_zero_fills_and_respects_split_params():
    datamodule = NeuroFMDataModule(
        pipeline=[_pipeline(**PIPE_A), _pipeline(**PIPE_B)],
        batch_size=2,
        channel_layout="union",
    )
    datamodule.setup()

    assert datamodule.resolved_ch_names == ["C3", "Cz", "C4", "Fz", "Pz"]
    assert len(datamodule.train_groups[0]) == 2
    assert len(datamodule.val_groups[0]) == 1
    assert len(datamodule.test_groups[0]) == 1
    assert len(datamodule.train_groups[1]) == 3
    assert len(datamodule.val_groups[1]) == 1
    assert len(datamodule.test_groups[1]) == 0

    for window in datamodule.train_windows:
        assert window.signal.shape == (5, 1024)

    a_window = datamodule.train_groups[0][0]
    b_window = datamodule.train_groups[1][0]
    assert a_window.channel_mask.tolist() == [True, True, True, False, False]
    assert b_window.channel_mask.tolist() == [False, True, False, True, True]

    batch = next(iter(datamodule.train_dataloader()))
    assert batch["signal"].shape[1:] == (5, 1024)
    assert batch["channel_mask"].shape[1] == 5
    assert datamodule.n_chans == 5
    model = TinyEEGNet(n_chans=datamodule.n_chans, n_times=1024, n_outputs=2)
    logits = model(batch)
    assert logits.shape == (batch["signal"].shape[0], 2)


def test_union_align_ch_names_drops_channels_outside_the_list():
    datamodule = NeuroFMDataModule(
        pipeline=[_pipeline(**PIPE_A), _pipeline(**PIPE_B)],
        batch_size=2,
        channel_layout="union",
        align_ch_names=["C3", "Cz", "C4"],
    )
    datamodule.setup()
    assert datamodule.resolved_ch_names == ["C3", "Cz", "C4"]
    assert datamodule.n_chans == 3
    a_window = datamodule.train_groups[0][0]
    b_window = datamodule.train_groups[1][0]
    assert a_window.signal.shape[0] == 3
    assert a_window.channel_mask.tolist() == [True, True, True]
    assert b_window.channel_mask.tolist() == [False, True, False]


def test_adapter_keeps_native_shapes_and_single_dataset_batches():
    datamodule = NeuroFMDataModule(
        pipeline=[_pipeline(**PIPE_A), _pipeline(**PIPE_B)],
        batch_size=8,
        channel_layout="adapter",
    )
    datamodule.setup()
    assert datamodule.dataset_n_chans == {"ds_a": 3, "ds_b": 3}
    assert datamodule.n_chans is None
    assert datamodule.train_groups[0][0].signal.shape[0] == 3
    assert datamodule.train_groups[1][0].signal.shape[0] == 3

    seen = set()
    for batch in datamodule.train_dataloader():
        assert batch["signal"].shape[1] == 3
        assert isinstance(batch["dataset"], str)
        seen.add(batch["dataset"])
    assert seen == {"ds_a", "ds_b"}

    model = TinyEEGNet(n_chans=4, n_times=1024, n_outputs=2)
    wrapped = InputAdapters(model, datamodule.dataset_n_chans)
    assert any("adapters" in name for name, _ in wrapped.named_parameters())
    batch = next(iter(datamodule.train_dataloader()))
    logits = wrapped(batch)
    assert logits.shape == (batch["signal"].shape[0], 2)


def test_adapter_rejects_align_ch_names():
    with pytest.raises(ValueError, match="align.ch_names"):
        NeuroFMDataModule(
            pipeline=[_pipeline(**PIPE_A), _pipeline(**PIPE_B)],
            channel_layout="adapter",
            align_ch_names=["C3", "Cz"],
        )


def test_adapter_rejects_duplicate_dataset_names():
    datamodule = NeuroFMDataModule(
        pipeline=[
            _pipeline(name=None, n_channels=3, channel_map={}, split_params={}),
            _pipeline(name=None, n_channels=4, channel_map={}, split_params={}),
        ],
        channel_layout="adapter",
    )
    with pytest.raises(ValueError, match="unique dataset names"):
        datamodule.setup()


def test_resample_to_model_sfreq_warns_and_continues():
    datamodule = NeuroFMDataModule(
        pipeline=_pipeline(
            **PIPE_A,
            temporal_transforms=[{"name": "resample", "sfreq": 200}],
        ),
        expected_sfreq=200.0,
    )
    with pytest.warns(UserWarning, match="resampled to 200"):
        datamodule.setup()
    assert datamodule.train_groups[0][0].sfreq == 200.0


def test_recorded_sfreq_must_match_model_without_resample():
    datamodule = NeuroFMDataModule(
        pipeline=_pipeline(**PIPE_A),
        expected_sfreq=200.0,
    )
    with pytest.raises(ValueError, match="expects 200"):
        datamodule.setup()


def test_resample_to_other_rate_still_errors():
    datamodule = NeuroFMDataModule(
        pipeline=_pipeline(
            **PIPE_A,
            temporal_transforms=[{"name": "resample", "sfreq": 128}],
        ),
        expected_sfreq=200.0,
    )
    with pytest.raises(ValueError, match="expects 200"):
        datamodule.setup()


def test_datasets_must_share_recorded_sfreq_when_model_has_none():
    datamodule = NeuroFMDataModule(
        pipeline=[
            _pipeline(**PIPE_A, sfreq=256.0),
            _pipeline(**PIPE_B, sfreq=128.0),
        ],
        expected_sfreq=None,
    )
    with pytest.raises(ValueError, match="Recorded rates must match"):
        datamodule.setup()


def test_window_lengths_must_match_after_transforms():
    datamodule = NeuroFMDataModule(
        pipeline=[
            _pipeline(**PIPE_A),
            _pipeline(**PIPE_B, spatial_transforms=[{"name": "resample", "sfreq": 128}]),
        ],
        expected_sfreq=None,
    )
    with pytest.raises(ValueError, match="Window time lengths differ"):
        datamodule.setup()


def test_setup_is_cached():
    datamodule = NeuroFMDataModule(pipeline=_pipeline(**PIPE_A), batch_size=2)
    datamodule.setup()
    first = datamodule.train_windows
    datamodule.setup()
    assert datamodule.train_windows is first
