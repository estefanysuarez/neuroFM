"""Per-dataset input adapters onto a shared model channel count."""

from typing import Mapping

import torch
import torch.nn as nn


class InputAdapters(nn.Module):
    """Map each dataset's native channel count onto ``model.n_chans``."""

    def __init__(self, model: nn.Module, dataset_n_chans: Mapping[str, int]):
        super().__init__()
        if getattr(model, "n_chans", None) is None:
            raise ValueError("The wrapped model must define n_chans.")
        n_out = int(model.n_chans)
        if not dataset_n_chans:
            raise ValueError("dataset_n_chans must map each dataset to its channel count.")
        self.model = model
        self._name_to_key = {
            name: name.replace(".", "_") for name in dataset_n_chans
        }
        self.adapters = nn.ModuleDict(
            {
                self._name_to_key[name]: nn.Conv1d(n_in, n_out, kernel_size=1)
                for name, n_in in dataset_n_chans.items()
            }
        )

    @property
    def n_chans(self):
        return self.model.n_chans

    @property
    def n_outputs(self):
        return self.model.n_outputs

    @property
    def n_times(self):
        return self.model.n_times

    @property
    def sfreq(self):
        return self.model.sfreq

    @property
    def input_contract(self):
        return self.model.input_contract

    def forward(self, batch: dict) -> torch.Tensor:
        dataset = batch.get("dataset")
        if isinstance(dataset, (list, tuple)):
            dataset = dataset[0]
        if dataset is None:
            raise ValueError(
                "Adapter batches must include 'dataset' so the matching "
                "input projection can be selected."
            )
        key = self._name_to_key.get(dataset)
        if key is None or key not in self.adapters:
            raise KeyError(
                f"No input adapter for dataset '{dataset}'. "
                f"Known: {list(self._name_to_key)}"
            )
        signal_key = "signal" if "signal" in batch else "x"
        if signal_key not in batch:
            raise ValueError(
                f"Batch must contain 'signal' or 'x'. Got: {list(batch.keys())}"
            )
        projected = dict(batch)
        projected[signal_key] = self.adapters[key](batch[signal_key])
        return self.model(projected)
