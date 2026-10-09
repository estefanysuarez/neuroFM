# neuroFM

A flexible hub for neuroscience foundation-model experiments. Integrates Braindecode, NeuralBench, and EEG-FM-Bench without vendoring them. PyTorch Lightning trains models on EEG, MEG, and related signals.

## Quick Start

```bash
source .venv/bin/activate
pip install -e .
python experiments/train.py experiment=synthetic_eegnet
```

## Architecture

The repo uses one canonical `Recording` format and a fixed pipeline order:

1. **Sources** read data and return `Recording` objects.
2. **Spatial transforms** (pick, montage, rereference) run on the continuous signal, then cache.
3. **Windowing** extracts trials or fixed-size windows.
4. **Temporal transforms** (bandpass, resample) filter each window or the recording.
5. **Targets and splits** assign labels and partition data.
6. **Augmentations** apply random transforms on the train loader only.
7. **Tokenizers** reshape windows into patches (optional; models may tokenize internally).
8. **Collate** builds the batch shape the model contract requires.
9. **Training** uses PyTorch Lightning and the objective (classify, regress, masked prediction).

## How to add a dataset

1. Create a class in `src/neurofm/data/sources/` that returns a `Recording`:
   ```python
   from neurofm.registry import register
   from neurofm.data.recording import Recording
   
   @register("sources", "my_dataset")
   class MyDataset:
       def __init__(self, path: str):
           self.path = path
       
       def __call__(self) -> Recording:
           # Load and return a Recording with signal, sfreq, ch_names, etc.
           pass
   ```

2. Reference it in a config:
   ```yaml
   # configs/data/source/my_dataset.yaml
   source: my_dataset
   source_params:
     path: /data/my_dataset
   ```

## How to add a transform

Transforms can be spatial (recording-level, cacheable) or temporal (continuous or isolated) or window-level.

```python
from neurofm.registry import register
from neurofm.data.recording import Recording, Window

@register("transforms", "my_filter")
class MyFilter:
    """A temporal transform that filters in isolation."""
    
    level = "temporal"  # or "recording" or "window"
    scope = "isolated"  # only for temporal; or "continuous"
    
    def __init__(self, freq_hz: float):
        self.freq_hz = freq_hz
    
    def __call__(self, x: Recording | Window) -> Recording | Window:
        # Apply the transform and return
        return x
```

## How to add an augmentation

Augmentations are random, applied only on the train loader, and operate on windows.

```python
from neurofm.registry import register

@register("augmentations", "my_noise")
class MyNoise:
    def __init__(self, std: float = 0.01):
        self.std = std
    
    def __call__(self, window):
        # Add random noise
        return window
```

## How to add a tokenizer

Tokenizers reshape windows without learned parameters.

```python
from neurofm.registry import register

@register("tokenizers", "patch")
class PatchTokenizer:
    output = "(batch, n_tokens, dim)"  # or "(batch, channels, time)"
    
    def __init__(self, patch_size: int = 200):
        self.patch_size = patch_size
    
    def __call__(self, window):
        # Reshape into patches
        return window
```

## How to add a model

Custom models live in `src/neurofm/models/custom/`:

```python
import torch
from neurofm.registry import register
from neurofm.models.base import InputContract, SignalSpec

@register("models", "my_eegnet")
class MyEEGNet(torch.nn.Module):
    input_contract = InputContract(
        expects=SignalSpec(unit="V", sfreq=None, band_hz=None),
        owned_steps=set(),
        collate="tensor",
        accepts="window",
    )
    
    def __init__(self, n_chans: int, n_times: int, n_outputs: int):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(n_chans, 16, 3, padding=1)
        self.fc = torch.nn.Linear(16 * n_times, n_outputs)
    
    def forward(self, batch):
        x = batch["signal"]  # (batch, channels, time)
        x = self.conv1(x)
        x = x.flatten(1)
        return self.fc(x)
```

## How to run an experiment

Edit or create `configs/experiment/my_exp.yaml`:

```yaml
defaults:
  - override /data/source: synthetic
  - override /model: my_eegnet
  - override /training: default
  - override /objective: classify

mode: experiment
handoff: neurofm

data:
  spatial_transforms: []
  temporal_transforms: ["bandpass"]
  window_type: "fixed"
  window_length_seconds: 4
  target: "binary"
  split: "predefined"
  augmentations: []

model:
  n_chans: 4
  n_times: 1024
  n_outputs: 2

training:
  n_epochs: 10
  batch_size: 32
  learning_rate: 1e-3
```

Then:
```bash
python experiments/train.py experiment=my_exp
```

## Two ways to use external packages

### 1. Inside neuroFM (experiment mode)

This repo's Lightning trainer runs the data pipeline and trains the model. Use this for new architectures, ablations, and custom datasets.

```bash
python experiments/train.py experiment=bcic2a_eegnet mode=experiment
```

### 2. Official benchmark (official mode)

The repo exports recordings and calls the package's official CLI. Preprocessing and training stay on that package, so published results stay comparable.

```bash
python experiments/train.py experiment=bcic2a_neuralbench mode=official
```

## Dependencies

**Core:** `torch`, `lightning`, `torchmetrics`, `hydra-core`, `numpy`.

**Optional:** 
- `braindecode`: for Braindecode models and the Braindecode adapter.
- `neuralbench`: for NeuralBench official benchmarks.
- `eegfm`: for EEG-FM-Bench official benchmarks.

Install with:
```bash
pip install -e ".[braindecode,neuralbench,eegfm]"
```

## References

- [Braindecode](https://braindecode.org/)
- [NeuralBench](https://facebookresearch.github.io/neuroai/neuralbench/)
- [EEG-FM-Bench](https://github.com/xw1216/EEG-FM-Bench)
