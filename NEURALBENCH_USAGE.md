# NeuralBench models in neuroFM

One wrapper, `neuralbench`, loads any NeuralBench/braindecode architecture. Collate is `neuralbench` when that extra is installed, otherwise `tensor` (with a warning).

```bash
pip install -e ".[neuralbench]"

# EEGNet, REVE, or BIOT — same wrapper, different config
python experiments/train.py experiment=neuralbench_eegnet
python experiments/train.py experiment=neuralbench_reve
python experiments/train.py experiment=neuralbench_biot

# or override the architecture directly
python experiments/train.py model=neuralbench_eegnet data=synthetic
```

## Config

```yaml
# configs/model/neuralbench_eegnet.yaml
name: neuralbench
architecture: EEGNet   # or REVE, BIOT, Deep4Net, ...
n_chans: 4
n_times: 1024
n_outputs: 2
from_pretrained: false
```

`architecture` is a NeuralBench alias (`eegnet`, `reve`, `biot`) or a braindecode class name (`EEGNet`, `REVE`, `BIOT`, `Deep4Net`, ...). Extra constructor arguments can be passed as `kwargs` or as extra YAML keys.

Set `from_pretrained: true` to load the NeuralBench default checkpoint (`brain-bzh/reve-base` for REVE, `braindecode/biot-pretrained-six-datasets-18chs` for BIOT), or pass a HuggingFace id as a string.

`n_chans` and `n_times` must match the data windows (synthetic data uses 4 channels and 4 s windows).

## Official NeuralBench CLI

For published benchmark numbers, use NeuralBench's own trainer:

```bash
neuralbench eeg motor_imagery -m reve -w linear_probe_flatten
```
