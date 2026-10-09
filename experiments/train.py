"""Main entry point for training experiments."""

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

# Import components so they get registered
from neurofm.data.sources import SyntheticSource, MOABBSource, NeuralBenchSource, EEGFMBenchSource
from neurofm.models.custom import TinyEEGNet
from neurofm.models.external import NeuralBenchWrapper  # noqa: F401
from neurofm.models.adapters import InputAdapters
from neurofm.data.transforms import BandpassFilter, AlignChannels, Resample  # noqa: F401
from neurofm.data.augmentations import GaussianNoise
from neurofm.data.targets import BinaryTarget
from neurofm.data.splits import PredefinedSplit

from neurofm.data.pipeline import Pipeline
from neurofm.data.collate import tensor_collate  # noqa: F401
from neurofm.training import NeuroFMModule, NeuroFMDataModule
from neurofm.registry import models_registry
from neurofm.training.manifest import Manifest


def _to_container(value):
    if value is None:
        return None
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _pipeline_from_cfg(cfg) -> Pipeline:
    source_params = _to_container(cfg.get("source_params")) or {}
    return Pipeline(
        source=cfg.source,
        source_params=source_params,
        window=cfg.get("window", "fixed"),
        spatial_transforms=_to_container(cfg.get("spatial_transforms")) or [],
        temporal_transforms=_to_container(cfg.get("temporal_transforms")) or [],
        target=cfg.get("target", "binary"),
        split=cfg.get("split", "predefined"),
        window_transforms=_to_container(cfg.get("window_transforms")) or [],
        augmentations=_to_container(cfg.get("augmentations")) or [],
        tokenizer=cfg.get("tokenizer", None),
        name=cfg.get("name", None),
        channel_map=_to_container(cfg.get("channel_map")) or {},
        split_params=_to_container(cfg.get("split_params")) or {},
        target_params=_to_container(cfg.get("target_params")) or {},
    )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Train a model using Hydra configuration."""
    
    # Print config
    # print(OmegaConf.to_yaml(cfg))
    
    # Seed
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    
    # Create output directory
    output_dir = Path(cfg.get("output_dir", "outputs")) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline_cfg = cfg.data
    datasets_cfg = pipeline_cfg.get("datasets")
    if datasets_cfg is not None:
        pipelines = [_pipeline_from_cfg(entry) for entry in datasets_cfg]
    else:
        pipelines = _pipeline_from_cfg(pipeline_cfg)

    align_cfg = pipeline_cfg.get("align")
    align_ch_names = None
    if align_cfg is not None:
        align_ch_names = _to_container(align_cfg.get("ch_names"))
    channel_layout = pipeline_cfg.get("channel_layout", "union")
    
    # Create DataModule
    datamodule = NeuroFMDataModule(
        pipeline=pipelines,
        batch_size=cfg.training.get("batch_size", 32),
        num_workers=cfg.training.get("num_workers", 0),
        channel_layout=channel_layout,
        align_ch_names=align_ch_names,
    )
    
    # Build model
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    model_name = model_params.pop("name")
    model_cls = models_registry.get(model_name)
    model = model_cls(**model_params)
    print(f"Loaded model {model_name}, model has: \
          \n{model.n_chans} channels, \
          \n{model.n_times} time points, \
          \n{model.n_outputs} outputs, \
          \n{model.sfreq} Hz sampling rate, \
          \n{model.chs_info} channels info"
        )
    
    # Setup DataModule after the model contract chooses the collate
    datamodule.expected_sfreq = model.sfreq
    datamodule._collate_name = model.input_contract.collate
    datamodule.setup()

    if channel_layout == "adapter":
        model = InputAdapters(model, datamodule.dataset_n_chans)
    
    # Create Lightning Module
    lit_module = NeuroFMModule(
        model=model,
        objective=cfg.get("objective", "classify"),
        learning_rate=cfg.training.get("learning_rate", 1e-3),
    )
    
    # Create Trainer
    trainer = Trainer(
        max_epochs=cfg.training.get("n_epochs", 10),
        logger=CSVLogger(save_dir=str(output_dir), name="logs"),
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    # Train
    trainer.fit(lit_module, datamodule)
    
    # Test
    trainer.test(lit_module, datamodule)
    
    split = (
        pipelines[0].split if isinstance(pipelines, list) else pipelines.split
    )
    # Save manifest
    manifest = Manifest(
        package_versions={
            "torch": torch.__version__,
        },
        pipeline_config=OmegaConf.to_container(pipeline_cfg),
        model_config=OmegaConf.to_container(cfg.model),
        objective=cfg.get("objective", "classify"),
        mode=cfg.get("mode", "experiment"),
        handoff=cfg.get("handoff", "neurofm"),
        split=split,
        seed=seed,
    )
    manifest.save(output_dir / "manifest.json")
    
    print(f"Experiment saved to {output_dir}")


if __name__ == "__main__":
    main()
