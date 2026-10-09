"""Main entry point for training experiments."""

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

# Import components so they get registered
from neurofm.data.sources import SyntheticSource, MOABBSource, NeuralBenchSource, EEGFMBenchSource
from neurofm.models.custom import TinyEEGNet
from neurofm.models.external import NeuralBenchWrapper  # noqa: F401
from neurofm.data.transforms import BandpassFilter
from neurofm.data.augmentations import GaussianNoise
from neurofm.data.targets import BinaryTarget
from neurofm.data.splits import PredefinedSplit

from neurofm.data.pipeline import Pipeline
from neurofm.data.collate import tensor_collate  # noqa: F401
from neurofm.training import NeuroFMModule, NeuroFMDataModule
from neurofm.registry import models_registry
from neurofm.training.manifest import Manifest


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
    
    # Build pipeline
    pipeline_cfg = cfg.data
    source_params = pipeline_cfg.get("source_params")
    if source_params is not None:
        source_params = OmegaConf.to_container(source_params, resolve=True)
    pipeline = Pipeline(
        source=pipeline_cfg.source,
        source_params=source_params or {},
        window=pipeline_cfg.get("window", "fixed"),
        spatial_transforms=pipeline_cfg.get("spatial_transforms", []),
        temporal_transforms=pipeline_cfg.get("temporal_transforms", []),
        target=pipeline_cfg.get("target", "binary"),
        split=pipeline_cfg.get("split", "predefined"),
        window_transforms=pipeline_cfg.get("window_transforms", []),
        augmentations=pipeline_cfg.get("augmentations", []),
        tokenizer=pipeline_cfg.get("tokenizer", None),
    )
    
    # Create DataModule
    datamodule = NeuroFMDataModule(
        pipeline=pipeline,
        batch_size=cfg.training.get("batch_size", 32),
        num_workers=cfg.training.get("num_workers", 0),
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
    
    # Create Lightning Module
    lit_module = NeuroFMModule(
        model=model,
        objective=cfg.get("objective", "classify"),
        learning_rate=cfg.training.get("learning_rate", 1e-3),
    )
    
    # Setup DataModule after the model contract chooses the collate
    datamodule._collate_name = model.input_contract.collate
    datamodule.setup()
    
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
        split=pipeline.split,
        seed=seed,
    )
    manifest.save(output_dir / "manifest.json")
    
    print(f"Experiment saved to {output_dir}")


if __name__ == "__main__":
    main()
