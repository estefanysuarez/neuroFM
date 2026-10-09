"""Main entry point for training experiments."""

from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import torch
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

import neurofm.plugins  # noqa: F401  registers sources, transforms, models, collates

from neurofm.models.adapters import InputAdapters
from neurofm.data.pipeline import Pipeline
from neurofm.training import NeuroFMModule, NeuroFMDataModule
from neurofm.registry import models_registry
from neurofm.training.manifest import Manifest


def _to_container(value):
    """Turn a Hydra node into a plain dict/list the rest of the code can use."""
    if value is None:
        return None
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _expected_sfreq(model_cls, model_params: dict):
    """Sampling rate the model expects, without constructing it.

    Checked during data setup so a recording is resampled (or rejected)
    before any window is built. Order: explicit ``sfreq``, then the
    NeuralBench architecture table, then the model's input contract.
    """
    if model_params.get("sfreq") is not None:
        return model_params["sfreq"]
    architecture = model_params.get("architecture")
    if architecture:
        from neurofm.models.external.neuralbench import resolve_architecture
        return resolve_architecture(architecture).sfreq
    expects = getattr(getattr(model_cls, "input_contract", None), "expects", None)
    return getattr(expects, "sfreq", None)


def _pipeline_from_cfg(cfg) -> Pipeline:
    """One dataset's load → transform → target → split recipe."""
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
    """Train one model from the composed Hydra config.

    Order matters: data is set up before the model is built, because a
    union montage decides ``n_chans``, and the model's collate function
    is only known after the model exists.
    """

    # 1. Reproducibility and where this run writes logs and the manifest.
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    output_dir = Path(cfg.get("output_dir", "outputs")) / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. One Pipeline per dataset. ``data.datasets`` is the multi-dataset
    # case; a config without it is a single pipeline rooted at ``cfg.data``.
    pipeline_cfg = cfg.data
    datasets_cfg = pipeline_cfg.get("datasets")
    if datasets_cfg is not None:
        pipelines = [_pipeline_from_cfg(entry) for entry in datasets_cfg]
    else:
        pipelines = _pipeline_from_cfg(pipeline_cfg)

    # 3. How channels are shared across datasets.
    #    union: pad/reorder every recording onto one electrode list.
    #    adapter: keep native channel counts and project them in the model.
    # ``align.ch_names`` fixes that list; otherwise the union is discovered.
    align_cfg = pipeline_cfg.get("align")
    align_ch_names = None
    if align_cfg is not None:
        align_ch_names = _to_container(align_cfg.get("ch_names"))
    channel_layout = pipeline_cfg.get("channel_layout", "union")

    datamodule = NeuroFMDataModule(
        pipeline=pipelines,
        batch_size=cfg.training.get("batch_size", 32),
        num_workers=cfg.training.get("num_workers", 0),
        channel_layout=channel_layout,
        align_ch_names=align_ch_names,
    )

    # 4. Resolve the model class, but do not construct it yet.
    model_params = OmegaConf.to_container(cfg.model, resolve=True)
    model_name = model_params.pop("name")
    model_cls = models_registry.get(model_name)

    # 5. Load and preprocess now. This resolves the union montage (or each
    # dataset's channel count) and checks sampling rates against the model.
    datamodule.expected_sfreq = _expected_sfreq(model_cls, model_params)
    datamodule.setup()

    # 6. Union layout: the model must have exactly as many channels as the
    # shared montage, so a configured n_chans that disagrees is replaced.
    if channel_layout == "union":
        n_chans = datamodule.n_chans
        if n_chans is None:
            raise RuntimeError("Union layout did not resolve a channel list.")
        configured = model_params.get("n_chans")
        if configured is not None and int(configured) != n_chans:
            print(
                f"Overriding model.n_chans={configured} with union length "
                f"{n_chans}: {datamodule.resolved_ch_names}"
            )
        else:
            print(
                f"Set model.n_chans={n_chans} from union "
                f"{datamodule.resolved_ch_names}"
            )
        model_params["n_chans"] = n_chans

    model = model_cls(**model_params)
    print(f"Loaded model {model_name}, model has: \
          \n{model.n_chans} channels, \
          \n{model.n_times} time points, \
          \n{model.n_outputs} outputs, \
          \n{model.sfreq} Hz sampling rate, \
          \n{model.chs_info} channels info"
        )

    # 7. Batch assembly follows the model (tensor vs. the model's own format).
    datamodule.set_collate(model.input_contract.collate)

    # 8. Adapter layout: a 1x1 conv per dataset maps its channels onto the model.
    if channel_layout == "adapter":
        model = InputAdapters(model, datamodule.dataset_n_chans)

    # 9. Lightning owns the loss, optimizer, and the fit/test loop.
    lit_module = NeuroFMModule(
        model=model,
        objective=cfg.get("objective", "classify"),
        learning_rate=cfg.training.get("learning_rate", 1e-3),
    )
    trainer = Trainer(
        max_epochs=cfg.training.get("n_epochs", 10),
        logger=CSVLogger(save_dir=str(output_dir), name="logs"),
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    trainer.fit(lit_module, datamodule)
    trainer.test(lit_module, datamodule)

    # 10. Record what was trained so the run can be reproduced later.
    # The manifest stores the first pipeline's split name only.
    split = (
        pipelines[0].split if isinstance(pipelines, list) else pipelines.split
    )
    manifest = Manifest(
        package_versions={
            "torch": torch.__version__,
        },
        pipeline_config=OmegaConf.to_container(pipeline_cfg),
        model_config={
            **OmegaConf.to_container(cfg.model),
            "n_chans": model.n_chans,
        },
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
