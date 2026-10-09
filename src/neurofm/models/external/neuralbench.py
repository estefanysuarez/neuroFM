"""Generic wrapper for NeuralBench / braindecode models."""

from __future__ import annotations

import inspect
import json
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from neurofm.models.base import InputContract, NeuroModel, SignalSpec
from neurofm.registry import register

try:
    import neuralbench  # noqa: F401

    NEURALBENCH_AVAILABLE = True
except ImportError:
    NEURALBENCH_AVAILABLE = False


@dataclass(frozen=True)
class ArchitectureSpec:
    """NeuralBench defaults for a named architecture."""

    class_name: str
    sfreq: Optional[float] = None
    band_hz: Optional[tuple[float, float]] = None
    owned_steps: frozenset[str] = frozenset()
    default_pretrained: Optional[str] = None
    needs_positions: bool = False


# Keys are case-insensitive aliases used in configs.
ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "eegnet": ArchitectureSpec("EEGNet", sfreq=120.0, band_hz=(0.1, 75.0)),
    "reve": ArchitectureSpec(
        "REVE",
        sfreq=200.0,
        band_hz=(0.5, 99.5),
        default_pretrained="brain-bzh/reve-base",
        needs_positions=True,
    ),
    "biot": ArchitectureSpec(
        "BIOT",
        sfreq=200.0,
        default_pretrained="braindecode/biot-pretrained-six-datasets-18chs",
    ),
}


def _select_collate() -> str:
    """Use NeuralBench collate when the extra is installed."""
    if NEURALBENCH_AVAILABLE:
        return "neuralbench"
    warnings.warn(
        "neuralbench is not installed; using tensor collate. "
        "Install with: pip install -e '.[neuralbench]'",
        UserWarning,
        stacklevel=3,
    )
    return "tensor"


def resolve_architecture(name: str) -> ArchitectureSpec:
    """Look up a known spec, or treat ``name`` as a braindecode class name."""
    key = name.strip()
    spec = ARCHITECTURES.get(key.lower())
    if spec is not None:
        return spec
    return ArchitectureSpec(class_name=key)


def _filter_kwargs(fn, kwargs: dict) -> dict:
    """Drop keys the callable does not accept unless it has **kwargs."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _drop_unset(kwargs: dict) -> dict:
    """Omit arguments left unset so a checkpoint config can fill them in."""
    return {key: value for key, value in kwargs.items() if value is not None}


def _hub_config(repo_id: str) -> dict:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return {}
    try:
        path = hf_hub_download(repo_id, "config.json")
    except Exception:
        return {}
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    return config if isinstance(config, dict) else {}


def _safetensor_keys(repo_id: str) -> set[str]:
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
    except ImportError:
        return set()
    try:
        path = hf_hub_download(repo_id, "model.safetensors")
        with safe_open(path, framework="pt") as handle:
            return set(handle.keys())
    except Exception:
        return set()


def _pretrained_overrides(
    init_kwargs: dict, *, config: dict, weight_names: set[str]
) -> tuple[dict, bool]:
    """Kwargs for ``from_pretrained``, and whether the checkpoint has no task head.

    Unset geometry (``None``) is omitted so values stored in ``config.json`` are
    used. A checkpoint that contains ``cls_query_token`` was pretrained with
    attention pooling; that flag has to be on or those weights are skipped.
    """
    overrides = _drop_unset(init_kwargs)
    if "attention_pooling" not in overrides and "cls_query_token" in weight_names:
        overrides["attention_pooling"] = True
    n_outputs = overrides.get("n_outputs", config.get("n_outputs"))
    return overrides, n_outputs is None


@contextmanager
def _optional_classification_head(cls):
    """Allow ``cls`` to initialize when the checkpoint has no ``n_outputs``.

    braindecode reads ``self.n_outputs`` while building the task head and raises
    if it was neither passed nor stored in the checkpoint.
    """
    saved_property = cls.__dict__.get("n_outputs")
    saved_build = cls.__dict__.get("_build_head")
    if saved_property is None:
        for base in cls.__mro__[1:]:
            saved_property = base.__dict__.get("n_outputs")
            if isinstance(saved_property, property):
                break

    def n_outputs(self):
        if getattr(self, "_n_outputs", None) is None:
            return None
        return saved_property.fget(self)

    def _build_head(self, n_outputs):
        if n_outputs is None:
            self.final_layer = None
            return None
        return saved_build(self, n_outputs)

    # braindecode rebinds this property on each model class, so replacing it
    # on EEGModuleMixin does not affect REVE.
    cls.n_outputs = property(n_outputs)
    if saved_build is not None:
        cls._build_head = _build_head
    try:
        yield
    finally:
        if "n_outputs" in cls.__dict__:
            if saved_property is not None and saved_property is not cls.__dict__.get("n_outputs"):
                cls.n_outputs = saved_property
            elif saved_property is None:
                del cls.n_outputs
        if saved_build is not None:
            cls._build_head = saved_build


def _build_braindecode_model(
    class_name: str,
    *,
    n_chans: int,
    n_times: int,
    n_outputs: int,
    sfreq: Optional[float],
    from_pretrained: Optional[str],
    extra: dict,
) -> nn.Module:
    try:
        import braindecode.models as bd_models
    except ImportError as exc:
        raise ImportError(
            "braindecode is required to build NeuralBench models. "
            "Install with: pip install -e '.[braindecode]' or '.[neuralbench]'"
        ) from exc

    try:
        cls = getattr(bd_models, class_name)
    except AttributeError as exc:
        available = sorted(
            name
            for name, obj in vars(bd_models).items()
            if inspect.isclass(obj) and name[0].isupper()
        )
        raise ValueError(
            f"Unknown architecture {class_name!r}. "
            f"Known aliases: {sorted(ARCHITECTURES)}. "
            f"braindecode classes include: {available}"
        ) from exc

    init_kwargs = {
        "n_chans": n_chans,
        "n_times": n_times,
        "n_outputs": n_outputs,
        **extra,
    }
    if sfreq is not None:
        init_kwargs["sfreq"] = sfreq

    if from_pretrained:
        if not hasattr(cls, "from_pretrained"):
            raise ValueError(f"{class_name} does not support from_pretrained")
        overrides, backbone_only = _pretrained_overrides(
            init_kwargs,
            config=_hub_config(from_pretrained),
            weight_names=_safetensor_keys(from_pretrained),
        )
        filtered = _filter_kwargs(cls.from_pretrained, overrides)
        if backbone_only:
            warnings.warn(
                f"{class_name} checkpoint {from_pretrained!r} has no n_outputs; "
                "loading the backbone without a classification head.",
                UserWarning,
                stacklevel=2,
            )
            with _optional_classification_head(cls):
                return cls.from_pretrained(from_pretrained, **filtered)
        return cls.from_pretrained(from_pretrained, **filtered)
    return cls(**_filter_kwargs(cls.__init__, init_kwargs))


def _default_positions(
    n_chans: int, batch_size: int, *, device, dtype
) -> torch.Tensor:
    theta = torch.linspace(0, 2 * torch.pi, n_chans + 1, device=device, dtype=dtype)[:-1]
    xyz = torch.stack(
        [torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1
    )
    return xyz.unsqueeze(0).expand(batch_size, -1, -1)


@register("models", "neuralbench")
class NeuralBenchWrapper(NeuroModel):
    """One wrapper for NeuralBench architectures (EEGNet, REVE, BIOT, ...).

    The architecture is selected from config. Collate is ``neuralbench`` when
    that extra is installed, otherwise ``tensor`` with a warning.

    Signal properties (``n_chans``, ``n_times``, ``n_outputs``, ``sfreq``,
    ``chs_info``, ``input_window_seconds``, ``input_shape``) are read from the
    built module, including values braindecode inferred. Unset properties are
    ``None``.

    ``from_pretrained`` leaves unset geometry to the checkpoint. A checkpoint
    with no classification head, such as ``brain-bzh/reve-base``, is loaded as
    a backbone and ``forward`` returns encoder features.
    """

    def __init__(
        self,
        architecture: str = "EEGNet",
        n_chans: Optional[int] = None,
        n_times: Optional[int] = None,
        n_outputs: Optional[int] = None,
        from_pretrained: bool | str | None = False,
        sfreq: Optional[float] = None,
        kwargs: Optional[dict] = None,
        **extra,
    ):
        super().__init__()
        spec = resolve_architecture(architecture)
        self.architecture = spec.class_name
        self.needs_positions = spec.needs_positions
        resolved_sfreq = sfreq if sfreq is not None else spec.sfreq

        if from_pretrained is True:
            pretrained = spec.default_pretrained
        elif from_pretrained:
            pretrained = str(from_pretrained)
        else:
            pretrained = None

        constructor_kwargs = {**(kwargs or {}), **extra}
        self.model = _build_braindecode_model(
            spec.class_name,
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            sfreq=resolved_sfreq,
            from_pretrained=pretrained,
            extra=constructor_kwargs,
        )
        self.input_contract = InputContract(
            expects=SignalSpec(unit="V", sfreq=spec.sfreq, band_hz=spec.band_hz),
            owned_steps=set(spec.owned_steps),
            collate=_select_collate(),
            accepts="window",
        )

    def _effective(self, name: str):
        """Read a signal property from the built braindecode module.

        braindecode raises ``ValueError`` when a property was neither given
        nor inferred. Surface that as ``None``, matching ``NeuroModel``.
        """
        try:
            return getattr(self.model, name)
        except ValueError:
            return None

    @property
    def n_chans(self) -> Optional[int]:
        return self._effective("n_chans")

    @property
    def n_times(self) -> Optional[int]:
        return self._effective("n_times")

    @property
    def n_outputs(self) -> Optional[int]:
        return self._effective("n_outputs")

    @property
    def sfreq(self) -> Optional[float]:
        return self._effective("sfreq")

    @property
    def chs_info(self):
        return self._effective("chs_info")

    @property
    def input_window_seconds(self) -> Optional[float]:
        return self._effective("input_window_seconds")

    @property
    def input_shape(self) -> Optional[tuple[int, int, int]]:
        return self._effective("input_shape")

    def forward(self, batch: dict) -> torch.Tensor:
        x = batch.get("signal", batch.get("x"))
        if x is None:
            raise ValueError(
                f"Batch must contain 'signal' or 'x'. Got: {list(batch.keys())}"
            )

        pos = batch.get("ch_positions")
        if pos is not None and pos.ndim == 2:
            pos = pos.unsqueeze(0).expand(x.size(0), -1, -1).to(device=x.device, dtype=x.dtype)
        elif pos is None and self.needs_positions:
            pos = _default_positions(x.size(1), x.size(0), device=x.device, dtype=x.dtype)

        call_kwargs = {}
        try:
            params = inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            params = {}
        if pos is not None and "pos" in params:
            call_kwargs["pos"] = pos
        if getattr(self.model, "final_layer", None) is None and "return_features" in params:
            call_kwargs["return_features"] = True

        out = self.model(x, **call_kwargs)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("logits", out.get("features"))
        return out

