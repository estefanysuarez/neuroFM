"""Generic wrapper for NeuralBench / braindecode models."""

from __future__ import annotations

import importlib
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


# Vanilla class -> wrapper that spline-interpolates arbitrary channels onto
# the montage the checkpoint was trained on.
_INTERPOLATED_CLASSES = {
    "BIOT": "InterpolatedBIOT",
    "Labram": "InterpolatedLaBraM",
    "SignalJEPA": "InterpolatedSignalJEPA",
}

# 10-20 / 10-10 names preferred when the caller only supplies ``n_chans``.
# MNE's head-sphere fit ignores electrodes with z < 0 and y > 0, so the
# selector below drops those before choosing positions.
_STANDARD_CHANNEL_NAMES = (
    "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
)

# Keys are case-insensitive aliases used in configs.
ARCHITECTURES: dict[str, ArchitectureSpec] = {
    "eegnet": ArchitectureSpec("EEGNet", sfreq=120.0, band_hz=(0.1, 75.0)),
    "reve": ArchitectureSpec(
        "REVE",
        sfreq=256.0,
        band_hz=(0.5, 99.5),
        default_pretrained="brain-bzh/reve-base",
        needs_positions=True,
    ),
    "biot": ArchitectureSpec(
        "BIOT",
        sfreq=200.0,
        default_pretrained="braindecode/biot-pretrained-six-datasets-18chs",
    ),
    "bendr": ArchitectureSpec(
        "BENDR",
        sfreq=250.0,
        default_pretrained="braindecode/braindecode-bendr",
    ),
    "cbramod": ArchitectureSpec(
        "CBraMod",
        sfreq=250.0,
        default_pretrained="braindecode/cbramod-pretrained",
    ),
    "codebrain": ArchitectureSpec(
        "CodeBrain",
        sfreq=200.0,
        default_pretrained="braindecode/codebrain-pretrained",
    ),
    "eegpt": ArchitectureSpec(
        "EEGPT",
        sfreq=250.0,
        default_pretrained="braindecode/eegpt-pretrained",
    ),
    "labram": ArchitectureSpec(
        "Labram",
        default_pretrained="braindecode/labram-pretrained",
    ),
    "luna": ArchitectureSpec(
        "LUNA",
        default_pretrained="PulpBio/LUNA",
    ),
    "signaljepa": ArchitectureSpec(
        "SignalJEPA",
        sfreq=128.0,
        default_pretrained="braindecode/signal-jepa",
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


def _is_channel_layout_error(exc: BaseException) -> bool:
    """True when ``exc`` means the channel count or montage was rejected."""
    message = str(exc).lower()
    if "n_chans" in message and "chs_info" in message:
        return True
    if "chs_info is required" in message:
        return True
    if "canonical" in message and "channel" in message:
        return True
    if "size mismatch" in message and "channel" in message:
        return True
    return False


def _canonical_channel_count(bd_models, class_name: str) -> Optional[int]:
    interpolated = _INTERPOLATED_CLASSES.get(class_name)
    if interpolated is None:
        return None
    cls = getattr(bd_models, interpolated, None)
    target = getattr(cls, "_TARGET_CHS_INFO", None)
    if not target:
        return None
    return len(target)


def _usable_electrode(loc) -> bool:
    """Keep electrodes MNE can use when fitting a head sphere.

    ``fit_sphere_to_headshape`` drops points with ``z < 0`` and ``y > 0``
    (nose and face). Fewer than four remaining points raises.
    """
    z, y = float(loc[2]), float(loc[1])
    return not (z < -1e-6 and y > 1e-6)


def _farthest_indices(xyz, n_chans: int) -> list[int]:
    """Pick ``n_chans`` rows spread over the scalp, starting at the vertex."""
    import numpy as np

    start = int(np.argmax(xyz[:, 2]))
    chosen = [start]
    remaining = set(range(len(xyz))) - {start}
    while len(chosen) < n_chans and remaining:
        selected = xyz[chosen]
        pool = np.fromiter(remaining, dtype=int)
        distance = np.linalg.norm(
            xyz[pool][:, None, :] - selected[None, :, :], axis=-1
        ).min(axis=1)
        nxt = int(pool[int(np.argmax(distance))])
        chosen.append(nxt)
        remaining.remove(nxt)
    return chosen


def _standard_chs_info(n_chans: int) -> list[dict]:
    """Build ``n_chans`` MNE channel dicts on a standard scalp montage.

    Used when a model rejects ``n_chans`` and the caller did not pass
    ``chs_info``. Positions are real electrodes so spline interpolation can
    run; they are not the recording's true layout.
    """
    import mne
    import numpy as np

    montage = mne.channels.make_standard_montage("colin27_1005")
    positions = montage.get_positions()["ch_pos"]
    preferred = [name for name in _STANDARD_CHANNEL_NAMES if name in positions]
    extras = [
        name
        for name in montage.ch_names
        if name in positions and name not in _STANDARD_CHANNEL_NAMES
    ]

    def _collect(names: list[str]):
        kept_names = []
        kept_locs = []
        for name in names:
            loc = np.asarray(positions[name], dtype=float)[:3]
            if _usable_electrode(loc):
                kept_names.append(name)
                kept_locs.append(loc)
        return kept_names, kept_locs

    preferred_names, preferred_locs = _collect(preferred)
    if n_chans <= len(preferred_names):
        names, locs = preferred_names, np.stack(preferred_locs)
    else:
        extra_names, extra_locs = _collect(extras)
        names = preferred_names + extra_names
        locs = np.stack(preferred_locs + extra_locs)
    if n_chans > len(names):
        raise ValueError(
            f"Cannot place {n_chans} channels on the colin27_1005 montage "
            f"({len(names)} usable electrodes)."
        )
    chosen = _farthest_indices(locs, n_chans)
    return [
        {"ch_name": names[index], "kind": "eeg", "loc": locs[index]}
        for index in chosen
    ]


def _user_chs_info(n_chans: Optional[int], extra: dict) -> list[dict]:
    """Channel geometry for an interpolated model, length ``n_chans``."""
    provided = extra.get("chs_info")
    if provided is not None and (n_chans is None or len(provided) == n_chans):
        return list(provided)
    if n_chans is None:
        raise ValueError(
            "Cannot interpolate channels without n_chans or chs_info."
        )
    if n_chans < 4:
        raise ValueError(
            "Channel interpolation needs at least 4 electrodes to fit a "
            f"head sphere, got n_chans={n_chans}."
        )
    return _standard_chs_info(int(n_chans))


def _probe_channel_layout(model: nn.Module, n_chans: int) -> None:
    """Run one dummy forward so montage checks that live in ``forward`` fire."""
    n_times = getattr(model, "n_times", None)
    if not isinstance(n_times, int) or n_times <= 0:
        return
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, int(n_chans), int(n_times)))
    finally:
        model.train(training)


def _error_summary(exc: BaseException) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    summary = lines[0] if lines else exc.__class__.__name__
    if summary.endswith(":") and len(lines) > 1:
        summary = lines[1]
    marker = " different from chs_info="
    if marker in summary:
        summary = summary.split(marker)[0] + " different from the canonical channel set"
    if len(summary) > 240:
        summary = summary[:240] + "..."
    return summary


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

    try:
        model = _instantiate_braindecode_model(
            cls,
            n_chans=n_chans,
            n_times=n_times,
            n_outputs=n_outputs,
            sfreq=sfreq,
            from_pretrained=from_pretrained,
            extra=extra,
        )
        canonical = _canonical_channel_count(bd_models, class_name)
        # Labram accepts a non-canonical count at init and raises in forward.
        if (
            canonical is not None
            and n_chans is not None
            and n_chans != canonical
        ):
            _probe_channel_layout(model, n_chans)
        return model
    except (ValueError, RuntimeError) as exc:
        interpolated_name = _INTERPOLATED_CLASSES.get(class_name)
        canonical = _canonical_channel_count(bd_models, class_name)
        wrong_count = canonical is None or n_chans != canonical
        if (
            interpolated_name is None
            or n_chans is None
            or not wrong_count
            or not _is_channel_layout_error(exc)
        ):
            raise
        position_note = ""
        if extra.get("chs_info") is None:
            position_note = (
                " Channel positions were taken from a standard scalp montage "
                "because chs_info was not provided."
            )
        warnings.warn(
            f"{class_name} rejected n_chans={n_chans} ({_error_summary(exc)}). "
            f"Using {interpolated_name} to project inputs onto the canonical "
            f"{canonical}-channel montage.{position_note}",
            UserWarning,
            stacklevel=3,
        )
        try:
            chs_info = _user_chs_info(n_chans, extra)
            return _instantiate_braindecode_model(
                getattr(bd_models, interpolated_name),
                n_chans=n_chans,
                n_times=n_times,
                n_outputs=n_outputs,
                sfreq=sfreq,
                from_pretrained=from_pretrained,
                extra=extra,
                chs_info=chs_info,
                strict=False,
            )
        except Exception as interpolated_exc:
            if "patch_embedding" in str(interpolated_exc) and sfreq is not None:
                raise RuntimeError(
                    f"{interpolated_name} matched the channel montage, but "
                    f"{from_pretrained} could not be loaded at sfreq={sfreq}. "
                    "The checkpoint's spectral layer is sized for its training "
                    "sampling rate (200 Hz for the BIOT checkpoints). "
                    "Set sfreq to that rate."
                ) from interpolated_exc
            raise interpolated_exc from exc


def _annotated_type_params(cls) -> set[str]:
    """Constructor arguments annotated as a class (``type[nn.Module]``)."""
    names: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for name, annotation in getattr(init, "__annotations__", {}).items():
            text = str(annotation)
            if text in {"type", "Type"} or "type[" in text or "Module" in text:
                names.add(name)
    return names


def _import_qualname(path: str):
    module_name, _, qualname = path.rpartition(".")
    if not module_name:
        raise ValueError(path)
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _decoded_hidden_types(cls, config: dict) -> dict:
    """Decode ``'pkg.Class'`` config values the hub coder will not see.

    ``Interpolated*`` constructors take backbone-only arguments such as
    ``activation`` through ``**kwargs``. The hub coder only rewrites fields
    that appear as typed parameters on the class being loaded, so those
    strings would otherwise be passed through and fail at ``activation()``.
    """
    try:
        own = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return {}
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in own.values()):
        return {}
    explicit = {name for name in own if name not in {"kwargs", "args"}}
    hidden = _annotated_type_params(cls) - explicit
    decoded = {}
    for key in hidden:
        value = config.get(key)
        if not isinstance(value, str) or "." not in value:
            continue
        try:
            obj = _import_qualname(value)
        except (ImportError, AttributeError, ValueError):
            continue
        if inspect.isclass(obj):
            decoded[key] = obj
    return decoded


def _instantiate_braindecode_model(
    cls,
    *,
    n_chans: int,
    n_times: int,
    n_outputs: int,
    sfreq: Optional[float],
    from_pretrained: Optional[str],
    extra: dict,
    chs_info=None,
    strict: Optional[bool] = None,
) -> nn.Module:
    init_kwargs = {
        "n_chans": n_chans,
        "n_times": n_times,
        "n_outputs": n_outputs,
        **extra,
    }
    if chs_info is not None:
        init_kwargs["chs_info"] = chs_info
    if sfreq is not None:
        init_kwargs["sfreq"] = sfreq

    if from_pretrained:
        if not hasattr(cls, "from_pretrained"):
            raise ValueError(f"{cls.__name__} does not support from_pretrained")
        config = _hub_config(from_pretrained)
        overrides, backbone_only = _pretrained_overrides(
            init_kwargs,
            config=config,
            weight_names=_safetensor_keys(from_pretrained),
        )
        for key, value in _decoded_hidden_types(cls, config).items():
            overrides.setdefault(key, value)
        filtered = _filter_kwargs(cls.from_pretrained, overrides)
        if strict is not None:
            filtered["strict"] = strict
        if backbone_only:
            warnings.warn(
                f"{cls.__name__} checkpoint {from_pretrained!r} has no n_outputs; "
                "loading the backbone without a classification head.",
                UserWarning,
                stacklevel=3,
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

    BIOT, Labram, and SignalJEPA are pretrained on a fixed montage. If
    ``n_chans`` does not match that montage, the resulting error is caught
    and the model is rebuilt as the corresponding ``Interpolated*`` wrapper,
    which projects the input onto the canonical channels.
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

