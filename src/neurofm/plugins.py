"""Load built-in components so their ``@register`` decorators run.

``import neurofm.plugins`` once. Each package below imports its components,
and those imports register them. A new component is included by exporting
it from its package ``__init__``.
"""

from importlib import import_module

_PACKAGES = (
    "neurofm.data.sources",
    "neurofm.data.transforms",
    "neurofm.data.augmentations",
    "neurofm.data.targets",
    "neurofm.data.splits",
    "neurofm.data.tokenizers",
    "neurofm.data.collate",
    "neurofm.models.custom",
    "neurofm.models.external",
)


def load() -> None:
    """Import every built-in component package."""
    for package in _PACKAGES:
        import_module(package)


load()
