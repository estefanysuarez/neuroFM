"""Global registry for sources, transforms, augmentations, tokenizers, splits, targets, models, and collates."""

from typing import Any, Callable, Dict, Type, TypeVar

T = TypeVar("T")


class Registry:
    """A registry for named components. Each component is registered once by name."""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type[Any]] = {}

    def register(self, component_name: str) -> Callable[[T], T]:
        """Decorator to register a component by name."""
        def decorator(cls: T) -> T:
            if component_name in self._registry:
                raise ValueError(
                    f"Component '{component_name}' already registered in {self.name}"
                )
            self._registry[component_name] = cls
            return cls
        return decorator

    def get(self, component_name: str) -> Type[Any]:
        """Retrieve a registered component by name."""
        if component_name not in self._registry:
            raise KeyError(
                f"Component '{component_name}' not found in {self.name}. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[component_name]

    def list(self) -> Dict[str, Type[Any]]:
        """List all registered components."""
        return dict(self._registry)


# Global registries
sources_registry = Registry("sources")
transforms_registry = Registry("transforms")
augmentations_registry = Registry("augmentations")
tokenizers_registry = Registry("tokenizers")
splits_registry = Registry("splits")
targets_registry = Registry("targets")
models_registry = Registry("models")
collates_registry = Registry("collates")


# Convenience decorators
def register(registry_name: str, component_name: str):
    """Register a component in a named registry."""
    registries = {
        "sources": sources_registry,
        "transforms": transforms_registry,
        "augmentations": augmentations_registry,
        "tokenizers": tokenizers_registry,
        "splits": splits_registry,
        "targets": targets_registry,
        "models": models_registry,
        "collates": collates_registry,
    }
    if registry_name not in registries:
        raise ValueError(f"Registry '{registry_name}' does not exist")
    return registries[registry_name].register(component_name)
