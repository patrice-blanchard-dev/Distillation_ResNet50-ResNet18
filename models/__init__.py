"""Model registry for CIFAR distillation experiments."""

from .resnet_cifar import resnet18_cifar, resnet50_cifar

__all__ = ["resnet18_cifar", "resnet50_cifar", "get_model"]


def get_model(name: str, num_classes: int = 100):
    """Instantiate a model by registry name."""
    name = name.lower().strip()
    registry = {
        "resnet18_cifar": resnet18_cifar,
        "resnet50_cifar": resnet50_cifar,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(registry)}")
    return registry[name](num_classes=num_classes)
