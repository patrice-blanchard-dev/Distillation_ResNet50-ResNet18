"""Dataset factory functions used by the training scripts."""

from .cifar100 import get_cifar100, get_cifar100_fulltrain

__all__ = ["get_cifar100", "get_cifar100_fulltrain"]
