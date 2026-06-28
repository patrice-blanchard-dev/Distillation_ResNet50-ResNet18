"""CIFAR-friendly ResNet model builders."""

import torch.nn as nn
import torchvision.models as tvm


def _adapt_resnet_for_cifar(net: nn.Module, num_classes: int = 100) -> nn.Module:
    """Adapt an ImageNet ResNet stem and classifier to 32x32 CIFAR images."""
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    in_features = net.fc.in_features
    net.fc = nn.Linear(in_features, num_classes)
    return net


def resnet18_cifar(num_classes: int = 100) -> nn.Module:
    """Build a randomly initialized ResNet-18 for CIFAR classification."""
    net = tvm.resnet18(weights=None)
    return _adapt_resnet_for_cifar(net, num_classes)


def resnet50_cifar(num_classes: int = 100) -> nn.Module:
    """Build a randomly initialized ResNet-50 for CIFAR classification."""
    net = tvm.resnet50(weights=None)
    return _adapt_resnet_for_cifar(net, num_classes)
