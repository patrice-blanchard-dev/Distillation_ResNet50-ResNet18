"""Reusable forward-hook utilities for feature-based distillation methods."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Return a nested module by its ``named_modules`` path.

    ResNet layers are commonly addressed with names such as ``layer2.1``.
    Raising with a short list of available names makes CLI configuration errors
    much easier to diagnose.
    """
    modules = dict(model.named_modules())
    if name not in modules:
        available = sorted(k for k in modules.keys() if k)
        raise ValueError(f"Layer {name!r} not found in model. Available examples: {available[:20]}")
    return modules[name]


def parse_layer_names(value: str | Sequence[str]) -> List[str]:
    """Normalize a comma-separated string or sequence into clean layer names."""
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]


def attach_feature_hooks(
    model: nn.Module,
    layer_names: Iterable[str],
) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    """Attach forward hooks and collect each requested layer's output tensor."""
    features: Dict[str, torch.Tensor] = {}
    hooks: List[torch.utils.hooks.RemovableHandle] = []

    for layer_name in layer_names:
        module = get_module_by_name(model, layer_name)

        def _make_hook(name: str):
            def _hook(_module, _inputs, output):
                # Some modules return tuples; feature losses operate on tensors.
                features[name] = output[0] if isinstance(output, tuple) else output

            return _hook

        hooks.append(module.register_forward_hook(_make_hook(layer_name)))

    return features, hooks


def remove_hooks(hooks: Iterable[torch.utils.hooks.RemovableHandle]) -> None:
    """Remove all hook handles, ignoring handles that were already removed."""
    for handle in hooks:
        try:
            handle.remove()
        except Exception:
            # Hook cleanup should never mask the original training/evaluation
            # error, so teardown is intentionally best-effort.
            pass
