from __future__ import annotations

"""Attention Transfer helpers for feature-map based distillation.

The implementation follows Zagoruyko and Komodakis, "Paying More Attention to
Attention", ICLR 2017. Training orchestration stays in ``scripts/train_at.py``;
this module only owns hook management and the AT loss.
"""

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hooks import attach_feature_hooks, parse_layer_names, remove_hooks


def _attention_map(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Convert a convolution feature map into a normalized attention map."""
    if feat.ndim != 4:
        raise ValueError(f'Expected 4D feature map [B,C,H,W], got shape={tuple(feat.shape)}')
    am = feat.pow(2).mean(dim=1)          # [B, H, W]
    am = am.flatten(1)                    # [B, H*W]
    am = am / (am.norm(p=2, dim=1, keepdim=True) + eps)
    return am


def attention_transfer_loss(
    teacher_feats: Dict[str, torch.Tensor],
    student_feats: Dict[str, torch.Tensor],
    teacher_layers: Sequence[str],
    student_layers: Sequence[str],
    normalize: bool = True,
) -> torch.Tensor:
    """Compute summed MSE between paired teacher/student attention maps."""
    if len(teacher_layers) != len(student_layers):
        raise ValueError(
            f'teacher_layers and student_layers must have the same length, got {len(teacher_layers)} and {len(student_layers)}'
        )

    losses: List[torch.Tensor] = []
    for t_name, s_name in zip(teacher_layers, student_layers):
        ft = teacher_feats.get(t_name, None)
        fs = student_feats.get(s_name, None)
        if ft is None or fs is None:
            raise RuntimeError(
                f'Missing hooked features for pair teacher={t_name!r}, student={s_name!r}. '
                f'Check that the layer names exist and are reached during forward.'
            )

        if ft.ndim != 4 or fs.ndim != 4:
            raise ValueError(
                f'AT expects 4D conv features. Got teacher {tuple(ft.shape)} and student {tuple(fs.shape)}'
            )

        if fs.shape[-2:] != ft.shape[-2:]:
            fs = F.interpolate(fs, size=ft.shape[-2:], mode='bilinear', align_corners=False)

        at_t = _attention_map(ft) if normalize else ft.pow(2).mean(dim=1).flatten(1)
        at_s = _attention_map(fs) if normalize else fs.pow(2).mean(dim=1).flatten(1)
        losses.append(F.mse_loss(at_s, at_t, reduction='mean'))

    if not losses:
        return torch.tensor(0.0, device=next(iter(student_feats.values())).device)
    return torch.stack(losses).sum()


class ATDistiller:
    """Lightweight helper that keeps AT hooks and loss state together."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        beta: float = 10.0,
        teacher_layers: Sequence[str] = ('layer2.1', 'layer3.1'),
        student_layers: Sequence[str] = ('layer2.1', 'layer3.1'),
    ):
        if beta < 0:
            raise ValueError(f'beta must be >= 0, got {beta}')
        self.teacher = teacher
        self.student = student
        self.beta = float(beta)
        self.teacher_layers = list(teacher_layers)
        self.student_layers = list(student_layers)

        self.teacher_feats, self.teacher_hooks = attach_feature_hooks(self.teacher, self.teacher_layers)
        self.student_feats, self.student_hooks = attach_feature_hooks(self.student, self.student_layers)

    def remove_hooks(self) -> None:
        """Detach all forward hooks registered by this helper."""
        remove_hooks(self.teacher_hooks)
        remove_hooks(self.student_hooks)

    @torch.no_grad()
    def teacher_forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run the frozen teacher while recording configured feature maps."""
        self.teacher.eval()
        return self.teacher(images)

    def student_forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run the student while recording configured feature maps."""
        return self.student(images)

    def at_loss(self) -> torch.Tensor:
        """Return the current Attention Transfer loss from the latest forwards."""
        return attention_transfer_loss(
            teacher_feats=self.teacher_feats,
            student_feats=self.student_feats,
            teacher_layers=self.teacher_layers,
            student_layers=self.student_layers,
            normalize=True,
        )

    def total_loss(self, loss_ce: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Combine task cross-entropy with the beta-weighted AT penalty."""
        loss_at = self.at_loss()
        total = loss_ce + self.beta * loss_at
        return total, loss_at
