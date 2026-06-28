from __future__ import annotations

"""FitNet hint-loss utilities.

FitNets first align an intermediate student representation with an intermediate
teacher representation through a learned 1x1 regressor. The final logits-based
distillation stage is implemented in ``scripts/train_fitnet.py``.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hooks import attach_feature_hooks, get_module_by_name, parse_layer_names, remove_hooks


def collect_trainable_params_through_layer(model: nn.Module, layer_name: str) -> List[nn.Parameter]:
    """Return parameters from the start of the network through ``layer_name`` inclusive.

    This matches the FitNets stage-1 idea: optimize the student part up to the
    guided layer, plus the learned regressor, before the final KD stage.
    """
    target = get_module_by_name(model, layer_name)
    params: List[nn.Parameter] = []
    seen: set[int] = set()

    for _name, module in model.named_modules():
        for param in module.parameters(recurse=False):
            pid = id(param)
            if pid not in seen:
                params.append(param)
                seen.add(pid)
        if module is target:
            break

    if not params:
        raise RuntimeError(f'No trainable parameters found through layer {layer_name!r}')
    return params


class FitNetHintDistiller(nn.Module):
    """Paper-style FitNet helper for stage-1 hint training.

    This helper only implements the hint regression stage. The final stage using
    KD on logits is handled in ``train_fitnet.py``.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        teacher_hint_layer: str = 'layer3.1',
        student_guided_layer: str = 'layer2.1',
    ) -> None:
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.teacher_hint_layer = str(teacher_hint_layer)
        self.student_guided_layer = str(student_guided_layer)

        self.teacher_feats, self.teacher_hooks = attach_feature_hooks(self.teacher, [self.teacher_hint_layer])
        self.student_feats, self.student_hooks = attach_feature_hooks(self.student, [self.student_guided_layer])
        self.regressor: nn.Conv2d | None = None
        self.mse = nn.MSELoss(reduction='mean')

    def remove_hooks(self) -> None:
        """Detach teacher and student hooks registered for hint extraction."""
        remove_hooks(self.teacher_hooks)
        remove_hooks(self.student_hooks)

    @torch.no_grad()
    def teacher_forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run the teacher in eval mode while capturing the hint feature."""
        self.teacher.eval()
        return self.teacher(images)

    def student_forward(self, images: torch.Tensor) -> torch.Tensor:
        """Run the student while capturing the guided feature."""
        return self.student(images)

    def _build_regressor(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> nn.Conv2d:
        """Create the 1x1 projection once feature channel counts are known."""
        if student_feat.ndim != 4 or teacher_feat.ndim != 4:
            raise ValueError(
                'FitNet expects conv features [B,C,H,W], '
                f'got {tuple(student_feat.shape)} and {tuple(teacher_feat.shape)}'
            )
        regressor = nn.Conv2d(
            in_channels=int(student_feat.shape[1]),
            out_channels=int(teacher_feat.shape[1]),
            kernel_size=1,
            bias=False,
        ).to(student_feat.device)
        self.regressor = regressor
        return regressor

    def build_regressor_from_batch(self, images: torch.Tensor) -> None:
        """Initialize the projection layer from one representative batch."""
        with torch.no_grad():
            _ = self.teacher_forward(images)
        _ = self.student_forward(images)
        teacher_feat = self.teacher_feats.get(self.teacher_hint_layer)
        student_feat = self.student_feats.get(self.student_guided_layer)
        if teacher_feat is None or student_feat is None:
            raise RuntimeError(
                'Missing hooked features for the FitNet hint pair. '
                'Check the configured teacher/student layer names.'
            )
        if self.regressor is None:
            self._build_regressor(student_feat, teacher_feat)

    def hint_loss(self) -> torch.Tensor:
        """Compute MSE between projected student features and teacher hints."""
        teacher_feat = self.teacher_feats.get(self.teacher_hint_layer)
        student_feat = self.student_feats.get(self.student_guided_layer)
        if teacher_feat is None or student_feat is None:
            raise RuntimeError(
                'Missing hooked features for the FitNet hint pair. '
                'Run teacher_forward/student_forward before calling hint_loss().'
            )
        if self.regressor is None:
            self._build_regressor(student_feat, teacher_feat)
        assert self.regressor is not None
        projected = self.regressor(student_feat)
        if projected.shape[-2:] != teacher_feat.shape[-2:]:
            projected = F.interpolate(projected, size=teacher_feat.shape[-2:], mode='bilinear', align_corners=False)
        return self.mse(projected, teacher_feat.detach())
