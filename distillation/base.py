"""Shared base abstractions for distillation experiments."""

import torch
import torch.nn as nn

from .hooks import attach_feature_hooks

__all__ = ["Distiller", "attach_feature_hooks"]


class Distiller(nn.Module):
    """Base module storing a teacher, a student and the common CE criterion."""

    def __init__(self, teacher: nn.Module, student: nn.Module, label_smoothing: float = 0.1):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.ce = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))

    def forward(self, images, labels):
        """Compute a distillation loss for one batch in concrete subclasses."""
        raise NotImplementedError

    def trainable_parameters(self):
        """Return parameters optimized by default: only the student network."""
        return self.student.parameters()
