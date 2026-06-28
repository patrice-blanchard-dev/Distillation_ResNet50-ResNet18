from __future__ import annotations

"""Classic Hinton-style Knowledge Distillation loss."""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class KDConfig:
    """Validated hyperparameters for the KD objective."""

    temperature: float = 4.0
    alpha: float = 0.9
    hard_label_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if not (0.0 <= self.hard_label_smoothing < 1.0):
            raise ValueError(
                f"hard_label_smoothing must be in [0, 1), got {self.hard_label_smoothing}"
            )


class KDLoss:
    """Hinton-style KD loss.

    Total loss:
        (1 - alpha) * CE(student, labels)
      + alpha * T^2 * KL( softmax(t/T) || softmax(s/T) )

    The hard-label branch uses optional label smoothing, but defaults to 0.0,
    which is generally the safest choice when KD already softens supervision.
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.9, hard_label_smoothing: float = 0.0):
        self.cfg = KDConfig(
            temperature=float(temperature),
            alpha=float(alpha),
            hard_label_smoothing=float(hard_label_smoothing),
        )

    @torch.no_grad()
    def teacher_probs(self, teacher_logits: Tensor) -> Tensor:
        """Return temperature-scaled teacher probabilities."""
        return F.softmax(teacher_logits / self.cfg.temperature, dim=1)

    def hard_loss(self, student_logits: Tensor, labels: Tensor) -> Tensor:
        """Cross-entropy between student logits and ground-truth labels."""
        return F.cross_entropy(
            student_logits,
            labels,
            label_smoothing=self.cfg.hard_label_smoothing,
        )

    def soft_loss(self, student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
        """KL divergence between softened student and teacher distributions."""
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"student/teacher logits shape mismatch: {tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
            )
        temperature = self.cfg.temperature
        log_p_student = F.log_softmax(student_logits / temperature, dim=1)
        p_teacher = self.teacher_probs(teacher_logits)
        return F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (temperature ** 2)

    def combine(self, hard_loss: Tensor, soft_loss: Tensor) -> Tensor:
        """Blend hard-label and soft-label losses according to ``alpha``."""
        alpha = self.cfg.alpha
        return (1.0 - alpha) * hard_loss + alpha * soft_loss

    def __call__(self, student_logits: Tensor, teacher_logits: Tensor, labels: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Compute total KD loss and scalar metrics for logging."""
        hard = self.hard_loss(student_logits, labels)
        soft = self.soft_loss(student_logits, teacher_logits)
        total = self.combine(hard, soft)
        return total, {
            "loss_total": float(total.detach().item()),
            "loss_ce": float(hard.detach().item()),
            "loss_kd": float(soft.detach().item()),
        }


@torch.no_grad()
def forward_teacher(teacher: torch.nn.Module, images: Tensor) -> Tensor:
    """Run the teacher in eval mode without tracking gradients."""
    teacher.eval()
    return teacher(images)
