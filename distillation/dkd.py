from __future__ import annotations

"""Decoupled Knowledge Distillation (DKD).

This module implements the core DKD loss introduced in:
  Zhao et al., "Decoupled Knowledge Distillation", CVPR 2022.

Paper idea:
- Standard KD mixes two very different effects in one KL term.
- DKD decouples them into:
    * TCKD: target-class knowledge distillation
    * NCKD: non-target-class knowledge distillation
- The final distillation term is:
    T^2 * (alpha * TCKD + beta * NCKD)
- In practice, the official implementation also uses a warmup on the
  distillation branch during the first training epochs.

This file intentionally focuses on the *loss* and light helpers.
The training loop and experiment orchestration live in ``train_dkd.py``.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class DKDConfig:
    """Validated hyperparameters for Decoupled Knowledge Distillation."""

    temperature: float = 4.0
    alpha: float = 1.0
    beta: float = 8.0
    warmup_epochs: int = 20
    hard_label_smoothing: float = 0.0
    mask_value: float = 1000.0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.alpha < 0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if self.beta < 0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")
        if self.warmup_epochs < 0:
            raise ValueError(f"warmup_epochs must be >= 0, got {self.warmup_epochs}")
        if not (0.0 <= self.hard_label_smoothing < 1.0):
            raise ValueError(
                f"hard_label_smoothing must be in [0, 1), got {self.hard_label_smoothing}"
            )
        if self.mask_value <= 0:
            raise ValueError(f"mask_value must be > 0, got {self.mask_value}")


def _get_gt_mask(logits: Tensor, target: Tensor) -> Tensor:
    """Build a boolean mask selecting the ground-truth class per sample."""
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D [B, C], got shape={tuple(logits.shape)}")
    if target.ndim != 1 or target.shape[0] != logits.shape[0]:
        raise ValueError(
            f"target must be 1D [B] aligned with logits batch, got target={tuple(target.shape)}, logits={tuple(logits.shape)}"
        )
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(1, target.view(-1, 1), True)
    return mask


def _get_other_mask(logits: Tensor, target: Tensor) -> Tensor:
    """Build the complement mask selecting non-ground-truth classes."""
    return ~_get_gt_mask(logits, target)


def _cat_mask(prob: Tensor, gt_mask: Tensor, other_mask: Tensor) -> Tensor:
    """Collapse a full C-class distribution into 2 classes: {gt, non-gt}."""
    gt_prob = (prob * gt_mask).sum(dim=1, keepdim=True)
    other_prob = (prob * other_mask).sum(dim=1, keepdim=True)
    return torch.cat([gt_prob, other_prob], dim=1)


def _mask_logits(logits: Tensor, gt_mask: Tensor, mask_value: float) -> Tensor:
    """Suppress the ground-truth class before softmax for NCKD."""
    return logits.masked_fill(gt_mask, -mask_value)


def dkd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    *,
    temperature: float,
    alpha: float,
    beta: float,
    mask_value: float = 1000.0,
) -> Tuple[Tensor, Dict[str, float]]:
    """Compute the DKD distillation branch only.

    Returns:
        loss_dkd, metrics where metrics contains raw TCKD/NCKD values before
        alpha/beta weighting but after the T^2 scaling convention used in DKD.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            "student/teacher logits shape mismatch: "
            f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if alpha < 0 or beta < 0:
        raise ValueError(f"alpha and beta must be >= 0, got alpha={alpha}, beta={beta}")

    t = float(temperature)
    gt_mask = _get_gt_mask(student_logits, labels)
    other_mask = ~gt_mask

    student_prob_t = F.softmax(student_logits / t, dim=1)
    teacher_prob_t = F.softmax(teacher_logits / t, dim=1)

    # TCKD: 2-way distribution between target class and all remaining classes
    student_prob_2 = _cat_mask(student_prob_t, gt_mask, other_mask)
    teacher_prob_2 = _cat_mask(teacher_prob_t, gt_mask, other_mask)
    loss_tckd = F.kl_div(
        torch.log(student_prob_2.clamp_min(1e-12)),
        teacher_prob_2,
        reduction="batchmean",
    )

    # NCKD: remove gt class, renormalize the remaining classes
    student_logits_ng = _mask_logits(student_logits / t, gt_mask, mask_value)
    teacher_logits_ng = _mask_logits(teacher_logits / t, gt_mask, mask_value)
    loss_nckd = F.kl_div(
        F.log_softmax(student_logits_ng, dim=1),
        F.softmax(teacher_logits_ng, dim=1),
        reduction="batchmean",
    )

    scale = t * t
    loss_tckd_scaled = loss_tckd * scale
    loss_nckd_scaled = loss_nckd * scale
    total = alpha * loss_tckd_scaled + beta * loss_nckd_scaled
    metrics = {
        "loss_tckd": float(loss_tckd_scaled.detach().item()),
        "loss_nckd": float(loss_nckd_scaled.detach().item()),
        "loss_dkd": float(total.detach().item()),
    }
    return total, metrics


class DKDLoss:
    """High-level DKD loss helper with hard-label branch and warmup.

    Total loss:
        CE(student, labels) + warmup(epoch) * DKD(student, teacher, labels)

    Warmup follows the official DKD practice:
        distill_factor = min(epoch / warmup_epochs, 1.0)
    with the convention that warmup_epochs == 0 disables warmup and immediately
    uses factor 1.0.
    """

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 1.0,
        beta: float = 8.0,
        warmup_epochs: int = 20,
        hard_label_smoothing: float = 0.0,
        mask_value: float = 1000.0,
    ) -> None:
        self.cfg = DKDConfig(
            temperature=float(temperature),
            alpha=float(alpha),
            beta=float(beta),
            warmup_epochs=int(warmup_epochs),
            hard_label_smoothing=float(hard_label_smoothing),
            mask_value=float(mask_value),
        )

    def hard_loss(self, student_logits: Tensor, labels: Tensor) -> Tensor:
        """Cross-entropy branch on hard labels."""
        return F.cross_entropy(
            student_logits,
            labels,
            label_smoothing=self.cfg.hard_label_smoothing,
        )

    def distill_factor(self, epoch: int) -> float:
        """Return the DKD warmup multiplier for the current epoch."""
        if self.cfg.warmup_epochs == 0:
            return 1.0
        if epoch <= 0:
            return 0.0
        return min(float(epoch) / float(self.cfg.warmup_epochs), 1.0)

    def soft_loss(self, student_logits: Tensor, teacher_logits: Tensor, labels: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the decoupled distillation branch."""
        return dkd_loss(
            student_logits,
            teacher_logits,
            labels,
            temperature=self.cfg.temperature,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            mask_value=self.cfg.mask_value,
        )

    def combine(self, hard_loss: Tensor, soft_loss: Tensor, epoch: int) -> Tensor:
        """Add hard-label loss and warmup-scaled DKD loss."""
        return hard_loss + self.distill_factor(epoch) * soft_loss

    def __call__(
        self,
        student_logits: Tensor,
        teacher_logits: Tensor,
        labels: Tensor,
        *,
        epoch: int,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute total DKD loss and scalar metrics for logging."""
        hard = self.hard_loss(student_logits, labels)
        soft, metrics = self.soft_loss(student_logits, teacher_logits, labels)
        factor = self.distill_factor(epoch)
        total = hard + factor * soft
        return total, {
            "loss_total": float(total.detach().item()),
            "loss_ce": float(hard.detach().item()),
            "loss_dkd": float((factor * soft).detach().item()),
            "loss_tckd": metrics["loss_tckd"],
            "loss_nckd": metrics["loss_nckd"],
            "distill_factor": float(factor),
        }


@torch.no_grad()
def forward_teacher(teacher: torch.nn.Module, images: Tensor) -> Tensor:
    """Run the teacher in eval mode without tracking gradients."""
    teacher.eval()
    return teacher(images)
