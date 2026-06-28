"""Training, evaluation, augmentation and checkpoint helpers."""

from __future__ import annotations

import os
import random
import tempfile
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm


def _atomic_torch_save(obj, path: str) -> None:
    """Save a torch object atomically to reduce the risk of corrupted checkpoints.

    PyTorch's zipfile writer can reject some temporary filenames on certain setups
    (for example hidden files such as ``.__tmp_ckpt_*``). Use a normal visible
    filename with a ``.pth.tmp`` suffix in the same directory, then atomically
    replace the target file.
    """
    save_dir = os.path.dirname(path) or "."
    os.makedirs(save_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix="tmp_ckpt_", suffix=".pth.tmp", dir=save_dir)
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_checkpoint(state, is_best, save_dir, filename: str = "checkpoint.pth"):
    """Save the latest checkpoint and optionally mirror it as ``best.pth``."""
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, filename)
    _atomic_torch_save(state, checkpoint_path)
    if is_best:
        _atomic_torch_save(state, os.path.join(save_dir, "best.pth"))


def get_scheduler(optimizer, epochs, scheduler_type: str = "cosine", milestones=(150, 200), gamma: float = 0.1):
    """Create the learning-rate scheduler requested by a training script."""
    scheduler_type = scheduler_type.lower().strip()
    if scheduler_type == "multistep":
        return MultiStepLR(optimizer, milestones=list(milestones), gamma=gamma)
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unknown scheduler_type {scheduler_type}")


def _autocast_context(device: torch.device, enabled: bool):
    """Return CUDA autocast when enabled, otherwise a no-op context manager."""
    if enabled and device.type == "cuda":
        return autocast(device_type="cuda")
    return nullcontext()


def _sample_beta(alpha: float, device: torch.device) -> float:
    """Draw a Mixup/CutMix interpolation factor from a symmetric Beta law."""
    if alpha <= 0:
        return 1.0
    concentration = torch.tensor([alpha], device=device, dtype=torch.float32)
    beta_dist = torch.distributions.Beta(concentration, concentration)
    return float(beta_dist.sample().item())


def mixup_data(x: Tensor, y: Tensor, alpha: float = 1.0, device: str | torch.device = "cuda"):
    """Apply Mixup to a batch and return mixed inputs plus paired labels."""
    if alpha <= 0 or x.size(0) <= 1:
        return x, y, y, 1.0

    device = torch.device(device)
    lam = _sample_beta(alpha, device)
    index = torch.randperm(x.size(0), device=device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def _rand_bbox(width: int, height: int, lam: float, device: torch.device) -> Tuple[int, int, int, int]:
    """Sample a CutMix rectangle whose area follows the sampled lambda."""
    cut_ratio = float((1.0 - lam) ** 0.5)
    cut_w = max(1, int(width * cut_ratio))
    cut_h = max(1, int(height * cut_ratio))

    cx = int(torch.randint(0, width, (1,), device=device).item())
    cy = int(torch.randint(0, height, (1,), device=device).item())

    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, height)

    # Extremely small images could still degenerate at borders.
    if x2 <= x1:
        x2 = min(x1 + 1, width)
    if y2 <= y1:
        y2 = min(y1 + 1, height)
    return x1, y1, x2, y2


def cutmix_data(x: Tensor, y: Tensor, alpha: float = 1.0, device: str | torch.device = "cuda"):
    """Apply CutMix to a batch and return mixed inputs plus paired labels."""
    if alpha <= 0 or x.size(0) <= 1:
        return x, y, y, 1.0

    device = torch.device(device)
    lam = _sample_beta(alpha, device)
    batch_size, _, h, w = x.size()
    index = torch.randperm(batch_size, device=device)
    mixed_x = x.clone()

    x1, y1, x2, y2 = _rand_bbox(w, h, lam, device)
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    replaced_area = float((x2 - x1) * (y2 - y1))
    lam = 1.0 - replaced_area / float(w * h)
    return mixed_x, y, y[index], lam


def _mixed_top1_correct(logits: Tensor, targets_a: Tensor, targets_b: Tensor, lam: float) -> float:
    """Compute lambda-weighted top-1 correctness for mixed-label batches."""
    preds = logits.argmax(dim=1)
    correct_a = preds.eq(targets_a).sum().item()
    correct_b = preds.eq(targets_b).sum().item()
    return float(lam * correct_a + (1.0 - lam) * correct_b)


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    use_cutmix: bool = False,
    cutmix_alpha: float = 1.0,
    scaler=None,
    max_grad_norm: Optional[float] = None,
):
    """Train one supervised epoch with optional Mixup, CutMix and AMP."""
    model.train()
    running_loss = 0.0
    running_correct = 0.0
    total = 0
    amp_enabled = (scaler is not None) and (device.type == "cuda")

    for images, labels in tqdm(dataloader, desc="Train", leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        mixed = False
        if use_mixup and use_cutmix:
            if random.random() < 0.5:
                images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=mixup_alpha, device=device)
            else:
                images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha, device=device)
            mixed = True
        elif use_mixup:
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=mixup_alpha, device=device)
            mixed = True
        elif use_cutmix:
            images, targets_a, targets_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha, device=device)
            mixed = True

        optimizer.zero_grad(set_to_none=True)

        with _autocast_context(device, amp_enabled):
            outputs = model(images)
            if mixed:
                loss = lam * criterion(outputs, targets_a) + (1.0 - lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs, labels)

        if not torch.isfinite(loss).all():
            raise FloatingPointError(f"Non-finite loss detected during training: {float(loss.detach().cpu())}")

        if amp_enabled:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        if mixed:
            running_correct += _mixed_top1_correct(outputs.detach(), targets_a, targets_b, lam)
        else:
            preds = outputs.argmax(dim=1)
            running_correct += float(preds.eq(labels).sum().item())
        total += batch_size

    return running_loss / max(total, 1), 100.0 * running_correct / max(total, 1)


@torch.inference_mode()
def eval_one_epoch(model, dataloader, criterion, device, desc: str = "Val"):
    """Evaluate loss and top-1 accuracy on a validation-like dataloader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    amp_enabled = device.type == "cuda"

    for images, labels in tqdm(dataloader, desc=desc, leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _autocast_context(device, amp_enabled):
            outputs = model(images)
            loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += float(loss.item()) * batch_size
        preds = outputs.argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += batch_size

    return running_loss / max(total, 1), 100.0 * correct / max(total, 1)


@torch.inference_mode()
def eval_on_test(model, dataloader, device, num_classes: int):
    """Evaluate test accuracy, per-class accuracy and confusion matrix."""
    model.eval()
    total_correct = 0
    total_seen = 0
    conf_mat = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    amp_enabled = device.type == "cuda"

    for images, labels in tqdm(dataloader, desc="Test", leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _autocast_context(device, amp_enabled):
            outputs = model(images)

        preds = outputs.argmax(dim=1)
        total_correct += preds.eq(labels).sum().item()
        total_seen += labels.size(0)

        flat = (labels * num_classes + preds).to(torch.int64)
        conf_mat += torch.bincount(flat, minlength=num_classes * num_classes).cpu().reshape(num_classes, num_classes)

    per_class_total = conf_mat.sum(dim=1).float().clamp(min=1.0)
    per_class_acc = conf_mat.diag().float() / per_class_total
    overall_acc = 100.0 * total_correct / max(total_seen, 1)
    return overall_acc, per_class_acc.cpu(), conf_mat.cpu()


@torch.inference_mode()
def collect_logits_and_labels(model, dataloader, device):
    """Collect logits and labels for downstream ROC/AUC computation."""
    model.eval()
    all_logits = []
    all_labels = []
    amp_enabled = device.type == "cuda"

    for images, labels in tqdm(dataloader, desc="Collect ROC", leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _autocast_context(device, amp_enabled):
            outputs = model(images)

        all_logits.append(outputs.cpu())
        all_labels.append(labels.cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)
