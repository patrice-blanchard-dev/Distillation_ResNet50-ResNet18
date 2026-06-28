"""Train a student with FitNet hint pretraining followed by KD."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

if __package__ is None or __package__ == '':
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
else:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    DATA_DIR,
    EPOCHS_STUDENT,
    NUM_CLASSES,
    NUM_WORKERS,
    SAVE_DIR,
    SEED,
    TRAIN_VAL_SPLIT,
)

try:
    from models import get_model  # type: ignore
except ImportError:
    from resnet_cifar import resnet18_cifar, resnet50_cifar  # type: ignore

    def get_model(name: str, num_classes: int = 100):
        name = name.lower().strip()
        if name == 'resnet18_cifar':
            return resnet18_cifar(num_classes=num_classes)
        if name == 'resnet50_cifar':
            return resnet50_cifar(num_classes=num_classes)
        raise ValueError(f'Unknown model: {name}')

try:
    from datasets import get_cifar100, get_cifar100_fulltrain  # type: ignore
except ImportError:
    from cifar100 import get_cifar100, get_cifar100_fulltrain  # type: ignore

try:
    from utils.main_utils import (  # type: ignore
        collect_logits_and_labels,
        cutmix_data,
        eval_on_test,
        get_scheduler,
        mixup_data,
        save_checkpoint,
    )
except ImportError:
    from main_utils import (  # type: ignore
        collect_logits_and_labels,
        cutmix_data,
        eval_on_test,
        get_scheduler,
        mixup_data,
        save_checkpoint,
    )

try:
    from distillation.fitnet import FitNetHintDistiller, collect_trainable_params_through_layer, parse_layer_names  # type: ignore
except ImportError:
    from fitnet import FitNetHintDistiller, collect_trainable_params_through_layer, parse_layer_names  # type: ignore

try:
    from distillation.kd import KDLoss  # type: ignore
except ImportError:
    from kd import KDLoss  # type: ignore


def parse_args():
    parser = argparse.ArgumentParser(description='Train FitNet student on CIFAR-100 (paper-style two-stage training)')
    parser.add_argument('--teacher_model', type=str, default='resnet50_cifar')
    parser.add_argument('--student_model', type=str, default='resnet18_cifar')
    parser.add_argument('--teacher_checkpoint', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--save_subdir', type=str, default='fitnet')

    parser.add_argument('--stage2_epochs', type=int, default=EPOCHS_STUDENT)
    parser.add_argument('--epochs', type=int, default=0, help='Alias for --stage2_epochs when > 0 for search-script compatibility')
    parser.add_argument('--hint_epochs', type=int, default=60)

    parser.add_argument('--stage2_lr', type=float, default=0.08)
    parser.add_argument('--lr', type=float, default=0.0, help='Alias for --stage2_lr when > 0 for search-script compatibility')
    parser.add_argument('--hint_lr', type=float, default=5e-4)

    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'multistep'])
    parser.add_argument('--milestones', type=int, nargs='+', default=[150, 200])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=0.0)

    parser.add_argument('--kd_temperature', type=float, default=4.0)
    parser.add_argument('--kd_alpha', type=float, default=0.9)
    parser.add_argument('--hint_weight', type=float, default=1.0)
    parser.add_argument('--hard_label_smoothing', type=float, default=0.0)
    parser.add_argument('--teacher_hint_layers', type=str, default='layer3.1')
    parser.add_argument('--student_guided_layers', type=str, default='layer2.1')

    parser.add_argument('--no_mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--no_cutmix', action='store_true')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)

    parser.add_argument('--full_train', action='store_true')
    parser.add_argument('--skip_test_metrics', action='store_true')
    parser.add_argument('--skip_plots', action='store_true')

    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--es_patience', type=int, default=15)
    parser.add_argument('--es_min_delta', type=float, default=0.05)
    parser.add_argument('--es_start_epoch', type=int, default=30)

    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--allow_tf32', action='store_true')
    parser.add_argument('--resume', type=str, default='')

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='distill_cifar100')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_tags', type=str, default='fitnet')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_group', type=str, default='')
    parser.add_argument('--wandb_job_type', type=str, default='train')
    parser.add_argument('--wandb_run_name', type=str, default='')
    parser.add_argument('--wandb_log_artifacts', action='store_true')
    return parser.parse_args()


def set_seed(seed: int, benchmark: bool = False, allow_tf32: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = not benchmark
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def _safe_torch_load(path: str, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _atomic_json_save(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='tmp_json_', suffix='.json.tmp', dir=os.path.dirname(path) or '.')
    os.close(fd)
    try:
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    _atomic_json_save(path, payload)


def _save_history(history: Dict[str, List[Any]], save_dir: str) -> str:
    history_path = os.path.join(save_dir, 'history.json')
    _save_json(history_path, history)
    return history_path


def _save_progress(save_dir: str, payload: Dict[str, Any]) -> str:
    progress_path = os.path.join(save_dir, 'progress.json')
    _save_json(progress_path, payload)
    return progress_path


def _load_existing_history(save_dir: str) -> Optional[Dict[str, List[Any]]]:
    history_path = os.path.join(save_dir, 'history.json')
    if not os.path.isfile(history_path):
        return None
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _init_wandb(args, save_dir: str):
    if not args.use_wandb or args.wandb_mode == 'disabled':
        return None
    try:
        import wandb  # type: ignore
    except Exception as exc:
        print(f'[W&B] import failed ({exc}) -> continuing without wandb.')
        return None

    os.environ['WANDB_MODE'] = args.wandb_mode
    tags = [t.strip() for t in args.wandb_tags.split(',') if t.strip()]
    run_name = args.wandb_run_name.strip() or f'fitnet/{args.student_model}/{args.exp_name}'
    run_group = args.wandb_group.strip() or None
    job_type = args.wandb_job_type.strip() or 'train'

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity.strip() or None,
        name=run_name,
        group=run_group,
        job_type=job_type,
        tags=tags if tags else None,
        config={
            'stage': 'fitnet',
            'fitnet_method': 'paper_two_stage_hint_then_kd',
            'teacher_model': args.teacher_model,
            'student_model': args.student_model,
            'teacher_checkpoint': args.teacher_checkpoint,
            'hint_epochs': args.hint_epochs,
            'stage2_epochs': _effective_stage2_epochs(args),
            'hint_lr': args.hint_lr,
            'stage2_lr': _effective_stage2_lr(args),
            'wd': args.wd,
            'momentum': args.momentum,
            'scheduler': args.scheduler,
            'milestones': args.milestones,
            'gamma': args.gamma,
            'kd_temperature': args.kd_temperature,
            'kd_alpha': args.kd_alpha,
            'hint_weight': args.hint_weight,
            'hard_label_smoothing': args.hard_label_smoothing,
            'teacher_hint_layers': parse_layer_names(args.teacher_hint_layers),
            'student_guided_layers': parse_layer_names(args.student_guided_layers),
            'mixup': (not args.no_mixup),
            'mixup_alpha': args.mixup_alpha,
            'cutmix': (not args.no_cutmix),
            'cutmix_alpha': args.cutmix_alpha,
            'full_train': args.full_train,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'seed': args.seed,
        },
        dir=save_dir,
        reinit=True,
    )
    return wandb


def _effective_stage2_epochs(args) -> int:
    return int(args.epochs) if int(args.epochs) > 0 else int(args.stage2_epochs)


def _effective_stage2_lr(args) -> float:
    return float(args.lr) if float(args.lr) > 0 else float(args.stage2_lr)


def _generate_curves(history: Dict[str, List[Any]], save_dir: str) -> Tuple[str, str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        print(f'[plot] matplotlib unavailable ({exc}); skipping curves.')
        return '', ''

    epochs = history.get('epoch', [])
    if not epochs:
        return '', ''

    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    train_acc = history.get('train_acc', [])
    val_acc = history.get('val_acc', [])

    loss_curve_path = os.path.join(save_dir, 'loss_curve.png')
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='train_loss')
    if any(v is not None for v in val_loss):
        xs = [e for e, v in zip(epochs, val_loss) if v is not None]
        ys = [v for v in val_loss if v is not None]
        plt.plot(xs, ys, label='val_loss')
    plt.xlabel('global epoch')
    plt.ylabel('loss')
    plt.title('FitNet training loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=180)
    plt.close()

    acc_curve_path = os.path.join(save_dir, 'acc_curve.png')
    plt.figure(figsize=(8, 5))
    if any(v is not None for v in train_acc):
        xs = [e for e, v in zip(epochs, train_acc) if v is not None]
        ys = [v for v in train_acc if v is not None]
        plt.plot(xs, ys, label='train_acc')
    if any(v is not None for v in val_acc):
        xs = [e for e, v in zip(epochs, val_acc) if v is not None]
        ys = [v for v in val_acc if v is not None]
        plt.plot(xs, ys, label='val_acc')
    plt.xlabel('global epoch')
    plt.ylabel('accuracy (%)')
    plt.title('FitNet training accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_curve_path, dpi=180)
    plt.close()
    return loss_curve_path, acc_curve_path


def _load_data(args):
    if args.full_train:
        train_loader, test_loader = get_cifar100_fulltrain(
            DATA_DIR,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        return train_loader, None, test_loader, (len(train_loader.dataset), 0, len(test_loader.dataset))

    train_loader, val_loader, test_loader, sizes = get_cifar100(
        DATA_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=TRAIN_VAL_SPLIT,
        seed=args.seed,
    )
    return train_loader, val_loader, test_loader, sizes


def _generate_roc_plot(student, dataloader, device, save_dir: str) -> str:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import auc, roc_curve
        from sklearn.preprocessing import label_binarize
        import numpy as np
    except Exception as exc:
        print(f'[plot] ROC dependencies unavailable ({exc}); skipping ROC.')
        return ''

    logits, labels = collect_logits_and_labels(student, dataloader, device)
    probs = torch.softmax(logits, dim=1).numpy()
    y_true = labels.numpy()
    y_bin = label_binarize(y_true, classes=list(range(probs.shape[1])))

    # Micro ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    # Macro ROC
    per_class_fpr = {}
    per_class_tpr = {}
    for i in range(probs.shape[1]):
        per_class_fpr[i], per_class_tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])

    all_fpr = np.unique(np.concatenate([per_class_fpr[i] for i in range(probs.shape[1])]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(probs.shape[1]):
        mean_tpr += np.interp(all_fpr, per_class_fpr[i], per_class_tpr[i])
    mean_tpr /= probs.shape[1]
    roc_auc_macro = auc(all_fpr, mean_tpr)

    _save_json(
        os.path.join(save_dir, 'roc_metrics.json'),
        {'auc_micro': float(roc_auc_micro), 'auc_macro': float(roc_auc_macro)},
    )

    roc_path = os.path.join(save_dir, 'roc_curve_micro_macro.png')
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_micro, tpr_micro, label=f'Micro ROC (AUC={roc_auc_micro:.3f})')
    plt.plot(all_fpr, mean_tpr, label=f'Macro ROC (AUC={roc_auc_macro:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC multi-class — CIFAR-100 (micro & macro)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(roc_path, dpi=180)
    plt.close()
    return roc_path


def _mark_progress(save_dir: str, **payload: Any) -> None:
    _save_progress(save_dir, payload)


def _load_teacher(model_name: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    teacher = get_model(model_name, NUM_CLASSES).to(device)
    ckpt = _safe_torch_load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    teacher.load_state_dict(state_dict)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == 'cuda':
        return autocast(device_type='cuda')
    return nullcontext()


def _maybe_mixed_batch(images, labels, device, use_mixup, mixup_alpha, use_cutmix, cutmix_alpha):
    mixed = False
    y_a = labels
    y_b = labels
    lam = 1.0
    if use_mixup and use_cutmix:
        if random.random() < 0.5:
            images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha, device=device)
        else:
            images, y_a, y_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha, device=device)
        mixed = True
    elif use_mixup:
        images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha, device=device)
        mixed = True
    elif use_cutmix:
        images, y_a, y_b, lam = cutmix_data(images, labels, alpha=cutmix_alpha, device=device)
        mixed = True
    return images, mixed, y_a, y_b, lam


def _mixed_top1_correct(logits: torch.Tensor, targets_a: torch.Tensor, targets_b: torch.Tensor, lam: float) -> float:
    preds = logits.argmax(dim=1)
    correct_a = preds.eq(targets_a).sum().item()
    correct_b = preds.eq(targets_b).sum().item()
    return float(lam * correct_a + (1.0 - lam) * correct_b)


def _optimizer_parameters(optimizer) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    for group in optimizer.param_groups:
        params.extend(group['params'])
    return params


def _save_stage1_checkpoint(
    save_dir: str,
    student: nn.Module,
    distiller: FitNetHintDistiller,
    hint_optimizer,
    scaler,
    run_cfg: Dict[str, Any],
    hint_epoch: int,
    global_epoch: int,
) -> None:
    save_checkpoint(
        {
            'epoch': 0,
            'global_epoch': int(global_epoch),
            'hint_epoch': int(hint_epoch),
            'hint_epochs_completed': int(hint_epoch),
            'stage1_completed': False,
            'state_dict': student.state_dict(),
            'fitnet_regressor': distiller.regressor.state_dict() if distiller.regressor is not None else None,
            'hint_optimizer': hint_optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler is not None else None,
            'best_val_acc': 0.0,
            'best_epoch': 0,
            'args': run_cfg,
        },
        is_best=False,
        save_dir=save_dir,
        filename='hint_checkpoint.pth',
    )


def _train_one_epoch_hint(
    distiller: FitNetHintDistiller,
    dataloader,
    optimizer,
    device: torch.device,
    hint_weight: float,
    scaler=None,
    max_grad_norm: Optional[float] = None,
) -> float:
    distiller.student.train()
    distiller.teacher.eval()
    assert distiller.regressor is not None
    distiller.regressor.train()

    total_loss = 0.0
    total_seen = 0
    amp_enabled = (scaler is not None) and (device.type == 'cuda')

    for images, _labels in tqdm(dataloader, desc='FitNet hint stage', leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_enabled):
            with torch.no_grad():
                _ = distiller.teacher_forward(images)
            _ = distiller.student_forward(images)
            loss_hint = float(hint_weight) * distiller.hint_loss()

        if not torch.isfinite(loss_hint).all():
            raise FloatingPointError(f'Non-finite hint loss detected during training: {float(loss_hint.detach().cpu())}')

        if amp_enabled:
            scaler.scale(loss_hint).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_hint.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer), max_grad_norm)
            optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss_hint.item()) * batch_size
        total_seen += batch_size

    return total_loss / max(total_seen, 1)


@torch.inference_mode()
def _eval_hint_stage(distiller: FitNetHintDistiller, dataloader, device: torch.device, hint_weight: float) -> float:
    distiller.student.eval()
    distiller.teacher.eval()
    if distiller.regressor is not None:
        distiller.regressor.eval()
    total_loss = 0.0
    total_seen = 0
    amp_enabled = device.type == 'cuda'

    for images, _labels in tqdm(dataloader, desc='Hint val', leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        with _autocast_context(device, amp_enabled):
            _ = distiller.teacher_forward(images)
            _ = distiller.student_forward(images)
            loss = float(hint_weight) * distiller.hint_loss()
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_seen += batch_size

    return total_loss / max(total_seen, 1)


def _train_one_epoch_stage2(
    teacher: nn.Module,
    student: nn.Module,
    dataloader,
    kd_loss_fn: KDLoss,
    optimizer,
    device: torch.device,
    use_mixup: bool,
    mixup_alpha: float,
    use_cutmix: bool,
    cutmix_alpha: float,
    scaler=None,
    max_grad_norm: Optional[float] = None,
) -> Tuple[float, float, float, float]:
    teacher.eval()
    student.train()
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_correct = 0.0
    total_seen = 0
    amp_enabled = (scaler is not None) and (device.type == 'cuda')

    for images, labels in tqdm(dataloader, desc='FitNet stage 2', leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images, mixed, y_a, y_b, lam = _maybe_mixed_batch(images, labels, device, use_mixup, mixup_alpha, use_cutmix, cutmix_alpha)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, amp_enabled):
            with torch.no_grad():
                logits_t = teacher(images)
            logits_s = student(images)

            if mixed:
                hard_a = kd_loss_fn.hard_loss(logits_s, y_a)
                hard_b = kd_loss_fn.hard_loss(logits_s, y_b)
                loss_ce = lam * hard_a + (1.0 - lam) * hard_b
            else:
                loss_ce = kd_loss_fn.hard_loss(logits_s, labels)
            loss_kd = kd_loss_fn.soft_loss(logits_s, logits_t)
            loss = kd_loss_fn.combine(loss_ce, loss_kd)

        if not torch.isfinite(loss).all():
            raise FloatingPointError(f'Non-finite stage-2 loss detected during training: {float(loss.detach().cpu())}')

        if amp_enabled:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
            optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_ce += float(loss_ce.item()) * batch_size
        total_kd += float(loss_kd.item()) * batch_size
        if mixed:
            total_correct += _mixed_top1_correct(logits_s.detach(), y_a, y_b, lam)
        else:
            preds = logits_s.argmax(dim=1)
            total_correct += float(preds.eq(labels).sum().item())
        total_seen += batch_size

    denom = max(total_seen, 1)
    return total_loss / denom, total_ce / denom, total_kd / denom, 100.0 * total_correct / denom


@torch.inference_mode()
def _eval_student(student: nn.Module, dataloader, device: torch.device) -> Tuple[float, float]:
    student.eval()
    criterion = nn.CrossEntropyLoss()
    amp_enabled = device.type == 'cuda'
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in tqdm(dataloader, desc='Val', leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _autocast_context(device, amp_enabled):
            logits = student(images)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int(logits.argmax(dim=1).eq(labels).sum().item())
        total_seen += batch_size

    denom = max(total_seen, 1)
    return total_loss / denom, 100.0 * total_correct / denom


def main():
    args = parse_args()
    stage2_epochs = _effective_stage2_epochs(args)
    stage2_lr = _effective_stage2_lr(args)

    if args.hint_epochs < 0:
        raise ValueError('--hint_epochs must be >= 0')
    if stage2_epochs <= 0:
        raise ValueError('stage-2 epochs must be > 0; use --epochs or --stage2_epochs')
    if stage2_lr <= 0:
        raise ValueError('stage-2 lr must be > 0; use --lr or --stage2_lr')
    if args.hint_lr <= 0:
        raise ValueError('--hint_lr must be > 0')
    if args.hint_weight <= 0:
        raise ValueError('--hint_weight must be > 0')
    if args.kd_temperature <= 0:
        raise ValueError('--kd_temperature must be > 0')
    if not (0.0 <= args.kd_alpha <= 1.0):
        raise ValueError('--kd_alpha must be in [0, 1]')
    if not (0.0 <= args.hard_label_smoothing < 1.0):
        raise ValueError('--hard_label_smoothing must be in [0, 1)')

    teacher_hint_layers = parse_layer_names(args.teacher_hint_layers)
    student_guided_layers = parse_layer_names(args.student_guided_layers)
    if len(teacher_hint_layers) != 1 or len(student_guided_layers) != 1:
        raise ValueError('FitNet paper-style training uses a single hint layer pair; provide exactly one teacher and one student layer')

    set_seed(args.seed, benchmark=args.benchmark, allow_tf32=args.allow_tf32)
    device = torch.device(f'cuda:{args.gpu}') if (args.gpu >= 0 and torch.cuda.is_available()) else torch.device('cpu')

    save_dir = os.path.join(SAVE_DIR, args.save_subdir, args.student_model, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    run_cfg = vars(args).copy()
    run_cfg['resolved_stage2_epochs'] = stage2_epochs
    run_cfg['resolved_stage2_lr'] = stage2_lr
    _save_json(os.path.join(save_dir, 'run_config.json'), run_cfg)

    print(f'Device: {device}')
    print(f'Save dir: {save_dir}')

    wandb = _init_wandb(args, save_dir)

    train_loader, val_loader, test_loader, sizes = _load_data(args)
    print('Dataset sizes:', sizes)

    teacher = _load_teacher(args.teacher_model, args.teacher_checkpoint, device)
    student = get_model(args.student_model, NUM_CLASSES).to(device)

    kd_loss_fn = KDLoss(
        temperature=float(args.kd_temperature),
        alpha=float(args.kd_alpha),
        hard_label_smoothing=float(args.hard_label_smoothing),
    )
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    distiller = FitNetHintDistiller(
        teacher=teacher,
        student=student,
        teacher_hint_layer=teacher_hint_layers[0],
        student_guided_layer=student_guided_layers[0],
    )

    try:
        first_batch_images = next(iter(train_loader))[0].to(device, non_blocking=True)
    except StopIteration as exc:
        raise RuntimeError('Training loader is empty; cannot initialize FitNet regressor.') from exc
    distiller.build_regressor_from_batch(first_batch_images)
    assert distiller.regressor is not None

    guided_params = collect_trainable_params_through_layer(student, student_guided_layers[0])
    hint_optimizer = optim.SGD(
        list(guided_params) + list(distiller.regressor.parameters()),
        lr=args.hint_lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    stage2_optimizer = optim.SGD(student.parameters(), lr=stage2_lr, momentum=args.momentum, weight_decay=args.wd)
    stage2_scheduler = get_scheduler(
        stage2_optimizer,
        stage2_epochs,
        scheduler_type=args.scheduler,
        milestones=tuple(args.milestones),
        gamma=args.gamma,
    )

    history: Dict[str, List[Any]] = {
        'epoch': [],
        'phase': [],
        'hint_loss': [],
        'hint_val_loss': [],
        'train_loss': [],
        'train_loss_ce': [],
        'train_loss_kd': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
    }
    existing_history = _load_existing_history(save_dir)
    if existing_history is not None and not args.resume:
        history = existing_history

    start_stage2_epoch = 1
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    last_epoch_ran = 0
    hint_completed_epochs = 0
    global_epoch = int(history['epoch'][-1]) if history.get('epoch') else 0
    stage1_completed = False

    if args.resume:
        ckpt = _safe_torch_load(args.resume, map_location='cpu')
        student.load_state_dict(ckpt['state_dict'])
        if distiller.regressor is not None and ckpt.get('fitnet_regressor') is not None:
            distiller.regressor.load_state_dict(ckpt['fitnet_regressor'])
        if scaler is not None and ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])

        hint_completed_epochs = int(ckpt.get('hint_epochs_completed', 0) or 0)
        stage1_completed = bool(ckpt.get('stage1_completed', False))
        best_val_acc = float(ckpt.get('best_val_acc', 0.0) or 0.0)
        best_epoch = int(ckpt.get('best_epoch', 0) or 0)
        global_epoch = int(ckpt.get('global_epoch', global_epoch) or global_epoch)

        if stage1_completed:
            if ckpt.get('optimizer') is not None:
                stage2_optimizer.load_state_dict(ckpt['optimizer'])
            if ckpt.get('scheduler') is not None:
                stage2_scheduler.load_state_dict(ckpt['scheduler'])
            start_stage2_epoch = int(ckpt.get('epoch', 0) or 0) + 1
            print(f'Resumed from {args.resume} at stage-2 epoch {start_stage2_epoch}')
        else:
            if ckpt.get('hint_optimizer') is not None:
                hint_optimizer.load_state_dict(ckpt['hint_optimizer'])
            print(f'Resumed from {args.resume} at hint epoch {hint_completed_epochs + 1}')

    use_mixup = not args.no_mixup
    use_cutmix = not args.no_cutmix

    try:
        if not stage1_completed and args.hint_epochs > 0:
            for hint_epoch in range(hint_completed_epochs + 1, args.hint_epochs + 1):
                global_epoch += 1
                print(f'\nHint stage epoch {hint_epoch}/{args.hint_epochs}')
                hint_loss = _train_one_epoch_hint(
                    distiller=distiller,
                    dataloader=train_loader,
                    optimizer=hint_optimizer,
                    device=device,
                    hint_weight=float(args.hint_weight),
                    scaler=scaler,
                    max_grad_norm=(args.max_grad_norm if args.max_grad_norm > 0 else None),
                )
                hint_val_loss: Optional[float] = None
                if val_loader is not None:
                    hint_val_loss = _eval_hint_stage(distiller, val_loader, device, hint_weight=float(args.hint_weight))

                hint_completed_epochs = hint_epoch
                history['epoch'].append(global_epoch)
                history['phase'].append('hint')
                history['hint_loss'].append(float(hint_loss))
                history['hint_val_loss'].append(float(hint_val_loss) if hint_val_loss is not None else None)
                history['train_loss'].append(float(hint_loss))
                history['train_loss_ce'].append(0.0)
                history['train_loss_kd'].append(0.0)
                history['train_acc'].append(None)
                history['val_loss'].append(float(hint_val_loss) if hint_val_loss is not None else None)
                history['val_acc'].append(None)
                history['lr'].append(float(hint_optimizer.param_groups[0]['lr']))

                _save_history(history, save_dir)
                _save_stage1_checkpoint(
                    save_dir=save_dir,
                    student=student,
                    distiller=distiller,
                    hint_optimizer=hint_optimizer,
                    scaler=scaler,
                    run_cfg=run_cfg,
                    hint_epoch=hint_epoch,
                    global_epoch=global_epoch,
                )
                _mark_progress(
                    save_dir,
                    status='hint_stage_running',
                    phase='hint',
                    epoch=int(global_epoch),
                    hint_epoch=int(hint_epoch),
                    hint_epochs_completed=int(hint_completed_epochs),
                    hint_loss=float(hint_loss),
                    hint_val_loss=None if hint_val_loss is None else float(hint_val_loss),
                    best_val_acc=None,
                )

                if wandb is not None:
                    payload = {
                        'epoch': global_epoch,
                        'hint/epoch': hint_epoch,
                        'hint/loss': float(hint_loss),
                        'hint/lr': float(hint_optimizer.param_groups[0]['lr']),
                    }
                    if hint_val_loss is not None:
                        payload['hint/val_loss'] = float(hint_val_loss)
                    wandb.log(payload, step=global_epoch)

            stage1_completed = True
            _mark_progress(
                save_dir,
                status='hint_stage_completed',
                phase='hint',
                epoch=int(global_epoch),
                hint_epochs_completed=int(hint_completed_epochs),
            )
        elif args.hint_epochs == 0:
            stage1_completed = True

        for epoch in range(start_stage2_epoch, stage2_epochs + 1):
            last_epoch_ran = epoch
            global_epoch += 1
            print(f'\nStage 2 epoch {epoch}/{stage2_epochs}')
            train_loss, train_ce, train_kd, train_acc = _train_one_epoch_stage2(
                teacher=teacher,
                student=student,
                dataloader=train_loader,
                kd_loss_fn=kd_loss_fn,
                optimizer=stage2_optimizer,
                device=device,
                use_mixup=use_mixup,
                mixup_alpha=args.mixup_alpha,
                use_cutmix=use_cutmix,
                cutmix_alpha=args.cutmix_alpha,
                scaler=scaler,
                max_grad_norm=(args.max_grad_norm if args.max_grad_norm > 0 else None),
            )
            val_loss: Optional[float] = None
            val_acc: Optional[float] = None
            is_best = False
            if val_loader is not None:
                val_loss, val_acc = _eval_student(student, val_loader, device)
                prev_best = best_val_acc
                is_best = float(val_acc) > prev_best
                if is_best:
                    best_val_acc = float(val_acc)
                    best_epoch = epoch
                if args.early_stop and epoch >= args.es_start_epoch:
                    improved = float(val_acc) > prev_best + args.es_min_delta
                    no_improve = 0 if improved else no_improve + 1
                else:
                    no_improve = 0
            else:
                best_epoch = epoch

            stage2_scheduler.step()
            lr_next = float(stage2_optimizer.param_groups[0]['lr'])

            print(f'Train total: {train_loss:.4f} | CE: {train_ce:.4f} | KD: {train_kd:.4f} | acc: {train_acc:.2f}')
            if val_loader is not None and val_loss is not None and val_acc is not None:
                print(f'Val   loss: {val_loss:.4f}, acc: {val_acc:.2f}')

            history['epoch'].append(global_epoch)
            history['phase'].append('stage2')
            history['hint_loss'].append(None)
            history['hint_val_loss'].append(None)
            history['train_loss'].append(float(train_loss))
            history['train_loss_ce'].append(float(train_ce))
            history['train_loss_kd'].append(float(train_kd))
            history['train_acc'].append(float(train_acc))
            history['lr'].append(lr_next)
            history['val_loss'].append(float(val_loss) if val_loss is not None else None)
            history['val_acc'].append(float(val_acc) if val_acc is not None else None)

            if wandb is not None:
                payload = {
                    'epoch': global_epoch,
                    'stage2/epoch': epoch,
                    'train/loss_total': float(train_loss),
                    'train/loss_ce': float(train_ce),
                    'train/loss_kd': float(train_kd),
                    'train/acc': float(train_acc),
                    'lr': lr_next,
                }
                if val_loader is not None and val_loss is not None and val_acc is not None:
                    payload['val/loss'] = float(val_loss)
                    payload['val/acc'] = float(val_acc)
                    payload['val/best_acc_so_far'] = float(best_val_acc)
                wandb.log(payload, step=global_epoch)

            save_checkpoint(
                {
                    'epoch': epoch,
                    'global_epoch': int(global_epoch),
                    'state_dict': student.state_dict(),
                    'fitnet_regressor': distiller.regressor.state_dict() if distiller.regressor is not None else None,
                    'optimizer': stage2_optimizer.state_dict(),
                    'scheduler': stage2_scheduler.state_dict() if stage2_scheduler is not None else None,
                    'scaler': scaler.state_dict() if scaler is not None else None,
                    'best_val_acc': float(best_val_acc),
                    'best_epoch': int(best_epoch),
                    'hint_epochs_completed': int(hint_completed_epochs),
                    'stage1_completed': True,
                    'args': run_cfg,
                },
                is_best=is_best,
                save_dir=save_dir,
            )
            _save_history(history, save_dir)
            _mark_progress(
                save_dir,
                status='running',
                phase='stage2',
                epoch=int(global_epoch),
                stage2_epoch=int(epoch),
                lr=lr_next,
                train_loss=float(train_loss),
                train_loss_ce=float(train_ce),
                train_loss_kd=float(train_kd),
                train_acc=float(train_acc),
                val_loss=None if val_loss is None else float(val_loss),
                val_acc=None if val_acc is None else float(val_acc),
                best_val_acc=None if val_loader is None else float(best_val_acc),
                best_epoch=None if val_loader is None else int(best_epoch),
                no_improve=None if val_loader is None else int(no_improve),
                hint_epochs_completed=int(hint_completed_epochs),
                stage1_completed=True,
                epochs_completed=int(epoch),
            )

            if val_loader is not None and args.early_stop and epoch >= args.es_start_epoch and no_improve >= args.es_patience:
                msg = f'Early stopping triggered at epoch {epoch}.'
                print(msg)
                _mark_progress(
                    save_dir,
                    status='early_stopped',
                    phase='stage2',
                    message=msg,
                    epoch=int(global_epoch),
                    best_val_acc=float(best_val_acc),
                    best_epoch=int(best_epoch),
                    hint_epochs_completed=int(hint_completed_epochs),
                    epochs_completed=int(epoch),
                )
                break

        checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
        best_path = os.path.join(save_dir, 'best.pth')
        if val_loader is not None and os.path.isfile(best_path):
            best_ckpt = _safe_torch_load(best_path, map_location='cpu')
            student.load_state_dict(best_ckpt['state_dict'])
            best_val_acc = float(best_ckpt.get('best_val_acc', best_val_acc))
            best_epoch = int(best_ckpt.get('best_epoch', best_epoch))
            print(f'Loaded best checkpoint from epoch {best_epoch} (val_acc={best_val_acc:.2f})')
        elif os.path.isfile(checkpoint_path):
            shutil.copyfile(checkpoint_path, best_path)

        history_path = _save_history(history, save_dir)
        loss_curve_path, acc_curve_path = ('', '')
        if not args.skip_plots:
            loss_curve_path, acc_curve_path = _generate_curves(history, save_dir)

        summary = {
            'teacher_model': args.teacher_model,
            'student_model': args.student_model,
            'teacher_checkpoint': args.teacher_checkpoint,
            'exp_name': args.exp_name,
            'save_dir': save_dir,
            'hint_epochs_requested': int(args.hint_epochs),
            'hint_epochs_completed': int(hint_completed_epochs),
            'stage2_epochs_requested': int(stage2_epochs),
            'stage2_epochs_completed': int(last_epoch_ran),
            'epochs_requested': int(stage2_epochs),
            'epochs_completed': int(last_epoch_ran),
            'best_epoch': int(best_epoch),
            'best_val_acc': None if val_loader is None else float(best_val_acc),
            'full_train': bool(args.full_train),
            'history_path': history_path,
            'fitnet_method': 'paper_two_stage_hint_then_kd',
            'hint_weight': float(args.hint_weight),
            'kd_temperature': float(args.kd_temperature),
            'kd_alpha': float(args.kd_alpha),
            'teacher_hint_layers': teacher_hint_layers,
            'student_guided_layers': student_guided_layers,
        }

        test_acc: Optional[float] = None
        roc_path: Optional[str] = None
        conf_path: Optional[str] = None
        if not args.skip_test_metrics:
            test_acc, per_class_acc, conf_mat = eval_on_test(student, test_loader, device, NUM_CLASSES)
            summary['test_acc'] = float(test_acc)
            summary['per_class_acc_mean'] = float(per_class_acc.mean().item())
            conf_path = os.path.join(save_dir, 'confusion_matrix.pt')
            torch.save(conf_mat, conf_path)
            summary['confusion_matrix_path'] = conf_path
            if not args.skip_plots:
                roc_path = _generate_roc_plot(student, test_loader, device, save_dir)
                if roc_path:
                    summary['roc_path'] = roc_path
            print(f'Test acc: {float(test_acc):.2f}')

        if loss_curve_path:
            summary['loss_curve_path'] = loss_curve_path
        if acc_curve_path:
            summary['acc_curve_path'] = acc_curve_path

        summary_path = os.path.join(save_dir, 'summary.json')
        _save_json(summary_path, summary)
        _mark_progress(
            save_dir,
            status='completed',
            phase='done',
            epoch=int(global_epoch),
            epochs_completed=int(last_epoch_ran),
            best_val_acc=None if val_loader is None else float(best_val_acc),
            best_epoch=None if val_loader is None else int(best_epoch),
            hint_epochs_completed=int(hint_completed_epochs),
            test_acc=None if test_acc is None else float(test_acc),
        )

        if wandb is not None:
            final_payload = {
                'summary/best_epoch': int(best_epoch),
                'summary/hint_epochs_completed': int(hint_completed_epochs),
                'summary/stage2_epochs_completed': int(last_epoch_ran),
            }
            if val_loader is not None:
                final_payload['summary/best_val_acc'] = float(best_val_acc)
            if test_acc is not None:
                final_payload['summary/test_acc'] = float(test_acc)
            wandb.log(final_payload, step=global_epoch)
            if args.wandb_log_artifacts:
                try:
                    artifact = wandb.Artifact(f'fitnet-{args.exp_name}', type='model')
                    best_path = os.path.join(save_dir, 'best.pth')
                    if os.path.isfile(best_path):
                        artifact.add_file(best_path)
                    artifact.add_file(summary_path)
                    artifact.add_file(history_path)
                    if loss_curve_path:
                        artifact.add_file(loss_curve_path)
                    if acc_curve_path:
                        artifact.add_file(acc_curve_path)
                    if roc_path:
                        artifact.add_file(roc_path)
                    if conf_path:
                        artifact.add_file(conf_path)
                    wandb.log_artifact(artifact)
                except Exception as exc:
                    print(f'[W&B] artifact logging failed ({exc})')
            wandb.finish()

    except Exception as exc:
        _mark_progress(save_dir, status='failed', phase='error', message=str(exc))
        if wandb is not None:
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass
        raise
    finally:
        distiller.remove_hooks()


if __name__ == '__main__':
    main()
