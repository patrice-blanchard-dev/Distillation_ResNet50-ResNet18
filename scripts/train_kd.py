"""Train a student with classic Knowledge Distillation."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

if __package__ is None or __package__ == '':
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

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
from datasets import get_cifar100, get_cifar100_fulltrain
from distillation.kd import KDLoss
from models import get_model
from utils.main_utils import (
    collect_logits_and_labels,
    cutmix_data,
    eval_on_test,
    get_scheduler,
    mixup_data,
    save_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train KD student on CIFAR-100')
    parser.add_argument('--teacher_model', type=str, default='resnet50_cifar')
    parser.add_argument('--student_model', type=str, default='resnet18_cifar')
    parser.add_argument('--teacher_checkpoint', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--save_subdir', type=str, default='kd')

    parser.add_argument('--epochs', type=int, default=EPOCHS_STUDENT)
    parser.add_argument('--lr', type=float, default=0.08)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'multistep'])
    parser.add_argument('--milestones', type=int, nargs='+', default=[150, 200])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--max_grad_norm', type=float, default=0.0)

    parser.add_argument('--kd_temperature', type=float, default=4.0)
    parser.add_argument('--kd_alpha', type=float, default=0.9)
    parser.add_argument('--hard_label_smoothing', type=float, default=0.0)

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
    parser.add_argument('--wandb_tags', type=str, default='kd')
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


def _atomic_json_save(path: str, payload: Dict) -> None:
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


def _save_json(path: str, payload: Dict) -> None:
    _atomic_json_save(path, payload)


def _save_history(history: Dict[str, List[float]], save_dir: str) -> str:
    history_path = os.path.join(save_dir, 'history.json')
    _save_json(history_path, history)
    return history_path


def _save_progress(save_dir: str, payload: Dict) -> str:
    progress_path = os.path.join(save_dir, 'progress.json')
    _save_json(progress_path, payload)
    return progress_path


def _load_existing_history(save_dir: str) -> Optional[Dict[str, List[float]]]:
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
    run_name = args.wandb_run_name.strip() or f'kd/{args.student_model}/{args.exp_name}'
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
            'stage': 'kd',
            'teacher_model': args.teacher_model,
            'student_model': args.student_model,
            'teacher_checkpoint': args.teacher_checkpoint,
            'epochs': args.epochs,
            'lr': args.lr,
            'wd': args.wd,
            'momentum': args.momentum,
            'scheduler': args.scheduler,
            'milestones': args.milestones,
            'gamma': args.gamma,
            'kd_temperature': args.kd_temperature,
            'kd_alpha': args.kd_alpha,
            'hard_label_smoothing': args.hard_label_smoothing,
            'mixup': (not args.no_mixup),
            'mixup_alpha': args.mixup_alpha,
            'cutmix': (not args.no_cutmix),
            'cutmix_alpha': args.cutmix_alpha,
            'max_grad_norm': args.max_grad_norm,
            'full_train': args.full_train,
            'skip_test_metrics': args.skip_test_metrics,
            'skip_plots': args.skip_plots,
            'early_stop': args.early_stop,
            'es_patience': args.es_patience,
            'es_min_delta': args.es_min_delta,
            'es_start_epoch': args.es_start_epoch,
            'seed': args.seed,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers,
            'benchmark': args.benchmark,
            'allow_tf32': args.allow_tf32,
            'save_dir': save_dir,
        },
        reinit=True,
    )
    return wandb


def _generate_curves(history: Dict[str, List[float]], save_dir: str) -> Tuple[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib unavailable, skipping training curves.')
        return '', ''

    epochs = history['epoch']
    if not epochs:
        return '', ''

    loss_curve_path = os.path.join(save_dir, 'loss_curve.png')
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train KD total')
    plt.plot(epochs, history['train_loss_ce'], label='Train CE')
    plt.plot(epochs, history['train_loss_kd'], label='Train KD')
    if history['val_loss']:
        plt.plot(epochs, history['val_loss'], label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('KD Loss vs Epoch')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(loss_curve_path)
    plt.close()

    acc_curve_path = os.path.join(save_dir, 'accuracy_curve.png')
    plt.figure()
    plt.plot(epochs, history['train_acc'], label='Train acc')
    if history['val_acc']:
        plt.plot(epochs, history['val_acc'], label='Val acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy vs Epoch')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(acc_curve_path)
    plt.close()
    return loss_curve_path, acc_curve_path


def _load_data(args):
    if args.full_train:
        train_loader, test_loader = get_cifar100_fulltrain(
            DATA_DIR,
            args.batch_size,
            args.num_workers,
            seed=args.seed,
        )
        val_loader = None
        sizes = {'train': len(train_loader.dataset), 'test': len(test_loader.dataset)}
    else:
        train_loader, val_loader, test_loader, split_sizes = get_cifar100(
            DATA_DIR,
            args.batch_size,
            args.num_workers,
            val_split=TRAIN_VAL_SPLIT,
            seed=args.seed,
        )
        sizes = {'train': split_sizes[0], 'val': split_sizes[1], 'test': split_sizes[2]}
    return train_loader, val_loader, test_loader, sizes


def _maybe_generate_roc(model, dataloader, device, save_dir: str) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import auc, roc_curve
        from sklearn.preprocessing import label_binarize
    except Exception as exc:
        print(f'ROC skipped ({exc}).')
        return None

    logits, labels = collect_logits_and_labels(model, dataloader, device)
    probs = torch.softmax(logits, dim=1).numpy()
    y_true = labels.numpy()
    y_bin = label_binarize(y_true, classes=list(range(probs.shape[1])))

    fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

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
    plt.savefig(roc_path)
    plt.close()
    return roc_path


def _mark_progress(save_dir: str, status: str, message: Optional[str] = None, **extra) -> None:
    payload: Dict[str, object] = {'status': status}
    if message:
        payload['message'] = message
    payload.update(extra)
    _save_progress(save_dir, payload)


@torch.no_grad()
def _load_teacher(model_name: str, checkpoint_path: str, device: torch.device) -> nn.Module:
    teacher = get_model(model_name, NUM_CLASSES).to(device)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'Teacher checkpoint not found: {checkpoint_path}')
    ckpt = _safe_torch_load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    teacher.load_state_dict(state_dict, strict=True)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def _maybe_mixed_batch(images, labels, device, use_mixup: bool, mixup_alpha: float, use_cutmix: bool, cutmix_alpha: float):
    mixed = False
    y_a = y_b = labels
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


def _autocast_context(device: torch.device, enabled: bool):
    if enabled and device.type == 'cuda':
        return autocast(device_type='cuda')
    return nullcontext()


def _train_one_epoch_kd(
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
    student.train()
    teacher.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_kd = 0.0
    total_correct = 0.0
    total_seen = 0
    amp_enabled = (scaler is not None) and (device.type == 'cuda')

    for images, labels in tqdm(dataloader, desc='Train KD', leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images, mixed, y_a, y_b, lam = _maybe_mixed_batch(
            images, labels, device, use_mixup, mixup_alpha, use_cutmix, cutmix_alpha
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = teacher(images)

        with _autocast_context(device, amp_enabled):
            student_logits = student(images)
            if mixed:
                ce_loss = lam * kd_loss_fn.hard_loss(student_logits, y_a) + (1.0 - lam) * kd_loss_fn.hard_loss(student_logits, y_b)
                kd_term = kd_loss_fn.soft_loss(student_logits, teacher_logits)
                loss = kd_loss_fn.combine(ce_loss, kd_term)
            else:
                loss, metrics = kd_loss_fn(student_logits, teacher_logits, labels)
                ce_loss = kd_loss_fn.hard_loss(student_logits, labels)
                kd_term = kd_loss_fn.soft_loss(student_logits, teacher_logits)

        if not torch.isfinite(loss).all():
            raise FloatingPointError(f'Non-finite KD loss detected: {float(loss.detach().cpu())}')

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
        total_ce += float(ce_loss.item()) * batch_size
        total_kd += float(kd_term.item()) * batch_size
        preds = student_logits.detach().argmax(dim=1)
        if mixed:
            correct = lam * preds.eq(y_a).sum().item() + (1.0 - lam) * preds.eq(y_b).sum().item()
        else:
            correct = preds.eq(labels).sum().item()
        total_correct += float(correct)
        total_seen += batch_size

    denom = max(total_seen, 1)
    return total_loss / denom, total_ce / denom, total_kd / denom, 100.0 * total_correct / denom


@torch.inference_mode()
def _eval_student(student: nn.Module, dataloader, device: torch.device) -> Tuple[float, float]:
    student.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    amp_enabled = device.type == 'cuda'
    for images, labels in tqdm(dataloader, desc='Val', leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _autocast_context(device, amp_enabled):
            logits = student(images)
            loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += logits.argmax(dim=1).eq(labels).sum().item()
        total_seen += batch_size
    return total_loss / max(total_seen, 1), 100.0 * total_correct / max(total_seen, 1)


def main() -> None:
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    set_seed(args.seed, benchmark=args.benchmark, allow_tf32=args.allow_tf32)

    save_dir = os.path.join(SAVE_DIR, args.save_subdir, args.student_model, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    _save_json(os.path.join(save_dir, 'run_config.json'), vars(args))
    _mark_progress(save_dir, status='starting', message='initializing KD run')
    print('Save dir:', save_dir)

    wandb = _init_wandb(args, save_dir)
    train_loader, val_loader, test_loader, sizes = _load_data(args)
    print('Dataset sizes:', sizes)

    teacher = _load_teacher(args.teacher_model, args.teacher_checkpoint, device)
    student = get_model(args.student_model, NUM_CLASSES).to(device)
    optimizer = optim.SGD(student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = get_scheduler(optimizer, args.epochs, scheduler_type=args.scheduler, milestones=tuple(args.milestones), gamma=args.gamma)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    kd_loss_fn = KDLoss(
        temperature=args.kd_temperature,
        alpha=args.kd_alpha,
        hard_label_smoothing=args.hard_label_smoothing,
    )

    history = {
        'epoch': [], 'train_loss': [], 'train_loss_ce': [], 'train_loss_kd': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []
    }
    existing_history = _load_existing_history(save_dir)
    if existing_history is not None:
        history = existing_history

    start_epoch = 1
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    last_epoch_ran = 0

    if args.resume:
        ckpt = _safe_torch_load(args.resume, map_location='cpu')
        student.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt and ckpt['scheduler'] is not None:
            scheduler.load_state_dict(ckpt['scheduler'])
        if scaler is not None and 'scaler' in ckpt and ckpt['scaler'] is not None:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        best_val_acc = float(ckpt.get('best_val_acc', 0.0) or 0.0)
        best_epoch = int(ckpt.get('best_epoch', 0) or 0)
        print(f'Resumed from {args.resume} at epoch {start_epoch}')

    use_mixup = not args.no_mixup
    use_cutmix = not args.no_cutmix

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            last_epoch_ran = epoch
            print(f'\nEpoch {epoch}/{args.epochs}')
            train_loss, train_ce, train_kd, train_acc = _train_one_epoch_kd(
                teacher, student, train_loader, kd_loss_fn, optimizer, device,
                use_mixup=use_mixup, mixup_alpha=args.mixup_alpha,
                use_cutmix=use_cutmix, cutmix_alpha=args.cutmix_alpha,
                scaler=scaler, max_grad_norm=(args.max_grad_norm if args.max_grad_norm > 0 else None),
            )
            lr_current = float(optimizer.param_groups[0]['lr'])
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

            print(f'Train total: {train_loss:.4f} | CE: {train_ce:.4f} | KD: {train_kd:.4f} | acc: {train_acc:.2f}')
            if val_loader is not None and val_loss is not None and val_acc is not None:
                print(f'Val   loss: {val_loss:.4f}, acc: {val_acc:.2f}')

            history['epoch'].append(epoch)
            history['train_loss'].append(float(train_loss))
            history['train_loss_ce'].append(float(train_ce))
            history['train_loss_kd'].append(float(train_kd))
            history['train_acc'].append(float(train_acc))
            history['lr'].append(lr_current)
            if val_loader is not None:
                history['val_loss'].append(float(val_loss))
                history['val_acc'].append(float(val_acc))

            if wandb is not None:
                payload = {
                    'epoch': epoch,
                    'train/loss_total': float(train_loss),
                    'train/loss_ce': float(train_ce),
                    'train/loss_kd': float(train_kd),
                    'train/acc': float(train_acc),
                    'lr': lr_current,
                }
                if val_loader is not None:
                    payload['val/loss'] = float(val_loss)
                    payload['val/acc'] = float(val_acc)
                    payload['val/best_acc_so_far'] = float(best_val_acc)
                wandb.log(payload, step=epoch)

            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': student.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'scaler': scaler.state_dict() if scaler is not None else None,
                    'best_val_acc': float(best_val_acc),
                    'best_epoch': int(best_epoch),
                    'args': vars(args),
                },
                is_best=is_best,
                save_dir=save_dir,
            )
            _save_history(history, save_dir)
            _mark_progress(
                save_dir,
                status='running',
                epoch=int(epoch),
                lr=lr_current,
                train_loss=float(train_loss),
                train_loss_ce=float(train_ce),
                train_loss_kd=float(train_kd),
                train_acc=float(train_acc),
                val_loss=None if val_loss is None else float(val_loss),
                val_acc=None if val_acc is None else float(val_acc),
                best_val_acc=None if val_loader is None else float(best_val_acc),
                best_epoch=None if val_loader is None else int(best_epoch),
                no_improve=None if val_loader is None else int(no_improve),
            )
            scheduler.step()

            if val_loader is not None and args.early_stop and epoch >= args.es_start_epoch and no_improve >= args.es_patience:
                msg = f'Early stopping triggered at epoch {epoch}.'
                print(msg)
                _mark_progress(save_dir, status='early_stopped', message=msg, epoch=int(epoch), best_val_acc=float(best_val_acc), best_epoch=int(best_epoch))
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
            'epochs_requested': args.epochs,
            'epochs_completed': last_epoch_ran,
            'best_epoch': best_epoch,
            'best_val_acc': None if val_loader is None else float(best_val_acc),
            'full_train': args.full_train,
            'history_path': history_path,
            'kd_temperature': args.kd_temperature,
            'kd_alpha': args.kd_alpha,
        }

        test_acc: Optional[float] = None
        roc_path: Optional[str] = None
        if not args.skip_test_metrics:
            test_acc, per_class_acc, conf_mat = eval_on_test(student, test_loader, device, NUM_CLASSES)
            print(f'\nTest accuracy: {test_acc:.2f}')
            conf_path = os.path.join(save_dir, 'confusion_matrix.pt')
            torch.save(conf_mat, conf_path)

            class_metrics = {
                'per_class_acc': [float(x) for x in per_class_acc.tolist()],
                'mean_per_class_acc': float(per_class_acc.mean().item()),
            }
            _save_json(os.path.join(save_dir, 'class_metrics.json'), class_metrics)
            _save_json(os.path.join(save_dir, 'test_metrics.json'), {'test_acc': float(test_acc)})
            summary['test_acc'] = float(test_acc)
            summary['mean_per_class_acc'] = float(per_class_acc.mean().item())

            if not args.skip_plots:
                roc_path = _maybe_generate_roc(student, test_loader, device, save_dir)

            roc_metrics_path = os.path.join(save_dir, 'roc_metrics.json')
            roc_metrics = None
            if os.path.isfile(roc_metrics_path):
                with open(roc_metrics_path, 'r', encoding='utf-8') as f:
                    roc_metrics = json.load(f)
                summary['auc_micro'] = float(roc_metrics['auc_micro'])
                summary['auc_macro'] = float(roc_metrics['auc_macro'])

            if wandb is not None:
                log_payload = {
                    'test/acc': float(test_acc),
                    'test/mean_per_class_acc': float(per_class_acc.mean().item()),
                }
                if roc_metrics is not None:
                    log_payload['test/auc_micro'] = float(roc_metrics['auc_micro'])
                    log_payload['test/auc_macro'] = float(roc_metrics['auc_macro'])
                wandb.log(log_payload)
                if os.path.isfile(conf_path):
                    wandb.save(conf_path)
                if os.path.isfile(roc_metrics_path):
                    wandb.save(roc_metrics_path)
                if roc_path is not None and os.path.isfile(roc_path):
                    wandb.save(roc_path)
        else:
            print('\nTest metrics skipped.')

        _save_json(os.path.join(save_dir, 'summary.json'), summary)
        _mark_progress(
            save_dir,
            status='completed',
            epoch=int(last_epoch_ran),
            best_val_acc=None if val_loader is None else float(best_val_acc),
            best_epoch=None if val_loader is None else int(best_epoch),
            test_acc=None if test_acc is None else float(test_acc),
            history_path=history_path,
            summary_path=os.path.join(save_dir, 'summary.json'),
        )

        if wandb is not None:
            for path in [history_path, loss_curve_path, acc_curve_path, os.path.join(save_dir, 'summary.json'), os.path.join(save_dir, 'progress.json')]:
                if path and os.path.isfile(path):
                    wandb.save(path)
            if args.wandb_log_artifacts:
                artifact = wandb.Artifact(
                    name=f'kd-{args.student_model}-{args.exp_name}'.replace('/', '-'),
                    type='model',
                    metadata={
                        'stage': 'kd',
                        'teacher_model': args.teacher_model,
                        'student_model': args.student_model,
                        'kd_temperature': float(args.kd_temperature),
                        'kd_alpha': float(args.kd_alpha),
                        'best_val_acc': None if val_loader is None else float(best_val_acc),
                        'test_acc': None if test_acc is None else float(test_acc),
                    },
                )
                for filename in ['best.pth', 'history.json', 'summary.json', 'progress.json', 'run_config.json', 'loss_curve.png', 'accuracy_curve.png', 'confusion_matrix.pt', 'class_metrics.json', 'test_metrics.json', 'roc_metrics.json', 'roc_curve_micro_macro.png']:
                    path = os.path.join(save_dir, filename)
                    if os.path.isfile(path):
                        artifact.add_file(path, name=filename)
                if wandb.run is not None:
                    wandb.run.log_artifact(artifact)
                    print(f'[W&B] Logged artifact: {artifact.name}')
            wandb.finish()

    except Exception as exc:
        _mark_progress(save_dir, status='failed', message=str(exc), epoch=int(last_epoch_ran))
        if wandb is not None:
            wandb.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
