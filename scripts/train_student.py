"""Train the ResNet-18 student baseline without distillation."""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

# Allow running both as a module (python -m scripts.train_teacher) and as a script
# (python scripts/train_teacher.py) by ensuring the project root is on sys.path.
if __package__ is None or __package__ == '':
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler

from config import (
    BASE_LR_STUDENT,
    BATCH_SIZE,
    DATA_DIR,
    EPOCHS_STUDENT,
    NUM_CLASSES,
    NUM_WORKERS,
    SAVE_DIR,
    SEED,
    TRAIN_VAL_SPLIT,
    WEIGHT_DECAY,
)

# ---- Robust imports for both flat and package layouts ----
try:
    from models import get_model  # type: ignore
except Exception:
    from resnet_cifar import resnet18_cifar, resnet50_cifar  # type: ignore

    def get_model(name: str, num_classes: int = 100):
        name = name.lower().strip()
        if name == "resnet18_cifar":
            return resnet18_cifar(num_classes=num_classes)
        if name == "resnet50_cifar":
            return resnet50_cifar(num_classes=num_classes)
        raise ValueError(f"Unknown model: {name}")

try:
    from datasets import get_cifar100, get_cifar100_fulltrain  # type: ignore
except Exception:
    from cifar100 import get_cifar100, get_cifar100_fulltrain  # type: ignore

try:
    from utils.main_utils import (  # type: ignore
        collect_logits_and_labels,
        eval_on_test,
        eval_one_epoch,
        get_scheduler,
        save_checkpoint,
        train_one_epoch,
    )
except Exception:
    from main_utils import (  # type: ignore
        collect_logits_and_labels,
        eval_on_test,
        eval_one_epoch,
        get_scheduler,
        save_checkpoint,
        train_one_epoch,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Training student baseline on CIFAR-100')
    parser.add_argument('--model', type=str, default='resnet18_cifar')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='default')
    parser.add_argument('--save_subdir', type=str, default='student')

    parser.add_argument('--epochs', type=int, default=EPOCHS_STUDENT)
    parser.add_argument('--lr', type=float, default=BASE_LR_STUDENT)
    parser.add_argument('--wd', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'multistep'])
    parser.add_argument('--milestones', type=int, nargs='+', default=[150, 200])
    parser.add_argument('--gamma', type=float, default=0.1)

    parser.add_argument('--no_mixup', action='store_true', help='Disable Mixup')
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--no_cutmix', action='store_true', help='Disable CutMix')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    parser.add_argument('--max_grad_norm', type=float, default=0.0, help='Gradient clipping norm; 0 disables clipping')

    parser.add_argument('--full_train', action='store_true', help='Use full CIFAR-100 train split (train+val)')
    parser.add_argument('--skip_test_metrics', action='store_true', help='Do not evaluate on test set')
    parser.add_argument('--skip_plots', action='store_true', help='Do not generate training curves / ROC plot')

    parser.add_argument('--early_stop', action='store_true', help='Enable early stopping on val_acc')
    parser.add_argument('--es_patience', type=int, default=15)
    parser.add_argument('--es_min_delta', type=float, default=0.05)
    parser.add_argument('--es_start_epoch', type=int, default=30)

    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=NUM_WORKERS)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--benchmark', action='store_true', help='Enable cuDNN benchmark for speed (less deterministic)')
    parser.add_argument('--allow_tf32', action='store_true', help='Allow TF32 on supported CUDA devices')
    parser.add_argument('--resume', type=str, default='', help='Path to a checkpoint to resume from')

    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='distill_cifar100')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_tags', type=str, default='teacher')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_group', type=str, default='')
    parser.add_argument('--wandb_job_type', type=str, default='train')
    parser.add_argument('--wandb_run_name', type=str, default='')
    parser.add_argument('--wandb_log_artifacts', action='store_true', help='Log best checkpoint as a W&B artifact')
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
    run_name = args.wandb_run_name.strip() or f'student/{args.model}/{args.exp_name}'
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
            'stage': 'student',
            'model': args.model,
            'exp_name': args.exp_name,
            'save_subdir': args.save_subdir,
            'epochs': args.epochs,
            'lr': args.lr,
            'wd': args.wd,
            'momentum': args.momentum,
            'label_smoothing': args.label_smoothing,
            'scheduler': args.scheduler,
            'milestones': args.milestones,
            'gamma': args.gamma,
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


def _atomic_json_save(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix='.__tmp_json_', dir=os.path.dirname(path) or '.')
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
    plt.plot(epochs, history['train_loss'], label='Train loss')
    if history['val_loss']:
        plt.plot(epochs, history['val_loss'], label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs Epoch')
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


def _log_best_checkpoint_artifact(wandb, args, save_dir: str, best_val_acc: Optional[float], test_acc: Optional[float], stopped_epoch: Optional[int]):
    if wandb is None or not args.wandb_log_artifacts:
        return

    best_ckpt_path = os.path.join(save_dir, 'best.pth')
    if not os.path.isfile(best_ckpt_path):
        print(f'[W&B] best checkpoint not found: {best_ckpt_path}')
        return

    artifact = wandb.Artifact(
        name=f'student-{args.model}-{args.exp_name}'.replace('/', '-'),
        type='model',
        metadata={
            'stage': 'student',
            'model': args.model,
            'exp_name': args.exp_name,
            'epochs': int(args.epochs),
            'lr': float(args.lr),
            'wd': float(args.wd),
            'label_smoothing': float(args.label_smoothing),
            'scheduler': args.scheduler,
            'mixup': bool(not args.no_mixup),
            'mixup_alpha': float(args.mixup_alpha),
            'cutmix': bool(not args.no_cutmix),
            'cutmix_alpha': float(args.cutmix_alpha),
            'max_grad_norm': float(args.max_grad_norm),
            'best_val_acc': None if best_val_acc is None else float(best_val_acc),
            'test_acc': None if test_acc is None else float(test_acc),
            'stopped_epoch': None if stopped_epoch is None else int(stopped_epoch),
        },
    )

    for filename in [
        'best.pth',
        'history.json',
        'summary.json',
        'progress.json',
        'run_config.json',
        'loss_curve.png',
        'accuracy_curve.png',
        'confusion_matrix.pt',
        'roc_curve_micro_macro.png',
        'class_metrics.json',
        'test_metrics.json',
    ]:
        path = os.path.join(save_dir, filename)
        if os.path.isfile(path):
            artifact.add_file(path, name=filename)

    if wandb.run is not None:
        wandb.run.log_artifact(artifact)
        print(f'[W&B] Logged artifact: {artifact.name}')


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


def _write_running_progress(save_dir: str, *, epoch: int, lr: float, train_loss: float, train_acc: float, val_loss: Optional[float], val_acc: Optional[float], best_val_acc: Optional[float], best_epoch: Optional[int], no_improve: Optional[int], status: str = 'running', message: Optional[str] = None) -> None:
    payload: Dict[str, object] = {
        'status': status,
        'epoch': int(epoch),
        'lr': float(lr),
        'train_loss': float(train_loss),
        'train_acc': float(train_acc),
        'val_loss': None if val_loss is None else float(val_loss),
        'val_acc': None if val_acc is None else float(val_acc),
        'best_val_acc': None if best_val_acc is None else float(best_val_acc),
        'best_epoch': None if best_epoch is None else int(best_epoch),
        'no_improve': None if no_improve is None else int(no_improve),
    }
    if message:
        payload['message'] = message
    _save_progress(save_dir, payload)


def _mark_progress_terminal(save_dir: str, status: str, message: Optional[str] = None, **extra) -> None:
    payload: Dict[str, object] = {'status': status}
    if message:
        payload['message'] = message
    payload.update(extra)
    _save_progress(save_dir, payload)


def main() -> None:
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    set_seed(args.seed, benchmark=args.benchmark, allow_tf32=args.allow_tf32)

    save_dir = os.path.join(SAVE_DIR, args.save_subdir, args.model, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print('Save dir:', save_dir)

    _save_json(os.path.join(save_dir, 'run_config.json'), vars(args))
    _mark_progress_terminal(save_dir, status='starting', message='initializing run')

    wandb = _init_wandb(args, save_dir)

    train_loader, val_loader, test_loader, sizes = _load_data(args)
    print('Dataset sizes:', sizes)

    model = get_model(args.model, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = get_scheduler(optimizer, args.epochs, scheduler_type=args.scheduler, milestones=tuple(args.milestones), gamma=args.gamma)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None

    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    existing_history = _load_existing_history(save_dir)
    if existing_history is not None:
        history = existing_history

    start_epoch = 1
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0
    last_epoch_ran = 0

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f'Resume checkpoint not found: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
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

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                device,
                use_mixup=use_mixup,
                mixup_alpha=args.mixup_alpha,
                use_cutmix=use_cutmix,
                cutmix_alpha=args.cutmix_alpha,
                scaler=scaler,
                max_grad_norm=(args.max_grad_norm if args.max_grad_norm > 0 else None),
            )

            lr_current = float(optimizer.param_groups[0]['lr'])
            val_loss: Optional[float] = None
            val_acc: Optional[float] = None
            is_best = False

            if val_loader is not None:
                val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device, desc='Val')
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

            print(f'Train loss: {train_loss:.4f}, acc: {train_acc:.2f}')
            if val_loader is not None and val_loss is not None and val_acc is not None:
                print(f'Val   loss: {val_loss:.4f}, acc: {val_acc:.2f}')

            history['epoch'].append(epoch)
            history['train_loss'].append(float(train_loss))
            history['train_acc'].append(float(train_acc))
            history['lr'].append(lr_current)
            if val_loader is not None and val_loss is not None and val_acc is not None:
                history['val_loss'].append(float(val_loss))
                history['val_acc'].append(float(val_acc))

            if wandb is not None:
                log_dict = {
                    'epoch': epoch,
                    'train/loss': float(train_loss),
                    'train/acc': float(train_acc),
                    'lr': lr_current,
                }
                if val_loader is not None and val_loss is not None and val_acc is not None:
                    log_dict['val/loss'] = float(val_loss)
                    log_dict['val/acc'] = float(val_acc)
                    log_dict['val/best_acc_so_far'] = float(best_val_acc)
                wandb.log(log_dict, step=epoch)

            save_checkpoint(
                {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
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
            _write_running_progress(
                save_dir,
                epoch=epoch,
                lr=lr_current,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                best_val_acc=(None if val_loader is None else best_val_acc),
                best_epoch=(None if val_loader is None else best_epoch),
                no_improve=(None if val_loader is None else no_improve),
                status='running',
            )
            scheduler.step()

            if val_loader is not None and args.early_stop and epoch >= args.es_start_epoch and no_improve >= args.es_patience:
                message = f'Early stopping triggered at epoch {epoch} (no improvement for {args.es_patience} epochs).'
                print(message)
                _mark_progress_terminal(
                    save_dir,
                    status='early_stopped',
                    message=message,
                    epoch=int(epoch),
                    best_val_acc=float(best_val_acc),
                    best_epoch=int(best_epoch),
                )
                break

        checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
        best_path = os.path.join(save_dir, 'best.pth')

        if val_loader is not None and os.path.isfile(best_path):
            best_ckpt = torch.load(best_path, map_location='cpu')
            model.load_state_dict(best_ckpt['state_dict'])
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
            'model': args.model,
            'exp_name': args.exp_name,
            'save_dir': save_dir,
            'epochs_requested': args.epochs,
            'epochs_completed': last_epoch_ran,
            'best_epoch': best_epoch,
            'best_val_acc': None if val_loader is None else float(best_val_acc),
            'full_train': args.full_train,
            'history_path': history_path,
        }

        test_acc: Optional[float] = None
        if not args.skip_test_metrics:
            test_acc, per_class_acc, conf_mat = eval_on_test(model, test_loader, device, NUM_CLASSES)
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

            roc_path = None
            if not args.skip_plots:
                roc_path = _maybe_generate_roc(model, test_loader, device, save_dir)
            else:
                roc_path = None

            if wandb is not None:
                log_payload = {
                    'test/acc': float(test_acc),
                    'test/mean_per_class_acc': float(per_class_acc.mean().item()),
                }
                if val_loader is not None:
                    log_payload['best_val_acc'] = float(best_val_acc)
                wandb.log(log_payload)
                if os.path.isfile(conf_path):
                    wandb.save(conf_path)
                if roc_path is not None and os.path.isfile(roc_path):
                    wandb.save(roc_path)
        else:
            print('\nTest metrics skipped.')

        _save_json(os.path.join(save_dir, 'summary.json'), summary)
        _mark_progress_terminal(
            save_dir,
            status='completed',
            message='run finished successfully',
            epoch=int(last_epoch_ran),
            best_val_acc=(None if val_loader is None else float(best_val_acc)),
            best_epoch=(None if val_loader is None else int(best_epoch)),
            test_acc=(None if test_acc is None else float(test_acc)),
            history_path=history_path,
            summary_path=os.path.join(save_dir, 'summary.json'),
        )

        if wandb is not None:
            for path in [history_path, loss_curve_path, acc_curve_path, os.path.join(save_dir, 'summary.json'), os.path.join(save_dir, 'progress.json')]:
                if path and os.path.isfile(path):
                    wandb.save(path)
            _log_best_checkpoint_artifact(
                wandb=wandb,
                args=args,
                save_dir=save_dir,
                best_val_acc=None if val_loader is None else best_val_acc,
                test_acc=test_acc,
                stopped_epoch=last_epoch_ran if args.early_stop else None,
            )
            wandb.finish()

    except Exception as exc:
        _mark_progress_terminal(
            save_dir,
            status='failed',
            message=str(exc),
            epoch=int(last_epoch_ran),
            best_val_acc=(None if best_val_acc is None else float(best_val_acc)),
            best_epoch=(None if best_epoch is None else int(best_epoch)),
        )
        if wandb is not None:
            wandb.finish(exit_code=1)
        raise


if __name__ == '__main__':
    main()
