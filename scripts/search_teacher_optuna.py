"""Run an Optuna search for teacher hyperparameters."""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent
_PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == 'scripts' else _THIS_DIR
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna

from config import SAVE_DIR, SEED


TRAIN_TEACHER_MODULE = 'scripts.train_teacher'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROGRESS_FILENAME = 'progress.json'
_SUMMARY_FILENAME = 'summary.json'
_HISTORY_FILENAME = 'history.json'
_LOG_FILENAME = 'subprocess.log'


class TrialExecutionError(RuntimeError):
    """Raised when a subprocess trial fails or produces unusable outputs."""


def parse_args():
    parser = argparse.ArgumentParser(description='Optuna search for teacher on CIFAR-100')
    parser.add_argument('--model', type=str, default='resnet50_cifar')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--n_trials', type=int, default=12)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--study_name', type=str, default='teacher_optuna')
    parser.add_argument('--storage', type=str, default='', help='Optional Optuna storage, e.g. sqlite:///teacher_optuna.db')
    parser.add_argument('--save_subdir', type=str, default='teacher_optuna', help='Subdirectory inside SAVE_DIR')
    parser.add_argument('--campaign_id', type=str, default='v1', help='Identifier used to group all W&B runs from the same Optuna campaign')
    parser.add_argument('--search_preset', type=str, default='quick', choices=['quick', 'full'])
    parser.add_argument('--early_stop_search', action='store_true', help='Enable early stopping during each Optuna trial')
    parser.add_argument('--es_patience', type=int, default=10)
    parser.add_argument('--es_min_delta', type=float, default=0.05)
    parser.add_argument('--es_start_epoch', type=int, default=15)
    parser.add_argument('--timeout_per_trial', type=int, default=0, help='Optional timeout in seconds for each subprocess trial')
    parser.add_argument('--n_startup_trials', type=int, default=4)
    parser.add_argument('--n_warmup_steps', type=int, default=5, help='Minimum reported epochs before pruning can happen')
    parser.add_argument('--interval_steps', type=int, default=1, help='Pruner interval in reported epochs')
    parser.add_argument('--poll_interval', type=float, default=2.0, help='Seconds between progress polling while a trial is running')
    parser.add_argument('--sampler_seed', type=int, default=SEED)
    parser.add_argument('--no_live_logs', action='store_true', help='Do not stream train_teacher output live in the terminal; only save it to subprocess.log')

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='distill_cifar100')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_tags', type=str, default='teacher,search,optuna')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    return parser.parse_args()


def _search_space(preset: str) -> Dict[str, Any]:
    if preset == 'quick':
        return {
            'lr': (0.05, 0.12),
            'wd': (5e-4, 1.5e-3),
            'label_smoothing_values': [0.0, 0.1],
            'augmentations': ['mixup_only', 'mixup_cutmix'],
            'mixup_alpha': (0.15, 0.45),
            'cutmix_alpha': (0.8, 1.1),
        }
    return {
        'lr': (0.02, 0.15),
        'wd': (2e-4, 2e-3),
        'label_smoothing_values': [0.0, 0.05, 0.1, 0.15],
        'augmentations': ['mixup_only', 'cutmix_only', 'mixup_cutmix'],
        'mixup_alpha': (0.1, 0.5),
        'cutmix_alpha': (0.6, 1.2),
    }


def sample_params(trial: optuna.Trial, preset: str) -> Dict[str, Any]:
    space = _search_space(preset)
    aug_mode = trial.suggest_categorical('augmentation', space['augmentations'])
    use_mixup = aug_mode in {'mixup_only', 'mixup_cutmix'}
    use_cutmix = aug_mode in {'cutmix_only', 'mixup_cutmix'}

    params: Dict[str, Any] = {
        'lr': trial.suggest_float('lr', *space['lr'], log=True),
        'wd': trial.suggest_float('wd', *space['wd'], log=True),
        'label_smoothing': trial.suggest_categorical('label_smoothing', space['label_smoothing_values']),
        'use_mixup': use_mixup,
        'use_cutmix': use_cutmix,
        'mixup_alpha': 0.0,
        'cutmix_alpha': 0.0,
        'scheduler': 'cosine',
    }
    if use_mixup:
        params['mixup_alpha'] = trial.suggest_float('mixup_alpha', *space['mixup_alpha'])
    if use_cutmix:
        params['cutmix_alpha'] = trial.suggest_float('cutmix_alpha', *space['cutmix_alpha'])
    return params


def _enqueue_baselines(study: optuna.Study, preset: str) -> None:
    baselines: List[Dict[str, Any]] = [
        {
            'augmentation': 'mixup_cutmix',
            'lr': 0.1,
            'wd': 5e-4,
            'label_smoothing': 0.1,
            'mixup_alpha': 0.2,
            'cutmix_alpha': 1.0,
        },
        {
            'augmentation': 'mixup_only',
            'lr': 0.05,
            'wd': 1e-3,
            'label_smoothing': 0.1,
            'mixup_alpha': 0.4,
        },
    ]
    if preset == 'full':
        baselines.append(
            {
                'augmentation': 'cutmix_only',
                'lr': 0.08,
                'wd': 5e-4,
                'label_smoothing': 0.05,
                'cutmix_alpha': 1.0,
            }
        )

    existing = {tuple(sorted(t.params.items())) for t in study.trials if t.params}
    for params in baselines:
        key = tuple(sorted(params.items()))
        if key not in existing:
            study.enqueue_trial(params)


def build_command(args, exp_name: str, params: Dict[str, Any], trial_id: int) -> List[str]:
    cmd = [
        sys.executable,
        '-m',
        TRAIN_TEACHER_MODULE,
        '--model', args.model,
        '--gpu', str(args.gpu),
        '--epochs', str(args.epochs),
        '--exp_name', exp_name,
        '--lr', str(params['lr']),
        '--wd', str(params['wd']),
        '--label_smoothing', str(params['label_smoothing']),
        '--scheduler', params['scheduler'],
        '--mixup_alpha', str(params['mixup_alpha']),
        '--cutmix_alpha', str(params['cutmix_alpha']),
        '--save_subdir', args.save_subdir,
        '--skip_test_metrics',
        '--skip_plots',
        '--seed', str(args.seed),
        '--benchmark',
    ]
    if args.batch_size > 0:
        cmd += ['--batch_size', str(args.batch_size)]
    if args.num_workers >= 0:
        cmd += ['--num_workers', str(args.num_workers)]
    if args.early_stop_search:
        cmd += [
            '--early_stop',
            '--es_patience', str(args.es_patience),
            '--es_min_delta', str(args.es_min_delta),
            '--es_start_epoch', str(args.es_start_epoch),
        ]
    if not params['use_mixup']:
        cmd.append('--no_mixup')
    if not params['use_cutmix']:
        cmd.append('--no_cutmix')

    if args.use_wandb:
        run_name = (
            f"trial_{trial_id:03d}"
            f"_lr{params['lr']:.4g}"
            f"_wd{params['wd']:.4g}"
            f"_ls{params['label_smoothing']}"
            f"_mu{params['mixup_alpha']:.3f}"
            f"_cm{params['cutmix_alpha']:.3f}"
        )
        run_group = f'teacher-optuna/{args.study_name}/{args.model}/{args.campaign_id}/{args.search_preset}'
        tags = ','.join([args.wandb_tags, args.model, 'cifar100', args.search_preset])
        cmd += [
            '--use_wandb',
            '--wandb_project', args.wandb_project,
            '--wandb_entity', args.wandb_entity,
            '--wandb_tags', tags,
            '--wandb_mode', args.wandb_mode,
            '--wandb_group', run_group,
            '--wandb_job_type', 'optuna',
            '--wandb_run_name', run_name,
        ]
    return cmd



def _read_json_if_valid(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _read_progress(run_dir: Path) -> Optional[Dict[str, Any]]:
    payload = _read_json_if_valid(run_dir / _PROGRESS_FILENAME)
    if payload is not None:
        return payload

    hist = _read_json_if_valid(run_dir / _HISTORY_FILENAME)
    if hist is None:
        return None
    epochs = hist.get('epoch', [])
    val_acc = hist.get('val_acc', [])
    train_acc = hist.get('train_acc', [])
    train_loss = hist.get('train_loss', [])
    val_loss = hist.get('val_loss', [])
    if not epochs:
        return None
    return {
        'epoch': int(epochs[-1]),
        'train_acc': float(train_acc[-1]) if train_acc else None,
        'train_loss': float(train_loss[-1]) if train_loss else None,
        'val_acc': float(val_acc[-1]) if val_acc else None,
        'val_loss': float(val_loss[-1]) if val_loss else None,
        'best_val_acc': float(max(val_acc)) if val_acc else None,
        'status': 'running',
    }


def _load_trial_metrics(run_dir: Path) -> Dict[str, Any]:
    summary_path = run_dir / _SUMMARY_FILENAME
    summary_payload = _read_json_if_valid(summary_path)
    if summary_payload is not None and 'best_val_acc' in summary_payload:
        return summary_payload

    progress_payload = _read_progress(run_dir)
    if progress_payload is not None:
        return {
            'best_val_acc': progress_payload.get('best_val_acc', 0.0) or 0.0,
            'epochs_ran': int(progress_payload.get('epoch', 0) or 0),
            'status': progress_payload.get('status', 'unknown'),
            'progress_path': str(run_dir / _PROGRESS_FILENAME),
        }

    raise TrialExecutionError(f'No summary/progress/history file found in {run_dir}')


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def run_one_trial(args, trial: optuna.Trial) -> float:
    params = sample_params(trial, args.search_preset)
    exp_name = f"optuna_{trial.number:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cmd = build_command(args, exp_name, params, trial.number)
    run_dir = Path(SAVE_DIR) / args.save_subdir / args.model / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / _LOG_FILENAME

    trial.set_user_attr('exp_name', exp_name)
    trial.set_user_attr('params_full', params)
    trial.set_user_attr('run_dir', str(run_dir))
    trial.set_user_attr('log_path', str(log_path))
    print(f"\n[Trial {trial.number}] Running: {' '.join(cmd)}")

    start_time = time.time()
    last_reported_epoch = -1
    last_reported_value: Optional[float] = None

    log_file = None
    if args.no_live_logs:
        log_file = open(log_path, 'w', encoding='utf-8')

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=None if not args.no_live_logs else log_file,
            stderr=None if not args.no_live_logs else subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            while True:
                returncode = proc.poll()
                progress = _read_progress(run_dir)
                if progress is not None:
                    epoch = int(progress.get('epoch', 0) or 0)
                    best_val = progress.get('best_val_acc', None)
                    if best_val is None:
                        best_val = progress.get('val_acc', None)
                    if best_val is not None and epoch > last_reported_epoch:
                        metric_value = float(best_val)
                        trial.report(metric_value, step=epoch)
                        trial.set_user_attr('last_reported_epoch', epoch)
                        trial.set_user_attr('last_reported_best_val_acc', metric_value)
                        last_reported_epoch = epoch
                        last_reported_value = metric_value
                        if trial.should_prune():
                            _terminate_process(proc)
                            duration_s = time.time() - start_time
                            trial.set_user_attr('status', 'pruned')
                            trial.set_user_attr('duration_s', duration_s)
                            raise optuna.TrialPruned(
                                f'Trial {trial.number} pruned at epoch {epoch} with best_val_acc={metric_value:.4f}'
                            )

                if returncode is not None:
                    break

                if args.timeout_per_trial > 0 and (time.time() - start_time) > args.timeout_per_trial:
                    _terminate_process(proc)
                    trial.set_user_attr('status', 'timeout')
                    trial.set_user_attr('duration_s', time.time() - start_time)
                    raise TrialExecutionError(f'Trial {trial.number} timed out after {args.timeout_per_trial}s')

                time.sleep(max(args.poll_interval, 0.2))

        except Exception:
            if proc.poll() is None:
                _terminate_process(proc)
            raise
    finally:
        if log_file is not None:
            log_file.close()

    duration_s = time.time() - start_time
    if proc.returncode != 0:
        try:
            log_tail = log_path.read_text(encoding='utf-8')[-4000:]
        except Exception:
            log_tail = ''
        trial.set_user_attr('status', 'failed')
        trial.set_user_attr('duration_s', duration_s)
        trial.set_user_attr('returncode', proc.returncode)
        trial.set_user_attr('log_tail', log_tail)
        raise TrialExecutionError(f'Trial {trial.number} failed with return code {proc.returncode}')

    metrics = _load_trial_metrics(run_dir)
    best_val_acc = float(metrics.get('best_val_acc', last_reported_value or 0.0) or 0.0)
    epochs_ran = int(metrics.get('epochs_ran', metrics.get('epoch', args.epochs) or args.epochs))

    trial.set_user_attr('status', 'ok')
    trial.set_user_attr('duration_s', duration_s)
    trial.set_user_attr('epochs_ran', epochs_ran)
    trial.set_user_attr('metrics_path', str(run_dir / (_SUMMARY_FILENAME if (run_dir / _SUMMARY_FILENAME).is_file() else _PROGRESS_FILENAME)))
    return best_val_acc


def _save_study_outputs(study: optuna.Study, args) -> Dict[str, str]:
    out_dir = Path(SAVE_DIR) / 'optuna_results'
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f'teacher_{args.model}_{args.save_subdir}_{args.study_name}_{stamp}'
    best_path = out_dir / f'{prefix}_best.json'
    trials_json_path = out_dir / f'{prefix}_trials.json'
    trials_csv_path = out_dir / f'{prefix}_trials.csv'

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    payload = {
        'study_name': study.study_name,
        'model': args.model,
        'search_preset': args.search_preset,
        'n_trials_requested': args.n_trials,
        'n_trials_total': len(study.trials),
        'n_trials_complete': len(completed_trials),
        'n_trials_failed': len(failed_trials),
        'n_trials_pruned': len(pruned_trials),
        'best_trial': study.best_trial.number if completed_trials else None,
        'best_value': study.best_value if completed_trials else None,
        'best_params': study.best_trial.params if completed_trials else None,
        'best_user_attrs': dict(study.best_trial.user_attrs) if completed_trials else None,
    }
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    serializable_trials = []
    for t in study.trials:
        serializable_trials.append(
            {
                'number': t.number,
                'state': str(t.state),
                'value': t.value,
                'params': t.params,
                'user_attrs': dict(t.user_attrs),
            }
        )
    with open(trials_json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_trials, f, indent=2)

    with open(trials_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'number', 'state', 'value', 'duration_s', 'epochs_ran', 'exp_name', 'lr', 'wd', 'label_smoothing',
                'augmentation', 'mixup_alpha', 'cutmix_alpha', 'status', 'metrics_path', 'run_dir', 'log_path',
            ],
        )
        writer.writeheader()
        for t in study.trials:
            row = {
                'number': t.number,
                'state': str(t.state),
                'value': t.value,
                'duration_s': t.user_attrs.get('duration_s', ''),
                'epochs_ran': t.user_attrs.get('epochs_ran', ''),
                'exp_name': t.user_attrs.get('exp_name', ''),
                'lr': t.params.get('lr', ''),
                'wd': t.params.get('wd', ''),
                'label_smoothing': t.params.get('label_smoothing', ''),
                'augmentation': t.params.get('augmentation', ''),
                'mixup_alpha': t.params.get('mixup_alpha', ''),
                'cutmix_alpha': t.params.get('cutmix_alpha', ''),
                'status': t.user_attrs.get('status', ''),
                'metrics_path': t.user_attrs.get('metrics_path', ''),
                'run_dir': t.user_attrs.get('run_dir', ''),
                'log_path': t.user_attrs.get('log_path', ''),
            }
            writer.writerow(row)

    return {
        'best': str(best_path),
        'trials_json': str(trials_json_path),
        'trials_csv': str(trials_csv_path),
    }


def main() -> None:
    args = parse_args()

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=args.n_warmup_steps,
        interval_steps=args.interval_steps,
    )
    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed,
        multivariate=True,
        group=True,
        constant_liar=True,
        n_startup_trials=args.n_startup_trials,
    )

    if args.storage.strip():
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction='maximize',
            load_if_exists=True,
            pruner=pruner,
            sampler=sampler,
        )
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            direction='maximize',
            pruner=pruner,
            sampler=sampler,
        )

    _enqueue_baselines(study, args.search_preset)

    def objective(trial: optuna.Trial) -> float:
        try:
            return run_one_trial(args, trial)
        except optuna.TrialPruned:
            raise
        except TrialExecutionError as exc:
            trial.set_user_attr('status', trial.user_attrs.get('status', 'failed'))
            trial.set_user_attr('error', str(exc))
            raise

    study.optimize(objective, n_trials=args.n_trials, catch=(TrialExecutionError,))

    outputs = _save_study_outputs(study, args)
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\nStudy saved:\n- {outputs['best']}\n- {outputs['trials_json']}\n- {outputs['trials_csv']}")
    if complete_trials:
        print(f"Best trial: #{study.best_trial.number} | value={study.best_value:.4f}")
        print(f'Best params: {study.best_trial.params}')
    else:
        print('No completed trials were found.')


if __name__ == '__main__':
    main()
