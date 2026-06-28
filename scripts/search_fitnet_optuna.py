"""Run an Optuna search for FitNet settings."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_THIS_FILE = Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent
PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == 'scripts' else _THIS_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna

from config import SAVE_DIR, SEED

TRAIN_FITNET_MODULE = 'scripts.train_fitnet'
_PROGRESS_FILENAME = 'progress.json'
_SUMMARY_FILENAME = 'summary.json'
_HISTORY_FILENAME = 'history.json'
_LOG_FILENAME = 'subprocess.log'


class TrialExecutionError(RuntimeError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='Optuna search for FitNet on CIFAR-100')
    parser.add_argument('--teacher_model', type=str, default='resnet50_cifar')
    parser.add_argument('--student_model', type=str, default='resnet18_cifar')
    parser.add_argument('--teacher_checkpoint', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40, help='Stage-2 epochs')
    parser.add_argument('--hint_epochs', type=int, default=60)
    parser.add_argument('--n_trials', type=int, default=12)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--study_name', type=str, default='fitnet_optuna')
    parser.add_argument('--storage', type=str, default='')
    parser.add_argument('--save_subdir', type=str, default='fitnet_optuna')
    parser.add_argument('--campaign_id', type=str, default='v1')
    parser.add_argument('--search_preset', type=str, default='quick', choices=['quick', 'full'])
    parser.add_argument('--early_stop_search', action='store_true')
    parser.add_argument('--es_patience', type=int, default=10)
    parser.add_argument('--es_min_delta', type=float, default=0.05)
    parser.add_argument('--es_start_epoch', type=int, default=15)
    parser.add_argument('--timeout_per_trial', type=int, default=0)
    parser.add_argument('--n_startup_trials', type=int, default=4)
    parser.add_argument('--n_warmup_steps', type=int, default=5)
    parser.add_argument('--interval_steps', type=int, default=1)
    parser.add_argument('--poll_interval', type=float, default=2.0)
    parser.add_argument('--sampler_seed', type=int, default=SEED)
    parser.add_argument('--no_live_logs', action='store_true')
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--allow_tf32', action='store_true')

    parser.add_argument('--no_mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--no_cutmix', action='store_true')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    parser.add_argument('--hard_label_smoothing', type=float, default=0.0)
    parser.add_argument('--teacher_hint_layers', type=str, default='layer3.1')
    parser.add_argument('--student_guided_layers', type=str, default='layer2.1')

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='distill_cifar100')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_tags', type=str, default='fitnet,search,optuna')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    return parser.parse_args()


def _search_space(preset: str) -> Dict[str, Any]:
    # Guided by the first FitNet mini-grid (20 hint + 20 stage-2 epochs):
    # - top 4 all use lr=0.08
    # - the best run is T=3.0, alpha=0.9, lr=0.08, wd=5e-4 -> 68.54
    # - very good neighbors: T in [3, 4], alpha in [0.7, 0.9], wd ~5e-4
    # - hint stage is extremely stable across runs with hint_lr=5e-4
    # => concentrate Optuna around this high-value region instead of re-searching wide.
    if preset == 'quick':
        return {
            'hint_lr': (3.5e-4, 8.0e-4),
            'kd_temperature': (2.8, 4.3),
            'kd_alpha': (0.68, 0.92),
            'lr': (0.072, 0.088),
            'wd': (4.0e-4, 7.0e-4),
        }
    return {
        'hint_lr': (2.5e-4, 1.0e-3),
        'kd_temperature': (2.5, 4.8),
        'kd_alpha': (0.62, 0.95),
        'lr': (0.065, 0.09),
        'wd': (3.5e-4, 8.5e-4),
    }


def sample_params(trial: optuna.Trial, preset: str) -> Dict[str, Any]:
    space = _search_space(preset)
    return {
        'hint_lr': trial.suggest_float('hint_lr', *space['hint_lr'], log=True),
        'kd_temperature': trial.suggest_float('kd_temperature', *space['kd_temperature']),
        'kd_alpha': trial.suggest_float('kd_alpha', *space['kd_alpha']),
        'lr': trial.suggest_float('lr', *space['lr'], log=True),
        'wd': trial.suggest_float('wd', *space['wd'], log=True),
    }


def _enqueue_baselines(study: optuna.Study, preset: str) -> None:
    # Seed the study with the best configurations observed in the mini-grid.
    baselines = [
        {'hint_lr': 5e-4, 'kd_temperature': 3.0, 'kd_alpha': 0.9, 'lr': 0.08, 'wd': 5e-4},
        {'hint_lr': 5e-4, 'kd_temperature': 3.0, 'kd_alpha': 0.7, 'lr': 0.08, 'wd': 5e-4},
        {'hint_lr': 5e-4, 'kd_temperature': 4.0, 'kd_alpha': 0.9, 'lr': 0.08, 'wd': 5e-4},
        {'hint_lr': 5e-4, 'kd_temperature': 4.0, 'kd_alpha': 0.7, 'lr': 0.08, 'wd': 5e-4},
        # local refinements around the best point
        {'hint_lr': 3.5e-4, 'kd_temperature': 3.2, 'kd_alpha': 0.88, 'lr': 0.08, 'wd': 5e-4},
        {'hint_lr': 6.5e-4, 'kd_temperature': 3.4, 'kd_alpha': 0.86, 'lr': 0.082, 'wd': 5.5e-4},
    ]
    if preset == 'full':
        baselines.extend([
            {'hint_lr': 4e-4, 'kd_temperature': 2.8, 'kd_alpha': 0.82, 'lr': 0.076, 'wd': 4.5e-4},
            {'hint_lr': 7.5e-4, 'kd_temperature': 4.2, 'kd_alpha': 0.9, 'lr': 0.084, 'wd': 6e-4},
        ])
    existing = {tuple(sorted(t.params.items())) for t in study.trials if t.params}
    for params in baselines:
        key = tuple(sorted(params.items()))
        if key not in existing:
            study.enqueue_trial(params)


def build_command(args, exp_name: str, params: Dict[str, Any], trial_id: int) -> List[str]:
    cmd = [
        sys.executable, '-m', TRAIN_FITNET_MODULE,
        '--teacher_model', args.teacher_model,
        '--student_model', args.student_model,
        '--teacher_checkpoint', args.teacher_checkpoint,
        '--gpu', str(args.gpu),
        '--epochs', str(args.epochs),
        '--hint_epochs', str(args.hint_epochs),
        '--exp_name', exp_name,
        '--lr', str(params['lr']),
        '--hint_lr', str(params['hint_lr']),
        '--wd', str(params['wd']),
        '--kd_temperature', str(params['kd_temperature']),
        '--kd_alpha', str(params['kd_alpha']),
        '--hard_label_smoothing', str(args.hard_label_smoothing),
        '--teacher_hint_layers', args.teacher_hint_layers,
        '--student_guided_layers', args.student_guided_layers,
        '--mixup_alpha', str(args.mixup_alpha),
        '--cutmix_alpha', str(args.cutmix_alpha),
        '--save_subdir', args.save_subdir,
        '--skip_test_metrics', '--skip_plots', '--seed', str(args.seed),
    ]
    if args.batch_size > 0:
        cmd += ['--batch_size', str(args.batch_size)]
    if args.num_workers >= 0:
        cmd += ['--num_workers', str(args.num_workers)]
    if args.benchmark:
        cmd.append('--benchmark')
    if args.allow_tf32:
        cmd.append('--allow_tf32')
    if args.no_mixup:
        cmd.append('--no_mixup')
    if args.no_cutmix:
        cmd.append('--no_cutmix')
    if args.early_stop_search:
        cmd += [
            '--early_stop',
            '--es_patience', str(args.es_patience),
            '--es_min_delta', str(args.es_min_delta),
            '--es_start_epoch', str(args.es_start_epoch),
        ]
    if args.use_wandb:
        run_name = (
            f"trial_{trial_id:03d}_hlr{params['hint_lr']:.4g}_T{params['kd_temperature']:.4g}_"
            f"a{params['kd_alpha']:.4g}_lr{params['lr']:.4g}_wd{params['wd']:.4g}"
        )
        run_group = f'fitnet-optuna/{args.study_name}/{args.student_model}/{args.campaign_id}/{args.search_preset}'
        tags = ','.join([args.wandb_tags, args.teacher_model, args.student_model, 'cifar100', args.search_preset])
        cmd += [
            '--use_wandb', '--wandb_project', args.wandb_project,
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
    val_acc = [v for v in hist.get('val_acc', []) if v is not None]
    val_loss = [v for v in hist.get('val_loss', []) if v is not None]
    train_acc = [v for v in hist.get('train_acc', []) if v is not None]
    train_loss = [v for v in hist.get('train_loss', []) if v is not None]
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
    summary = _read_json_if_valid(run_dir / _SUMMARY_FILENAME)
    if summary is not None and 'best_val_acc' in summary:
        return summary
    progress = _read_progress(run_dir)
    if progress is not None:
        return {
            'best_val_acc': progress.get('best_val_acc', 0.0) or 0.0,
            'epochs_ran': int(progress.get('epoch', 0) or 0),
            'status': progress.get('status', 'unknown'),
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
    run_dir = Path(SAVE_DIR) / args.save_subdir / args.student_model / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / _LOG_FILENAME

    trial.set_user_attr('exp_name', exp_name)
    trial.set_user_attr('params_full', params)
    trial.set_user_attr('run_dir', str(run_dir))
    trial.set_user_attr('log_path', str(log_path))
    print(f"\n[Trial {trial.number}] Running: {' '.join(cmd)}")

    start_time = time.time()
    last_reported_epoch = -1
    log_file = open(log_path, 'w', encoding='utf-8') if args.no_live_logs else None
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
                        trial.report(float(best_val), step=epoch)
                        last_reported_epoch = epoch
                        if trial.should_prune():
                            _terminate_process(proc)
                            raise optuna.TrialPruned(f'Pruned at epoch {epoch} with best_val_acc={best_val:.4f}')

                if returncode is not None:
                    if returncode != 0:
                        raise TrialExecutionError(f'Trial subprocess exited with code {returncode}. See {log_path}')
                    break

                if args.timeout_per_trial > 0 and (time.time() - start_time) > args.timeout_per_trial:
                    _terminate_process(proc)
                    raise TrialExecutionError(f'Trial timed out after {args.timeout_per_trial} seconds')

                time.sleep(max(args.poll_interval, 0.2))
        finally:
            if proc.poll() is None:
                _terminate_process(proc)
    finally:
        if log_file is not None:
            log_file.close()

    metrics = _load_trial_metrics(run_dir)
    best_val_acc = float(metrics.get('best_val_acc', 0.0) or 0.0)
    epochs_ran = int(metrics.get('epochs_completed', metrics.get('epochs_ran', args.epochs)) or args.epochs)

    trial.set_user_attr('duration_sec', round(time.time() - start_time, 2))
    trial.set_user_attr('best_val_acc', best_val_acc)
    trial.set_user_attr('epochs_ran', epochs_ran)

    print(
        f"[Trial {trial.number}] val_acc={best_val_acc:.2f} | "
        f"hint_lr={params['hint_lr']:.4g} | T={params['kd_temperature']:.4g} | "
        f"alpha={params['kd_alpha']:.4g} | lr={params['lr']:.4g} | wd={params['wd']:.4g}"
    )
    return best_val_acc


def _study_to_records(study: optuna.Study) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in study.trials:
        row = {
            'trial_number': t.number,
            'state': str(t.state).split('.')[-1],
            'value': t.value if t.value is not None else None,
            'hint_lr': t.params.get('hint_lr'),
            'kd_temperature': t.params.get('kd_temperature'),
            'kd_alpha': t.params.get('kd_alpha'),
            'lr': t.params.get('lr'),
            'wd': t.params.get('wd'),
            'exp_name': t.user_attrs.get('exp_name'),
            'run_dir': t.user_attrs.get('run_dir'),
            'duration_sec': t.user_attrs.get('duration_sec'),
            'epochs_ran': t.user_attrs.get('epochs_ran'),
        }
        rows.append(row)
    rows.sort(key=lambda x: (x['value'] is None, -(x['value'] or 0.0)))
    return rows


def _write_trials_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)


def _write_trials_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'trial_number', 'state', 'value', 'hint_lr', 'kd_temperature', 'kd_alpha', 'lr', 'wd',
        'exp_name', 'run_dir', 'duration_sec', 'epochs_ran',
    ]
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed,
        n_startup_trials=args.n_startup_trials,
        multivariate=True,
        group=True,
        constant_liar=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=args.n_warmup_steps,
        interval_steps=args.interval_steps,
    )
    storage = args.storage.strip() or None
    study = optuna.create_study(
        study_name=args.study_name,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=bool(storage),
    )
    _enqueue_baselines(study, args.search_preset)

    def objective(trial: optuna.Trial) -> float:
        return run_one_trial(args, trial)

    study.optimize(objective, n_trials=args.n_trials, catch=(TrialExecutionError,))

    out_dir = Path(SAVE_DIR) / 'optuna_results'
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'fitnet_{args.student_model}_from_{args.teacher_model}_{args.study_name}'
    rows = _study_to_records(study)
    _write_trials_json(out_dir / f'{tag}_trials.json', rows)
    _write_trials_csv(out_dir / f'{tag}_trials.csv', rows)

    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best_trial = study.best_trial if complete_trials else None

    best_payload = {
        'best_value': best_trial.value if best_trial is not None else None,
        'best_params': best_trial.params if best_trial is not None else {},
        'best_trial_number': best_trial.number if best_trial is not None else None,
        'study_name': args.study_name,
        'teacher_model': args.teacher_model,
        'student_model': args.student_model,
        'teacher_checkpoint': args.teacher_checkpoint,
        'search_preset': args.search_preset,
        'n_trials_requested': args.n_trials,
        'n_trials_completed': len(study.trials),
        'save_subdir': args.save_subdir,
        'hint_epochs': args.hint_epochs,
        'teacher_hint_layers': args.teacher_hint_layers,
        'student_guided_layers': args.student_guided_layers,
        'fitnet_method': 'paper_two_stage_hint_then_kd',
    }
    with open(out_dir / f'{tag}_best.json', 'w', encoding='utf-8') as f:
        json.dump(best_payload, f, indent=2)

    print('\n=== BEST FITNET OPTUNA TRIAL ===')
    print(json.dumps(best_payload, indent=2))


if __name__ == '__main__':
    main()
