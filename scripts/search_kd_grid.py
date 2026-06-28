"""Run a deterministic grid search for Knowledge Distillation settings."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

_THIS_FILE = Path(__file__).resolve()
_THIS_DIR = _THIS_FILE.parent
PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == 'scripts' else _THIS_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SAVE_DIR

TRAIN_KD_MODULE = 'scripts.train_kd'


def parse_args():
    parser = argparse.ArgumentParser(description='Grid search for KD on CIFAR-100')
    parser.add_argument('--teacher_model', type=str, default='resnet50_cifar')
    parser.add_argument('--student_model', type=str, default='resnet18_cifar')
    parser.add_argument('--teacher_checkpoint', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--save_subdir', type=str, default='kd_grid')
    parser.add_argument('--campaign_id', type=str, default='v1')
    parser.add_argument('--grid_preset', type=str, default='quick', choices=['quick', 'full'])
    parser.add_argument('--max_trials', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--allow_tf32', action='store_true')
    parser.add_argument('--early_stop_search', action='store_true')
    parser.add_argument('--es_patience', type=int, default=10)
    parser.add_argument('--es_min_delta', type=float, default=0.05)
    parser.add_argument('--es_start_epoch', type=int, default=15)

    parser.add_argument('--no_mixup', action='store_true')
    parser.add_argument('--mixup_alpha', type=float, default=0.2)
    parser.add_argument('--no_cutmix', action='store_true')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0)
    parser.add_argument('--hard_label_smoothing', type=float, default=0.0)

    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='distill_cifar100')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_tags', type=str, default='kd,search,grid')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    return parser.parse_args()


QUICK_GRID: Dict[str, List[Any]] = {
    'kd_temperature': [2.0, 4.0, 6.0],
    'kd_alpha': [0.7, 0.9],
    'lr': [0.05, 0.08],
    'wd': [5e-4],
}

FULL_GRID: Dict[str, List[Any]] = {
    'kd_temperature': [2.0, 4.0, 6.0, 8.0],
    'kd_alpha': [0.5, 0.7, 0.9],
    'lr': [0.03, 0.05, 0.08],
    'wd': [3e-4, 5e-4, 1e-3],
}


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    tmp.replace(path)


def grid_to_list(grid_dict: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid_dict.keys())
    values = [list(grid_dict[k]) for k in keys]
    return [dict(zip(keys, vals)) for vals in product(*values)]


def cfg_key(cfg: Dict[str, Any]) -> Tuple[Any, ...]:
    return (cfg['kd_temperature'], cfg['kd_alpha'], cfg['lr'], cfg['wd'])


def make_grid(preset: str) -> List[Dict[str, Any]]:
    grid_def = QUICK_GRID if preset == 'quick' else FULL_GRID
    raw_grid = grid_to_list(grid_def)
    unique = {cfg_key(cfg): cfg for cfg in raw_grid}
    return list(unique.values())


def build_command(args, exp_name: str, cfg: Dict[str, Any], idx: int) -> List[str]:
    cmd = [
        sys.executable, '-m', TRAIN_KD_MODULE,
        '--teacher_model', args.teacher_model,
        '--student_model', args.student_model,
        '--teacher_checkpoint', args.teacher_checkpoint,
        '--gpu', str(args.gpu),
        '--epochs', str(args.epochs),
        '--exp_name', exp_name,
        '--lr', str(cfg['lr']),
        '--wd', str(cfg['wd']),
        '--kd_temperature', str(cfg['kd_temperature']),
        '--kd_alpha', str(cfg['kd_alpha']),
        '--hard_label_smoothing', str(args.hard_label_smoothing),
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
        cmd += ['--early_stop', '--es_patience', str(args.es_patience), '--es_min_delta', str(args.es_min_delta), '--es_start_epoch', str(args.es_start_epoch)]
    if args.use_wandb:
        run_name = f"grid_{idx:03d}_T{cfg['kd_temperature']}_a{cfg['kd_alpha']}_lr{cfg['lr']}_wd{cfg['wd']}"
        run_group = f'kd-grid/{args.student_model}/{args.campaign_id}/{args.grid_preset}'
        tags = ','.join([args.wandb_tags, args.teacher_model, args.student_model, 'cifar100', args.grid_preset])
        cmd += [
            '--use_wandb', '--wandb_project', args.wandb_project,
            '--wandb_entity', args.wandb_entity,
            '--wandb_tags', tags,
            '--wandb_mode', args.wandb_mode,
            '--wandb_group', run_group,
            '--wandb_job_type', 'grid',
            '--wandb_run_name', run_name,
        ]
    return cmd


def _load_best_val_acc(run_dir: Path) -> float:
    summary_path = run_dir / 'summary.json'
    if summary_path.is_file():
        with summary_path.open('r', encoding='utf-8') as f:
            summary = json.load(f)
        if 'best_val_acc' in summary and summary['best_val_acc'] is not None:
            return float(summary['best_val_acc'])
    history_path = run_dir / 'history.json'
    if history_path.is_file():
        with history_path.open('r', encoding='utf-8') as f:
            hist = json.load(f)
        return float(max(hist.get('val_acc', [0.0]) or [0.0]))
    raise FileNotFoundError(f'No summary.json or history.json found in {run_dir}')


def main():
    args = parse_args()
    grid = make_grid(args.grid_preset)
    if args.max_trials > 0:
        grid = grid[: args.max_trials]

    out_dir = Path(SAVE_DIR) / 'grid_results'
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f'kd_{args.student_model}_from_{args.teacher_model}_{args.save_subdir}'
    results_path = out_dir / f'{tag}_results.json'
    best_path = out_dir / f'{tag}_best.json'
    meta_path = out_dir / f'{tag}_meta.json'

    meta = {
        'teacher_model': args.teacher_model,
        'student_model': args.student_model,
        'teacher_checkpoint': args.teacher_checkpoint,
        'epochs': args.epochs,
        'grid_preset': args.grid_preset,
        'grid_size': len(grid),
        'save_subdir': args.save_subdir,
        'campaign_id': args.campaign_id,
        'mixup_enabled': not args.no_mixup,
        'mixup_alpha': args.mixup_alpha,
        'cutmix_enabled': not args.no_cutmix,
        'cutmix_alpha': args.cutmix_alpha,
        'hard_label_smoothing': args.hard_label_smoothing,
    }
    _atomic_json_write(meta_path, meta)
    print(f"Using preset='{args.grid_preset}' with {len(grid)} unique configs for {args.epochs} epochs each.")

    results: List[Dict[str, Any]] = []
    for i, cfg in enumerate(grid):
        exp_name = f'grid_{i:03d}'
        cmd = build_command(args, exp_name, cfg, i)
        print(f"\n[{i + 1}/{len(grid)}] Running: {' '.join(cmd)}")
        rec = dict(cfg)
        rec['exp_name'] = exp_name
        rec['status'] = 'ok'
        start = time.time()
        try:
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        except subprocess.CalledProcessError as exc:
            rec['status'] = 'failed'
            rec['best_val_acc'] = 0.0
            rec['error'] = str(exc)
            rec['duration_sec'] = round(time.time() - start, 2)
            results.append(rec)
            continue
        rec['duration_sec'] = round(time.time() - start, 2)
        run_dir = Path(SAVE_DIR) / args.save_subdir / args.student_model / exp_name
        rec['run_dir'] = str(run_dir)
        rec['best_val_acc'] = _load_best_val_acc(run_dir)
        results.append(rec)
        results.sort(key=lambda x: x.get('best_val_acc', 0.0), reverse=True)
        _atomic_json_write(results_path, results)
        _atomic_json_write(best_path, results[0] if results else {})

    results.sort(key=lambda x: x.get('best_val_acc', 0.0), reverse=True)
    _atomic_json_write(results_path, results)
    _atomic_json_write(best_path, results[0] if results else {})
    print('\nTop results:')
    for rank, rec in enumerate(results[:5], start=1):
        print(f"#{rank}: val_acc={rec.get('best_val_acc', 0.0):.2f} | T={rec['kd_temperature']} alpha={rec['kd_alpha']} lr={rec['lr']} wd={rec['wd']} | {rec['exp_name']}")


if __name__ == '__main__':
    main()
