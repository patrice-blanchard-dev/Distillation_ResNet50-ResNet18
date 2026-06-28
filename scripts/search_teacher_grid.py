"""Run a deterministic grid search for teacher hyperparameters."""

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
PROJECT_ROOT = _THIS_DIR.parent if _THIS_DIR.name == "scripts" else _THIS_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SAVE_DIR

TRAIN_TEACHER_MODULE = "scripts.train_teacher"


def parse_args():
    parser = argparse.ArgumentParser(description="Grid search for teacher on CIFAR-100")
    parser.add_argument("--model", type=str, default="resnet50_cifar")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--save_subdir", type=str, default="teacher_grid", help="Subdirectory inside SAVE_DIR")
    parser.add_argument("--campaign_id", type=str, default="v1", help="Identifier used to group all W&B runs from the same grid campaign")
    parser.add_argument("--grid_preset", type=str, default="quick", choices=["quick", "full"], help="quick = small practical grid, full = larger grid with deduplication")
    parser.add_argument("--max_trials", type=int, default=0, help="Optional cap for debugging (0 = all)")
    parser.add_argument("--batch_size", type=int, default=0, help="Override training batch size if > 0")
    parser.add_argument("--num_workers", type=int, default=-1, help="Override dataloader workers if >= 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark", action="store_true", help="Enable cuDNN benchmark for speed inside each trial")
    parser.add_argument("--allow_tf32", action="store_true", help="Allow TF32 on supported CUDA devices")
    parser.add_argument("--early_stop_search", action="store_true", help="Enable early stopping during grid search trials")
    parser.add_argument("--es_patience", type=int, default=10)
    parser.add_argument("--es_min_delta", type=float, default=0.05)
    parser.add_argument("--es_start_epoch", type=int, default=15)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="distill_cifar100")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_tags", type=str, default="teacher,search,grid")
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


QUICK_GRID: Dict[str, List[Any]] = {
    "lr": [0.05, 0.1, 0.2],
    "wd": [5e-4, 1e-3],
    "label_smoothing": [0.1],
    "scheduler": ["cosine"],
    "use_mixup": [True],
    "mixup_alpha": [0.2, 0.4],
    "use_cutmix": [True],
    "cutmix_alpha": [1.0],
}

FULL_GRID: Dict[str, List[Any]] = {
    "lr": [0.05, 0.1, 0.2],
    "wd": [5e-4, 1e-3],
    "label_smoothing": [0.0, 0.1],
    "scheduler": ["cosine"],
    "use_mixup": [True],
    "mixup_alpha": [0.2, 0.4],
    "use_cutmix": [False, True],
    "cutmix_alpha": [0.8, 1.0],
}


def grid_to_list(grid_dict: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid_dict.keys())
    values = [list(grid_dict[k]) for k in keys]
    return [dict(zip(keys, vals)) for vals in product(*values)]


def normalize_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg)
    if not cfg["use_mixup"]:
        cfg["mixup_alpha"] = 0.0
    if not cfg["use_cutmix"]:
        cfg["cutmix_alpha"] = 0.0
    return cfg


def cfg_key(cfg: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        cfg["lr"],
        cfg["wd"],
        cfg["label_smoothing"],
        cfg["scheduler"],
        bool(cfg["use_mixup"]),
        float(cfg["mixup_alpha"]),
        bool(cfg["use_cutmix"]),
        float(cfg["cutmix_alpha"]),
    )


def make_grid(preset: str) -> List[Dict[str, Any]]:
    grid_def = QUICK_GRID if preset == "quick" else FULL_GRID
    raw_grid = grid_to_list(grid_def)
    unique = {}
    for cfg in raw_grid:
        norm = normalize_cfg(cfg)
        unique[cfg_key(norm)] = norm
    return list(unique.values())


def build_command(args, exp_name: str, cfg: Dict[str, Any], idx: int) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        TRAIN_TEACHER_MODULE,
        "--model", args.model,
        "--gpu", str(args.gpu),
        "--epochs", str(args.epochs),
        "--lr", str(cfg["lr"]),
        "--wd", str(cfg["wd"]),
        "--label_smoothing", str(cfg["label_smoothing"]),
        "--scheduler", cfg["scheduler"],
        "--exp_name", exp_name,
        "--mixup_alpha", str(cfg["mixup_alpha"]),
        "--cutmix_alpha", str(cfg["cutmix_alpha"]),
        "--save_subdir", args.save_subdir,
        "--skip_test_metrics",
        "--skip_plots",
        "--seed", str(args.seed),
    ]

    if args.batch_size > 0:
        cmd += ["--batch_size", str(args.batch_size)]
    if args.num_workers >= 0:
        cmd += ["--num_workers", str(args.num_workers)]
    if args.benchmark:
        cmd.append("--benchmark")
    if args.allow_tf32:
        cmd.append("--allow_tf32")

    if not cfg["use_mixup"]:
        cmd.append("--no_mixup")
    if not cfg["use_cutmix"]:
        cmd.append("--no_cutmix")

    if args.early_stop_search:
        cmd += [
            "--early_stop",
            "--es_patience", str(args.es_patience),
            "--es_min_delta", str(args.es_min_delta),
            "--es_start_epoch", str(args.es_start_epoch),
        ]

    if args.use_wandb:
        run_name = (
            f"grid_{idx:03d}"
            f"_lr{cfg['lr']}"
            f"_wd{cfg['wd']}"
            f"_ls{cfg['label_smoothing']}"
            f"_mu{cfg['mixup_alpha']}"
            f"_cm{cfg['cutmix_alpha']}"
        )
        run_group = f"teacher-grid/{args.model}/{args.campaign_id}/{args.grid_preset}"
        tags = ",".join([args.wandb_tags, args.model, "cifar100", args.grid_preset])
        cmd += [
            "--use_wandb",
            "--wandb_project", args.wandb_project,
            "--wandb_entity", args.wandb_entity,
            "--wandb_tags", tags,
            "--wandb_mode", args.wandb_mode,
            "--wandb_group", run_group,
            "--wandb_job_type", "grid",
            "--wandb_run_name", run_name,
        ]
    return cmd


def _load_best_val_acc(run_dir: Path) -> float:
    summary_path = run_dir / "summary.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        if "best_val_acc" in summary:
            return float(summary["best_val_acc"])

    history_path = run_dir / "history.json"
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8") as f:
            hist = json.load(f)
        return float(max(hist.get("val_acc", [0.0]) or [0.0]))
    raise FileNotFoundError(f"No summary.json or history.json found in {run_dir}")


def main():
    args = parse_args()
    grid = make_grid(args.grid_preset)
    if args.max_trials > 0:
        grid = grid[: args.max_trials]

    out_dir = Path(SAVE_DIR) / "grid_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"{args.save_subdir}_{args.model}_results.json"
    best_path = out_dir / f"{args.save_subdir}_{args.model}_best.json"
    meta_path = out_dir / f"{args.save_subdir}_{args.model}_meta.json"

    meta = {
        "model": args.model,
        "epochs": args.epochs,
        "grid_preset": args.grid_preset,
        "grid_size": len(grid),
        "save_subdir": args.save_subdir,
        "campaign_id": args.campaign_id,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(
        f"Using preset='{args.grid_preset}' with {len(grid)} unique configs for {args.epochs} epochs each."
    )

    results: List[Dict[str, Any]] = []
    for i, cfg in enumerate(grid):
        exp_name = f"grid_{i:03d}"
        cmd = build_command(args, exp_name, cfg, i)
        print(f"\n[{i + 1}/{len(grid)}] Running: {' ' .join(cmd)}")

        rec = dict(cfg)
        rec["exp_name"] = exp_name
        rec["status"] = "ok"
        rec["grid_preset"] = args.grid_preset
        rec["epochs"] = args.epochs
        start = time.time()

        try:
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        except subprocess.CalledProcessError as exc:
            print(f"[WARN] Trial failed for {exp_name}: {exc}")
            rec["status"] = "failed"
            rec["best_val_acc"] = 0.0
            rec["history_path"] = ""
            rec["duration_sec"] = round(time.time() - start, 2)
            rec["error"] = str(exc)
            results.append(rec)
            continue

        rec["duration_sec"] = round(time.time() - start, 2)
        run_dir = Path(SAVE_DIR) / args.save_subdir / args.model / exp_name
        rec["history_path"] = str(run_dir / "history.json")
        try:
            rec["best_val_acc"] = _load_best_val_acc(run_dir)
        except FileNotFoundError:
            rec["status"] = "missing_metrics"
            rec["best_val_acc"] = 0.0
        results.append(rec)

        results.sort(key=lambda x: x.get("best_val_acc", 0.0), reverse=True)
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        best_path.write_text(json.dumps(results[0] if results else {}, indent=2), encoding="utf-8")

    results.sort(key=lambda x: x.get("best_val_acc", 0.0), reverse=True)
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    best_path.write_text(json.dumps(results[0] if results else {}, indent=2), encoding="utf-8")

    print(f"\nSaved results: {results_path}")
    print(f"Saved best config: {best_path}")
    print("\nTop 5 configs:")
    for rank, rec in enumerate(results[:5], start=1):
        print(
            f"#{rank} | exp={rec['exp_name']} | val_acc={rec.get('best_val_acc', 0.0):.2f} | "
            f"lr={rec['lr']} | wd={rec['wd']} | ls={rec['label_smoothing']} | "
            f"mixup={rec['use_mixup']}({rec['mixup_alpha']}) | cutmix={rec['use_cutmix']}({rec['cutmix_alpha']}) | "
            f"status={rec['status']} | time={rec.get('duration_sec', 0.0):.1f}s"
        )


if __name__ == "__main__":
    main()
