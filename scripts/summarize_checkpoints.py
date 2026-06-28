"""Summarize parameter counts, weight memory and checkpoint sizes."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Tuple

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from config import NUM_CLASSES, SAVE_DIR
from models import resnet18_cifar, resnet50_cifar

ModelSpec = Tuple[Callable[[], nn.Module], str]


DEFAULT_MODELS: Dict[str, ModelSpec] = {
    "Teacher": (
        lambda: resnet50_cifar(num_classes=NUM_CLASSES),
        "teacher/resnet50_cifar/teacher_final_optuna_best/best.pth",
    ),
    "Student": (
        lambda: resnet18_cifar(num_classes=NUM_CLASSES),
        "student/resnet18_cifar/student_final_baseline/best.pth",
    ),
    "KD": (
        lambda: resnet18_cifar(num_classes=NUM_CLASSES),
        "kd/resnet18_cifar/kd_final_best/best.pth",
    ),
    "AT": (
        lambda: resnet18_cifar(num_classes=NUM_CLASSES),
        "at/resnet18_cifar/at_final_best/best.pth",
    ),
    "FitNet": (
        lambda: resnet18_cifar(num_classes=NUM_CLASSES),
        "fitnet_final/resnet18_cifar/fitnet_final_best_fulltrain/best.pth",
    ),
    "DKD": (
        lambda: resnet18_cifar(num_classes=NUM_CLASSES),
        "dkd_final/resnet18_cifar/dkd_final_optuna_best_fulltrain/best.pth",
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for checkpoint summarization."""
    parser = argparse.ArgumentParser(description="Summarize trained checkpoint sizes and parameter counts.")
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR, help="Root directory containing experiment outputs.")
    return parser.parse_args()


def safe_torch_load(path: Path):
    """Load a checkpoint across PyTorch versions with and without weights_only."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint):
    """Return the model state dict from common checkpoint layouts."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_weights(model: nn.Module, checkpoint_path: Path) -> None:
    """Load weights into a model from a checkpoint path."""
    checkpoint = safe_torch_load(checkpoint_path)
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def float32_weight_size_mb(num_parameters: int) -> float:
    """Return the theoretical float32 memory footprint in decimal megabytes."""
    return num_parameters * 4 / 1_000_000


def main() -> None:
    """Print a compact table for all configured checkpoints."""
    args = parse_args()
    save_dir = Path(args.save_dir)
    rows = []

    for name, (builder, relative_checkpoint) in DEFAULT_MODELS.items():
        checkpoint_path = save_dir / relative_checkpoint
        if not checkpoint_path.exists():
            print(f"[missing] {name}: {checkpoint_path}")
            continue

        model = builder()
        load_weights(model, checkpoint_path)
        total_params, trainable_params = count_parameters(model)
        weight_size_mb = float32_weight_size_mb(total_params)
        checkpoint_size_mb = os.path.getsize(checkpoint_path) / (1024**2)
        rows.append((name, total_params, trainable_params, weight_size_mb, checkpoint_size_mb))

    print("-" * 112)
    print(
        f"{'Model':<10} {'Parameters':>15} {'Trainable':>15} {'Millions':>12} "
        f"{'Weights f32 (MB)':>17} {'Checkpoint (MiB)':>17}"
    )
    print("-" * 112)
    for name, total_params, trainable_params, weight_size_mb, checkpoint_size_mb in rows:
        print(
            f"{name:<10} {total_params:>15} {trainable_params:>15} {total_params / 1e6:>12.3f} "
            f"{weight_size_mb:>17.2f} {checkpoint_size_mb:>17.2f}"
        )
    print("-" * 112)


if __name__ == "__main__":
    main()
