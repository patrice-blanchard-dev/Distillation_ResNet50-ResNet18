"""CIFAR-100 data loaders and augmentation policies."""

import random
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment


class Cutout:
    """Apply a zero mask on a random square region of a tensor image."""

    def __init__(self, size: int = 16):
        self.size = int(size)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(img):
            raise TypeError('Cutout expects a tensor image after ToTensor().')

        out = img.clone()
        _, h, w = out.shape
        half = self.size // 2

        cy = random.randint(0, h - 1)
        cx = random.randint(0, w - 1)

        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)

        out[:, y1:y2, x1:x2] = 0.0
        return out


def _make_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Create train/evaluation transforms with CIFAR-100 normalization."""
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761],
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
            Cutout(size=16),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def _make_worker_init_fn(base_seed: int):
    """Seed each dataloader worker deterministically from the base seed."""
    def worker_init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return worker_init_fn


def _build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool, seed: int):
    """Build a reproducible DataLoader with GPU-friendly defaults."""
    worker_init_fn = _make_worker_init_fn(seed)
    generator = torch.Generator().manual_seed(seed)

    kwargs: Dict[str, object] = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': True,
        'worker_init_fn': worker_init_fn,
        'generator': generator,
        'drop_last': False,
    }
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 4

    return DataLoader(**kwargs)


def get_cifar100(data_dir, batch_size, num_workers, val_split=0.1, seed=42):
    """Return train/validation/test loaders with a deterministic train split."""
    train_transform, eval_transform = _make_transforms()

    full_train_aug = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    full_train_clean = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=False,
        transform=eval_transform,
    )
    test_set = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    n_total = len(full_train_aug)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_set = Subset(full_train_aug, train_idx)
    val_set = Subset(full_train_clean, val_idx)

    train_loader = _build_loader(train_set, batch_size, num_workers, shuffle=True, seed=seed)
    val_loader = _build_loader(val_set, batch_size, num_workers, shuffle=False, seed=seed)
    test_loader = _build_loader(test_set, batch_size, num_workers, shuffle=False, seed=seed)

    return train_loader, val_loader, test_loader, (n_train, n_val, len(test_set))


def get_cifar100_fulltrain(data_dir, batch_size, num_workers, seed=42):
    """Return train/test loaders using all official training images."""
    train_transform, eval_transform = _make_transforms()

    train_set = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_set = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=eval_transform,
    )

    train_loader = _build_loader(train_set, batch_size, num_workers, shuffle=True, seed=seed)
    test_loader = _build_loader(test_set, batch_size, num_workers, shuffle=False, seed=seed)
    return train_loader, test_loader
