"""Central configuration values for the CIFAR-100 distillation experiments.

The training scripts expose most of these values as command-line arguments.
Keeping the defaults here makes experiments reproducible while avoiding magic
numbers spread across the project.
"""

from itertools import product
from typing import Any, Dict, List

# Default filesystem layout. These folders are intentionally ignored by Git
# because they contain downloaded data, checkpoints, plots and experiment logs.
DATA_DIR = "./data"
SAVE_DIR = "./save"
SEED = 42

# CIFAR-100 and data-loading defaults.
NUM_CLASSES = 100
BATCH_SIZE = 128
NUM_WORKERS = 4

# Baseline training schedules.
EPOCHS_TEACHER = 240
EPOCHS_STUDENT = 240

# Optimizer defaults used by teacher and student baselines.
BASE_LR_TEACHER = 0.1
BASE_LR_STUDENT = 0.05
WEIGHT_DECAY = 5e-4

TRAIN_VAL_SPLIT = 0.1  # 10% of CIFAR-100 train is held out for validation.

# Compact hyperparameter grids used by the grid-search scripts.
KD_GRID = {
    "T": [2.0, 4.0, 6.0],
    "alpha": [0.5, 0.7, 0.9],  # Weight of KD relative to cross-entropy.
    "lr": [0.05, 0.1],
}

AT_GRID = {
    "beta": [0.5, 1.0],
    "lr": [0.05, 0.1],
}

FITNET_GRID = {
    "beta": [10.0, 50.0],
    "lr": [0.05, 0.1],
}

DKD_GRID = {
    "T": [4.0, 6.0],
    "dkd_alpha": [1.0, 2.0],
    "dkd_beta": [4.0, 6.0, 8.0],
    "lr": [0.05, 0.08],
}


def grid_to_list(grid_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Expand a dictionary of hyperparameter choices into a list of configs."""
    keys = list(grid_dict.keys())
    values = [grid_dict[k] for k in keys]
    combos: List[Dict[str, Any]] = []
    for vals in product(*values):
        cfg: Dict[str, Any] = {}
        for k, v in zip(keys, vals):
            cfg[k] = v
        combos.append(cfg)
    return combos
