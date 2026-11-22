"""Shared experiment defaults. Change these together as a team."""

from pathlib import Path
import random

import numpy as np

NUM_CLASSES = 43
IMAGE_SIZE = (32, 32)  # (width, height), as expected by OpenCV
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.2
EXPECTED_TRAIN_IMAGES = 39_209
EXPECTED_TEST_IMAGES = 12_630

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_DATA_ROOT = PROJECT_ROOT / "Dataset" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

CONDITIONS = ("clean", "brightness", "noise", "motion_blur")
DEGRADATION_PARAMS = {
    "brightness": {"factor": 0.5},
    "noise": {"std": 0.05},  # standard deviation on the normalized [0, 1] scale
    "motion_blur": {"kernel_size": 9},  # horizontal motion
}


def get_data_root(use_drive=False, drive_data_root=None):
    """Select the configured Drive folder or the repository's raw data folder."""
    if use_drive:
        if not drive_data_root:
            raise ValueError("Set DRIVE_DATA_ROOT to the folder containing Train.csv and Test.csv.")
        return Path(drive_data_root).expanduser().resolve()
    return REPO_DATA_ROOT


def seed_everything(seed=RANDOM_SEED):
    """Seed Python, NumPy, and PyTorch (if installed); return the seed.

    CUDA backend choices are made reproducible where possible. Exact equality
    across hardware/library versions is not guaranteed. Future DataLoaders with
    random workers must also seed those workers and their torch.Generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed
