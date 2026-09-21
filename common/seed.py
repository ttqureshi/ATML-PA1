"""Reproducibility helpers shared across tasks."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 6304) -> None:
    """Fix common RNG sources so splits and runs are reproducible.

    Assumption: we set seeds for Python, NumPy, and PyTorch (CPU/CUDA).
    Full bit-wise reproducibility across GPUs/OS is not guaranteed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Prefer determinism where cheap; some ops may still be nondeterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
