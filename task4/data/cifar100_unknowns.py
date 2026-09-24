"""CIFAR-100 near/far unknown loaders (evaluation only).

Hard rule: these loaders must never be used for training, checkpoint
selection, score design, or threshold calibration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100

from .cifar10 import make_cifar10_transforms
from .constants import FAR_UNKNOWN_CLASSES, NEAR_UNKNOWN_CLASSES


class FilteredCIFAR100(Dataset):
    """CIFAR-100 test subset restricted to a fixed fine-class name list."""

    def __init__(
        self,
        root: str | Path,
        class_names: Sequence[str],
        transform=None,
        download: bool = True,
        group: str = "unknown",
    ):
        self.base = CIFAR100(root=str(root), train=False, download=download, transform=transform)
        name_to_idx = {n: i for i, n in enumerate(self.base.classes)}
        missing = [n for n in class_names if n not in name_to_idx]
        if missing:
            raise ValueError(f"Unknown CIFAR-100 class names: {missing}")
        self.keep_labels = {name_to_idx[n] for n in class_names}
        self.class_names = list(class_names)
        self.group = group
        self.indices = [
            i for i, y in enumerate(self.base.targets) if int(y) in self.keep_labels
        ]
        # Map CIFAR-100 fine label → readable name for failure analysis.
        self.idx_to_name = {name_to_idx[n]: n for n in class_names}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        img, y = self.base[idx]
        y = int(y)
        return img, y, int(idx), self.idx_to_name[y]


def build_unknown_loaders(cfg: dict, *, download: bool = True) -> dict:
    """Build near / far / all-unknown eval loaders from CIFAR-100 **test** only."""
    root = Path(cfg["data"]["root"])
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    bs = cfg["training"]["batch_size"]
    nw = cfg["data"].get("num_workers", 2)
    eval_tf = make_cifar10_transforms(train=False, mean=mean, std=std)

    near_names = cfg["evaluation"]["near_classes"]
    far_names = cfg["evaluation"]["far_classes"]
    # Sanity: fixed lists from constants unless config overrides (cfg mirrors constants).
    assert list(near_names) == NEAR_UNKNOWN_CLASSES
    assert list(far_names) == FAR_UNKNOWN_CLASSES

    near_ds = FilteredCIFAR100(root, near_names, transform=eval_tf, download=download, group="near")
    far_ds = FilteredCIFAR100(root, far_names, transform=eval_tf, download=download, group="far")

    def _loader(ds):
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
        )

    return {
        "near": _loader(near_ds),
        "far": _loader(far_ds),
        "n_near": len(near_ds),
        "n_far": len(far_ds),
        "near_classes": near_names,
        "far_classes": far_names,
    }
