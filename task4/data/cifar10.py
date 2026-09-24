"""CIFAR-10 loaders for known-class train / val / test."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import RandAugment

from .make_splits import load_splits


def make_cifar10_transforms(
    train: bool,
    mean: Sequence[float],
    std: Sequence[float],
    pad: int = 4,
    randaugment: bool = False,
    randaugment_num_ops: int = 2,
    randaugment_magnitude: int = 9,
):
    """Build transforms.

    Train order (assignment): random crop → flip → [optional RandAugment] → ToTensor → Normalize.
    Eval: ToTensor → Normalize only (no augmentation).
    """
    tfms: list = []
    if train:
        tfms.append(transforms.RandomCrop(32, padding=pad))
        tfms.append(transforms.RandomHorizontalFlip())
        if randaugment:
            # After crop+flip, before ToTensor/Normalize (assignment).
            tfms.append(
                RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude)
            )
    tfms.append(transforms.ToTensor())
    tfms.append(transforms.Normalize(mean=list(mean), std=list(std)))
    return transforms.Compose(tfms)


class IndexedSubset(Dataset):
    """Subset that also returns the underlying CIFAR index (for caching / failure IDs)."""

    def __init__(self, base: Dataset, indices: Sequence[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        img, y = self.base[idx]
        return img, int(y), int(idx)


def build_cifar10_loaders(
    cfg: dict,
    *,
    randaugment: bool = False,
    download: bool = True,
) -> dict:
    """Return train/val/test loaders + a train_eval (unaugmented train) loader.

    ``train_eval`` is used for Mahalanobis stats (unaugmented CIFAR-10 train features).
    """
    root = Path(cfg["data"]["root"])
    splits = load_splits(cfg["data"]["splits"])
    mean, std = cfg["data"]["mean"], cfg["data"]["std"]
    pad = cfg["training"].get("pad", 4)
    bs = cfg["training"]["batch_size"]
    nw = cfg["data"].get("num_workers", 2)

    mcfg = cfg["methods"].get(cfg.get("method", "vanilla"), {})
    ra_ops = mcfg.get("randaugment_num_ops", 2)
    ra_mag = mcfg.get("randaugment_magnitude", 9)

    train_tf = make_cifar10_transforms(
        train=True,
        mean=mean,
        std=std,
        pad=pad,
        randaugment=randaugment,
        randaugment_num_ops=ra_ops,
        randaugment_magnitude=ra_mag,
    )
    eval_tf = make_cifar10_transforms(train=False, mean=mean, std=std)

    train_base = CIFAR10(root=str(root), train=True, download=download, transform=train_tf)
    train_eval_base = CIFAR10(
        root=str(root), train=True, download=download, transform=eval_tf
    )
    test_base = CIFAR10(root=str(root), train=False, download=download, transform=eval_tf)

    train_idx = splits["train_indices"]
    val_idx = splits["val_indices"]
    test_idx = list(range(len(test_base)))

    train_ds = IndexedSubset(train_base, train_idx)
    train_eval_ds = IndexedSubset(train_eval_base, train_idx)
    val_ds = IndexedSubset(train_eval_base, val_idx)  # val never augmented
    test_ds = IndexedSubset(test_base, test_idx)

    def _loader(ds, shuffle: bool, drop_last: bool = False):
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
        )

    return {
        "train": _loader(train_ds, shuffle=True, drop_last=True),
        "train_eval": _loader(train_eval_ds, shuffle=False),
        "val": _loader(val_ds, shuffle=False),
        "test": _loader(test_ds, shuffle=False),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "splits": splits,
    }
