"""Stratified 90/10 CIFAR-10 train split (seed 6304).

Writes JSON indices only — no images in git.
CIFAR-100 is never touched here (no-leakage).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import CIFAR10


def make_cifar10_splits(
    root: str | Path,
    out_path: str | Path,
    train_frac: float = 0.9,
    seed: int = 6304,
    download: bool = True,
) -> dict:
    root = Path(root)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trainset = CIFAR10(root=str(root), train=True, download=download)
    targets = np.asarray(trainset.targets)
    indices = np.arange(len(targets))

    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=train_frac, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(indices, targets))
    train_idx = np.sort(train_idx).tolist()
    val_idx = np.sort(val_idx).tolist()

    payload = {
        "dataset": "CIFAR-10",
        "seed": seed,
        "train_frac": train_frac,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_test": 10000,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "note": "Test = official CIFAR-10 test set (all indices). "
        "CIFAR-100 never used for splits / training / thresholds.",
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path} (train={len(train_idx)}, val={len(val_idx)})")
    return payload


def load_splits(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/cifar")
    p.add_argument("--out", default="task4/results/splits/cifar10_seed6304.json")
    p.add_argument("--seed", type=int, default=6304)
    p.add_argument("--train-frac", type=float, default=0.9)
    args = p.parse_args()
    make_cifar10_splits(args.root, args.out, args.train_frac, args.seed)
