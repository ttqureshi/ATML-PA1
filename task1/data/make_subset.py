"""Build reproducible STL-10 splits and the shared evaluation subset.

Outputs (JSON) under task1/results/splits/:
  - train_val_split.json : stratified 80/20 from official train (seed 6304)
  - eval_subset.json     : class-balanced subset of official test (seed 6304)

ASSUMPTION: dataset is STL-10 (assignment recommendation).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms


STL10_CLASSES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_train_val_split(
    labels: np.ndarray,
    train_frac: float,
    seed: int,
) -> Dict[str, List[int]]:
    indices = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        indices,
        train_size=train_frac,
        stratify=labels,
        random_state=seed,
    )
    return {
        "train_indices": train_idx.astype(int).tolist(),
        "val_indices": val_idx.astype(int).tolist(),
    }


def make_eval_subset(
    labels: np.ndarray,
    subset_size: int,
    seed: int,
    class_names: List[str],
) -> dict:
    """Class-balanced sample from the official test set.

    If a class has too few examples, take all of them and document imbalance.
    """
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[int]] = defaultdict(list)
    for i, y in enumerate(labels.tolist()):
        by_class[int(y)].append(i)

    n_classes = len(class_names)
    per_class = subset_size // n_classes
    selected: List[int] = []
    counts = {}
    notes = []

    for c in range(n_classes):
        pool = by_class[c]
        take = min(per_class, len(pool))
        chosen = rng.choice(pool, size=take, replace=False).tolist()
        selected.extend(chosen)
        counts[class_names[c]] = take
        if take < per_class:
            notes.append(
                f"Class '{class_names[c]}' has only {len(pool)} test images; "
                f"used all {take} (requested {per_class})."
            )

    # If we still need more to approach subset_size (rare), fill from leftovers.
    selected_set = set(selected)
    if len(selected) < subset_size:
        leftovers = [i for i in range(len(labels)) if i not in selected_set]
        need = min(subset_size - len(selected), len(leftovers))
        if need > 0:
            extra = rng.choice(leftovers, size=need, replace=False).tolist()
            selected.extend(extra)
            notes.append(f"Filled {need} extra images to approach subset_size.")

    selected = sorted(selected)
    return {
        "indices": selected,
        "per_class_counts": counts,
        "notes": notes,
        "requested_size": subset_size,
        "actual_size": len(selected),
    }


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    root = Path(cfg["dataset"]["root"])
    root.mkdir(parents=True, exist_ok=True)

    # Download once; we only need labels here (images loaded later).
    train_set = datasets.STL10(
        root=str(root),
        split="train",
        download=True,
        transform=transforms.ToTensor(),
    )
    test_set = datasets.STL10(
        root=str(root),
        split="test",
        download=True,
        transform=transforms.ToTensor(),
    )

    train_labels = np.array(train_set.labels)
    test_labels = np.array(test_set.labels)

    split = make_train_val_split(
        train_labels,
        train_frac=float(cfg["dataset"]["train_frac"]),
        seed=seed,
    )
    eval_subset = make_eval_subset(
        test_labels,
        subset_size=int(cfg["dataset"]["eval_subset_size"]),
        seed=seed,
        class_names=STL10_CLASSES,
    )

    out_dir = Path(cfg["paths"]["splits_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_val_path = out_dir / "train_val_split.json"
    eval_path = out_dir / "eval_subset.json"

    payload_split = {
        "seed": seed,
        "dataset": "stl10",
        "class_names": STL10_CLASSES,
        "n_train_official": len(train_set),
        "n_test_official": len(test_set),
        **split,
    }
    payload_eval = {
        "seed": seed,
        "dataset": "stl10",
        "class_names": STL10_CLASSES,
        "split": "test",
        **eval_subset,
    }

    train_val_path.write_text(json.dumps(payload_split, indent=2), encoding="utf-8")
    eval_path.write_text(json.dumps(payload_eval, indent=2), encoding="utf-8")

    print(f"Wrote {train_val_path}")
    print(f"  train={len(split['train_indices'])} val={len(split['val_indices'])}")
    print(f"Wrote {eval_path}")
    print(f"  eval_subset={eval_subset['actual_size']} counts={eval_subset['per_class_counts']}")
    if eval_subset["notes"]:
        print("Notes:")
        for n in eval_subset["notes"]:
            print(" -", n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Task 1 STL-10 splits.")
    parser.add_argument(
        "--config",
        default="task1/configs/default.yaml",
        help="Path to Task 1 YAML config.",
    )
    args = parser.parse_args()
    main(args.config)
