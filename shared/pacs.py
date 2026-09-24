"""PACS dataset loading, transforms, and stratified source splits.

PACS has four visual *domains* and the same seven *object classes* in each:
  Domains: photo, art_painting, cartoon, sketch
  Classes: dog, elephant, giraffe, guitar, horse, house, person

Tasks 2 and 3 both use:
  Sources (labeled): photo, art_painting, cartoon
  Target:            sketch

Folder layout expected under ``data/pacs/`` (standard PACS layout)::

    data/pacs/
      photo/dog/*.jpg
      photo/elephant/*.jpg
      ...
      art_painting/...
      cartoon/...
      sketch/...

ASSUMPTIONS
-----------
1. Domain and class folder names match the constants below (case-sensitive).
2. Stratified 80/20 train/val is built *per source domain* with seed 6304 and
   saved once to ``shared/splits/pacs_sketch_seed6304.json`` so Task 3 reuses
   the exact same IDs.
3. Sketch labels are never used for training or checkpoint selection in Task 2;
   they appear only in final evaluation / class analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms


# ---- Assignment constants -------------------------------------------------

DOMAINS: tuple[str, ...] = ("photo", "art_painting", "cartoon", "sketch")
SOURCE_DOMAINS: tuple[str, ...] = ("photo", "art_painting", "cartoon")
TARGET_DOMAIN: str = "sketch"

CLASS_NAMES: tuple[str, ...] = (
    "dog",
    "elephant",
    "giraffe",
    "guitar",
    "horse",
    "house",
    "person",
)
CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES: int = len(CLASS_NAMES)

# ImageNet normalization for ResNet18_Weights.IMAGENET1K_V1.
# Hardcoded because some torchvision builds omit mean/std from weights.meta.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PacsExample:
    """One image record used in splits and datasets."""

    path: str  # absolute or repo-relative path string
    domain: str
    class_name: str
    class_idx: int
    # Stable ID for reproducibility across machines: domain/class/filename
    uid: str


def _image_extensions() -> set[str]:
    return {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def discover_pacs(root: str | Path) -> list[PacsExample]:
    """Walk ``root/{domain}/{class}/*`` and return every image found."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"PACS root not found: {root}\n"
            "Download PACS into data/pacs/ (see shared/README.md)."
        )

    examples: list[PacsExample] = []
    for domain in DOMAINS:
        domain_dir = root / domain
        if not domain_dir.is_dir():
            raise FileNotFoundError(f"Missing domain folder: {domain_dir}")
        for class_name in CLASS_NAMES:
            class_dir = domain_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            for path in sorted(class_dir.iterdir()):
                if path.suffix.lower() not in _image_extensions():
                    continue
                uid = f"{domain}/{class_name}/{path.name}"
                examples.append(
                    PacsExample(
                        path=str(path.resolve()),
                        domain=domain,
                        class_name=class_name,
                        class_idx=CLASS_TO_IDX[class_name],
                        uid=uid,
                    )
                )
    if not examples:
        raise RuntimeError(f"No images found under {root}")
    return examples


def build_source_splits(
    examples: Sequence[PacsExample],
    *,
    train_frac: float = 0.8,
    seed: int = 6304,
) -> dict:
    """Stratified 80/20 train/val *within each source domain*.

    Returns a JSON-serializable dict keyed by domain, then by split name,
    listing example UIDs (not absolute paths) so the file is portable.
    """
    by_domain: dict[str, list[PacsExample]] = {d: [] for d in SOURCE_DOMAINS}
    for ex in examples:
        if ex.domain in by_domain:
            by_domain[ex.domain].append(ex)

    splits: dict = {
        "seed": seed,
        "train_frac": train_frac,
        "source_domains": list(SOURCE_DOMAINS),
        "target_domain": TARGET_DOMAIN,
        "class_names": list(CLASS_NAMES),
        "domains": {},
    }

    for domain, domain_examples in by_domain.items():
        y = np.array([ex.class_idx for ex in domain_examples], dtype=np.int64)
        indices = np.arange(len(domain_examples))
        sss = StratifiedShuffleSplit(
            n_splits=1, train_size=train_frac, random_state=seed
        )
        train_idx, val_idx = next(sss.split(indices, y))
        train_uids = [domain_examples[i].uid for i in sorted(train_idx.tolist())]
        val_uids = [domain_examples[i].uid for i in sorted(val_idx.tolist())]
        splits["domains"][domain] = {
            "train": train_uids,
            "val": val_uids,
            "n_train": len(train_uids),
            "n_val": len(val_uids),
        }

    # Target: all sketch UIDs (labels exist on disk but are unused in training).
    target = [ex for ex in examples if ex.domain == TARGET_DOMAIN]
    splits["target"] = {
        "all": [ex.uid for ex in target],
        "n_all": len(target),
        "note": (
            "Task 2 may use these images without class labels during adaptation. "
            "Task 3 must not load Sketch until final evaluation."
        ),
    }
    return splits


def save_splits(splits: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")


def load_splits(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_transforms(*, train: bool, image_size: int = 224, resize_size: int = 256):
    """Assignment preprocessing for ResNet-18 ImageNet weights.

    Train: resize to ``resize_size``, RandomResizedCrop(image_size), HFlip, Normalize.
    Eval:  resize to ``resize_size``, CenterCrop(image_size), Normalize.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if train:
        return transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class PacsImageDataset(Dataset):
    """Simple image dataset indexed by a list of PacsExample records."""

    def __init__(
        self,
        examples: Sequence[PacsExample],
        transform: Callable | None = None,
        *,
        return_domain: bool = True,
        return_label: bool = True,
    ) -> None:
        self.examples = list(examples)
        self.transform = transform
        self.return_domain = return_domain
        self.return_label = return_label
        self.domain_to_idx = {d: i for i, d in enumerate(DOMAINS)}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        ex = self.examples[index]
        image = Image.open(ex.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        out = {"image": image, "uid": ex.uid}
        if self.return_label:
            out["label"] = ex.class_idx
        if self.return_domain:
            out["domain"] = self.domain_to_idx[ex.domain]
            out["domain_name"] = ex.domain
        return out


def index_by_uid(examples: Iterable[PacsExample]) -> dict[str, PacsExample]:
    return {ex.uid: ex for ex in examples}


def examples_from_uids(
    uid_list: Sequence[str], uid_index: dict[str, PacsExample]
) -> list[PacsExample]:
    missing = [u for u in uid_list if u not in uid_index]
    if missing:
        raise KeyError(f"Split UIDs not found on disk (first few): {missing[:5]}")
    return [uid_index[u] for u in uid_list]
