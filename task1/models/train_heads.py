"""Train linear classifier heads on frozen backbone features.

Protocol (assignment):
  - Stratified 80/20 from official STL-10 train (seed 6304)
  - AdamW lr=1e-3, weight_decay=1e-4
  - Max 50 epochs, early stop after 5 epochs without val-acc improvement
  - Backbone frozen; only the linear head is optimized
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import yaml

from common.seed import set_seed
from common.metrics import top1_accuracy, macro_f1, mean_max_confidence
from task1.data.make_subset import STL10_CLASSES
from task1.data.transforms import to_numpy_uint8
from task1.models.backbones import LinearHead, build_backbone, softmax_np


class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def extract_features(
    backbone,
    preprocess,
    dataset,
    indices: List[int],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract frozen features for a list of dataset indices."""
    feats = []
    labels = []
    backbone.eval()
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        tensors = []
        ys = []
        for i in batch_idx:
            img, y = dataset[i]
            # STL10 returns PIL when transform=None; ToTensor otherwise.
            rgb = to_numpy_uint8(img)
            tensors.append(preprocess(rgb))
            ys.append(int(y))
        x = torch.stack(tensors, dim=0).to(device)
        f = backbone(x).cpu().numpy()
        feats.append(f)
        labels.append(np.asarray(ys, dtype=np.int64))
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)


def train_linear_head(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    cfg: dict,
    device: torch.device,
) -> Tuple[LinearHead, dict]:
    set_seed(int(cfg["seed"]))
    in_dim = train_x.shape[1]
    num_classes = len(STL10_CLASSES)
    head = LinearHead(in_dim, num_classes).to(device)

    train_loader = DataLoader(
        FeatureDataset(train_x, train_y),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        FeatureDataset(val_x, val_y),
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
    )

    opt = torch.optim.AdamW(
        head.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    best_state = None
    patience = int(cfg["training"]["early_stop_patience"])
    bad_epochs = 0
    history = []

    for epoch in range(1, int(cfg["training"]["max_epochs"]) + 1):
        head.train()
        total_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = head(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * len(yb)
            n += len(yb)

        # Validation
        head.eval()
        logits_all, ys_all = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits_all.append(head(xb).cpu().numpy())
                ys_all.append(yb.numpy())
        logits_np = np.concatenate(logits_all, axis=0)
        ys_np = np.concatenate(ys_all, axis=0)
        preds = logits_np.argmax(axis=1)
        acc = top1_accuracy(ys_np, preds)
        f1 = macro_f1(ys_np, preds)
        conf = mean_max_confidence(softmax_np(logits_np))
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(n, 1),
                "val_acc": acc,
                "val_macro_f1": f1,
                "val_mean_max_conf": conf,
            }
        )
        print(
            f"  epoch {epoch:02d}  loss={total_loss/max(n,1):.4f}  "
            f"val_acc={acc:.4f}  val_f1={f1:.4f}"
        )

        if acc > best_acc + 1e-6:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  Early stopping at epoch {epoch} (best val_acc={best_acc:.4f})")
                break

    assert best_state is not None
    head.load_state_dict(best_state)
    summary = {
        "best_val_acc": best_acc,
        "history": history,
        "feature_dim": in_dim,
        "num_classes": num_classes,
    }
    return head, summary


def train_all_heads(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    splits = json.loads(Path(cfg["paths"]["splits_dir"], "train_val_split.json").read_text())
    train_idx = splits["train_indices"]
    val_idx = splits["val_indices"]

    root = Path(cfg["dataset"]["root"])
    # PIL images (no transform) so preprocess owns resize/normalize.
    train_set = datasets.STL10(root=str(root), split="train", download=True)

    heads_dir = Path(cfg["paths"]["heads_dir"])
    feat_dir = Path(cfg["paths"]["features_dir"])
    heads_dir.mkdir(parents=True, exist_ok=True)
    feat_dir.mkdir(parents=True, exist_ok=True)

    backbone_names = [m for m in cfg["models"] if m != "clip_zeroshot"]
    for name in backbone_names:
        print(f"\n=== Training linear head for {name} ===")
        backbone, info = build_backbone(name, device)

        # Cache train/val features so re-runs are cheap.
        cache_train = feat_dir / f"{name}_train_features.npz"
        if cache_train.exists():
            data = np.load(cache_train)
            train_x, train_y = data["features"], data["labels"]
            val_x, val_y = data["val_features"], data["val_labels"]
            print(f"  Loaded cached features from {cache_train}")
        else:
            print("  Extracting train features...")
            train_x, train_y = extract_features(
                backbone, info.preprocess, train_set, train_idx, device,
                batch_size=int(cfg["training"]["batch_size"]),
            )
            print("  Extracting val features...")
            val_x, val_y = extract_features(
                backbone, info.preprocess, train_set, val_idx, device,
                batch_size=int(cfg["training"]["batch_size"]),
            )
            np.savez_compressed(
                cache_train,
                features=train_x,
                labels=train_y,
                val_features=val_x,
                val_labels=val_y,
            )

        head, summary = train_linear_head(train_x, train_y, val_x, val_y, cfg, device)
        ckpt = {
            "model_name": name,
            "state_dict": head.state_dict(),
            "feature_dim": info.feature_dim,
            "class_names": STL10_CLASSES,
            "summary": summary,
            "seed": int(cfg["seed"]),
        }
        out = heads_dir / f"{name}_linear_head.pt"
        torch.save(ckpt, out)
        (heads_dir / f"{name}_train_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"  Saved {out} (best val_acc={summary['best_val_acc']:.4f})")


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task1/configs/default.yaml")
    args = parser.parse_args()
    train_all_heads(args.config)
