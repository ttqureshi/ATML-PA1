"""Extract and cache logits / features for identical score inputs.

Saves under task4/results/cache/{model}/{split}.npz
CIFAR-100 is extracted only for final evaluation caches — never for training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.seed import set_seed
from task4.data.cifar10 import build_cifar10_loaders
from task4.data.cifar100_unknowns import build_unknown_loaders
from task4.models.resnet_cifar import build_resnet18_cifar


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_dummy = int(ckpt.get("num_dummy", 0))
    # Infer num_dummy from state dict if needed.
    if num_dummy == 0 and any(k.startswith("fc_dummy") for k in ckpt["model_state"]):
        w = ckpt["model_state"].get("fc_dummy.weight")
        if w is not None:
            num_dummy = int(w.shape[0])
    model = build_resnet18_cifar(num_classes=10, num_dummy=num_dummy)
    if num_dummy > 0 and model.fc_dummy is None:
        model.attach_dummy_heads(num_dummy)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, ckpt


@torch.no_grad()
def extract_known_split(model, loader, device, has_dummy: bool) -> dict:
    feats, logits, labels, indices = [], [], [], []
    dummies = [] if has_dummy else None
    for batch in tqdm(loader, leave=False):
        images, y, idx = batch[0].to(device), batch[1], batch[2]
        f, z = model.forward_features(images)
        feats.append(f.cpu().numpy())
        logits.append(z.cpu().numpy())
        labels.append(y.numpy())
        indices.append(idx.numpy())
        if has_dummy:
            d = model.fc_dummy(f)
            dummies.append(d.cpu().numpy())
    out = {
        "features": np.concatenate(feats, axis=0).astype(np.float32),
        "logits": np.concatenate(logits, axis=0).astype(np.float32),
        "labels": np.concatenate(labels, axis=0).astype(np.int64),
        "indices": np.concatenate(indices, axis=0).astype(np.int64),
    }
    if has_dummy:
        out["dummy_logits"] = np.concatenate(dummies, axis=0).astype(np.float32)
    return out


@torch.no_grad()
def extract_unknown_split(model, loader, device, has_dummy: bool) -> dict:
    feats, logits, cifar100_labels, indices, names = [], [], [], [], []
    dummies = [] if has_dummy else None
    for batch in tqdm(loader, leave=False):
        images = batch[0].to(device)
        y100, idx, name = batch[1], batch[2], batch[3]
        f, z = model.forward_features(images)
        feats.append(f.cpu().numpy())
        logits.append(z.cpu().numpy())
        cifar100_labels.append(y100.numpy())
        indices.append(idx.numpy())
        names.extend(list(name))
        if has_dummy:
            d = model.fc_dummy(f)
            dummies.append(d.cpu().numpy())
    out = {
        "features": np.concatenate(feats, axis=0).astype(np.float32),
        "logits": np.concatenate(logits, axis=0).astype(np.float32),
        "cifar100_labels": np.concatenate(cifar100_labels, axis=0).astype(np.int64),
        "indices": np.concatenate(indices, axis=0).astype(np.int64),
        "class_names": np.asarray(names),
    }
    if has_dummy:
        out["dummy_logits"] = np.concatenate(dummies, axis=0).astype(np.float32)
    return out


def extract_for_model(cfg: dict, method: str, device: torch.device) -> Path:
    ckpt_path = Path(cfg["paths"]["checkpoints_dir"]) / f"{method}_best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}")

    model, ckpt = load_model(ckpt_path, device)
    has_dummy = model.fc_dummy is not None

    # Known loaders: never use RandAugment for extraction.
    known = build_cifar10_loaders(cfg, randaugment=False)
    unknowns = build_unknown_loaders(cfg)

    cache_root = Path(cfg["paths"]["cache_dir"]) / method
    cache_root.mkdir(parents=True, exist_ok=True)

    splits = {
        "train_eval": known["train_eval"],
        "val": known["val"],
        "test": known["test"],
    }
    for name, loader in splits.items():
        print(f"[{method}] extract known/{name}")
        arr = extract_known_split(model, loader, device, has_dummy)
        np.savez_compressed(cache_root / f"{name}.npz", **arr)

    for group in ("near", "far"):
        print(f"[{method}] extract unknown/{group}")
        arr = extract_unknown_split(model, unknowns[group], device, has_dummy)
        np.savez_compressed(cache_root / f"{group}.npz", **arr)

    meta = {
        "method": method,
        "checkpoint": str(ckpt_path).replace("\\", "/"),
        "has_dummy": has_dummy,
        "num_dummy": int(getattr(model, "num_dummy", 0)),
        "val_acc_at_ckpt": ckpt.get("meta", {}).get("val_acc"),
    }
    np.savez_compressed(cache_root / "meta.npz", **{k: np.asarray(v) for k, v in meta.items()})
    print(f"[{method}] cache → {cache_root}")
    return cache_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["vanilla", "gcsc", "proser"])
    parser.add_argument("--base-config", type=str, default="task4/configs/base.yaml")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["method"] = args.method
    set_seed(int(cfg["seed"]))
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    extract_for_model(cfg, args.method, device)


if __name__ == "__main__":
    main()
