"""Shared Task 2 training loop.

Design
------
All methods (Source-only, DAN, DANN, CDAN) share this file. Each method only
plugs in its ``total_loss(...)``. That keeps initialization, sampling,
augmentation, BN policy, optimizer, and early stopping identical — which is
required for a fair comparison.

Checkpoint rule (critical)
--------------------------
Best checkpoint = highest *mean macro-F1 across the three source validation
domains*. Sketch labels are never consulted here.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.metrics import macro_f1, top1_accuracy
from common.seed import set_seed
from shared.pacs import (
    SOURCE_DOMAINS,
    TARGET_DOMAIN,
    PacsImageDataset,
    examples_from_uids,
    index_by_uid,
    discover_pacs,
    load_splits,
    make_transforms,
)
from shared.pacs_protocol import (
    cyclic_loader,
    make_loader,
    set_eval_mode,
    set_train_mode_with_frozen_bn,
    unpack_batch,
)
from task2.methods.cdan import CDANMethod
from task2.methods.dan import DANMethod
from task2.methods.dann import DANNMethod
from task2.methods.source_only import SourceOnlyMethod
from task2.models.backbone import ResNet18Classifier


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_merged_config(base_path: Path, method_path: Path | None) -> dict:
    with open(base_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if method_path is not None:
        with open(method_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, override)
        # Promote method-specific block if present under methods.<name>
        method_name = cfg.get("method")
        if method_name and method_name in cfg.get("methods", {}):
            # Allow per-file keys like ``dan: {lambda_mmd: ...}`` to win.
            if method_name in override:
                cfg["methods"][method_name] = deep_update(
                    cfg["methods"][method_name], override[method_name]
                )
    return cfg


def build_method(cfg: dict):
    name = cfg["method"]
    mcfg = cfg["methods"][name]
    if name == "source_only":
        return SourceOnlyMethod()
    if name == "dan":
        return DANMethod(
            lambda_mmd=mcfg.get("lambda_mmd", 1.0),
            kernel_multipliers=mcfg.get("kernel_multipliers", [0.5, 1.0, 2.0]),
        )
    if name == "dann":
        return DANNMethod(
            feature_dim=cfg["model"]["feature_dim"],
            lambda_domain=mcfg.get("lambda_domain", 1.0),
            grl_gamma=mcfg.get("grl_gamma", 10.0),
            grl_max=mcfg.get("grl_max", 1.0),
            disc_hidden=mcfg.get("disc_hidden", 256),
            disc_dropout=mcfg.get("disc_dropout", 0.5),
        )
    if name == "cdan":
        return CDANMethod(
            feature_dim=cfg["model"]["feature_dim"],
            num_classes=cfg["model"]["num_classes"],
            lambda_domain=mcfg.get("lambda_domain", 1.0),
            grl_gamma=mcfg.get("grl_gamma", 10.0),
            grl_max=mcfg.get("grl_max", 1.0),
            disc_hidden=mcfg.get("disc_hidden", 256),
            disc_dropout=mcfg.get("disc_dropout", 0.5),
        )
    raise ValueError(f"Unknown method: {name}")


def build_dataloaders(cfg: dict):
    root = Path(cfg["data"]["root"])
    splits = load_splits(cfg["data"]["splits"])
    examples = discover_pacs(root)
    uid_index = index_by_uid(examples)

    train_tf = make_transforms(
        train=True,
        image_size=cfg["data"]["image_size"],
        resize_size=cfg["data"]["resize_size"],
    )
    eval_tf = make_transforms(
        train=False,
        image_size=cfg["data"]["image_size"],
        resize_size=cfg["data"]["resize_size"],
    )

    source_train_loaders = {}
    source_val_loaders = {}
    train_sizes = []
    for domain in SOURCE_DOMAINS:
        train_ex = examples_from_uids(splits["domains"][domain]["train"], uid_index)
        val_ex = examples_from_uids(splits["domains"][domain]["val"], uid_index)
        train_sizes.append(len(train_ex))
        source_train_loaders[domain] = make_loader(
            PacsImageDataset(train_ex, transform=train_tf),
            batch_size=cfg["training"]["source_per_domain"],
            shuffle=True,
            num_workers=cfg["training"]["num_workers"],
            drop_last=True,
        )
        source_val_loaders[domain] = make_loader(
            PacsImageDataset(val_ex, transform=eval_tf),
            batch_size=64,
            shuffle=False,
            num_workers=cfg["training"]["num_workers"],
            drop_last=False,
        )

    target_ex = examples_from_uids(splits["target"]["all"], uid_index)
    # Task 2: target images used without labels for adaptation methods.
    # Labels remain in the dataset object for final eval only.
    target_train_loader = make_loader(
        PacsImageDataset(target_ex, transform=train_tf),
        batch_size=cfg["training"]["target_batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        drop_last=True,
    )
    target_eval_loader = make_loader(
        PacsImageDataset(target_ex, transform=eval_tf),
        batch_size=64,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        drop_last=False,
    )

    steps_per_epoch = max(1, int(round(float(np.mean(train_sizes)) / cfg["training"]["source_per_domain"])))
    return {
        "source_train": source_train_loaders,
        "source_val": source_val_loaders,
        "target_train": target_train_loader,
        "target_eval": target_eval_loader,
        "steps_per_epoch": steps_per_epoch,
    }


@torch.no_grad()
def evaluate_loader(model: torch.nn.Module, loader, device: torch.device) -> dict:
    set_eval_mode(model)
    all_y, all_pred = [], []
    for batch in loader:
        images, labels, _ = unpack_batch(batch, device)
        _, logits = model(images)
        pred = logits.argmax(dim=1)
        all_y.append(labels.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
    y = np.concatenate(all_y)
    pred = np.concatenate(all_pred)
    return {
        "accuracy": top1_accuracy(y, pred),
        "macro_f1": macro_f1(y, pred),
        "n": int(len(y)),
    }


def evaluate_sources(model, source_val_loaders, device) -> dict:
    per_domain = {}
    f1s, accs = [], []
    for domain, loader in source_val_loaders.items():
        metrics = evaluate_loader(model, loader, device)
        per_domain[domain] = metrics
        f1s.append(metrics["macro_f1"])
        accs.append(metrics["accuracy"])
    return {
        "per_domain": per_domain,
        "mean_macro_f1": float(np.mean(f1s)),
        "mean_accuracy": float(np.mean(accs)),
    }


def run_training(cfg: dict, device: torch.device) -> dict:
    set_seed(cfg["seed"])
    loaders = build_dataloaders(cfg)
    model = ResNet18Classifier(
        num_classes=cfg["model"]["num_classes"],
        feature_dim=cfg["model"]["feature_dim"],
    ).to(device)
    method = build_method(cfg).to(device)

    params = list(model.parameters()) + [p for p in method.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    source_cyclic = {d: cyclic_loader(loaders["source_train"][d]) for d in SOURCE_DOMAINS}
    target_cyclic = cyclic_loader(loaders["target_train"])
    use_target = cfg["method"] != "source_only"

    max_epochs = cfg["training"]["max_epochs"]
    steps = loaders["steps_per_epoch"]
    patience = cfg["training"]["early_stop_patience"]

    best_score = -1.0
    best_state = None
    best_epoch = -1
    epochs_without_improve = 0
    history = []

    method_name = cfg["method"]
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    curves_dir = Path(cfg["paths"]["curves_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    # Tag controlled-study DAN runs by lambda for unique filenames.
    run_tag = method_name
    if method_name == "dan":
        lam = cfg["methods"]["dan"].get("lambda_mmd", 1.0)
        run_tag = f"dan_lambda{lam}"

    print(f"Training {run_tag} on {device} | steps/epoch={steps}")

    for epoch in range(1, max_epochs + 1):
        set_train_mode_with_frozen_bn(model)
        method.train()
        # Discriminator (if any) should also train; BN freeze only on backbone.
        if hasattr(method, "discriminator"):
            method.discriminator.train()

        running = {"loss": 0.0, "cls_loss": 0.0, "align_loss": 0.0}
        t0 = time.time()

        for step in tqdm(range(steps), desc=f"{run_tag} epoch {epoch}", leave=False):
            progress = ((epoch - 1) * steps + step) / max(max_epochs * steps, 1)

            # Domain-balanced source batch: concat 8 from each source domain.
            source_images, source_labels = [], []
            for domain in SOURCE_DOMAINS:
                batch = next(source_cyclic[domain])
                imgs, labels, _ = unpack_batch(batch, device)
                source_images.append(imgs)
                source_labels.append(labels)
            source_images = torch.cat(source_images, dim=0)
            source_labels = torch.cat(source_labels, dim=0)

            source_feats, source_logits = model(source_images)

            target_feats = target_logits = None
            if use_target:
                tbatch = next(target_cyclic)
                timgs, _, _ = unpack_batch(tbatch, device)
                # Labels deliberately unused in the loss.
                target_feats, target_logits = model(timgs)

            loss_dict = method.total_loss(
                source_logits=source_logits,
                source_labels=source_labels,
                source_features=source_feats,
                target_features=target_feats if use_target else source_feats,
                target_logits=target_logits if use_target else source_logits,
                progress=progress,
                device=device,
            )
            loss = loss_dict["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip = cfg["training"].get("grad_clip_norm", None)
            if clip is not None and float(clip) > 0:
                torch.nn.utils.clip_grad_norm_(params, max_norm=float(clip))
            optimizer.step()

            running["loss"] += float(loss.item())
            running["cls_loss"] += float(loss_dict["cls_loss"].item())
            running["align_loss"] += float(loss_dict["align_loss"].item())

        for k in running:
            running[k] /= max(steps, 1)

        # Checkpoint selection: source validation only.
        source_metrics = evaluate_sources(model, loaders["source_val"], device)
        mean_f1 = source_metrics["mean_macro_f1"]

        record = {
            "epoch": epoch,
            "train": running,
            "source_val": source_metrics,
            "seconds": round(time.time() - t0, 2),
        }
        history.append(record)
        print(
            f"[{run_tag}] epoch {epoch:02d}  "
            f"loss={running['loss']:.4f} cls={running['cls_loss']:.4f} "
            f"align={running['align_loss']:.4f}  "
            f"mean_src_f1={mean_f1:.4f}"
        )

        if mean_f1 > best_score + 1e-6:
            best_score = mean_f1
            best_epoch = epoch
            epochs_without_improve = 0
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "method": {k: v.detach().cpu().clone() for k, v in method.state_dict().items()},
                "epoch": epoch,
                "mean_source_val_macro_f1": mean_f1,
                "source_val": source_metrics,
                "config_method": method_name,
                "run_tag": run_tag,
                "seed": cfg["seed"],
            }
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"Early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")

    # Restore best weights and save.
    model.load_state_dict(best_state["model"])
    method.load_state_dict(best_state["method"])

    ckpt_path = ckpt_dir / f"{run_tag}_best.pt"
    # Canonical Source-only path required by Task 3.
    if method_name == "source_only":
        ckpt_path = Path(cfg["paths"]["source_only_checkpoint"])
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_state["model"],
            "method_state_dict": best_state["method"],
            "epoch": best_state["epoch"],
            "mean_source_val_macro_f1": best_state["mean_source_val_macro_f1"],
            "source_val": best_state["source_val"],
            "method": method_name,
            "run_tag": run_tag,
            "seed": cfg["seed"],
            "cfg_snapshot": {
                "method": method_name,
                "methods": cfg["methods"].get(method_name, {}),
                "training": cfg["training"],
                "seed": cfg["seed"],
            },
        },
        ckpt_path,
    )

    curves_path = curves_dir / f"{run_tag}_history.json"
    curves_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    summary = {
        "run_tag": run_tag,
        "method": method_name,
        "checkpoint": str(ckpt_path),
        "best_epoch": best_epoch,
        "best_mean_source_val_macro_f1": best_score,
        "best_source_val": best_state["source_val"],
        "history_path": str(curves_path),
    }
    summary_path = Path(cfg["paths"]["tables_dir"]) / f"{run_tag}_train_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved checkpoint -> {ckpt_path}")
    print(f"Saved curves     -> {curves_path}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Train one Task 2 UDA method.")
    p.add_argument("--base-config", type=Path, default=Path("task2/configs/base.yaml"))
    p.add_argument("--method-config", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--lambda-mmd",
        type=float,
        default=None,
        help="Optional override for DAN controlled study.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_merged_config(args.base_config, args.method_config)
    if args.lambda_mmd is not None:
        cfg.setdefault("methods", {}).setdefault("dan", {})["lambda_mmd"] = args.lambda_mmd
        cfg["method"] = "dan"
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_training(cfg, device)


if __name__ == "__main__":
    main()
