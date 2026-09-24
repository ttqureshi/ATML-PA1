"""Shared Task 3 training loop (DAN-DG and SAM only).

Design
------
- ERM is *not* trained here: load Task 2 ``source_only_best.pt``.
- No Sketch / target loaders anywhere in this file (hard no-leakage rule).
- Checkpoint = highest mean macro-F1 across the three source validation domains.
- SAM uses two forward/backward passes with frozen-BN on both.
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
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.metrics import macro_f1, top1_accuracy
from common.seed import set_seed
from shared.pacs import (
    SOURCE_DOMAINS,
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
from task2.models.backbone import ResNet18Classifier
from task3.evaluation.metrics import with_worst_domain
from task3.methods.dan_dg import DANDGMethod
from task3.methods.sam import SAMMethod, sam_weight_perturbation


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
        method_name = cfg.get("method")
        if method_name and method_name in cfg.get("methods", {}):
            if method_name in override:
                cfg["methods"][method_name] = deep_update(
                    cfg["methods"][method_name], override[method_name]
                )
    return cfg


def build_method(cfg: dict):
    name = cfg["method"]
    mcfg = cfg["methods"][name]
    if name == "dan_dg":
        return DANDGMethod(
            lambda_dg=mcfg.get("lambda_dg", 1.0),
            kernel_multipliers=mcfg.get("kernel_multipliers", [0.5, 1.0, 2.0]),
            source_per_domain=cfg["training"]["source_per_domain"],
        )
    if name == "sam":
        return SAMMethod(rho=mcfg.get("rho", 0.05))
    if name == "erm":
        raise ValueError(
            "ERM is load-only from Task 2. Use task3.evaluate_source / "
            "evaluate_final with paths.erm_checkpoint — do not train here."
        )
    raise ValueError(f"Unknown Task 3 method: {name}")


def build_dataloaders(cfg: dict):
    """Source-only loaders. Sketch is never opened here."""
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

    steps_per_epoch = max(
        1,
        int(round(float(np.mean(train_sizes)) / cfg["training"]["source_per_domain"])),
    )
    return {
        "source_train": source_train_loaders,
        "source_val": source_val_loaders,
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
    return with_worst_domain(
        {
            "per_domain": per_domain,
            "mean_macro_f1": float(np.mean(f1s)),
            "mean_accuracy": float(np.mean(accs)),
        }
    )


def _next_source_batch(source_cyclic, device):
    source_images, source_labels = [], []
    for domain in SOURCE_DOMAINS:
        batch = next(source_cyclic[domain])
        imgs, labels, _ = unpack_batch(batch, device)
        source_images.append(imgs)
        source_labels.append(labels)
    return torch.cat(source_images, dim=0), torch.cat(source_labels, dim=0)


def _sam_step(model, method, optimizer, images, labels, cfg, params):
    """Two-pass SAM update; frozen BN already set by caller."""
    rho = float(cfg["methods"]["sam"].get("rho", 0.05))

    # Pass 1: ascent direction at θ.
    optimizer.zero_grad(set_to_none=True)
    _, logits = model(images)
    loss_dict = method.total_loss(source_logits=logits, source_labels=labels)
    loss_dict["loss"].backward()
    clip = cfg["training"].get("grad_clip_norm", None)
    if clip is not None and float(clip) > 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(clip))

    with sam_weight_perturbation(params, rho):
        # Pass 2: gradients at θ+ε (BN still frozen; stay in train+frozen-BN).
        set_train_mode_with_frozen_bn(model)
        optimizer.zero_grad(set_to_none=True)
        _, logits_pert = model(images)
        loss_dict2 = method.total_loss(source_logits=logits_pert, source_labels=labels)
        loss_dict2["loss"].backward()
        if clip is not None and float(clip) > 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=float(clip))

    optimizer.step()
    # Logging uses the perturbed-point CE (the loss that drove the update).
    return loss_dict2


def _erm_style_step(model, method, optimizer, images, labels, cfg, params):
    """Single-pass step used by DAN-DG (and would be ERM if we trained it)."""
    optimizer.zero_grad(set_to_none=True)
    feats, logits = model(images)
    loss_dict = method.total_loss(
        source_logits=logits,
        source_labels=labels,
        source_features=feats,
    )
    loss_dict["loss"].backward()
    clip = cfg["training"].get("grad_clip_norm", None)
    if clip is not None and float(clip) > 0:
        torch.nn.utils.clip_grad_norm_(params, max_norm=float(clip))
    optimizer.step()
    return loss_dict


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
    method_name = cfg["method"]
    use_sam = method_name == "sam"

    max_epochs = cfg["training"]["max_epochs"]
    steps = loaders["steps_per_epoch"]
    patience = cfg["training"]["early_stop_patience"]

    best_score = -1.0
    best_state = None
    best_epoch = -1
    epochs_without_improve = 0
    history = []

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    curves_dir = Path(cfg["paths"]["curves_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    run_tag = method_name
    if method_name == "dan_dg":
        lam = cfg["methods"]["dan_dg"].get("lambda_dg", 1.0)
        run_tag = f"dan_dg_lambda{lam}"
    elif method_name == "sam":
        rho = cfg["methods"]["sam"].get("rho", 0.05)
        run_tag = f"sam_rho{rho}"

    print(f"Training {run_tag} on {device} | steps/epoch={steps} | NO Sketch")

    for epoch in range(1, max_epochs + 1):
        set_train_mode_with_frozen_bn(model)
        method.train()

        running = {"loss": 0.0, "cls_loss": 0.0, "align_loss": 0.0}
        t0 = time.time()

        for _step in tqdm(range(steps), desc=f"{run_tag} epoch {epoch}", leave=False):
            images, labels = _next_source_batch(source_cyclic, device)
            set_train_mode_with_frozen_bn(model)

            if use_sam:
                loss_dict = _sam_step(model, method, optimizer, images, labels, cfg, params)
            else:
                loss_dict = _erm_style_step(
                    model, method, optimizer, images, labels, cfg, params
                )

            running["loss"] += float(loss_dict["loss"].item())
            running["cls_loss"] += float(loss_dict["cls_loss"].item())
            running["align_loss"] += float(loss_dict["align_loss"].item())

        for k in running:
            running[k] /= max(steps, 1)

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
            f"mean_src_f1={mean_f1:.4f} worst_f1={source_metrics['worst_macro_f1']:.4f}"
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

    model.load_state_dict(best_state["model"])
    method.load_state_dict(best_state["method"])

    ckpt_path = ckpt_dir / f"{run_tag}_best.pt"
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
    p = argparse.ArgumentParser(description="Train one Task 3 DG method (no Sketch).")
    p.add_argument("--base-config", type=Path, default=Path("task3/configs/base.yaml"))
    p.add_argument("--method-config", type=Path, required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--lambda-dg",
        type=float,
        default=None,
        help="Optional override for DAN-DG controlled study.",
    )
    p.add_argument(
        "--rho",
        type=float,
        default=None,
        help="Optional override for SAM ρ (not used in main controlled study).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_merged_config(args.base_config, args.method_config)
    if args.lambda_dg is not None:
        cfg.setdefault("methods", {}).setdefault("dan_dg", {})["lambda_dg"] = args.lambda_dg
        cfg["method"] = "dan_dg"
    if args.rho is not None:
        cfg.setdefault("methods", {}).setdefault("sam", {})["rho"] = args.rho
        cfg["method"] = "sam"
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    run_training(cfg, device)


if __name__ == "__main__":
    main()
