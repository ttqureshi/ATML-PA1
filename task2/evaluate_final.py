"""Final Task 2 evaluation — run ONLY after all checkpoints are fixed.

This script is allowed to use Sketch labels. It must not be used to revise
training settings, λ values, or method design.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.seed import set_seed
from shared.pacs import (
    SOURCE_DOMAINS,
    PacsImageDataset,
    discover_pacs,
    examples_from_uids,
    index_by_uid,
    load_splits,
    make_transforms,
)
from shared.pacs_protocol import make_loader, set_eval_mode, unpack_batch
from task2.evaluation.class_analysis import (
    accuracy_deltas,
    confusion_matrix,
    per_class_accuracy,
    top_confusions,
)
from task2.evaluation.domain_separability import domain_separability_score
from task2.models.backbone import ResNet18Classifier
from task2.train import evaluate_loader, evaluate_sources, load_merged_config


def collect_features(model, loader, device):
    set_eval_mode(model)
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            images, y, _ = unpack_batch(batch, device)
            f, _ = model(images)
            feats.append(f.cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(feats), np.concatenate(labels)


def collect_predictions(model, loader, device):
    set_eval_mode(model)
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            images, y, _ = unpack_batch(batch, device)
            _, logits = model(images)
            ys.append(y.cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds)


def load_model_from_ckpt(ckpt_path: Path, cfg: dict, device: torch.device):
    model = ResNet18Classifier(
        num_classes=cfg["model"]["num_classes"],
        feature_dim=cfg["model"]["feature_dim"],
    ).to(device)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    return model, blob


def build_eval_loaders(cfg: dict):
    root = Path(cfg["data"]["root"])
    splits = load_splits(cfg["data"]["splits"])
    uid_index = index_by_uid(discover_pacs(root))
    eval_tf = make_transforms(
        train=False,
        image_size=cfg["data"]["image_size"],
        resize_size=cfg["data"]["resize_size"],
    )
    source_val = {}
    source_val_examples = []
    for domain in SOURCE_DOMAINS:
        ex = examples_from_uids(splits["domains"][domain]["val"], uid_index)
        source_val_examples.extend(ex)
        source_val[domain] = make_loader(
            PacsImageDataset(ex, transform=eval_tf),
            batch_size=64,
            shuffle=False,
            num_workers=cfg["training"]["num_workers"],
        )
    # Pooled source-val for domain-separability (equal count vs target later).
    source_val_pooled = make_loader(
        PacsImageDataset(source_val_examples, transform=eval_tf),
        batch_size=64,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )
    target_ex = examples_from_uids(splits["target"]["all"], uid_index)
    target_loader = make_loader(
        PacsImageDataset(target_ex, transform=eval_tf),
        batch_size=64,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )
    return source_val, source_val_pooled, target_loader


def evaluate_checkpoint(cfg: dict, ckpt_path: Path, device: torch.device) -> dict:
    model, blob = load_model_from_ckpt(ckpt_path, cfg, device)
    source_val, source_val_pooled, target_loader = build_eval_loaders(cfg)

    source_metrics = evaluate_sources(model, source_val, device)
    target_metrics = evaluate_loader(model, target_loader, device)

    src_feats, _ = collect_features(model, source_val_pooled, device)
    tgt_feats, tgt_labels = collect_features(model, target_loader, device)
    y_true, y_pred = collect_predictions(model, target_loader, device)

    sep_cfg = cfg["evaluation"]["domain_separability"]
    separability = domain_separability_score(
        src_feats,
        tgt_feats,
        test_size=sep_cfg["test_size"],
        C=sep_cfg["C"],
        seed=sep_cfg["seed"],
    )

    per_class = per_class_accuracy(y_true, y_pred, num_classes=cfg["model"]["num_classes"])
    return {
        "checkpoint": str(ckpt_path),
        "train_blob_meta": {
            "epoch": blob.get("epoch"),
            "mean_source_val_macro_f1": blob.get("mean_source_val_macro_f1"),
            "method": blob.get("method"),
            "run_tag": blob.get("run_tag"),
        },
        "source_val": source_metrics,
        "target": target_metrics,
        "domain_separability": separability,
        "target_per_class": per_class,
        "target_confusions_top": top_confusions(y_true, y_pred, k=12),
        "target_confusion_matrix": confusion_matrix(
            y_true, y_pred, num_classes=cfg["model"]["num_classes"]
        ),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=Path("task2/configs/base.yaml"))
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="",
        help="Comma-separated checkpoint paths. Empty = discover main comparison ckpts.",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_merged_config(args.base_config, None)
    set_seed(cfg["seed"])
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    if args.checkpoints.strip():
        ckpts = [Path(p.strip()) for p in args.checkpoints.split(",") if p.strip()]
    else:
        # Default main comparison filenames.
        ckpts = [
            Path(cfg["paths"]["source_only_checkpoint"]),
            ckpt_dir / "dan_lambda1.0_best.pt",
            ckpt_dir / "dann_best.pt",
            ckpt_dir / "cdan_best.pt",
        ]

    results = {}
    for ckpt in ckpts:
        if not ckpt.exists():
            print(f"SKIP missing checkpoint: {ckpt}")
            continue
        print(f"Evaluating {ckpt} ...")
        results[ckpt.stem] = evaluate_checkpoint(cfg, ckpt, device)

    # Attach Δacc / per-class Δ vs Source-only when available.
    so_key = None
    for k, v in results.items():
        if v["train_blob_meta"].get("method") == "source_only":
            so_key = k
            break
    if so_key is not None:
        so_acc = results[so_key]["target"]["accuracy"]
        so_per = results[so_key]["target_per_class"]
        for k, v in results.items():
            v["target_accuracy_change_vs_source_only"] = float(
                v["target"]["accuracy"] - so_acc
            )
            v["target_per_class_delta_vs_source_only"] = accuracy_deltas(
                so_per, v["target_per_class"]
            )

    # Compact table-friendly rows (drop large y_true/y_pred from summary table).
    table_rows = []
    for k, v in results.items():
        row = {
            "run": k,
            "method": v["train_blob_meta"].get("method"),
            "source_val_per_domain": {
                d: {
                    "accuracy": v["source_val"]["per_domain"][d]["accuracy"],
                    "macro_f1": v["source_val"]["per_domain"][d]["macro_f1"],
                }
                for d in v["source_val"]["per_domain"]
            },
            "mean_source_accuracy": v["source_val"]["mean_accuracy"],
            "mean_source_macro_f1": v["source_val"]["mean_macro_f1"],
            "target_accuracy": v["target"]["accuracy"],
            "target_macro_f1": v["target"]["macro_f1"],
            "target_accuracy_change_vs_source_only": v.get(
                "target_accuracy_change_vs_source_only"
            ),
            "domain_separability": v["domain_separability"]["held_out_accuracy"],
        }
        table_rows.append(row)
        # Keep predictions only in per-run detail files, not the summary.
        detail = {kk: vv for kk, vv in v.items() if kk not in ("y_true", "y_pred")}
        detail_path = Path(cfg["paths"]["tables_dir"]) / f"eval_{k}.json"
        detail_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")

    out = {
        "note": (
            "Sketch labels used only in this final evaluation stage. "
            "Do not revise training settings based on these numbers."
        ),
        "table": table_rows,
        "runs": {k: f"eval_{k}.json" for k in results},
    }
    out_path = Path(cfg["paths"]["tables_dir"]) / "task2_main_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
