"""Task 3 *source-side* evaluation — NO Sketch images loaded.

Run after DAN-DG / SAM checkpoints exist (ERM loads Task 2 Source-only).
Reports per-source Acc/F1, mean, worst-domain, 3-way source-domain
separability, and the common Δsharp proxy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

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
from task2.models.backbone import ResNet18Classifier
from task3.evaluation.metrics import summarize_per_domain
from task3.evaluation.sharpness import build_fixed_sharpness_batch, sharpness_proxy
from task3.evaluation.source_domain_separability import source_domain_separability_score
from task3.train import evaluate_sources, load_merged_config


def load_model_from_ckpt(ckpt_path: Path, cfg: dict, device: torch.device):
    model = ResNet18Classifier(
        num_classes=cfg["model"]["num_classes"],
        feature_dim=cfg["model"]["feature_dim"],
    ).to(device)
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model_state_dict"])
    return model, blob


def collect_features_by_domain(model, loaders_by_domain, device):
    set_eval_mode(model)
    out = {}
    with torch.no_grad():
        for domain, loader in loaders_by_domain.items():
            feats = []
            for batch in loader:
                images, _, _ = unpack_batch(batch, device)
                f, _ = model(images)
                feats.append(f.cpu().numpy())
            out[domain] = np.concatenate(feats, axis=0)
    return out


def build_source_val(cfg: dict):
    root = Path(cfg["data"]["root"])
    splits = load_splits(cfg["data"]["splits"])
    uid_index = index_by_uid(discover_pacs(root))
    eval_tf = make_transforms(
        train=False,
        image_size=cfg["data"]["image_size"],
        resize_size=cfg["data"]["resize_size"],
    )
    loaders = {}
    examples_by_domain = {}
    for domain in SOURCE_DOMAINS:
        ex = examples_from_uids(splits["domains"][domain]["val"], uid_index)
        examples_by_domain[domain] = ex
        loaders[domain] = make_loader(
            PacsImageDataset(ex, transform=eval_tf),
            batch_size=64,
            shuffle=False,
            num_workers=cfg["training"]["num_workers"],
        )
    return loaders, examples_by_domain, eval_tf


def evaluate_checkpoint_source(cfg: dict, ckpt_path: Path, device: torch.device) -> dict:
    model, blob = load_model_from_ckpt(ckpt_path, cfg, device)
    source_val, examples_by_domain, eval_tf = build_source_val(cfg)

    source_metrics = evaluate_sources(model, source_val, device)

    feats_by_domain = collect_features_by_domain(model, source_val, device)
    sep_cfg = cfg["evaluation"]["source_domain_separability"]
    separability = source_domain_separability_score(
        feats_by_domain,
        test_size=sep_cfg["test_size"],
        C=sep_cfg["C"],
        seed=sep_cfg["seed"],
    )

    sharp_cfg = cfg["evaluation"]["sharpness"]
    sharp_examples = build_fixed_sharpness_batch(
        examples_by_domain,
        per_domain=sharp_cfg["per_domain"],
        seed=sharp_cfg["seed"],
    )
    sharp_ds = PacsImageDataset(sharp_examples, transform=eval_tf)
    # One batch: 32 × 3 = 96 images.
    sharp_loader = make_loader(
        sharp_ds,
        batch_size=len(sharp_examples),
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(sharp_loader))
    images, labels, _ = unpack_batch(batch, device)
    sharpness = sharpness_proxy(
        model, images, labels, rho=float(sharp_cfg["rho"])
    )

    method = blob.get("method")
    if method is None and "source_only" in str(ckpt_path):
        method = "erm"
    run_tag = blob.get("run_tag") or ckpt_path.stem

    return {
        "checkpoint": str(ckpt_path),
        "method": method if method != "source_only" else "erm",
        "run_tag": run_tag if run_tag != "source_only_best" else "erm",
        "train_blob_meta": {
            "epoch": blob.get("epoch"),
            "mean_source_val_macro_f1": blob.get("mean_source_val_macro_f1"),
            "method": blob.get("method"),
            "run_tag": blob.get("run_tag"),
        },
        "source_val": source_metrics,
        "source_domain_separability": separability,
        "sharpness": sharpness,
        "note": "No Sketch loaded in this script.",
    }


def default_checkpoints(cfg: dict) -> list[tuple[str, Path]]:
    """Named (label, path) pairs for main comparison."""
    ckpt_dir = Path(cfg["paths"]["checkpoints_dir"])
    return [
        ("erm", Path(cfg["paths"]["erm_checkpoint"])),
        ("dan_dg_lambda1.0", ckpt_dir / "dan_dg_lambda1.0_best.pt"),
        ("sam_rho0.05", ckpt_dir / "sam_rho0.05_best.pt"),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=Path("task3/configs/base.yaml"))
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="",
        help="Comma-separated paths. Empty = main ERM + DAN-DG λ=1 + SAM ρ=0.05.",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = load_merged_config(args.base_config, None)
    set_seed(cfg["seed"])
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.checkpoints.strip():
        ckpts = [("custom", Path(p.strip())) for p in args.checkpoints.split(",") if p.strip()]
    else:
        ckpts = default_checkpoints(cfg)

    results = {}
    table_rows = []
    for label, ckpt in ckpts:
        if not ckpt.exists():
            print(f"SKIP missing checkpoint: {ckpt}")
            continue
        print(f"[source-eval] {ckpt} ...")
        detail = evaluate_checkpoint_source(cfg, ckpt, device)
        key = detail["run_tag"]
        results[key] = detail

        row = {
            "run": key,
            "method": detail["method"],
            "source_val_per_domain": summarize_per_domain(detail["source_val"]),
            "mean_source_accuracy": detail["source_val"]["mean_accuracy"],
            "mean_source_macro_f1": detail["source_val"]["mean_macro_f1"],
            "worst_source_accuracy": detail["source_val"]["worst_accuracy"],
            "worst_source_macro_f1": detail["source_val"]["worst_macro_f1"],
            "worst_macro_f1_domain": detail["source_val"]["worst_macro_f1_domain"],
            "source_domain_separability": detail["source_domain_separability"][
                "held_out_accuracy"
            ],
            "delta_sharp": detail["sharpness"]["delta_sharp"],
        }
        table_rows.append(row)

        detail_path = Path(cfg["paths"]["tables_dir"]) / f"source_eval_{key}.json"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_path.write_text(json.dumps(detail, indent=2), encoding="utf-8")

    out = {
        "note": (
            "Source-side diagnostics only. Sketch not loaded. "
            "Do not use these numbers to revise Task 3 using Task 2 Sketch scores."
        ),
        "table": table_rows,
        "runs": {k: f"source_eval_{k}.json" for k in results},
    }
    out_path = Path(cfg["paths"]["tables_dir"]) / "task3_source_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
