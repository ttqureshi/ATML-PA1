"""Final Task 3 evaluation — Sketch labels allowed ONLY here.

Run after all Task 3 decisions (architectures, λ_DG, ρ, checkpoints) are fixed.
Adds Sketch Acc/F1, ΔSketch vs ERM, per-class analysis, and merges source-side
diagnostics from ``evaluate_source`` when available (or recomputes them).
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
    TARGET_DOMAIN,
    PacsImageDataset,
    discover_pacs,
    examples_from_uids,
    index_by_uid,
    load_splits,
    make_transforms,
)
from shared.pacs_protocol import make_loader, unpack_batch
from task2.evaluation.class_analysis import (
    accuracy_deltas,
    confusion_matrix,
    per_class_accuracy,
    top_confusions,
)
from task3.evaluate_source import (
    default_checkpoints,
    evaluate_checkpoint_source,
    load_model_from_ckpt,
)
from task3.evaluation.metrics import summarize_per_domain
from task3.train import evaluate_loader, load_merged_config


def build_sketch_loader(cfg: dict):
    root = Path(cfg["data"]["root"])
    splits = load_splits(cfg["data"]["splits"])
    uid_index = index_by_uid(discover_pacs(root))
    eval_tf = make_transforms(
        train=False,
        image_size=cfg["data"]["image_size"],
        resize_size=cfg["data"]["resize_size"],
    )
    target_ex = examples_from_uids(splits["target"]["all"], uid_index)
    # Sanity: only Sketch UIDs.
    for ex in target_ex:
        if ex.domain != TARGET_DOMAIN:
            raise RuntimeError(f"Non-Sketch example in target split: {ex.uid}")
    loader = make_loader(
        PacsImageDataset(target_ex, transform=eval_tf),
        batch_size=64,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )
    return loader


def collect_predictions(model, loader, device):
    from shared.pacs_protocol import set_eval_mode

    set_eval_mode(model)
    ys, preds = [], []
    with torch.no_grad():
        for batch in loader:
            images, y, _ = unpack_batch(batch, device)
            _, logits = model(images)
            ys.append(y.cpu().numpy())
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=Path("task3/configs/base.yaml"))
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="",
        help="Comma-separated checkpoint paths. Empty = main comparison.",
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

    sketch_loader = build_sketch_loader(cfg)
    results = {}

    for _label, ckpt in ckpts:
        if not ckpt.exists():
            print(f"SKIP missing checkpoint: {ckpt}")
            continue
        print(f"[final-eval] {ckpt} ...")
        # Source diagnostics (no Sketch inside that helper).
        source_detail = evaluate_checkpoint_source(cfg, ckpt, device)
        model, blob = load_model_from_ckpt(ckpt, cfg, device)
        sketch_metrics = evaluate_loader(model, sketch_loader, device)
        y_true, y_pred = collect_predictions(model, sketch_loader, device)
        per_class = per_class_accuracy(
            y_true, y_pred, num_classes=cfg["model"]["num_classes"]
        )

        key = source_detail["run_tag"]
        results[key] = {
            **source_detail,
            "sketch": sketch_metrics,
            "sketch_per_class": per_class,
            "sketch_confusions_top": top_confusions(y_true, y_pred, k=12),
            "sketch_confusion_matrix": confusion_matrix(
                y_true, y_pred, num_classes=cfg["model"]["num_classes"]
            ),
            "note": (
                "Sketch labels used only in this final evaluation stage. "
                "Do not revise training settings based on these numbers."
            ),
        }

    # ΔSketch vs ERM.
    erm_key = None
    for k, v in results.items():
        if v.get("method") in ("erm", "source_only") or "erm" in k or "source_only" in k:
            erm_key = k
            break
    if erm_key is not None:
        erm_acc = results[erm_key]["sketch"]["accuracy"]
        erm_per = results[erm_key]["sketch_per_class"]
        for k, v in results.items():
            v["sketch_accuracy_change_vs_erm"] = float(v["sketch"]["accuracy"] - erm_acc)
            v["sketch_per_class_delta_vs_erm"] = accuracy_deltas(erm_per, v["sketch_per_class"])

    table_rows = []
    for k, v in results.items():
        row = {
            "run": k,
            "method": v["method"],
            "source_val_per_domain": summarize_per_domain(v["source_val"]),
            "mean_source_accuracy": v["source_val"]["mean_accuracy"],
            "mean_source_macro_f1": v["source_val"]["mean_macro_f1"],
            "worst_source_accuracy": v["source_val"]["worst_accuracy"],
            "worst_source_macro_f1": v["source_val"]["worst_macro_f1"],
            "sketch_accuracy": v["sketch"]["accuracy"],
            "sketch_macro_f1": v["sketch"]["macro_f1"],
            "sketch_accuracy_change_vs_erm": v.get("sketch_accuracy_change_vs_erm"),
            "source_domain_separability": v["source_domain_separability"]["held_out_accuracy"],
            "delta_sharp": v["sharpness"]["delta_sharp"],
        }
        table_rows.append(row)
        detail_path = Path(cfg["paths"]["tables_dir"]) / f"eval_{k}.json"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_path.write_text(json.dumps(v, indent=2), encoding="utf-8")

    out = {
        "note": (
            "Sketch labels used only here. Main comparison uses λ_DG=1 and ρ=0.05. "
            "Do not pick a post-hoc winner from Sketch."
        ),
        "table": table_rows,
        "runs": {k: f"eval_{k}.json" for k in results},
    }
    out_path = Path(cfg["paths"]["tables_dir"]) / "task3_main_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
