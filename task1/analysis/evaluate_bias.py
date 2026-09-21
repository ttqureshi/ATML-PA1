"""Evaluate Task 1 models on clean images and controlled interventions.

Produces machine-readable tables under task1/results/tables/:
  - clean_baseline.json
  - color_results.json
  - translation_results.json
  - patch_shuffle_results.json
  - cue_conflict_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import datasets

from common.metrics import (
    macro_f1,
    mean_max_confidence,
    prediction_consistency,
    top1_accuracy,
)
from common.seed import set_seed
from task1.data.make_subset import STL10_CLASSES
from task1.data.transforms import (
    grayscale,
    hue_rotate,
    patch_shuffle,
    translate,
)
from task1.models.backbones import (
    LinearHead,
    build_backbone,
    clip_zeroshot_logits,
    softmax_np,
)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_eval_images(cfg: dict) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
    """Load the shared 224x224 clean evaluation subset."""
    meta = json.loads(Path(cfg["paths"]["splits_dir"], "eval_subset.json").read_text())
    indices = meta["indices"]
    root = Path(cfg["dataset"]["root"])
    size = int(cfg["dataset"]["image_size"])
    ds = datasets.STL10(root=str(root), split="test", download=True)
    images, labels = [], []
    for idx in indices:
        img, y = ds[idx]
        rgb = np.asarray(img.convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.uint8)
        images.append(rgb)
        labels.append(int(y))
    return images, np.asarray(labels, dtype=np.int64), indices


def load_heads(cfg: dict, device: torch.device) -> Dict[str, object]:
    """Load backbones + linear heads, and the CLIP zero-shot pathway."""
    bundle = {}
    heads_dir = Path(cfg["paths"]["heads_dir"])
    for name in cfg["models"]:
        if name == "clip_zeroshot":
            continue
        backbone, info = build_backbone(name, device)
        ckpt = torch.load(heads_dir / f"{name}_linear_head.pt", map_location=device)
        head = LinearHead(ckpt["feature_dim"], len(STL10_CLASSES)).to(device)
        head.load_state_dict(ckpt["state_dict"])
        head.eval()
        bundle[name] = {"backbone": backbone, "head": head, "info": info, "kind": "linear"}

    # Zero-shot CLIP reuses the CLIP visual backbone.
    if "clip_zeroshot" in cfg["models"]:
        if "clip_vit_b_32" not in bundle:
            backbone, info = build_backbone("clip_vit_b_32", device)
        else:
            backbone = bundle["clip_vit_b_32"]["backbone"]
            info = bundle["clip_vit_b_32"]["info"]
        bundle["clip_zeroshot"] = {
            "backbone": backbone,
            "head": None,
            "info": info,
            "kind": "zeroshot",
        }
    return bundle


@torch.no_grad()
def predict_model(
    model_bundle: dict,
    images: List[np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return preds, probs, features for a list of RGB uint8 images."""
    backbone = model_bundle["backbone"]
    head = model_bundle["head"]
    preprocess = model_bundle["info"].preprocess
    kind = model_bundle["kind"]

    preds, probs, feats = [], [], []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        x = torch.stack([preprocess(img) for img in batch], dim=0).to(device)
        f = backbone(x)
        if kind == "zeroshot":
            logits = clip_zeroshot_logits(backbone, f, STL10_CLASSES, device)
        else:
            logits = head(f)
        logits_np = logits.cpu().numpy()
        p = softmax_np(logits_np)
        preds.append(p.argmax(axis=1))
        probs.append(p)
        feats.append(f.cpu().numpy())
    return (
        np.concatenate(preds),
        np.concatenate(probs),
        np.concatenate(feats),
    )


def summarize(y_true, y_pred, probs, y_ref_pred=None) -> dict:
    out = {
        "top1_accuracy": top1_accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "mean_max_confidence": mean_max_confidence(probs),
    }
    if y_ref_pred is not None:
        out["prediction_consistency"] = prediction_consistency(y_ref_pred, y_pred)
        out["accuracy_change"] = out["top1_accuracy"] - top1_accuracy(y_true, y_ref_pred)
    return out


def eval_clean(cfg, models, images, labels, device) -> dict:
    results = {}
    clean_preds = {}
    for name, bundle in models.items():
        pred, prob, _ = predict_model(bundle, images, device, cfg["training"]["batch_size"])
        results[name] = summarize(labels, pred, prob)
        clean_preds[name] = pred
        print(f"[clean] {name}: acc={results[name]['top1_accuracy']:.4f}")
    return {"metrics": results, "clean_preds": {k: v.tolist() for k, v in clean_preds.items()}}


def eval_color(cfg, models, images, labels, clean_preds, device) -> dict:
    gray_imgs = [grayscale(im) for im in images]
    hue_imgs = [hue_rotate(im, cfg["color"]["hue_degrees"]) for im in images]
    out = {"grayscale": {}, "hue_rotate": {}}
    for name, bundle in models.items():
        ref = np.asarray(clean_preds[name])
        for tag, imgs in [("grayscale", gray_imgs), ("hue_rotate", hue_imgs)]:
            pred, prob, _ = predict_model(bundle, imgs, device, cfg["training"]["batch_size"])
            out[tag][name] = summarize(labels, pred, prob, ref)
            print(f"[{tag}] {name}: acc={out[tag][name]['top1_accuracy']:.4f}")
    out["assumptions"] = {
        "extra_color_intervention": "fixed hue rotation",
        "hue_degrees": cfg["color"]["hue_degrees"],
        "note": (
            "Grayscale removes chromatic information. "
            "Hue rotation changes color identity while preserving geometry/value."
        ),
    }
    return out


def eval_translation(cfg, models, images, labels, clean_preds, device) -> dict:
    displacements = cfg["translation"]["displacements"]
    directions = cfg["translation"]["directions"]
    out = {}
    for name, bundle in models.items():
        ref = np.asarray(clean_preds[name])
        out[name] = []
        for d in displacements:
            # Average metrics across the four cardinal directions.
            accs, cons, f1s, confs = [], [], [], []
            for direction in directions:
                imgs = [translate(im, d, direction) for im in images]
                pred, prob, _ = predict_model(
                    bundle, imgs, device, cfg["training"]["batch_size"]
                )
                m = summarize(labels, pred, prob, ref)
                accs.append(m["top1_accuracy"])
                cons.append(m["prediction_consistency"])
                f1s.append(m["macro_f1"])
                confs.append(m["mean_max_confidence"])
            out[name].append(
                {
                    "displacement": d,
                    "top1_accuracy": float(np.mean(accs)),
                    "prediction_consistency": float(np.mean(cons)),
                    "macro_f1": float(np.mean(f1s)),
                    "mean_max_confidence": float(np.mean(confs)),
                }
            )
            print(
                f"[translate δ={d}] {name}: "
                f"acc={out[name][-1]['top1_accuracy']:.4f} "
                f"cons={out[name][-1]['prediction_consistency']:.4f}"
            )
    return out


def eval_patch_shuffle(cfg, models, images, labels, clean_preds, device) -> dict:
    """One non-identity permutation per image (seed 6304), reused across models."""
    set_seed(int(cfg["seed"]))
    rng = np.random.default_rng(int(cfg["seed"]))
    grid = int(cfg["patch_shuffle"]["grid_size"])
    shuffled = []
    perms = []
    for im in images:
        out, perm = patch_shuffle(im, grid_size=grid, rng=rng)
        shuffled.append(out)
        perms.append(perm.tolist())

    results = {"metrics": {}, "permutations_saved": True, "n_images": len(images)}
    for name, bundle in models.items():
        ref = np.asarray(clean_preds[name])
        pred, prob, _ = predict_model(
            bundle, shuffled, device, cfg["training"]["batch_size"]
        )
        results["metrics"][name] = summarize(labels, pred, prob, ref)
        print(
            f"[patch_shuffle] {name}: "
            f"acc={results['metrics'][name]['top1_accuracy']:.4f}"
        )

    # Persist permutations so the exact same shuffled images can be regenerated.
    perm_path = Path(cfg["paths"]["tables_dir"]) / "patch_shuffle_perms.json"
    perm_path.write_text(json.dumps(perms), encoding="utf-8")
    return results


def eval_cue_conflict(cfg, models, device) -> dict:
    cue_meta = json.loads(Path(cfg["paths"]["cue_dir"], "cue_conflicts.json").read_text())
    img_dir = Path(cfg["paths"]["cue_dir"]) / "images"
    accepted = [r for r in cue_meta["records"] if r.get("accepted") and r.get("file")]
    if not accepted:
        return {"error": "No accepted cue-conflict images found."}

    images = []
    content_ids = []
    style_ids = []
    for r in accepted:
        images.append(np.asarray(Image.open(img_dir / r["file"]).convert("RGB"), dtype=np.uint8))
        content_ids.append(STL10_CLASSES.index(r["content_class"]))
        style_ids.append(STL10_CLASSES.index(r["style_class"]))
    content_ids = np.asarray(content_ids)
    style_ids = np.asarray(style_ids)

    results = {
        "n_accepted": len(accepted),
        "n_rejected": cue_meta.get("rejected_count", None),
        "models": {},
        "examples": [],
    }
    all_preds = {}
    for name, bundle in models.items():
        pred, prob, _ = predict_model(bundle, images, device, cfg["training"]["batch_size"])
        all_preds[name] = pred
        n_shape = int(((pred == content_ids) & (pred != style_ids)).sum())
        n_texture = int(((pred == style_ids) & (pred != content_ids)).sum())
        n_other = int(len(pred) - n_shape - n_texture)
        denom = n_shape + n_texture
        shape_bias = (100.0 * n_shape / denom) if denom > 0 else float("nan")
        coverage = 100.0 * denom / len(pred)
        results["models"][name] = {
            "n_shape": n_shape,
            "n_texture": n_texture,
            "n_other": n_other,
            "shape_bias_pct": shape_bias,
            "coverage_pct": coverage,
            "mean_max_confidence": mean_max_confidence(prob),
        }
        print(
            f"[cue] {name}: shape_bias={shape_bias:.1f}% "
            f"coverage={coverage:.1f}% (S={n_shape} T={n_texture} O={n_other})"
        )

    # Save a handful of informative examples for the report (agreements / disagreements).
    model_names = list(all_preds.keys())
    for i, r in enumerate(accepted[:50]):  # scan early examples only
        votes = {m: int(all_preds[m][i]) for m in model_names}
        labels = {m: STL10_CLASSES[votes[m]] for m in model_names}
        kinds = {}
        for m, p in votes.items():
            if p == content_ids[i] and p != style_ids[i]:
                kinds[m] = "shape"
            elif p == style_ids[i] and p != content_ids[i]:
                kinds[m] = "texture"
            else:
                kinds[m] = "other"
        # Keep rows where models disagree or all pick texture/other.
        if len(set(kinds.values())) > 1 or all(k != "shape" for k in kinds.values()):
            results["examples"].append(
                {
                    "file": r["file"],
                    "content_class": r["content_class"],
                    "style_class": r["style_class"],
                    "predictions": labels,
                    "decision_kind": kinds,
                }
            )
        if len(results["examples"]) >= 12:
            break
    return results


def run_all_evaluations(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tables = Path(cfg["paths"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)

    print("Loading eval subset...")
    images, labels, _ = load_eval_images(cfg)
    print(f"Eval images: {len(images)}")

    print("Loading models...")
    models = load_heads(cfg, device)

    clean = eval_clean(cfg, models, images, labels, device)
    (tables / "clean_baseline.json").write_text(
        json.dumps({"metrics": clean["metrics"]}, indent=2), encoding="utf-8"
    )

    color = eval_color(cfg, models, images, labels, clean["clean_preds"], device)
    (tables / "color_results.json").write_text(json.dumps(color, indent=2), encoding="utf-8")

    trans = eval_translation(cfg, models, images, labels, clean["clean_preds"], device)
    (tables / "translation_results.json").write_text(
        json.dumps(trans, indent=2), encoding="utf-8"
    )

    patch = eval_patch_shuffle(cfg, models, images, labels, clean["clean_preds"], device)
    (tables / "patch_shuffle_results.json").write_text(
        json.dumps(patch, indent=2), encoding="utf-8"
    )

    cue = eval_cue_conflict(cfg, models, device)
    (tables / "cue_conflict_results.json").write_text(
        json.dumps(cue, indent=2), encoding="utf-8"
    )
    print(f"Wrote evaluation tables to {tables}")


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task1/configs/default.yaml")
    args = parser.parse_args()
    run_all_evaluations(args.config)
