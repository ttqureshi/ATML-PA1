"""Representation cosine stability between clean and transformed images.

IT = mean_i  cos( f(x_i), f(T(x_i)) )

Computed for grayscale, cue-conflict (matched pairs), translation, and
patch shuffling — as required by the assignment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import datasets

from common.seed import set_seed
from task1.analysis.evaluate_bias import load_eval_images, load_heads, predict_model
from task1.data.transforms import grayscale, patch_shuffle, translate


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cosine_stability(clean_feats: np.ndarray, trans_feats: np.ndarray) -> float:
    a = clean_feats.astype(np.float64)
    b = trans_feats.astype(np.float64)
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return float((a_n * b_n).sum(axis=1).mean())


def run_feature_similarity(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tables = Path(cfg["paths"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)

    images, labels, _ = load_eval_images(cfg)
    models = load_heads(cfg, device)

    # Only backbone feature extractors (skip duplicate zeroshot entry for features).
    backbone_names = [m for m in models if models[m]["kind"] == "linear"]

    # Build transformed image sets once.
    gray_imgs = [grayscale(im) for im in images]
    # Use δ=16 as a representative translation for the stability table.
    # Full translation curves already live in translation_results.json.
    trans16 = [translate(im, 16, "right") for im in images]

    rng = np.random.default_rng(int(cfg["seed"]))
    shuffled = [patch_shuffle(im, cfg["patch_shuffle"]["grid_size"], rng=rng)[0] for im in images]

    # Cue conflicts: compare stylized image features to their content image features.
    cue_meta = json.loads(Path(cfg["paths"]["cue_dir"], "cue_conflicts.json").read_text())
    img_dir = Path(cfg["paths"]["cue_dir"]) / "images"
    root = Path(cfg["dataset"]["root"])
    size = int(cfg["dataset"]["image_size"])
    test_set = datasets.STL10(root=str(root), split="test", download=True)
    cue_stylized, cue_content = [], []
    for r in cue_meta["records"]:
        if not r.get("accepted") or not r.get("file"):
            continue
        cue_stylized.append(np.asarray(Image.open(img_dir / r["file"]).convert("RGB"), np.uint8))
        content = test_set[r["content_idx"]][0].convert("RGB").resize((size, size), Image.BICUBIC)
        cue_content.append(np.asarray(content, dtype=np.uint8))

    results: Dict[str, dict] = {}
    for name in backbone_names:
        bundle = models[name]
        print(f"Features: {name}")
        _, _, f_clean = predict_model(bundle, images, device, cfg["training"]["batch_size"])
        _, _, f_gray = predict_model(bundle, gray_imgs, device, cfg["training"]["batch_size"])
        _, _, f_t16 = predict_model(bundle, trans16, device, cfg["training"]["batch_size"])
        _, _, f_patch = predict_model(bundle, shuffled, device, cfg["training"]["batch_size"])

        entry = {
            "grayscale": cosine_stability(f_clean, f_gray),
            "translation_16px_right": cosine_stability(f_clean, f_t16),
            "patch_shuffle": cosine_stability(f_clean, f_patch),
        }

        if cue_stylized:
            _, _, f_cue_c = predict_model(
                bundle, cue_content, device, cfg["training"]["batch_size"]
            )
            _, _, f_cue_s = predict_model(
                bundle, cue_stylized, device, cfg["training"]["batch_size"]
            )
            entry["cue_conflict_vs_content"] = cosine_stability(f_cue_c, f_cue_s)

        results[name] = entry
        print(f"  {entry}")

    out = {
        "definition": "mean cosine similarity between clean and transformed backbone features",
        "notes": [
            "Translation stability uses a single representative shift (16px right).",
            "Cue-conflict stability compares stylized features to the content image features.",
            "CLIP zeroshot uses the same visual features as clip_vit_b_32, so it is omitted here.",
        ],
        "results": results,
    }
    path = tables / "feature_similarity.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task1/configs/default.yaml")
    args = parser.parse_args()
    run_feature_similarity(args.config)
