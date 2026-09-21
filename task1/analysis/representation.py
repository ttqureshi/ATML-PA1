"""t-SNE / UMAP visualization of clean vs transformed representations.

Assignment rules followed here:
  - Fit ONE 2D projection per backbone on the COMBINED clean+transformed features.
  - Color = ground-truth class; marker style = clean vs transformed.
  - Keep image subset and random seed fixed.
  - Do not compare absolute coordinates across separately fitted backbones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.manifold import TSNE

from common.seed import set_seed
from task1.analysis.evaluate_bias import load_eval_images, load_heads, predict_model
from task1.data.make_subset import STL10_CLASSES
from task1.data.transforms import grayscale, patch_shuffle


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fit_projection(features: np.ndarray, method: str, seed: int, perplexity: float) -> np.ndarray:
    method = method.lower()
    if method == "tsne":
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        return reducer.fit_transform(features)
    if method == "umap":
        try:
            import umap
        except ImportError as e:
            raise ImportError("Install umap-learn to use UMAP visualizations.") from e
        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(features)
    raise ValueError(f"Unknown method: {method}")


def plot_embedding(
    xy: np.ndarray,
    labels: np.ndarray,
    is_transformed: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Class-colored scatter: circles=clean, crosses=transformed."""
    plt.figure(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")
    for cls, name in enumerate(STL10_CLASSES):
        color = cmap(cls % 10)
        mask_c = (labels == cls) & (~is_transformed)
        mask_t = (labels == cls) & is_transformed
        plt.scatter(
            xy[mask_c, 0],
            xy[mask_c, 1],
            s=20,
            marker="o",
            color=color,
            alpha=0.8,
            label=name,
        )
        plt.scatter(
            xy[mask_t, 0],
            xy[mask_t, 1],
            s=20,
            marker="x",
            color=color,
            alpha=0.8,
        )
    plt.title(title + "\n(o = clean, x = transformed)")
    plt.legend(markerscale=1.2, fontsize=8, ncol=2, loc="best")
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def run_representation_analysis(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images, labels, _ = load_eval_images(cfg)
    models = load_heads(cfg, device)
    method = cfg["representation"]["method"]
    perplexity = float(cfg["representation"]["perplexity"])
    seed = int(cfg["seed"])
    fig_dir = Path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "method": method,
        "perplexity": perplexity if method == "tsne" else None,
        "seed": seed,
        "n_points_per_condition": len(images),
        "interventions_plotted": ["grayscale", "patch_shuffle"],
        "note": (
            "One joint 2D fit per backbone on concatenated clean+transformed features. "
            "Absolute coordinates are not comparable across backbones."
        ),
    }
    (fig_dir / "representation_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    backbone_names = [m for m in models if models[m]["kind"] == "linear"]
    for name in backbone_names:
        bundle = models[name]
        _, _, f_clean = predict_model(bundle, images, device, cfg["training"]["batch_size"])

        # Grayscale
        gray_imgs = [grayscale(im) for im in images]
        print(f"Projection {name} / grayscale")
        _, _, f_gray = predict_model(bundle, gray_imgs, device, cfg["training"]["batch_size"])
        feats = np.concatenate([f_clean, f_gray], axis=0)
        labs = np.concatenate([labels, labels], axis=0)
        is_t = np.concatenate([np.zeros(len(labels), bool), np.ones(len(labels), bool)])
        xy = fit_projection(feats, method, seed, perplexity)
        out = fig_dir / f"{name}_grayscale_{method}.png"
        plot_embedding(xy, labs, is_t, f"{name}: clean vs grayscale", out)
        print(f"  Saved {out}")

        # Patch shuffle — regenerate with the same seed for every backbone.
        rng = np.random.default_rng(seed)
        shuffled = [
            patch_shuffle(im, cfg["patch_shuffle"]["grid_size"], rng=rng)[0] for im in images
        ]
        print(f"Projection {name} / patch_shuffle")
        _, _, f_patch = predict_model(bundle, shuffled, device, cfg["training"]["batch_size"])
        feats = np.concatenate([f_clean, f_patch], axis=0)
        xy = fit_projection(feats, method, seed, perplexity)
        out = fig_dir / f"{name}_patch_shuffle_{method}.png"
        plot_embedding(xy, labs, is_t, f"{name}: clean vs patch_shuffle", out)
        print(f"  Saved {out}")


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="task1/configs/default.yaml")
    args = parser.parse_args()
    run_representation_analysis(args.config)
