"""Visualize learned 512-d feature distributions for Task 2 methods.

For each checkpoint we:
  1) Extract features from balanced source-val + Sketch subsets
  2) Fit one t-SNE (seed 6304) on that method's features
  3) Color by domain (source vs sketch) and by class

This answers: did adaptation actually move Sketch toward sources in feature space?

Run from repo root (GPU optional; CPU is fine for a few thousand points):
  python -m task2.scripts.make_feature_tsne
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.seed import set_seed
from shared.pacs import (
    CLASS_NAMES,
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

CKPT_DIR = ROOT / "task2" / "results" / "checkpoints"
OUT_DIR = ROOT / "task2" / "results" / "figures" / "feature_tsne"
SPLITS = ROOT / "shared" / "splits" / "pacs_sketch_seed6304.json"
PACS_ROOT = ROOT / "data" / "pacs"

# Main comparison + lambda extremes for the lecture.
RUNS = [
    ("source_only", CKPT_DIR / "source_only_best.pt"),
    ("dan_lambda1.0", CKPT_DIR / "dan_lambda1.0_best.pt"),
    ("dann", CKPT_DIR / "dann_best.pt"),
    ("cdan", CKPT_DIR / "cdan_best.pt"),
    ("dan_lambda0.1", CKPT_DIR / "dan_lambda0.1_best.pt"),
    ("dan_lambda10.0", CKPT_DIR / "dan_lambda10.0_best.pt"),
]


@torch.no_grad()
def extract_features(model, loader, device):
    set_eval_mode(model)
    feats, labels, domains = [], [], []
    n_batches = 0
    for batch in loader:
        images, y, d = unpack_batch(batch, device)
        f, _ = model(images)
        feats.append(f.cpu().numpy())
        labels.append(y.cpu().numpy())
        domains.append(d.cpu().numpy())
        n_batches += 1
        if n_batches % 5 == 0:
            print(f"    batch {n_batches}", flush=True)
    return (
        np.concatenate(feats),
        np.concatenate(labels),
        np.concatenate(domains),
    )


def build_balanced_loaders(max_per_domain: int = 150, batch_size: int = 64):
    """Equal-ish counts from each source val domain + Sketch for fair t-SNE."""
    set_seed(6304)
    splits = load_splits(SPLITS)
    uid_index = index_by_uid(discover_pacs(PACS_ROOT))
    tf = make_transforms(train=False)

    rng = np.random.RandomState(6304)
    source_examples = []
    for domain in SOURCE_DOMAINS:
        ex = examples_from_uids(splits["domains"][domain]["val"], uid_index)
        if len(ex) > max_per_domain:
            idx = rng.choice(len(ex), size=max_per_domain, replace=False)
            ex = [ex[i] for i in sorted(idx.tolist())]
        source_examples.extend(ex)
        print(f"  source {domain}: {len(ex)}", flush=True)

    sketch = examples_from_uids(splits["target"]["all"], uid_index)
    n_src = len(source_examples)
    if len(sketch) > n_src:
        idx = rng.choice(len(sketch), size=n_src, replace=False)
        sketch = [sketch[i] for i in sorted(idx.tolist())]
    print(f"  sketch: {len(sketch)}", flush=True)

    src_loader = make_loader(
        PacsImageDataset(source_examples, transform=tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    sk_loader = make_loader(
        PacsImageDataset(sketch, transform=tf),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return src_loader, sk_loader, n_src


def plot_method(name: str, feats: np.ndarray, labels: np.ndarray, is_sketch: np.ndarray, out_dir: Path):
    set_seed(6304)
    # Standardize lightly for t-SNE stability
    x = feats.astype(np.float64)
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    emb = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=6304,
    ).fit_transform(x)

    # Panel A: domain coloring
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    ax.scatter(
        emb[~is_sketch, 0],
        emb[~is_sketch, 1],
        s=8,
        alpha=0.55,
        c="#1f77b4",
        label="source val (P/A/C)",
        rasterized=True,
    )
    ax.scatter(
        emb[is_sketch, 0],
        emb[is_sketch, 1],
        s=8,
        alpha=0.55,
        c="#d62728",
        label="Sketch",
        rasterized=True,
    )
    ax.set_title(f"{name}: features colored by DOMAIN")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])

    # Panel B: class coloring (Sketch marked with x)
    ax = axes[1]
    cmap = plt.get_cmap("tab10")
    for c, cname in enumerate(CLASS_NAMES):
        mask = labels == c
        ax.scatter(
            emb[mask & ~is_sketch, 0],
            emb[mask & ~is_sketch, 1],
            s=8,
            alpha=0.5,
            color=cmap(c),
            marker="o",
            label=f"{cname} (src)",
            rasterized=True,
        )
        ax.scatter(
            emb[mask & is_sketch, 0],
            emb[mask & is_sketch, 1],
            s=14,
            alpha=0.7,
            color=cmap(c),
            marker="x",
            label=f"{cname} (sk)",
            rasterized=True,
        )
    ax.set_title(f"{name}: features colored by CLASS (x = Sketch)")
    ax.legend(fontsize=6, loc="best", ncol=2, markerscale=1.5)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.suptitle(
        "t-SNE of 512-d ResNet features (same subset / seed 6304 per run)",
        fontsize=11,
    )
    fig.tight_layout()
    out = out_dir / f"{name}_tsne.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")

    # Also save a domain-only compact figure for the lecture
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(
        emb[~is_sketch, 0],
        emb[~is_sketch, 1],
        s=10,
        alpha=0.55,
        c="#1f77b4",
        label="source",
        rasterized=True,
    )
    ax.scatter(
        emb[is_sketch, 0],
        emb[is_sketch, 1],
        s=10,
        alpha=0.55,
        c="#d62728",
        label="Sketch",
        rasterized=True,
    )
    ax.set_title(f"{name}\nblue=source  red=Sketch")
    ax.legend(markerscale=1.8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out2 = out_dir / f"{name}_domain_only.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src_loader, sk_loader, n = build_balanced_loaders()
    print(f"points per side ~ {n}")

    summary = {}
    for name, ckpt in RUNS:
        if not ckpt.exists():
            print(f"SKIP missing {ckpt}")
            continue
        print(f"Extracting {name} ...")
        model = ResNet18Classifier().to(device)
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(blob["model_state_dict"])
        fs, ys, _ = extract_features(model, src_loader, device)
        ft, yt, _ = extract_features(model, sk_loader, device)
        feats = np.concatenate([fs, ft], axis=0)
        labels = np.concatenate([ys, yt], axis=0)
        is_sketch = np.concatenate(
            [np.zeros(len(fs), dtype=bool), np.ones(len(ft), dtype=bool)]
        )
        plot_method(name, feats, labels, is_sketch, OUT_DIR)

        # Quick centroid distance diagnostic (raw space, not t-SNE)
        mu_s = fs.mean(0)
        mu_t = ft.mean(0)
        summary[name] = {
            "n_source": int(len(fs)),
            "n_sketch": int(len(ft)),
            "centroid_l2": float(np.linalg.norm(mu_s - mu_t)),
            "mean_source_val_macro_f1_at_ckpt": blob.get("mean_source_val_macro_f1"),
        }
        print(f"  centroid L2(source, sketch) = {summary[name]['centroid_l2']:.3f}")

    out_json = OUT_DIR / "centroid_distances.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
