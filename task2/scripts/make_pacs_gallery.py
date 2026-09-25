"""Build PACS visual galleries so Task 2 domain gap is easy to see.

Outputs under task2/results/figures/pacs_gallery/:
  - domains_same_class.png   : one class × four domains
  - classes_across_domains.png : several classes × four domains
  - hard_vs_easy.png         : giraffe/dog/horse vs guitar (why Sketch fails)

Run from repo root:
  python -m task2.scripts.make_pacs_gallery
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.pacs import CLASS_NAMES, DOMAINS, SOURCE_DOMAINS, TARGET_DOMAIN

PACS_ROOT = ROOT / "data" / "pacs"
OUT_DIR = ROOT / "task2" / "results" / "figures" / "pacs_gallery"


def _list_images(domain: str, class_name: str) -> list[Path]:
    d = PACS_ROOT / domain / class_name
    if not d.is_dir():
        raise FileNotFoundError(d)
    files = sorted(
        p
        for p in d.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not files:
        raise FileNotFoundError(f"No images in {d}")
    return files


def load_rgb(path: Path, size: int = 224) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def pick_image(domain: str, class_name: str, which: int = 0) -> np.ndarray:
    files = _list_images(domain, class_name)
    return load_rgb(files[which % len(files)])


def save_grid(
    rows: list[tuple[str, list[np.ndarray]]],
    col_titles: list[str],
    out_path: Path,
    *,
    title: str,
    figsize: tuple[float, float],
) -> None:
    n_rows = len(rows)
    n_cols = len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)
    for r, (row_label, images) in enumerate(rows):
        for c, img in enumerate(images):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11, pad=6)
            if c == 0:
                ax.set_ylabel(row_label, fontsize=11, rotation=90, labelpad=8)
    fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout(rect=[0.02, 0.02, 1.0, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    if not PACS_ROOT.is_dir():
        raise SystemExit(f"PACS not found at {PACS_ROOT}")

    domains = list(DOMAINS)  # photo, art_painting, cartoon, sketch
    domain_titles = ["Photo", "Art painting", "Cartoon", "Sketch"]

    # 1) One class across domains — pick giraffe (hard on Sketch) and guitar (easy)
    for cls, which in [("giraffe", 2), ("guitar", 1), ("dog", 3), ("house", 0)]:
        imgs = [pick_image(d, cls, which=which) for d in domains]
        save_grid(
            [(cls, imgs)],
            domain_titles,
            OUT_DIR / f"domain_gap_{cls}.png",
            title=f"Same class '{cls}' across PACS domains",
            figsize=(10, 3.0),
        )

    # 2) Multi-class board (report-friendly overview)
    showcase = ["dog", "elephant", "giraffe", "guitar", "horse", "person"]
    rows = []
    for cls in showcase:
        rows.append((cls, [pick_image(d, cls, which=1) for d in domains]))
    save_grid(
        rows,
        domain_titles,
        OUT_DIR / "classes_across_domains.png",
        title="PACS: labels stay the same; appearance (domain) changes",
        figsize=(10, 12),
    )

    # 3) Hard vs easy on Sketch — why Source-only patterns make sense
    hard = ["giraffe", "dog", "horse"]
    easy = ["guitar", "house", "person"]
    rows = []
    for cls in hard + easy:
        # Show Photo vs Sketch only for a tight comparison
        rows.append(
            (
                cls,
                [pick_image("photo", cls, which=2), pick_image("sketch", cls, which=2)],
            )
        )
    save_grid(
        rows,
        ["Photo (source-like)", "Sketch (target)"],
        OUT_DIR / "photo_vs_sketch_hard_easy.png",
        title="Photo → Sketch: animals lose texture; guitar/house keep silhouette",
        figsize=(6.5, 11),
    )

    # 4) Source trio vs Sketch for one animal (the adaptation story)
    cls = "elephant"
    imgs = [pick_image(d, cls, which=4) for d in ["photo", "art_painting", "cartoon", "sketch"]]
    save_grid(
        [(cls, imgs)],
        domain_titles,
        OUT_DIR / "adaptation_story_elephant.png",
        title="Task 2 setup: train on Photo/Art/Cartoon labels; adapt using Sketch images without labels",
        figsize=(10, 3.0),
    )

    print(f"Gallery folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
