"""Build a visual gallery of Task 1 interventions for inspection / report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from task1.data.transforms import grayscale, hue_rotate, patch_shuffle, translate

ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = ROOT / "data" / "stl10" / "stl10_binary"
OUT_DIR = ROOT / "task1" / "results" / "figures" / "intervention_gallery"
CUE_DIR = ROOT / "task1" / "results" / "cue_conflicts" / "images"
SPLITS = ROOT / "task1" / "results" / "splits" / "eval_subset.json"

CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]


def load_stl10_test_image(index: int, size: int = 224) -> np.ndarray:
    """Read one STL-10 test image directly from the binary (no unlabeled file needed).

    Torchvision stores STL-10 with an H/W transpose after the channel-major reshape.
    We mirror that so gallery images match what the models saw.
    """
    x_path = BIN_DIR / "test_X.bin"
    img_size = 3 * 96 * 96
    with open(x_path, "rb") as f:
        f.seek(index * img_size)
        buf = np.frombuffer(f.read(img_size), dtype=np.uint8)
    # Mirror torchvision.datasets.STL10 loading:
    # reshape -> (3, 96, 96), then transpose to (3, 96, 96) with H/W swap, then HWC.
    img = buf.reshape(3, 96, 96)
    img = np.transpose(img, (0, 2, 1))  # (C, H, W) with H/W swap
    img = np.transpose(img, (1, 2, 0))  # HWC
    return np.asarray(Image.fromarray(img).resize((size, size), Image.BICUBIC), dtype=np.uint8)


def load_stl10_test_label(index: int) -> int:
    with open(BIN_DIR / "test_y.bin", "rb") as f:
        f.seek(index)
        # labels are 1..10 in the file
        return int(np.frombuffer(f.read(1), dtype=np.uint8)[0]) - 1


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads(SPLITS.read_text(encoding="utf-8"))
    indices = meta["indices"]

    want = ["airplane", "cat", "truck", "horse"]
    picked = []
    for name in want:
        cid = CLASSES.index(name)
        for idx in indices:
            if load_stl10_test_label(idx) == cid:
                picked.append((name, idx))
                break

    # 1) Color + spatial interventions
    fig, axes = plt.subplots(len(picked), 6, figsize=(14, 2.6 * len(picked)))
    titles = ["Clean", "Grayscale", "Hue +90°", "Translate 32px →", "Translate 32px ↑", "Patch shuffle 4×4"]
    for r, (name, idx) in enumerate(picked):
        clean = load_stl10_test_image(idx)
        variants = [
            clean,
            grayscale(clean),
            hue_rotate(clean, 90.0),
            translate(clean, 32, "right"),
            translate(clean, 32, "up"),
            patch_shuffle(clean, 4, rng=np.random.default_rng(6304 + idx))[0],
        ]
        for c, im in enumerate(variants):
            axes[r, c].imshow(im)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(titles[c], fontsize=10)
        axes[r, 0].set_ylabel(name, fontsize=11)
        axes[r, 0].axis("on")
        axes[r, 0].set_xticks([])
        axes[r, 0].set_yticks([])
        for spine in axes[r, 0].spines.values():
            spine.set_visible(False)
    fig.suptitle("Task 1 interventions (same source images)", fontsize=13, y=1.01)
    fig.tight_layout()
    p1 = OUT_DIR / "interventions_color_spatial.png"
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close()
    print("wrote", p1)

    # 2) Cue conflicts
    pairs = [("airplane", "bird"), ("cat", "dog"), ("car", "truck"), ("deer", "horse")]
    fig, axes = plt.subplots(len(pairs), 3, figsize=(9, 2.8 * len(pairs)))
    for r, (a, b) in enumerate(pairs):
        files = sorted(CUE_DIR.glob(f"{a}_shape__{b}_texture__*.png"))
        stylized = np.asarray(Image.open(files[0]).convert("RGB"))
        stem = files[0].stem
        c_idx = int(stem.split("_c")[-1].split("_s")[0])
        s_idx = int(stem.split("_s")[-1])
        content = load_stl10_test_image(c_idx)
        style = load_stl10_test_image(s_idx)
        for c, (im, title) in enumerate(
            [
                (content, f"Content / shape: {a}"),
                (style, f"Style / texture: {b}"),
                (stylized, f"Cue conflict\nshape={a}, texture={b}"),
            ]
        ):
            axes[r, c].imshow(im)
            axes[r, c].axis("off")
            axes[r, c].set_title(title, fontsize=10)
    fig.suptitle("Shape–texture cue conflicts (AdaIN)", fontsize=13, y=1.01)
    fig.tight_layout()
    p2 = OUT_DIR / "cue_conflicts_examples.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close()
    print("wrote", p2)

    # 3) Translation ladder
    name, idx = picked[0]
    clean = load_stl10_test_image(idx)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, d in enumerate([0, 8, 16, 32]):
        axes[0, i].imshow(translate(clean, d, "right"))
        axes[0, i].set_title(f"δ={d} right")
        axes[0, i].axis("off")
        axes[1, i].imshow(translate(clean, d, "up"))
        axes[1, i].set_title(f"δ={d} up")
        axes[1, i].axis("off")
    fig.suptitle(f"Translation ladder ({name})", fontsize=13)
    fig.tight_layout()
    p3 = OUT_DIR / "translation_ladder.png"
    fig.savefig(p3, dpi=140, bbox_inches="tight")
    plt.close()
    print("wrote", p3)
    print("DONE")


if __name__ == "__main__":
    main()
