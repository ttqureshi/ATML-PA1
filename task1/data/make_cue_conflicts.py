"""Generate shape-vs-texture cue-conflict images with AdaIN.

Public implementation basis:
  Huang & Belongie (2017) + pytorch-AdaIN weights (naoto0804/pytorch-AdaIN).

ASSUMPTIONS:
  - Content image supplies shape; style image supplies texture.
  - We generate both directions for each unordered class pair when possible.
  - Visual rejection is decided WITHOUT model predictions (assignment rule).
  - Rejection uses edge correlation with content + pixel correlation with content.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

# Allow `python task1/data/make_cue_conflicts.py` from the repo root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import datasets

from task1.data.make_subset import STL10_CLASSES


# ---------------------------------------------------------------------------
# AdaIN network (architecture from naoto0804/pytorch-AdaIN)
# ---------------------------------------------------------------------------

def build_vgg() -> nn.Sequential:
    # Matches the normalised VGG used by pytorch-AdaIN (first 31 children = relu4_1).
    return nn.Sequential(
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, (3, 3)),
        nn.ReLU(),
    )


def build_decoder() -> nn.Sequential:
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )


def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    size = feat.size()
    assert len(size) == 4
    n, c = size[:2]
    feat_var = feat.view(n, c, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(n, c, 1, 1)
    feat_mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


class AdaINStylizer:
    """Thin wrapper around pretrained AdaIN encoder/decoder."""

    WEIGHT_URLS = {
        "vgg_normalised.pth": (
            "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/vgg_normalised.pth"
        ),
        "decoder.pth": (
            "https://github.com/naoto0804/pytorch-AdaIN/releases/download/v0.0.0/decoder.pth"
        ),
    }

    def __init__(self, weights_dir: Path, device: torch.device):
        self.device = device
        weights_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_weights(weights_dir)

        vgg = build_vgg()
        vgg.load_state_dict(torch.load(weights_dir / "vgg_normalised.pth", map_location="cpu"))
        # relu4_1 is the first 31 modules in the sequential.
        self.encoder = nn.Sequential(*list(vgg.children())[:31]).to(device).eval()

        decoder = build_decoder()
        decoder.load_state_dict(torch.load(weights_dir / "decoder.pth", map_location="cpu"))
        self.decoder = decoder.to(device).eval()

        for p in list(self.encoder.parameters()) + list(self.decoder.parameters()):
            p.requires_grad_(False)

    def _ensure_weights(self, weights_dir: Path) -> None:
        for name, url in self.WEIGHT_URLS.items():
            path = weights_dir / name
            if path.exists():
                continue
            print(f"Downloading AdaIN weight: {name}")
            urllib.request.urlretrieve(url, path)

    @torch.no_grad()
    def stylize(
        self,
        content_rgb: np.ndarray,
        style_rgb: np.ndarray,
        alpha: float = 1.0,
    ) -> np.ndarray:
        content = self._to_tensor(content_rgb)
        style = self._to_tensor(style_rgb)
        c_feat = self.encoder(content)
        s_feat = self.encoder(style)
        t = adaptive_instance_normalization(c_feat, s_feat)
        t = alpha * t + (1.0 - alpha) * c_feat
        out = self.decoder(t)
        out = out.clamp(0.0, 1.0)
        arr = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).round()
        return arr.astype(np.uint8)

    def _to_tensor(self, rgb: np.ndarray) -> torch.Tensor:
        # AdaIN VGG expects RGB in [0, 1] (weights include a learned first 1x1).
        t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
        return t.unsqueeze(0).to(self.device)


# ---------------------------------------------------------------------------
# Visual rejection (no model predictions)
# ---------------------------------------------------------------------------

def _sobel_edges(gray: np.ndarray) -> np.ndarray:
    """Fast edge magnitude on a downsampled grayscale image."""
    g = gray.astype(np.float32)[::4, ::4]
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    return np.sqrt(gx * gx + gy * gy)


def edge_correlation(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    a = a_rgb.mean(axis=-1)
    b = b_rgb.mean(axis=-1)
    ea = _sobel_edges(a).ravel()
    eb = _sobel_edges(b).ravel()
    if ea.std() < 1e-6 or eb.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(ea, eb)[0, 1])


def pixel_correlation(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    a = a_rgb.astype(np.float32).ravel()
    b = b_rgb.astype(np.float32).ravel()
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def accept_stylization(
    content: np.ndarray,
    stylized: np.ndarray,
    min_edge_corr: float,
    max_pixel_corr: float,
) -> Tuple[bool, Dict[str, float]]:
    """Reject failed stylizations using visual heuristics only.

    Reject if:
      - edges no longer resemble the content (content destroyed), OR
      - stylized image is nearly identical to content (style transfer failed).
    """
    ecorr = edge_correlation(content, stylized)
    pcorr = pixel_correlation(content, stylized)
    ok = (ecorr >= min_edge_corr) and (pcorr <= max_pixel_corr)
    return ok, {"edge_corr_content": ecorr, "pixel_corr_content": pcorr}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def class_to_id(name: str) -> int:
    return STL10_CLASSES.index(name)


def collect_by_class(dataset, indices: List[int]) -> Dict[int, List[int]]:
    buckets: Dict[int, List[int]] = {i: [] for i in range(len(STL10_CLASSES))}
    for idx in indices:
        y = int(dataset.labels[idx])
        buckets[y].append(idx)
    return buckets


def resize_rgb(img: Image.Image, size: int) -> np.ndarray:
    img = img.convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = int(cfg["dataset"]["image_size"])
    root = Path(cfg["dataset"]["root"])
    eval_meta = json.loads(Path(cfg["paths"]["splits_dir"], "eval_subset.json").read_text())
    eval_indices = eval_meta["indices"]

    test_set = datasets.STL10(root=str(root), split="test", download=True)
    by_class = collect_by_class(test_set, eval_indices)

    stylizer = AdaINStylizer(Path(cfg["cue_conflict"]["adain_weights_dir"]), device)
    alpha = float(cfg["cue_conflict"]["style_alpha"])
    min_edge = float(cfg["cue_conflict"]["min_edge_corr_with_content"])
    max_pix = float(cfg["cue_conflict"]["max_pixel_corr_with_content"])
    pairs = cfg["cue_conflict"]["class_pairs"]
    min_valid = int(cfg["cue_conflict"]["min_valid"])

    out_dir = Path(cfg["paths"]["cue_dir"])
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    accepted = 0
    rejected = 0

    # Target roughly equal count per (pair, direction).
    n_directions = sum(2 for _ in pairs)  # both directions
    per_bucket = max(1, int(np.ceil(min_valid / n_directions)))

    for pair in pairs:
        a_name, b_name = pair
        a_id, b_id = class_to_id(a_name), class_to_id(b_name)
        for content_id, style_id, content_name, style_name in [
            (a_id, b_id, a_name, b_name),
            (b_id, a_id, b_name, a_name),
        ]:
            content_pool = by_class[content_id]
            style_pool = by_class[style_id]
            if not content_pool or not style_pool:
                print(f"Skipping {content_name}->{style_name}: empty pool")
                continue

            made = 0
            attempts = 0
            max_attempts = per_bucket * 20
            while made < per_bucket and attempts < max_attempts:
                attempts += 1
                c_idx = int(rng.choice(content_pool))
                s_idx = int(rng.choice(style_pool))
                content = resize_rgb(test_set[c_idx][0], size)
                style = resize_rgb(test_set[s_idx][0], size)
                stylized = stylizer.stylize(content, style, alpha=alpha)
                ok, scores = accept_stylization(content, stylized, min_edge, max_pix)

                rel_name = (
                    f"{content_name}_shape__{style_name}_texture__"
                    f"c{c_idx}_s{s_idx}.png"
                )
                if ok:
                    Image.fromarray(stylized).save(img_dir / rel_name)
                    accepted += 1
                    made += 1
                    records.append(
                        {
                            "file": rel_name,
                            "accepted": True,
                            "content_class": content_name,
                            "style_class": style_name,
                            "content_idx": c_idx,
                            "style_idx": s_idx,
                            **scores,
                        }
                    )
                else:
                    rejected += 1
                    records.append(
                        {
                            "file": None,
                            "accepted": False,
                            "content_class": content_name,
                            "style_class": style_name,
                            "content_idx": c_idx,
                            "style_idx": s_idx,
                            **scores,
                        }
                    )

            print(
                f"{content_name} (shape) / {style_name} (texture): "
                f"accepted={made}/{per_bucket} after {attempts} attempts"
            )

    meta = {
        "seed": seed,
        "method": "AdaIN (naoto0804/pytorch-AdaIN weights)",
        "alpha": alpha,
        "min_valid_requested": min_valid,
        "accepted_count": accepted,
        "rejected_count": rejected,
        "rejection_rule": {
            "min_edge_corr_with_content": min_edge,
            "max_pixel_corr_with_content": max_pix,
            "note": "Rejected without using any model predictions.",
        },
        "class_pairs": pairs,
        "records": records,
    }
    meta_path = out_dir / "cue_conflicts.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Accepted={accepted} Rejected={rejected}")
    print(f"Wrote {meta_path}")
    if accepted < min_valid:
        print(
            f"WARNING: only {accepted} valid conflicts (< {min_valid}). "
            "Consider loosening rejection thresholds in the config."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Task 1 cue-conflict images.")
    parser.add_argument("--config", default="task1/configs/default.yaml")
    args = parser.parse_args()
    main(args.config)
