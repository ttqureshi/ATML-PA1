"""Controlled image interventions for Task 1.

All transforms operate on a shared 224x224 RGB image (uint8/PIL or float
tensor in [0, 1]) BEFORE model-specific normalization.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ArrayLike = Union[np.ndarray, Image.Image, torch.Tensor]


def to_numpy_uint8(image: ArrayLike) -> np.ndarray:
    """Convert PIL / tensor / ndarray to HWC uint8 RGB."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)
        if t.dtype.is_floating_point:
            t = (t.clamp(0, 1) * 255.0).round()
        return t.numpy().astype(np.uint8)
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).round()
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def to_pil(image: ArrayLike) -> Image.Image:
    return Image.fromarray(to_numpy_uint8(image), mode="RGB")


def grayscale(image: ArrayLike) -> np.ndarray:
    """Remove color while preserving geometry (luminance replicated to 3 channels)."""
    rgb = to_numpy_uint8(image).astype(np.float32)
    # ITU-R BT.601 luma.
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    out = np.stack([y, y, y], axis=-1)
    return np.clip(out, 0, 255).astype(np.uint8)


def hue_rotate(image: ArrayLike, degrees: float = 90.0) -> np.ndarray:
    """Fixed hue rotation in HSV. Geometry / value channel are preserved.

    Assumption: OpenCV-style HSV with H in [0, 179] for uint8 images.
    We implement without requiring OpenCV so the dependency list stays light.
    """
    rgb = to_numpy_uint8(image).astype(np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    diff = mx - mn

    h = np.zeros_like(mx)
    mask = diff > 1e-8
    r_eq = (mx == r) & mask
    g_eq = (mx == g) & mask
    b_eq = (mx == b) & mask
    h[r_eq] = ((g[r_eq] - b[r_eq]) / diff[r_eq]) % 6.0
    h[g_eq] = (b[g_eq] - r[g_eq]) / diff[g_eq] + 2.0
    h[b_eq] = (r[b_eq] - g[b_eq]) / diff[b_eq] + 4.0
    h = h / 6.0  # [0, 1)

    s = np.zeros_like(mx)
    s[mx > 1e-8] = diff[mx > 1e-8] / mx[mx > 1e-8]
    v = mx

    h = (h + (degrees / 360.0)) % 1.0

    # HSV -> RGB
    c = v * s
    x = c * (1 - np.abs((h * 6) % 2 - 1))
    m = v - c
    z = np.zeros_like(c)
    h6 = (h * 6).astype(np.int32) % 6
    rgb_out = np.zeros_like(rgb)
    table = [
        (c, x, z),
        (x, c, z),
        (z, c, x),
        (z, x, c),
        (x, z, c),
        (c, z, x),
    ]
    for i, (rr, gg, bb) in enumerate(table):
        sel = h6 == i
        rgb_out[sel, 0] = rr[sel]
        rgb_out[sel, 1] = gg[sel]
        rgb_out[sel, 2] = bb[sel]
    rgb_out = rgb_out + m[..., None]
    return np.clip(rgb_out * 255.0, 0, 255).astype(np.uint8)


def translate(
    image: ArrayLike,
    pixels: int,
    direction: str,
) -> np.ndarray:
    """Translate with reflection padding, then center-crop back to original size.

    Assignment: reflection padding followed by a shifted crop.
    """
    rgb = to_numpy_uint8(image)
    h, w = rgb.shape[:2]
    pad = abs(int(pixels))
    if pad == 0:
        return rgb.copy()

    # Pad then crop with an offset so the object shifts in the requested direction.
    # Cropping opposite to the motion direction produces the intended translation.
    padded = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    direction = direction.lower()
    if direction == "up":
        y0, x0 = 2 * pad, pad
    elif direction == "down":
        y0, x0 = 0, pad
    elif direction == "left":
        y0, x0 = pad, 2 * pad
    elif direction == "right":
        y0, x0 = pad, 0
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return padded[y0 : y0 + h, x0 : x0 + w].copy()


def patch_shuffle(
    image: ArrayLike,
    grid_size: int = 4,
    rng: np.random.Generator | None = None,
    perm: Sequence[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Divide into a grid_size x grid_size pixel-space grid and permute patches.

    Returns (shuffled_image, permutation). If perm is given it is reused so that
    every model sees the exact same shuffled image (assignment requirement).
    """
    rgb = to_numpy_uint8(image)
    h, w = rgb.shape[:2]
    assert h % grid_size == 0 and w % grid_size == 0, (
        "Image size must be divisible by grid_size "
        f"(got {h}x{w}, grid={grid_size})."
    )
    ph, pw = h // grid_size, w // grid_size
    patches = [
        rgb[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw].copy()
        for i in range(grid_size)
        for j in range(grid_size)
    ]
    n = grid_size * grid_size
    if perm is None:
        if rng is None:
            rng = np.random.default_rng()
        # Non-identity permutation.
        while True:
            perm_arr = rng.permutation(n)
            if not np.array_equal(perm_arr, np.arange(n)):
                break
    else:
        perm_arr = np.asarray(perm, dtype=np.int64)

    shuffled = [patches[i] for i in perm_arr]
    out = np.zeros_like(rgb)
    for idx, patch in enumerate(shuffled):
        i, j = divmod(idx, grid_size)
        out[i * ph : (i + 1) * ph, j * pw : (j + 1) * pw] = patch
    return out, perm_arr


def average_over_directions(
    image: ArrayLike,
    pixels: int,
    directions: Iterable[str] = ("up", "down", "left", "right"),
) -> List[np.ndarray]:
    """Return the four translated images (caller averages metrics across them)."""
    return [translate(image, pixels, d) for d in directions]
