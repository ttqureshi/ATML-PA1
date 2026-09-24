"""Mahalanobis unknownness (shared diagonal covariance)."""

from __future__ import annotations

import numpy as np


def fit_mahalanobis(
    features: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 10,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate class means + shared diagonal covariance from train features.

    Returns (means (C,D), inv_diag (D,)) with eps added to every diagonal entry.
    """
    d = features.shape[1]
    means = np.zeros((num_classes, d), dtype=np.float64)
    centered = []
    for c in range(num_classes):
        feats_c = features[labels == c]
        if len(feats_c) == 0:
            raise ValueError(f"No training features for class {c}")
        means[c] = feats_c.mean(axis=0)
        centered.append(feats_c - means[c])
    centered = np.concatenate(centered, axis=0)
    var = centered.var(axis=0) + eps
    inv_diag = 1.0 / var
    return means.astype(np.float64), inv_diag.astype(np.float64)


def mahalanobis_unknownness(
    features: np.ndarray,
    means: np.ndarray,
    inv_diag: np.ndarray,
) -> np.ndarray:
    """u_Mah = min_c (f − μ_c)ᵀ Σ^{-1} (f − μ_c) with shared diagonal Σ."""
    diff = features[:, None, :] - means[None, :, :]
    dist = (diff * diff * inv_diag[None, None, :]).sum(axis=2)
    return dist.min(axis=1)
