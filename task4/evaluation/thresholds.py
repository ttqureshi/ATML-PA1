"""Threshold from known validation unknownness only (no CIFAR-100)."""

from __future__ import annotations

import numpy as np


def threshold_from_val(
    u_val: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """τ = percentile of u on CIFAR-10 validation (known only).

    Accept when u(x) ≤ τ ⇒ aims to accept ``percentile``% of known val examples.
    """
    return float(np.percentile(u_val, percentile))
