"""MSP unknownness: u = 1 - max softmax probability."""

from __future__ import annotations

import numpy as np


def msp_unknownness(logits: np.ndarray) -> np.ndarray:
    """u_MSP = 1 − max_k p_k(x)."""
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    p = exp / exp.sum(axis=1, keepdims=True)
    return 1.0 - p.max(axis=1)
