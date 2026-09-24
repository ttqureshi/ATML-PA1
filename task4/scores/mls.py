"""MLS unknownness: u = -max logit."""

from __future__ import annotations

import numpy as np


def mls_unknownness(logits: np.ndarray) -> np.ndarray:
    """u_MLS = − max_k z_k(x)."""
    return -logits.max(axis=1)
