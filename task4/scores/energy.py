"""Energy unknownness: u = -logsumexp(logits)."""

from __future__ import annotations

import numpy as np


def energy_unknownness(logits: np.ndarray) -> np.ndarray:
    """u_Energy = − log ∑_k exp(z_k)."""
    m = logits.max(axis=1, keepdims=True)
    logsumexp = m.squeeze(1) + np.log(np.exp(logits - m).sum(axis=1))
    return -logsumexp
