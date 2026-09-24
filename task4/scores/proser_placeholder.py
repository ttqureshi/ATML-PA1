"""PROSER placeholder detection score (reference DeltaP)."""

from __future__ import annotations

import numpy as np


def proser_placeholder_unknownness(
    known_logits: np.ndarray,
    dummy_logits: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Build [z_known, max_dummy], softmax, then u = p_dummy − max_k p_known.

    Larger ⇒ more unknown-like. Calibrate τ on CIFAR-10 val only.
    """
    max_dummy = dummy_logits.max(axis=1, keepdims=True)
    logits = np.concatenate([known_logits, max_dummy], axis=1)
    logits = logits / float(temperature)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    p = exp / exp.sum(axis=1, keepdims=True)
    p_dummy = p[:, -1]
    p_known_max = p[:, :-1].max(axis=1)
    return p_dummy - p_known_max
