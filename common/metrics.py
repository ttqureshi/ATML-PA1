"""Simple classification metrics used by Task 1."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def mean_max_confidence(probs: np.ndarray) -> float:
    """Mean of max softmax probability (or equivalent confidence)."""
    probs = np.asarray(probs)
    return float(probs.max(axis=1).mean())


def prediction_consistency(y_clean: np.ndarray, y_transformed: np.ndarray) -> float:
    """Fraction of images whose predicted class is unchanged after an intervention."""
    y_clean = np.asarray(y_clean)
    y_transformed = np.asarray(y_transformed)
    return float((y_clean == y_transformed).mean())
