"""Per-class Sketch analysis and confusion helpers.

Use Sketch *labels only here* — after all checkpoints and settings are fixed.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from shared.pacs import CLASS_NAMES


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {}
    for c in range(num_classes):
        mask = y_true == c
        n = int(mask.sum())
        acc = float((y_pred[mask] == c).mean()) if n else float("nan")
        out[CLASS_NAMES[c]] = {"accuracy": acc, "n": n}
    return out


def accuracy_deltas(
    baseline_per_class: dict, adapted_per_class: dict
) -> dict[str, float]:
    deltas = {}
    for name in CLASS_NAMES:
        b = baseline_per_class[name]["accuracy"]
        a = adapted_per_class[name]["accuracy"]
        deltas[name] = float(a - b)
    return deltas


def top_confusions(
    y_true: np.ndarray, y_pred: np.ndarray, k: int = 10
) -> list[dict]:
    """Most frequent (true → predicted) mistakes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    wrong = y_true != y_pred
    pairs = [
        (CLASS_NAMES[int(t)], CLASS_NAMES[int(p)])
        for t, p in zip(y_true[wrong], y_pred[wrong])
    ]
    counts = Counter(pairs)
    rows = []
    for (true_name, pred_name), count in counts.most_common(k):
        rows.append({"true": true_name, "pred": pred_name, "count": int(count)})
    return rows


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 7) -> list[list[int]]:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        cm[int(t), int(p)] += 1
    return cm.tolist()
