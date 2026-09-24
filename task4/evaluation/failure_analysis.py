"""Failure analysis: incorrectly accepted unknowns under Vanilla MLS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from task4.data.constants import CIFAR10_CLASSES


def collect_false_accepts(
    *,
    u: np.ndarray,
    tau: float,
    pred_known: np.ndarray,
    unknown_class_names: Sequence[str],
    group: str,
    top_k: int = 3,
) -> list[dict]:
    """Pick incorrectly accepted unknowns (u ≤ τ), preferring high confidence
    (lowest u among accepts) as informative overconfident failures.
    """
    accepted = np.where(u <= tau)[0]
    if len(accepted) == 0:
        return []
    # Most confidently accepted unknowns = smallest u among accepts.
    order = accepted[np.argsort(u[accepted])]
    picks = order[:top_k]
    rows = []
    for i in picks:
        pred = int(pred_known[i])
        rows.append(
            {
                "group": group,
                "unknown_class": str(unknown_class_names[i]),
                "predicted_cifar10": CIFAR10_CLASSES[pred],
                "predicted_cifar10_id": pred,
                "unknownness_u": float(u[i]),
                "threshold_tau": float(tau),
                "margin_below_tau": float(tau - u[i]),
                "semantically_plausible_hint": _plausibility_hint(
                    str(unknown_class_names[i]), CIFAR10_CLASSES[pred]
                ),
            }
        )
    return rows


def _plausibility_hint(unk: str, pred: str) -> str:
    """Coarse heuristic for report discussion (not used for any decision)."""
    vehicle_unk = {"bus", "pickup_truck", "motorcycle", "tractor"}
    vehicle_known = {"automobile", "truck"}
    animal_unk = {"wolf", "fox", "leopard", "camel"}
    animal_known = {"dog", "cat", "horse", "deer"}
    if unk in vehicle_unk and pred in vehicle_known:
        return "plausible_vehicle_confusion"
    if unk in animal_unk and pred in animal_known:
        return "plausible_animal_confusion"
    return "surprising_or_weak_semantic_link"


def save_failures(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"failures": rows, "n": len(rows)}, f, indent=2)
