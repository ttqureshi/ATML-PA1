"""OSR metrics: AUROC + validation-calibrated rejection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc_known_vs_unknown(u_known: np.ndarray, u_unknown: np.ndarray) -> float:
    """AUROC for ranking: unknown should have larger u than known.

    Labels: 0 = known, 1 = unknown. Score = u.
    """
    y = np.concatenate(
        [np.zeros(len(u_known), dtype=np.int32), np.ones(len(u_unknown), dtype=np.int32)]
    )
    s = np.concatenate([u_known, u_unknown])
    return float(roc_auc_score(y, s))


def closed_set_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """CSA from known-class logits only."""
    pred = logits.argmax(axis=1)
    return float((pred == labels).mean())


def rejection_at_threshold(
    u_known_test: np.ndarray,
    u_near: np.ndarray,
    u_far: np.ndarray,
    tau: float,
) -> dict:
    """Accept when u(x) ≤ τ (assignment).

    Returns known acceptance rate and near/far rejection rates + FPR@95TPR-style
    false-accept rates on unknowns.
    """
    known_accept = float((u_known_test <= tau).mean())
    near_reject = float((u_near > tau).mean())
    far_reject = float((u_far > tau).mean())
    # FPR@95TPR ≡ fraction of unknowns incorrectly accepted
    near_fpr = float((u_near <= tau).mean())
    far_fpr = float((u_far <= tau).mean())
    all_u = np.concatenate([u_near, u_far])
    all_fpr = float((all_u <= tau).mean())
    all_reject = float((all_u > tau).mean())
    return {
        "tau": float(tau),
        "known_test_accept_rate": known_accept,
        "near_reject_rate": near_reject,
        "far_reject_rate": far_reject,
        "all_unknown_reject_rate": all_reject,
        "near_fpr_at_95tpr": near_fpr,
        "far_fpr_at_95tpr": far_fpr,
        "all_unknown_fpr_at_95tpr": all_fpr,
    }
