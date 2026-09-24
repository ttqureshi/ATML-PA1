"""Domain-separability diagnostic.

Intuition
---------
After adaptation, freeze the backbone and ask a *new* logistic regression:
"given only the 512-d feature, can you tell source from target?"

- ~50% held-out accuracy → domains look alike (hard to tell apart)
- ≫50% → domain information still recoverable

Important: low separability ≠ successful adaptation. Features could be
domain-confused *and* class-confused (collapsed). Always read this score
next to Sketch accuracy / macro-F1.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def domain_separability_score(
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    test_size: float = 0.3,
    C: float = 1.0,
    seed: int = 6304,
) -> dict:
    """Binary LR: source=0 vs target=1. Returns held-out accuracy and chance."""
    source_features = np.asarray(source_features, dtype=np.float64)
    target_features = np.asarray(target_features, dtype=np.float64)

    # Equal counts (assignment): subsample the larger set.
    n = min(len(source_features), len(target_features))
    rng = np.random.RandomState(seed)
    if len(source_features) > n:
        idx = rng.choice(len(source_features), size=n, replace=False)
        source_features = source_features[idx]
    if len(target_features) > n:
        idx = rng.choice(len(target_features), size=n, replace=False)
        target_features = target_features[idx]

    X = np.concatenate([source_features, target_features], axis=0)
    y = np.concatenate(
        [np.zeros(n, dtype=np.int64), np.ones(n, dtype=np.int64)], axis=0
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    clf.fit(X_train, y_train)
    acc = float(clf.score(X_test, y_test))
    return {
        "held_out_accuracy": acc,
        "chance": 0.5,
        "n_per_domain": int(n),
        "test_size": test_size,
        "C": C,
        "seed": seed,
    }
