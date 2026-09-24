"""Source-domain separability (3-way) for Task 3 DG diagnostics.

Intuition
---------
Freeze the backbone and ask a *new* logistic regression: given only the 512-d
feature, can you tell Photo vs Art vs Cartoon?

- ~33.3% held-out accuracy → domains look interchangeable (strong invariance)
- ≫33.3% → domain identity still recoverable

Low separability ≠ success: features can be domain-confused *and* class-confused.
Always read next to source Acc/F1 and (later) Sketch.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shared.pacs import SOURCE_DOMAINS


def source_domain_separability_score(
    features_by_domain: dict[str, np.ndarray],
    *,
    test_size: float = 0.3,
    C: float = 1.0,
    seed: int = 6304,
) -> dict:
    """Multinomial LR over the three source domains. Chance = 1/3.

    ``features_by_domain`` maps domain name → (N_d, D) arrays. We balance by
    taking ``n = min_d N_d`` from each domain (assignment: balanced features).
    """
    missing = [d for d in SOURCE_DOMAINS if d not in features_by_domain]
    if missing:
        raise KeyError(f"Missing domains for separability: {missing}")

    arrays = {
        d: np.asarray(features_by_domain[d], dtype=np.float64) for d in SOURCE_DOMAINS
    }
    n = min(len(arrays[d]) for d in SOURCE_DOMAINS)
    if n < 2:
        raise ValueError("Need at least 2 features per source domain")

    rng = np.random.RandomState(seed)
    Xs, ys = [], []
    for domain_idx, domain in enumerate(SOURCE_DOMAINS):
        feats = arrays[domain]
        if len(feats) > n:
            idx = rng.choice(len(feats), size=n, replace=False)
            feats = feats[idx]
        Xs.append(feats)
        ys.append(np.full(n, domain_idx, dtype=np.int64))

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

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
        "chance": 1.0 / 3.0,
        "n_per_domain": int(n),
        "test_size": test_size,
        "C": C,
        "seed": seed,
        "domains": list(SOURCE_DOMAINS),
    }
