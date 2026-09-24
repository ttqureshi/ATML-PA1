"""DAN-DG: average pairwise MMD across the three *source* domains.

Intuition
---------
Task 2 DAN pulls *source* and *Sketch* feature clouds together.
DAN-DG never sees Sketch. Instead it asks: "make Photo, Art, and Cartoon
look alike in feature space." If Sketch benefits, invariance learned only from
observed domains transferred; if Sketch hurts while source separability drops,
we may have erased class signal (negative transfer of the DG kind).

Math (assignment)
-----------------
    L_DAN-DG = L_ERM + (λ_DG / 3) * Σ_{e < e'} MMD²(F(X_e), F(X_e'))

Unordered pairs: Photo–Art, Photo–Cartoon, Art–Cartoon.
MMD = same multi-kernel RBF as Task 2 (import ``mmd_rbf``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.pacs import SOURCE_DOMAINS
from task2.methods.dan import mmd_rbf


def split_domain_features(
    features: torch.Tensor,
    *,
    per_domain: int,
) -> dict[str, torch.Tensor]:
    """Undo domain-balanced concat (SOURCE_DOMAINS order, ``per_domain`` each)."""
    expect = per_domain * len(SOURCE_DOMAINS)
    if features.size(0) != expect:
        raise ValueError(
            f"Expected batch size {expect} (8×3 sources), got {features.size(0)}"
        )
    out = {}
    for i, domain in enumerate(SOURCE_DOMAINS):
        out[domain] = features[i * per_domain : (i + 1) * per_domain]
    return out


class DANDGMethod(nn.Module):
    name = "dan_dg"

    def __init__(
        self,
        lambda_dg: float = 1.0,
        kernel_multipliers: list[float] | tuple[float, ...] = (0.5, 1.0, 2.0),
        source_per_domain: int = 8,
    ) -> None:
        super().__init__()
        self.lambda_dg = float(lambda_dg)
        self.kernel_multipliers = tuple(float(m) for m in kernel_multipliers)
        self.source_per_domain = int(source_per_domain)

    def pairwise_mmd(self, source_features: torch.Tensor) -> torch.Tensor:
        """Average MMD² over the three unordered source pairs."""
        by_domain = split_domain_features(
            source_features, per_domain=self.source_per_domain
        )
        domains = list(SOURCE_DOMAINS)
        mmds = []
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                mmds.append(
                    mmd_rbf(
                        by_domain[domains[i]],
                        by_domain[domains[j]],
                        multipliers=self.kernel_multipliers,
                    )
                )
        return sum(mmds) / float(len(mmds))

    def total_loss(
        self,
        *,
        source_logits: torch.Tensor,
        source_labels: torch.Tensor,
        source_features: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        cls = F.cross_entropy(source_logits, source_labels)
        mmd = self.pairwise_mmd(source_features)
        loss = cls + self.lambda_dg * mmd
        return {"loss": loss, "cls_loss": cls, "align_loss": mmd}
