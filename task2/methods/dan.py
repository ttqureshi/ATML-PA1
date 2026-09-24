"""DAN: Maximum Mean Discrepancy (MMD) between source and target features.

Intuition
---------
Imagine source features as one cloud of points and target features as another.
MMD asks: "how different are these two clouds?" in a reproducing kernel Hilbert
space. If the clouds overlap, MMD is near zero.

DAN adds λ * MMD² to the classification loss so the backbone is encouraged to
pull the clouds together — *marginal* alignment (it does not know class labels
on the target side).

Math (assignment form)
----------------------
    L_DAN = L_cls + λ_MMD * || E_s[φ(F(x_s))] − E_t[φ(F(x_t))] ||²_H

We never build φ explicitly. With an RBF kernel k, the unbiased batch estimate
of MMD² is:

    MMD² = mean_{i≠i'} k(s_i, s_i') + mean_{j≠j'} k(t_j, t_j')
           − 2 mean_{i,j} k(s_i, t_j)

Multi-kernel: sum of three RBFs with bandwidths 0.5, 1, 2 × median pairwise
squared distance in the *combined* source+target batch (assignment).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pairwise_squared_distances(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return ||x_i - y_j||² matrix of shape (n_x, n_y)."""
    # (x-y)^2 = x^2 + y^2 - 2 x·y
    x2 = (x * x).sum(dim=1, keepdim=True)
    y2 = (y * y).sum(dim=1, keepdim=True)
    return x2 + y2.T - 2.0 * (x @ y.T)


def median_heuristic_bandwidth(features: torch.Tensor) -> torch.Tensor:
    """Median pairwise squared distance within a feature batch (scalar tensor)."""
    n = features.size(0)
    if n < 2:
        return torch.ones((), device=features.device, dtype=features.dtype)
    dist = _pairwise_squared_distances(features, features)
    # Upper triangle only (exclude diagonal zeros).
    tri = dist[torch.triu(torch.ones(n, n, device=features.device, dtype=torch.bool), 1)]
    med = tri.median()
    # Avoid divide-by-zero if all points identical.
    return torch.clamp(med, min=1e-6)


def mmd_rbf(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    multipliers: tuple[float, ...] = (0.5, 1.0, 2.0),
) -> torch.Tensor:
    """Multi-kernel RBF MMD² between source and target feature batches."""
    combined = torch.cat([source, target], dim=0)
    base = median_heuristic_bandwidth(combined)

    kernels = []
    for m in multipliers:
        gamma = 1.0 / (2.0 * (m * base))  # bandwidth σ² = m * median
        # Within / across kernel matrices
        k_ss = torch.exp(-gamma * _pairwise_squared_distances(source, source))
        k_tt = torch.exp(-gamma * _pairwise_squared_distances(target, target))
        k_st = torch.exp(-gamma * _pairwise_squared_distances(source, target))
        kernels.append((k_ss, k_tt, k_st))

    # Sum kernels, then unbiased within-domain means (exclude diagonal).
    n_s = source.size(0)
    n_t = target.size(0)
    mmd = source.new_zeros(())
    for k_ss, k_tt, k_st in kernels:
        # Exclude diagonal: sum all − trace, divide by n(n-1)
        ss = (k_ss.sum() - k_ss.diag().sum()) / max(n_s * (n_s - 1), 1)
        tt = (k_tt.sum() - k_tt.diag().sum()) / max(n_t * (n_t - 1), 1)
        st = k_st.mean()
        mmd = mmd + (ss + tt - 2.0 * st)
    return mmd


class DANMethod(nn.Module):
    name = "dan"

    def __init__(
        self,
        lambda_mmd: float = 1.0,
        kernel_multipliers: list[float] | tuple[float, ...] = (0.5, 1.0, 2.0),
    ) -> None:
        super().__init__()
        self.lambda_mmd = float(lambda_mmd)
        self.kernel_multipliers = tuple(float(m) for m in kernel_multipliers)

    def total_loss(
        self,
        *,
        source_logits: torch.Tensor,
        source_labels: torch.Tensor,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        cls = F.cross_entropy(source_logits, source_labels)
        mmd = mmd_rbf(
            source_features,
            target_features,
            multipliers=self.kernel_multipliers,
        )
        loss = cls + self.lambda_mmd * mmd
        return {"loss": loss, "cls_loss": cls, "align_loss": mmd}
