"""Manifold mixup for PROSER data placeholders (after layer2)."""

from __future__ import annotations

import torch


def manifold_mixup_layer2(
    h: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix layer2 features from *different* classes within a mini-batch.

    Assignment: λ ~ Beta(α, α) with α=2 → Beta(2,2).
    Returns (mixed_h, y_a, y_b) for the successfully mixed pairs.
    If a sample's shuffle partner has the same label, we re-sample partner
    indices until different (or fall back to a random different-class index).
    """
    b = h.size(0)
    if b < 2:
        raise ValueError("Need batch size >= 2 for manifold mixup.")

    device = h.device
    # Start from a random permutation; repair same-class collisions.
    perm = torch.randperm(b, device=device)
    for i in range(b):
        if y[perm[i]] != y[i]:
            continue
        # Find any index with a different label.
        candidates = (y != y[i]).nonzero(as_tuple=False).view(-1)
        if len(candidates) == 0:
            # Degenerate batch (all one class) — leave unpaired; caller should skip.
            continue
        # Prefer swapping with a different-class partner still in perm if possible.
        perm[i] = candidates[torch.randint(len(candidates), (1,), device=device)]

    # Keep only different-class pairs.
    mask = y != y[perm]
    if mask.sum() == 0:
        # Extremely rare for CIFAR-10 bs=64 half-batch; return empty-safe mix of self.
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
        return h, y, y

    h_a = h[mask]
    h_b = h[perm][mask]
    y_a = y[mask]
    y_b = y[perm][mask]

    lam = torch.distributions.Beta(alpha, alpha).sample().to(device)
    # Broadcast λ over feature dims.
    while lam.dim() < h_a.dim():
        lam = lam.view(-1, *([1] * (h_a.dim() - 1)))
    mixed = lam * h_a + (1.0 - lam) * h_b
    return mixed, y_a, y_b
