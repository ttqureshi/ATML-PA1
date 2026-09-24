"""SAM: Sharpness-Aware Minimization on the ERM classification loss.

Intuition
---------
Ordinary SGD finds a point θ that has low source loss. SAM asks for a point
whose *neighborhood* also has low loss: if a small nudge still keeps L small,
the solution is "flat" / locally stable. The hope in DG is that a stable source
solution transfers better to an unseen visual style (Sketch) — without ever
aligning domains explicitly.

Math (assignment, non-adaptive)
-------------------------------
    min_θ  max_{||ε||_2 ≤ ρ}  L_ERM(θ + ε)

Per batch:
  1. Forward/backward at θ → g = ∇L(θ)
  2. ε = ρ * g / ||g||_2
  3. Forward/backward at θ+ε → gradients used for the AdamW update
  4. Restore θ, then optimizer.step()

Frozen-BN policy (Task 2/3 shared): call ``set_train_mode_with_frozen_bn``
before *both* passes so BN running stats stay at ImageNet values.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMMethod(nn.Module):
    """ERM + SAM wrapper. Alignment loss is always zero (for logging parity)."""

    name = "sam"

    def __init__(self, rho: float = 0.05) -> None:
        super().__init__()
        self.rho = float(rho)

    def total_loss(
        self,
        *,
        source_logits: torch.Tensor,
        source_labels: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        cls = F.cross_entropy(source_logits, source_labels)
        align = torch.zeros((), device=cls.device)
        return {"loss": cls, "cls_loss": cls, "align_loss": align}


@contextmanager
def sam_weight_perturbation(parameters: Iterable[torch.nn.Parameter], rho: float):
    """Ascent ε = ρ g / ||g||, apply to weights, yield, then restore.

    Call *after* the first ``loss.backward()`` so ``.grad`` is populated.
    """
    params = [p for p in parameters if p.grad is not None]
    if not params:
        yield
        return

    # Global L2 norm of the concatenated gradient.
    grads = torch.cat([p.grad.detach().flatten() for p in params])
    grad_norm = torch.norm(grads, p=2).clamp_min(1e-12)
    scale = float(rho) / float(grad_norm.item())

    backups: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    with torch.no_grad():
        for p in params:
            eps = p.grad * scale
            backups.append((p, p.data.clone()))
            p.add_(eps)

    try:
        yield
    finally:
        with torch.no_grad():
            for p, data in backups:
                p.data.copy_(data)
