"""Gradient Reversal Layer (GRL) for DANN / CDAN.

Intuition
---------
A domain discriminator wants features that *reveal* domain (source vs target).
The feature extractor wants the opposite: features that *hide* domain.

GRL sits between them. Forward pass: identity (features unchanged).
Backward pass: multiply gradients by −α, so the extractor is pushed the
opposite way from the discriminator.

Schedule (assignment / Ganin et al.):
    α(p) = 2 / (1 + exp(−γ p)) − 1,   p ∈ [0, 1] = training progress.

Early training: α ≈ 0 → mostly learn the class task.
Late training:  α → 1 → stronger domain-confusion pressure.
"""

from __future__ import annotations

import torch
from torch.autograd import Function


class _GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Reverse and scale the incoming gradient.
        return -ctx.alpha * grad_output, None


class GradientReversal(torch.nn.Module):
    """Module wrapper around the GRL autograd Function."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.alpha)


def grl_lambda(progress: float, gamma: float = 10.0, max_value: float = 1.0) -> float:
    """Compute α(p) and optionally scale by ``max_value`` (controlled study).

    ``progress`` must be in [0, 1]. ``max_value`` defaults to 1.0 for the main
    DANN/CDAN runs; the optional GRL study would use {0.25, 0.5, 1.0}.
    """
    progress = float(min(max(progress, 0.0), 1.0))
    alpha = 2.0 / (1.0 + torch.exp(torch.tensor(-gamma * progress))) - 1.0
    return float(alpha.item()) * float(max_value)
