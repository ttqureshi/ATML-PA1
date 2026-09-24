"""Local sharpness proxy Δsharp (assignment diagnostic).

Intuition
---------
Take a fixed source-val mini-batch. Measure CE loss at θ, then take one
normalized gradient-ascent step of radius ρ=0.05, and measure CE again.
    Δsharp = L(θ+ε) − L(θ),   ε = ρ ∇L / ||∇L||_2

A smaller Δsharp means the loss rose less under that nudge — evidence of
*local* stability under this specific probe, not a proof of global flatness.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from shared.pacs import SOURCE_DOMAINS
from shared.pacs_protocol import set_eval_mode


def build_fixed_sharpness_batch(
    source_val_examples_by_domain: dict[str, Sequence],
    *,
    per_domain: int = 32,
    seed: int = 6304,
) -> list:
    """Pick a reproducible list of PacsExample objects (32 per source domain)."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    selected = []
    for domain in SOURCE_DOMAINS:
        examples = list(source_val_examples_by_domain[domain])
        if len(examples) < per_domain:
            raise ValueError(
                f"{domain} val has {len(examples)} < {per_domain} examples"
            )
        # Sample without replacement via randperm.
        perm = torch.randperm(len(examples), generator=rng).tolist()
        for i in perm[:per_domain]:
            selected.append(examples[i])
    return selected


@torch.enable_grad()
def sharpness_proxy(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    rho: float = 0.05,
) -> dict:
    """Compute Δsharp on one batch. Model is placed in eval mode (assignment)."""
    set_eval_mode(model)
    params = [p for p in model.parameters() if p.requires_grad]

    # --- L(θ) and ∇L ---
    model.zero_grad(set_to_none=True)
    _, logits = model(images)
    loss0 = F.cross_entropy(logits, labels)
    loss0.backward()

    grads = torch.cat([p.grad.detach().flatten() for p in params if p.grad is not None])
    grad_norm = torch.norm(grads, p=2).clamp_min(1e-12)
    scale = float(rho) / float(grad_norm.item())

    backups: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    with torch.no_grad():
        for p in params:
            if p.grad is None:
                continue
            backups.append((p, p.data.clone()))
            p.add_(p.grad * scale)

    # --- L(θ+ε) ---
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        _, logits_pert = model(images)
        loss1 = F.cross_entropy(logits_pert, labels)

    # Restore
    with torch.no_grad():
        for p, data in backups:
            p.data.copy_(data)
    model.zero_grad(set_to_none=True)

    loss0_v = float(loss0.detach().item())
    loss1_v = float(loss1.detach().item())
    return {
        "delta_sharp": loss1_v - loss0_v,
        "loss_at_theta": loss0_v,
        "loss_at_theta_plus_eps": loss1_v,
        "rho": float(rho),
        "grad_norm": float(grad_norm.item()),
        "n": int(labels.numel()),
    }
