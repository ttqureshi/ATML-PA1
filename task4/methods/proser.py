"""PROSER: classifier placeholders + data placeholders (Zhou et al., 2021).

Loss (assignment / paper):
  l1 = CE(f̂(x), y) + β · CE(f̂(x) \\ y, K+1)     # first half of batch
  l2 = CE(f̂(φ_post(x̃_pre)), K+1)                 # second half (mixup)
  l_total = l1 + γ · l2

with β=1, γ=0.1, C=5 dummy heads, mixup after layer2, λ ~ Beta(2,2).

f̂(x) = [z_known(x), max_c z_dummy,c(x)].
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from task4.methods.manifold_mixup import manifold_mixup_layer2


def augmented_logits(known: torch.Tensor, dummy: torch.Tensor) -> torch.Tensor:
    """[known logits, max dummy logit] → shape (B, K+1)."""
    max_dummy, _ = dummy.max(dim=1, keepdim=True)
    return torch.cat([known, max_dummy], dim=1)


def classifier_placeholder_loss(
    known_logits: torch.Tensor,
    dummy_logits: torch.Tensor,
    y: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Eq. 5: CE on augmented logits + β · CE after masking the true class."""
    k = known_logits.size(1)
    f_hat = augmented_logits(known_logits, dummy_logits)
    loss_ce = F.cross_entropy(f_hat, y)

    # Mask true-class logit so the dummy slot should win among remaining.
    f_masked = f_hat.clone()
    f_masked[torch.arange(f_masked.size(0), device=y.device), y] = -1e9
    dummy_target = torch.full_like(y, fill_value=k)
    loss_dummy = F.cross_entropy(f_masked, dummy_target)
    return loss_ce + beta * loss_dummy


def data_placeholder_loss(
    known_logits: torch.Tensor,
    dummy_logits: torch.Tensor,
    y_dummy_index: int,
) -> torch.Tensor:
    """Eq. 7: train mixed representations toward the dummy (K+1) slot."""
    f_hat = augmented_logits(known_logits, dummy_logits)
    target = torch.full(
        (f_hat.size(0),),
        fill_value=y_dummy_index,
        dtype=torch.long,
        device=f_hat.device,
    )
    return F.cross_entropy(f_hat, target)


def proser_batch_loss(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    beta: float = 1.0,
    gamma: float = 0.1,
    mixup_alpha: float = 2.0,
) -> dict:
    """Split batch in half: first → classifier placeholders; second → mixup."""
    b = images.size(0)
    half = b // 2
    if half < 2:
        raise ValueError("PROSER needs batch size >= 4 (two halves with mixup).")

    x1, y1 = images[:half], labels[:half]
    x2, y2 = images[half : 2 * half], labels[half : 2 * half]
    k = model.num_classes

    # --- l1: classifier placeholders on first half ---
    feat1, z1 = model.forward_features(x1)
    d1 = model.fc_dummy(feat1)
    l1 = classifier_placeholder_loss(z1, d1, y1, beta=beta)

    # --- l2: manifold mixup data placeholders on second half ---
    h2 = model.forward_pre(x2)
    h_mix, _, _ = manifold_mixup_layer2(h2, y2, alpha=mixup_alpha)
    feat_m, z_m = model.forward_from_pre(h_mix)
    d_m = model.fc_dummy(feat_m)
    l2 = data_placeholder_loss(z_m, d_m, y_dummy_index=k)

    total = l1 + gamma * l2
    return {"loss": total, "l1": l1.detach(), "l2": l2.detach()}
