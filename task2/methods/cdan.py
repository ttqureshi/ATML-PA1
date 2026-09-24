"""CDAN: Conditional Domain Adversarial Network.

Intuition
---------
Marginal alignment (DAN / DANN) can glue the wrong class regions together:
e.g. source-dogs align with target-horses if that shrinks the domain gap.

CDAN feeds the discriminator a *class-conditioned* signal:

    g(x) = vec(f ⊗ p)

where f is the 512-d feature and p = softmax(C(f)) is the classifier's soft
prediction. The outer product makes domain confusion class-aware: the disc
sees "what the model thinks this is" together with "how it looks".

Assignment constraints we follow strictly:
  - same disc width / dropout / GRL schedule / loss weight as DANN
  - do NOT detach f or p
  - do NOT use entropy conditioning
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from task2.models.domain_discriminator import DomainDiscriminator
from task2.models.grl import GradientReversal, grl_lambda


def outer_product_features(features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Full vectorized outer product g = vec(f ⊗ p), shape (B, D * C).

    For sample i: g_i = flatten(f_i ⊗ p_i). This is the *non-randomized*
    CDAN multilinear map (assignment does not ask for the random projection).
    """
    # (B, D, 1) * (B, 1, C) -> (B, D, C) -> (B, D*C)
    return torch.bmm(features.unsqueeze(2), probs.unsqueeze(1)).view(features.size(0), -1)


class CDANMethod(nn.Module):
    name = "cdan"

    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 7,
        lambda_domain: float = 1.0,
        grl_gamma: float = 10.0,
        grl_max: float = 1.0,
        disc_hidden: int = 256,
        disc_dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_domain = float(lambda_domain)
        self.grl_gamma = float(grl_gamma)
        self.grl_max = float(grl_max)
        self.grl = GradientReversal(alpha=0.0)
        self.discriminator = DomainDiscriminator(
            in_features=feature_dim * num_classes,
            hidden=disc_hidden,
            dropout=disc_dropout,
            num_domains=2,
        )

    def set_progress(self, progress: float) -> None:
        alpha = grl_lambda(progress, gamma=self.grl_gamma, max_value=self.grl_max)
        self.grl.set_alpha(alpha)

    def total_loss(
        self,
        *,
        source_logits: torch.Tensor,
        source_labels: torch.Tensor,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        target_logits: torch.Tensor,
        progress: float = 0.0,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        self.set_progress(progress)

        cls = F.cross_entropy(source_logits, source_labels)

        # Soft class predictions — NOT detached (assignment).
        source_probs = F.softmax(source_logits, dim=1)
        target_probs = F.softmax(target_logits, dim=1)

        # L2-normalize features before the outer product for discriminator stability.
        # (Same rationale as DANN; CE still uses raw logits.)
        source_f = F.normalize(source_features, p=2, dim=1)
        target_f = F.normalize(target_features, p=2, dim=1)

        source_g = outer_product_features(source_f, source_probs)
        target_g = outer_product_features(target_f, target_probs)
        g = torch.cat([source_g, target_g], dim=0)

        reversed_g = self.grl(g)
        domain_logits = self.discriminator(reversed_g)
        domain_labels = torch.cat(
            [
                torch.zeros(source_g.size(0), dtype=torch.long, device=g.device),
                torch.ones(target_g.size(0), dtype=torch.long, device=g.device),
            ],
            dim=0,
        )
        domain = F.cross_entropy(domain_logits, domain_labels)
        loss = cls + self.lambda_domain * domain
        return {
            "loss": loss,
            "cls_loss": cls,
            "align_loss": domain,
            "grl_alpha": torch.tensor(self.grl.alpha, device=g.device),
        }
