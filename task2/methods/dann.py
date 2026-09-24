"""DANN: domain-adversarial training with a binary domain discriminator.

Intuition
---------
Instead of measuring cloud distance with a kernel (DAN), we *train a small
network* to guess "source or target?" from the 512-d feature.

- Discriminator loss: get good at telling domains apart.
- Feature extractor (via GRL): make the discriminator fail.

At equilibrium, features are domain-confused (marginal alignment again), while
the classifier still uses source labels.

Only source examples contribute to L_cls.
Both source and target contribute to L_domain.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from task2.models.domain_discriminator import DomainDiscriminator
from task2.models.grl import GradientReversal, grl_lambda


class DANNMethod(nn.Module):
    name = "dann"

    def __init__(
        self,
        feature_dim: int = 512,
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
            in_features=feature_dim,
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
        progress: float = 0.0,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        self.set_progress(progress)

        cls = F.cross_entropy(source_logits, source_labels)

        # Domain labels: 0 = source, 1 = target
        feats = torch.cat([source_features, target_features], dim=0)
        # L2-normalize only for the discriminator path (stability under AdamW + GRL).
        # Classification still uses raw features/logits; MMD (DAN) is unchanged.
        feats_n = F.normalize(feats, p=2, dim=1)
        reversed_feats = self.grl(feats_n)
        domain_logits = self.discriminator(reversed_feats)
        domain_labels = torch.cat(
            [
                torch.zeros(source_features.size(0), dtype=torch.long, device=feats.device),
                torch.ones(target_features.size(0), dtype=torch.long, device=feats.device),
            ],
            dim=0,
        )
        domain = F.cross_entropy(domain_logits, domain_labels)
        loss = cls + self.lambda_domain * domain
        return {
            "loss": loss,
            "cls_loss": cls,
            "align_loss": domain,
            "grl_alpha": torch.tensor(self.grl.alpha, device=feats.device),
        }
