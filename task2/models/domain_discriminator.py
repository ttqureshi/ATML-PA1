"""Domain discriminator used by DANN and CDAN.

Architecture (assignment):
  Linear(in_dim → 256) → ReLU → Dropout(0.5) → Linear(256 → 2)

Input dimension:
  - DANN: 512 (raw feature f)
  - CDAN: 512 * 7 = 3584 (vectorized outer product f ⊗ p)
"""

from __future__ import annotations

import torch.nn as nn


class DomainDiscriminator(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden: int = 256,
        dropout: float = 0.5,
        num_domains: int = 2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_domains),
        )
        # Small last-layer init reduces early over-confident domain logits
        # (helps prevent the GRL feedback loop from exploding).
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.constant_(self.net[-1].bias, 0.0)

    def forward(self, x):
        return self.net(x)
