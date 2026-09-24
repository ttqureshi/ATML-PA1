"""Source-only ERM: supervised CE on labeled sources; ignore target images.

Intuition
---------
This is ordinary multi-domain supervised learning. We never look at Sketch
pixels during training. The Sketch score later tells us the *raw domain gap*:
how badly Photo/Art/Cartoon training fails on Sketch without any adaptation.

Also: this checkpoint is Task 3's ERM baseline — save it and reuse unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SourceOnlyMethod(nn.Module):
    """No alignment term; returns zero alignment loss for logging."""

    name = "source_only"

    def alignment_loss(self, **kwargs) -> torch.Tensor:
        # Keep a tensor on the right device for logging / backward no-ops.
        device = kwargs.get("device", torch.device("cpu"))
        return torch.zeros((), device=device)

    def total_loss(
        self,
        *,
        source_logits: torch.Tensor,
        source_labels: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        cls = F.cross_entropy(source_logits, source_labels)
        align = self.alignment_loss(device=cls.device)
        return {"loss": cls, "cls_loss": cls, "align_loss": align}
