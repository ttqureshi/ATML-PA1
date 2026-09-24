"""ResNet-18 backbone that returns the 512-d feature *and* class logits.

Why we return both:
  - Classification loss uses logits (and softmax probabilities for CDAN).
  - Alignment losses (MMD / domain disc) use the 512-d feature before the head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from task2.models.classifier_head import ClassifierHead


class ResNet18Classifier(nn.Module):
    """Full fine-tune ResNet-18 with a replaceable 7-class head."""

    def __init__(self, num_classes: int = 7, feature_dim: int = 512) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        base = resnet18(weights=weights)
        # Keep everything except the original ImageNet fc.
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.feature_dim = feature_dim
        self.classifier = ClassifierHead(feature_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened 512-d features."""
        feats = self.feature_extractor(x)  # (B, 512, 1, 1)
        return torch.flatten(feats, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (features, logits)."""
        feats = self.forward_features(x)
        logits = self.classifier(feats)
        return feats, logits
