"""Linear classifier head on top of the 512-d ResNet feature."""

from __future__ import annotations

import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 7) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, features):
        return self.fc(features)
