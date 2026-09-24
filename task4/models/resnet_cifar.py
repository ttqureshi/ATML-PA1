"""CIFAR-appropriate ResNet-18 (32×32, no ImageNet stem).

Changes vs torchvision ImageNet ResNet-18 (assignment):
- first conv: 3×3, stride 1, padding 1 (not 7×7 stride 2)
- remove initial max-pool
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18CIFAR(nn.Module):
    """ResNet-18 for CIFAR with optional dummy (PROSER) heads."""

    def __init__(self, num_classes: int = 10, num_dummy: int = 0):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        in_features = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()  # we own the heads

        self.backbone = backbone
        self.feature_dim = in_features
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)
        self.num_dummy = int(num_dummy)
        self.fc_dummy = (
            nn.Linear(in_features, self.num_dummy) if self.num_dummy > 0 else None
        )

    def forward_pre(self, x: torch.Tensor) -> torch.Tensor:
        """ϕ_pre: through layer2 (for PROSER manifold mixup)."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        return x

    def forward_from_pre(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ϕ_post + heads from a layer2 feature map."""
        x = self.backbone.layer3(h)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        feat = torch.flatten(x, 1)
        logits = self.fc(feat)
        return feat, logits

    def forward_dummy_from_pre(self, h: torch.Tensor) -> torch.Tensor:
        if self.fc_dummy is None:
            raise RuntimeError("No dummy head attached.")
        x = self.backbone.layer3(h)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        feat = torch.flatten(x, 1)
        return self.fc_dummy(feat)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (penultimate features f(x), known-class logits z(x))."""
        h = self.forward_pre(x)
        return self.forward_from_pre(h)

    def forward_dummy(self, x: torch.Tensor) -> torch.Tensor:
        if self.fc_dummy is None:
            raise RuntimeError("No dummy head attached.")
        feat, _ = self.forward_features(x)
        return self.fc_dummy(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, logits = self.forward_features(x)
        return logits

    def attach_dummy_heads(self, num_dummy: int = 5) -> None:
        """Append randomly initialized dummy classifiers (PROSER)."""
        self.num_dummy = int(num_dummy)
        self.fc_dummy = nn.Linear(self.feature_dim, self.num_dummy)
        nn.init.kaiming_normal_(self.fc_dummy.weight, mode="fan_out", nonlinearity="relu")
        if self.fc_dummy.bias is not None:
            nn.init.zeros_(self.fc_dummy.bias)


def build_resnet18_cifar(num_classes: int = 10, num_dummy: int = 0) -> ResNet18CIFAR:
    return ResNet18CIFAR(num_classes=num_classes, num_dummy=num_dummy)
