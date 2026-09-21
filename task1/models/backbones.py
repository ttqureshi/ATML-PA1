"""Frozen pretrained backbones and linear classifier heads for Task 1.

Backbones (assignment):
  - torchvision ResNet-50  (IMAGENET1K_V2)  -> GAP feature
  - torchvision ViT-B/16   (IMAGENET1K_V1)  -> final class token
  - OpenCLIP ViT-B-32      (pretrained=openai) -> L2-normalized image embedding

ASSUMPTION: backbones stay frozen; only a linear head is trained (except
CLIP zero-shot, which uses text prompts and no trained head).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from task1.data.make_subset import STL10_CLASSES
from task1.data.transforms import to_pil


@dataclass
class BackboneInfo:
    name: str
    feature_dim: int
    # Callable: PIL/np RGB -> normalized CHW float tensor for this backbone.
    preprocess: Callable


def _resize_center_crop(img: Image.Image, size: int = 224) -> Image.Image:
    # Common resize-then-center-crop used by torchvision weights.
    img = TF.resize(img, size, interpolation=InterpolationMode.BICUBIC)
    img = TF.center_crop(img, size)
    return img


def imagenet_preprocess(image, size: int = 224) -> torch.Tensor:
    img = to_pil(image)
    img = _resize_center_crop(img, size)
    t = TF.to_tensor(img)
    t = TF.normalize(t, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return t


def clip_preprocess(image, size: int = 224) -> torch.Tensor:
    """OpenAI CLIP normalization (distinct from ImageNet)."""
    img = to_pil(image)
    img = _resize_center_crop(img, size)
    t = TF.to_tensor(img)
    t = TF.normalize(
        t,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    return t


class FrozenResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        net = models.resnet50(weights=weights)
        # Everything except the final FC; GAP is already applied in forward below.
        self.stem = nn.Sequential(*list(net.children())[:-1])  # -> (B, 2048, 1, 1)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.feature_dim = 2048

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        return feat.flatten(1)


class FrozenViTB16(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        self.net = models.vit_b_16(weights=weights)
        # Remove classifier head; keep encoder + class token pathway.
        self.net.heads = nn.Identity()
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.feature_dim = 768

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # torchvision ViT with Identity heads returns the final class-token embedding.
        return self.net(x)


class FrozenCLIPViTB32(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "open_clip is required for CLIP. Install with: pip install open-clip-torch"
            ) from e

        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model = model
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.feature_dim = model.visual.output_dim

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.model.encode_image(x)
        return F.normalize(feat, dim=-1)

    @torch.no_grad()
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        feat = self.model.encode_text(text_tokens)
        return F.normalize(feat, dim=-1)


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def build_backbone(name: str, device: torch.device) -> Tuple[nn.Module, BackboneInfo]:
    name = name.lower()
    if name == "resnet50":
        model = FrozenResNet50().to(device)
        info = BackboneInfo("resnet50", model.feature_dim, imagenet_preprocess)
        return model, info
    if name == "vit_b_16":
        model = FrozenViTB16().to(device)
        info = BackboneInfo("vit_b_16", model.feature_dim, imagenet_preprocess)
        return model, info
    if name in ("clip_vit_b_32", "clip"):
        model = FrozenCLIPViTB32().to(device)
        info = BackboneInfo("clip_vit_b_32", model.feature_dim, clip_preprocess)
        return model, info
    raise ValueError(f"Unknown backbone: {name}")


@torch.no_grad()
def clip_zeroshot_logits(
    clip_model: FrozenCLIPViTB32,
    image_features: torch.Tensor,
    class_names: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Zero-shot logits from cosine similarities with fixed prompt.

    Prompt (assignment): 'a photo of a {class}.'
    Confidence later uses softmax over scaled similarities.
    """
    import open_clip

    class_names = class_names or STL10_CLASSES
    device = device or image_features.device
    templates = [f"a photo of a {c}." for c in class_names]
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    tokens = tokenizer(templates).to(device)
    text_features = clip_model.encode_text(tokens)
    # CLIP logit scale is learnable; use the model's scale.
    scale = clip_model.model.logit_scale.exp()
    logits = scale * image_features @ text_features.t()
    return logits


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
