"""Task 4 training methods (thin markers + PROSER losses)."""

from .proser import proser_batch_loss, augmented_logits
from .manifold_mixup import manifold_mixup_layer2

__all__ = ["proser_batch_loss", "augmented_logits", "manifold_mixup_layer2"]
