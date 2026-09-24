"""Shared PACS training protocol helpers for Tasks 2 and 3.

Two assignment rules live here so every method obeys them the same way:

1. BatchNorm running statistics stay frozen at ImageNet values.
   Only BN scale/bias (γ, β) remain trainable.
2. Domain-balanced batches: equal influence from each source domain
   (and, in Task 2 UDA, equal total source vs target mass).
"""

from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def freeze_batchnorm_running_stats(module: nn.Module) -> None:
    """Put every BatchNorm* submodule into eval mode (no running-stat updates).

    Call this *after* ``model.train()`` so the rest of the network stays in
    train mode (Dropout, etc.) while BN uses fixed ImageNet running mean/var.

    Why: if BN updated on mixed source+target batches, the running stats would
    themselves become a hidden adaptation mechanism. The assignment wants the
    comparison to isolate the *explicit* alignment losses (MMD / DANN / CDAN).
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            # Keep γ/β trainable; only freeze the running buffers' *updates*
            # by staying in eval(). Parameters still receive gradients.
            for param_name in ("weight", "bias"):
                param = getattr(m, param_name, None)
                if param is not None:
                    param.requires_grad = True


def set_train_mode_with_frozen_bn(module: nn.Module) -> None:
    """``module.train()`` then freeze BN running stats (assignment policy)."""
    module.train()
    freeze_batchnorm_running_stats(module)


def set_eval_mode(module: nn.Module) -> None:
    """Full evaluation mode (BN + Dropout both eval)."""
    module.eval()


class _CyclicLoader:
    """Yield batches forever by restarting an exhausted DataLoader iterator."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self._iter: Iterator | None = None

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter is None:
            self._iter = iter(self.loader)
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


def cyclic_loader(loader: DataLoader) -> _CyclicLoader:
    return _CyclicLoader(loader)


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )


def unpack_batch(batch: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Move a PacsImageDataset batch to device; return (images, labels, domains)."""
    images = batch["image"].to(device, non_blocking=True)
    labels = batch["label"].to(device, non_blocking=True) if "label" in batch else None
    domains = batch["domain"].to(device, non_blocking=True) if "domain" in batch else None
    return images, labels, domains
