from __future__ import annotations

from typing import Sequence

import torch
from torchvision.transforms import Normalize

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def normalize_faces(
    x: torch.Tensor,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    return Normalize(mean, std)(x)


def normalize_masked_sequence(
    frames: torch.Tensor,
    mask: torch.Tensor,
    *,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> torch.Tensor:
    """Normalize valid face crops while keeping padded frame slots zeroed."""
    normalized = frames.clone()
    if mask.any():
        normalized[mask] = normalize_faces(normalized[mask], mean=mean, std=std)
    return normalized
