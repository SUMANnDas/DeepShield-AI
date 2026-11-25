"""Native-resolution CIFAKE transforms implemented with NumPy and PyTorch."""

from __future__ import annotations

import random

import numpy as np
import torch


class ImageTransform:
    """Convert RGB uint8 images to normalized tensors with mild train augmentation."""

    def __init__(
        self,
        mean: tuple[float, float, float] | list[float],
        std: tuple[float, float, float] | list[float],
        *,
        training: bool = False,
        horizontal_flip_probability: float = 0.0,
        crop_padding: int = 0,
    ) -> None:
        if len(mean) != 3 or len(std) != 3 or any(float(value) <= 0 for value in std):
            raise ValueError("mean and std must contain three channels and positive std values")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be between 0 and 1")
        if crop_padding < 0 or crop_padding > 4:
            raise ValueError("crop_padding must be between 0 and 4 pixels")
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.training = training
        self.horizontal_flip_probability = horizontal_flip_probability if training else 0.0
        self.crop_padding = crop_padding if training else 0

    def __call__(self, image_rgb: np.ndarray) -> torch.Tensor:
        if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("Expected a uint8 RGB image with shape (height, width, 3)")
        image = image_rgb
        if self.training and self.crop_padding:
            pad = self.crop_padding
            padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
            max_offset = 2 * pad
            top = random.randint(0, max_offset)
            left = random.randint(0, max_offset)
            height, width = image.shape[:2]
            image = padded[top:top + height, left:left + width]
        if self.training and random.random() < self.horizontal_flip_probability:
            image = image[:, ::-1]
        contiguous = np.ascontiguousarray(image.transpose(2, 0, 1))
        tensor = torch.from_numpy(contiguous).to(dtype=torch.float32).div_(255.0)
        return (tensor - self.mean) / self.std
