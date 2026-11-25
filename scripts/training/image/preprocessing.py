"""Shared, deterministic image preparation for the 32x32 CIFAKE models."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image, ImageOps


def prepare_model_rgb(
    image: Image.Image,
    *,
    size: int = 32,
    fill_rgb: Sequence[int] = (127, 127, 127),
) -> tuple[np.ndarray, dict[str, object]]:
    """Orient an image and fit it into a square without changing its aspect ratio.

    Native-size CIFAKE images pass through unchanged. Other dimensions are resized
    to fit and padded with the run's RGB training mean. Those resized inputs are
    still outside the model's validated resolution domain.
    """
    if size < 1:
        raise ValueError("size must be positive")
    if len(fill_rgb) != 3 or any(not 0 <= int(value) <= 255 for value in fill_rgb):
        raise ValueError("fill_rgb must contain three byte values")

    oriented = ImageOps.exif_transpose(image).convert("RGB")
    source_width, source_height = oriented.size
    if source_width < 1 or source_height < 1:
        raise ValueError("Image dimensions must be positive")

    if (source_width, source_height) == (size, size):
        return np.ascontiguousarray(np.asarray(oriented, dtype=np.uint8)), {
            "method": "native_size",
            "source_width": source_width,
            "source_height": source_height,
            "resized_width": size,
            "resized_height": size,
            "padding": {"top": 0, "bottom": 0, "left": 0, "right": 0},
        }

    scale = min(size / source_width, size / source_height)
    resized_width = max(1, min(size, int(round(source_width * scale))))
    resized_height = max(1, min(size, int(round(source_height * scale))))
    resized = oriented.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2
    right = size - resized_width - left
    bottom = size - resized_height - top
    canvas = Image.new("RGB", (size, size), tuple(int(value) for value in fill_rgb))
    canvas.paste(resized, (left, top))
    return np.ascontiguousarray(np.asarray(canvas, dtype=np.uint8)), {
        "method": "aspect_preserving_bilinear_letterbox",
        "source_width": source_width,
        "source_height": source_height,
        "resized_width": resized_width,
        "resized_height": resized_height,
        "padding": {"top": top, "bottom": bottom, "left": left, "right": right},
    }
