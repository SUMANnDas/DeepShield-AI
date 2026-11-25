"""Shared project paths, reproducibility helpers, and runtime metadata."""

from __future__ import annotations

import importlib.metadata
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def set_reproducible_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch before creating loaders or models."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Force the deterministic SDPA math implementation. PyTorch's
        # memory-efficient attention backward may otherwise be nondeterministic
        # on CUDA and emit a warning while violating the experiment setting.
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
                torch.backends.cuda.enable_cudnn_sdp(False)
        torch.use_deterministic_algorithms(True, warn_only=False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    return device


def software_versions() -> dict[str, str | None]:
    names = (
        "torch", "torchvision", "numpy", "scipy", "opencv-python-headless",
        "Pillow", "scikit-learn", "matplotlib", "tqdm", "httpx",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions["python"] = sys.version.split()[0]
    versions["cuda_runtime"] = torch.version.cuda
    return versions


def gpu_metadata() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "device_count": 0}
    properties = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
        "total_memory_mib": round(properties.total_memory / (1024**2)),
    }
