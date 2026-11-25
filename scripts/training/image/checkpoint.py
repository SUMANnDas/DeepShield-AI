"""Append-only checkpoint writing and safe recovery helpers."""

from __future__ import annotations

import os
import random
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint_exclusive(path: Path, payload: dict[str, Any]) -> Path:
    """Write to a unique temporary path and atomically publish without replacing files."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        raise FileExistsError(f"Checkpoint target appeared during save; temporary retained at {temporary}")
    # On Windows os.rename fails when the destination already exists.
    os.rename(temporary, path)
    return path


def load_checkpoint(path: Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or "model_state" not in payload or "epoch" not in payload:
        raise ValueError(f"Invalid training checkpoint structure: {path}")
    return payload


def capture_rng_state() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    if not state:
        return
    random.setstate(tuple(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        np.asarray(numpy_state["keys"], dtype=np.uint32),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda") or []
    if torch.cuda.is_available() and cuda_states:
        # Checkpoints are commonly loaded with map_location=cuda for model and
        # optimizer tensors. CUDA generator state APIs still require CPU byte
        # tensors, so normalize the saved states before restoring them.
        torch.cuda.set_rng_state_all([generator_state.cpu() for generator_state in cuda_states])
