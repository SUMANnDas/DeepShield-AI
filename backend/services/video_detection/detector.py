from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import torch

from backend.services.video_model_registry import video_model_paths
from scripts.training.video.inference import load_video_for_inference, predict_video
from scripts.training.video.storage_budget import VideoStorageBudget


class VideoDetectionService:
    """Loads one verified video checkpoint at a time or wraps an explicit model."""

    def __init__(self, model=None, device: str = "cpu", max_bytes: int = 200_000_000):
        self._lock = threading.RLock()
        self._explicit_model = model is not None
        self.model = model
        self.device = device
        self.max_bytes = max_bytes
        self._loaded_key: tuple[str, str] | None = None
        self._runtime: dict[str, Any] | None = None
        self._identifier: str | None = None

    def _load(self, model_id: str, data_root: Path | None = None) -> None:
        root_key = str(Path(data_root).resolve()) if data_root is not None else "default"
        cache_key = (model_id, root_key)
        if self._loaded_key == cache_key and self.model is not None and self._runtime is not None:
            return
        self.model = None
        self._runtime = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        budget = VideoStorageBudget(root=data_root, minimum_free_bytes=0, project_root=None) if data_root else VideoStorageBudget()
        run_dir, checkpoint_path, _ = video_model_paths(model_id, data_root=budget.root)
        model, runtime, device, identifier = load_video_for_inference(
            run_dir,
            budget=budget,
            checkpoint_path=checkpoint_path,
        )
        self.model = model
        self.device = device
        self._runtime = runtime
        self._identifier = identifier
        self._loaded_key = cache_key

    def detect(
        self,
        path: str | Path,
        *,
        model_id: str | None = None,
        data_root: Path | None = None,
        include_heatmaps: bool = True,
    ) -> dict[str, Any]:
        p = Path(path)
        if not p.is_file() or p.stat().st_size > self.max_bytes:
            raise ValueError("invalid or oversized video input")
        with self._lock:
            if not self._explicit_model:
                if not model_id:
                    raise ValueError("model_id is required when no explicit video model is loaded")
                self._load(model_id, data_root=data_root)
                assert self.model is not None and self._runtime is not None
                norm = self._runtime["normalization"]
                start = time.perf_counter()
                result = predict_video(
                    p,
                    self.model,
                    device=self.device,
                    max_frames=self._runtime["max_frames"],
                    size=self._runtime["input_size"],
                    normalize=bool(norm.get("normalize_faces", True)),
                    mean=tuple(norm["mean"]),
                    std=tuple(norm["std"]),
                    mixed_precision=self._runtime["mixed_precision"],
                    include_heatmaps=include_heatmaps,
                )
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                result["model_id"] = model_id
                result["model_identifier"] = self._identifier
                result["device"] = str(self.device)
                result["inference_duration_ms"] = round(elapsed_ms, 3)
            else:
                result = predict_video(p, self.model, self.device)
        # Do not return host paths to clients.
        if result.get("metadata"):
            result["metadata"].pop("file_path", None)
        if result.get("report", {}).get("metadata"):
            result["report"]["metadata"].pop("file_path", None)
        return result
