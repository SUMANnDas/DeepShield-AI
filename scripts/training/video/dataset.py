from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .face_detection import HaarFaceDetector
from .preprocess_video import preprocess_video
from .storage_budget import GB, StorageBudgetExceeded, VideoStorageBudget
from .transforms import IMAGENET_MEAN, IMAGENET_STD, normalize_masked_sequence

LABEL_TO_INDEX = {
    "REAL": 0,
    "AUTHENTIC": 0,
    "FAKE": 1,
    "MANIPULATED": 1,
    "DEEPFAKE": 1,
    "SYNTHETIC": 1,
}


def _label_index(record: dict) -> int:
    value = record.get("label")
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return value
    normalized = str(value or record.get("label_name") or "").strip().upper()
    try:
        return LABEL_TO_INDEX[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported video label {value!r} for sample {record.get('sample_id')!r}") from exc


def _cache_key(relative_path: str, max_frames: int, size: int) -> str:
    return hashlib.sha256(f"{relative_path}\0{max_frames}\0{size}".encode("utf-8")).hexdigest()[:24]


def build_crop_cache(
    manifest_paths: list[Path],
    dataset_root: Path,
    cache_dir: Path,
    *,
    budget: VideoStorageBudget,
    max_frames: int = 12,
    size: int = 224,
    max_frame_pixels: int = 16_000_000,
    cache_limit_bytes: int = 2 * GB,
    workers: int = 6,
) -> dict[str, int]:
    """Deterministically materialize lossless uint8 face-crop sequences under an explicit cache cap."""
    root = Path(dataset_root).resolve()
    target_dir = budget.validate_location(cache_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen_rel: set[str] = set()
    for mpath in manifest_paths:
        with Path(mpath).open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                rel = rec.get("relative_path")
                if rel and rel not in seen_rel:
                    seen_rel.add(rel)
                    records.append(rec)

    lock = threading.Lock()
    local_state = threading.local()
    stats = {"cached": 0, "skipped_existing": 0, "empty_face_tracks": 0, "errors": 0, "cache_bytes": 0}
    used_cache_bytes = sum(p.stat().st_size for p in target_dir.glob("*.npz") if p.is_file())
    stats["cache_bytes"] = used_cache_bytes

    def _detector() -> HaarFaceDetector:
        det = getattr(local_state, "detector", None)
        if det is None:
            det = HaarFaceDetector()
            local_state.detector = det
        return det

    def _process_one(rec: dict[str, Any]) -> None:
        nonlocal used_cache_bytes
        rel = rec["relative_path"]
        out_path = budget.validate_location(target_dir / f"{_cache_key(rel, max_frames, size)}.npz")
        if out_path.is_file():
            with lock:
                stats["skipped_existing"] += 1
            return
        src_path = (root / rel).resolve()
        if root not in src_path.parents or not src_path.is_file():
            with lock:
                stats["errors"] += 1
            return
        prepared = preprocess_video(
            src_path,
            detector=_detector(),
            max_frames=max_frames,
            max_crops=max_frames,
            size=size,
            max_frame_pixels=max_frame_pixels,
        )
        if prepared.get("error"):
            crops = np.zeros((0, size, size, 3), dtype=np.uint8)
            frame_indices = np.zeros((0,), dtype=np.int32)
            with lock:
                stats["errors"] += 1
        else:
            frames = prepared.get("frames") or []
            if not frames:
                crops = np.zeros((0, size, size, 3), dtype=np.uint8)
                frame_indices = np.zeros((0,), dtype=np.int32)
                with lock:
                    stats["empty_face_tracks"] += 1
            else:
                crops = np.stack([item["crop"] for item in frames[:max_frames]], axis=0).astype(np.uint8)
                frame_indices = np.asarray([int(item["frame_index"]) for item in frames[:max_frames]], dtype=np.int32)
        buf = io.BytesIO()
        np.savez_compressed(buf, crops=crops, frame_indices=frame_indices)
        payload = buf.getvalue()
        with lock:
            if out_path.exists():
                stats["skipped_existing"] += 1
                return
            if used_cache_bytes + len(payload) > cache_limit_bytes:
                raise StorageBudgetExceeded(
                    f"Crop cache would exceed explicit {cache_limit_bytes / GB:.2f} GB cache cap"
                )
            budget.require_capacity(len(payload), purpose="writing deterministic video crop cache")
            with out_path.open("xb") as handle:
                handle.write(payload)
            used_cache_bytes += len(payload)
            stats["cache_bytes"] = used_cache_bytes
            stats["cached"] += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_process_one, rec) for rec in records]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()
    return stats


class VideoSequenceDataset(Dataset):
    def __init__(
        self,
        manifest_path,
        root,
        max_frames=12,
        size=224,
        detector=None,
        max_frame_pixels=16_000_000,
        *,
        normalize: bool = False,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        skip_errors: bool = False,
        crop_cache_dir: Path | None = None,
        horizontal_flip_prob: float = 0.0,
    ):
        if max_frames < 1 or size < 1 or max_frame_pixels < 1:
            raise ValueError("max_frames, size, and max_frame_pixels must be positive")
        self.records = []
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Manifest line {line_number} must contain an object")
                if not record.get("relative_path"):
                    raise ValueError(f"Manifest line {line_number} has no relative_path")
                record["_label_index"] = _label_index(record)
                self.records.append(record)
        self.root = Path(root).resolve()
        self.max_frames = max_frames
        self.size = size
        self.detector = detector or HaarFaceDetector()
        self.max_frame_pixels = max_frame_pixels
        self.normalize = bool(normalize)
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self.skip_errors = bool(skip_errors)
        self.crop_cache_dir = Path(crop_cache_dir).resolve() if crop_cache_dir is not None else None
        self.horizontal_flip_prob = float(horizontal_flip_prob)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        rel = record["relative_path"]
        frames = torch.zeros(self.max_frames, 3, self.size, self.size, dtype=torch.float32)
        mask = torch.zeros(self.max_frames, dtype=torch.bool)

        cached_file = (
            self.crop_cache_dir / f"{_cache_key(rel, self.max_frames, self.size)}.npz"
            if self.crop_cache_dir is not None
            else None
        )
        if cached_file is not None and cached_file.is_file():
            with np.load(cached_file, allow_pickle=False) as loaded:
                crops = loaded["crops"]
            count = min(self.max_frames, int(crops.shape[0]))
            for frame_index in range(count):
                crop = np.ascontiguousarray(crops[frame_index].transpose(2, 0, 1))
                frames[frame_index] = torch.from_numpy(crop).float() / 255.0
                mask[frame_index] = True
        else:
            path = (self.root / rel).resolve()
            if self.root not in path.parents:
                raise ValueError("manifest path escapes dataset root")
            if not path.is_file():
                raise FileNotFoundError(f"Video sample is missing: {rel}")
            prepared = preprocess_video(
                path,
                detector=self.detector,
                max_frames=self.max_frames,
                max_crops=self.max_frames,
                size=self.size,
                max_frame_pixels=self.max_frame_pixels,
            )
            if prepared.get("error"):
                if not self.skip_errors:
                    raise ValueError(f"Video preprocessing failed: {prepared['error']}")
                prepared = {"frames": []}

            for frame_index, item in enumerate(prepared["frames"][: self.max_frames]):
                crop = np.ascontiguousarray(item["crop"].transpose(2, 0, 1))
                frames[frame_index] = torch.from_numpy(crop).float() / 255.0
                mask[frame_index] = True

        if self.horizontal_flip_prob > 0.0 and mask.any() and torch.rand(1).item() < self.horizontal_flip_prob:
            frames[mask] = torch.flip(frames[mask], dims=[-1])

        if self.normalize and mask.any():
            frames = normalize_masked_sequence(frames, mask, mean=self.mean, std=self.std)

        video_id = str(record.get("video_id") or record.get("sample_id") or rel)
        source_video_id = str(
            record.get("source_video_id")
            or record.get("original_video_id")
            or record.get("video_id")
            or record.get("sample_id")
            or video_id
        )
        manipulation_method = str(
            record.get("manipulation_method")
            or record.get("manipulation_type")
            or ("original" if record["_label_index"] == 0 else "unknown")
        )
        return {
            "frames": frames,
            "mask": mask,
            "label": torch.tensor(record["_label_index"], dtype=torch.long),
            "video_id": video_id,
            "source_video_id": source_video_id,
            "manipulation_method": manipulation_method,
        }


def collate_sequences(batch):
    return {
        "frames": torch.stack([item["frames"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "video_id": [item["video_id"] for item in batch],
        "source_video_id": [item["source_video_id"] for item in batch],
        "manipulation_method": [item["manipulation_method"] for item in batch],
    }
