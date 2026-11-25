"""Manifest-first, read-only CIFAKE data loading."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from scripts.training.image.transforms import ImageTransform


LABEL_TO_INDEX = {"REAL": 0, "FAKE": 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}
EXPECTED_SIZE = (32, 32)


@dataclass(frozen=True)
class ManifestSample:
    relative_path: str
    absolute_path: Path
    label_name: str
    target: int
    split: str
    sample_id: str
    file_hash: str
    split_group_id: str


def resolve_image_path(dataset_root: Path, relative_path: str) -> Path:
    """Resolve a manifest path and reject absolute paths, traversal, and symlink escapes."""
    root = Path(dataset_root).resolve()
    normalized = str(relative_path).replace("\\", "/")
    candidate_part = Path(normalized)
    if candidate_part.is_absolute() or any(part == ".." for part in candidate_part.parts):
        raise ValueError(f"Unsafe manifest relative_path: {relative_path!r}")
    candidate = (root / candidate_part).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes dataset root: {relative_path!r}") from exc
    return candidate


def read_manifest(
    manifest_path: Path,
    dataset_root: Path,
    expected_split: str,
    *,
    check_files: bool = True,
) -> list[ManifestSample]:
    """Load one approved JSONL split and validate all fields before iteration."""
    if expected_split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split name: {expected_split}")
    manifest_path = Path(manifest_path)
    dataset_root = Path(dataset_root).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"CIFAKE dataset root not found: {dataset_root}")

    samples: list[ManifestSample] = []
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    required = {"relative_path", "label", "split", "sample_id", "file_hash", "split_group_id"}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            try:
                row: Any = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {manifest_path} line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest line {line_number} must contain a JSON object")
            missing = required - row.keys()
            if missing:
                raise ValueError(f"Manifest line {line_number} missing fields: {sorted(missing)}")
            if row.get("dataset_name") != "CIFAKE" or str(row.get("dataset_version")) != "3":
                raise ValueError(f"Manifest line {line_number} is not CIFAKE version 3")
            if row.get("task") != "ai_generated_image":
                raise ValueError(f"Manifest line {line_number} has unexpected task: {row.get('task')!r}")
            split = str(row["split"])
            if split != expected_split:
                raise ValueError(
                    f"Manifest line {line_number} says split={split!r}, expected {expected_split!r}"
                )
            label = str(row["label"]).strip().upper()
            if label not in LABEL_TO_INDEX:
                raise ValueError(f"Unexpected CIFAKE label on line {line_number}: {row['label']!r}")
            if row.get("preprocessing_status") != "validated":
                raise ValueError(f"Manifest line {line_number} is not marked validated")
            if row.get("image_width") != EXPECTED_SIZE[0] or row.get("image_height") != EXPECTED_SIZE[1]:
                raise ValueError(f"Manifest line {line_number} is not 32x32")
            relative_path = str(row["relative_path"])
            if relative_path in seen_paths:
                raise ValueError(f"Duplicate relative_path in {manifest_path}: {relative_path}")
            seen_paths.add(relative_path)
            sample_id = str(row["sample_id"])
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample_id in {manifest_path}: {sample_id}")
            seen_ids.add(sample_id)
            path = resolve_image_path(dataset_root, relative_path)
            if check_files and not path.is_file():
                raise FileNotFoundError(f"Image from manifest line {line_number} not found: {path}")
            samples.append(
                ManifestSample(
                    relative_path=relative_path,
                    absolute_path=path,
                    label_name=label,
                    target=LABEL_TO_INDEX[label],
                    split=split,
                    sample_id=sample_id,
                    file_hash=str(row["file_hash"]),
                    split_group_id=str(row["split_group_id"]),
                )
            )
    if not samples:
        raise ValueError(f"Manifest contains no samples: {manifest_path}")
    return samples


def assert_group_disjoint(*sample_sets: Iterable[ManifestSample]) -> None:
    seen: dict[str, str] = {}
    for samples in sample_sets:
        for sample in samples:
            for group in (sample.split_group_id, sample.file_hash):
                if not group:
                    continue
                previous = seen.get(group)
                if previous is not None and previous != sample.split:
                    raise ValueError(
                        f"Leakage group {group!r} appears in both {previous!r} and {sample.split!r}"
                    )
                seen[group] = sample.split


def decode_rgb(sample: ManifestSample) -> np.ndarray:
    image_bgr = cv2.imread(str(sample.absolute_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unreadable image for sample {sample.sample_id}: {sample.absolute_path}")
    if image_bgr.shape[:2] != (EXPECTED_SIZE[1], EXPECTED_SIZE[0]):
        raise ValueError(
            f"Unexpected image size for {sample.relative_path}: "
            f"{image_bgr.shape[1]}x{image_bgr.shape[0]} (expected 32x32)"
        )
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


class CIFAKEDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self,
        samples: list[ManifestSample],
        transform: ImageTransform | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def get_rgb(self, index: int) -> np.ndarray:
        return decode_rgb(self.samples[index])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        image = decode_rgb(sample)
        tensor = self.transform(image) if self.transform else torch.from_numpy(image.transpose(2, 0, 1).copy())
        if self.transform is None:
            tensor = tensor.to(dtype=torch.float32).div_(255.0)
        return tensor, sample.target


def compute_training_statistics(
    train_samples: list[ManifestSample],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute channel moments from train images only; validation/test are never opened."""
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    pixels = 0
    for sample in train_samples:
        rgb = decode_rgb(sample).astype(np.float64) / 255.0
        channel_sum += rgb.sum(axis=(0, 1))
        channel_squared_sum += np.square(rgb).sum(axis=(0, 1))
        pixels += int(rgb.shape[0] * rgb.shape[1])
    if pixels == 0:
        raise ValueError("Cannot compute normalization statistics from an empty training split")
    mean_array = channel_sum / pixels
    variance = np.maximum(channel_squared_sum / pixels - np.square(mean_array), 1e-12)
    std_array = np.sqrt(variance)
    return tuple(float(x) for x in mean_array), tuple(float(x) for x in std_array)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def make_loader(
    dataset: CIFAKEDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=_seed_worker,
        generator=generator,
        # Restart workers per epoch so the explicit epoch seed also controls
        # augmentation after a resumed run on Windows.
        persistent_workers=False,
    )


def build_train_validation_loaders(
    dataset_root: Path,
    train_manifest: Path,
    validation_manifest: Path,
    *,
    batch_size: int = 128,
    seed: int = 42,
    num_workers: int = 2,
    pin_memory: bool = False,
    mean: tuple[float, float, float] | list[float] | None = None,
    std: tuple[float, float, float] | list[float] | None = None,
    horizontal_flip_probability: float = 0.5,
    crop_padding: int = 2,
) -> tuple[DataLoader, DataLoader, tuple[tuple[float, ...], tuple[float, ...]]]:
    train_samples = read_manifest(train_manifest, dataset_root, "train")
    validation_samples = read_manifest(validation_manifest, dataset_root, "validation")
    assert_group_disjoint(train_samples, validation_samples)
    if mean is None or std is None:
        mean, std = compute_training_statistics(train_samples)
    train_transform = ImageTransform(
        mean, std, training=True,
        horizontal_flip_probability=horizontal_flip_probability,
        crop_padding=crop_padding,
    )
    eval_transform = ImageTransform(mean, std, training=False)
    train_dataset = CIFAKEDataset(train_samples, train_transform)
    validation_dataset = CIFAKEDataset(validation_samples, eval_transform)
    train_loader = make_loader(
        train_dataset, batch_size=batch_size, shuffle=True, seed=seed,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    validation_loader = make_loader(
        validation_dataset, batch_size=batch_size, shuffle=False, seed=seed,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    return train_loader, validation_loader, (tuple(mean), tuple(std))


def build_test_loader(
    dataset_root: Path,
    test_manifest: Path,
    *,
    mean: tuple[float, float, float] | list[float],
    std: tuple[float, float, float] | list[float],
    batch_size: int = 128,
    seed: int = 42,
    num_workers: int = 2,
    pin_memory: bool = False,
) -> DataLoader:
    """Create the isolated test loader; callers should invoke only after model lock."""
    samples = read_manifest(test_manifest, dataset_root, "test")
    dataset = CIFAKEDataset(samples, ImageTransform(mean, std, training=False))
    return make_loader(
        dataset, batch_size=batch_size, shuffle=False, seed=seed,
        num_workers=num_workers, pin_memory=pin_memory,
    )
