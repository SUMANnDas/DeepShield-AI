"""Streaming image validation, hashing, face analysis, and derived image writing."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from scripts.datasets.face_utils import FaceDetector
from scripts.datasets.manifest_utils import (
    annotation_for,
    config_hash,
    exclusive_jsonl,
    iter_media_files,
    json_config,
    log_file_error,
    relative_posix,
    refuse_existing,
    safe_output_name,
    sample_id,
    sha256_file,
    append_jsonl,
    load_annotations,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG: dict[str, Any] = {
    "target_size": [256, 256],
    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"],
    "face_detection_enabled": True,
    "face_min_size": 32,
    "face_scale_factor": 1.1,
    "face_min_neighbors": 5,
    "crop_faces": False,
    "face_margin_ratio": 0.15,
    "perceptual_hash_enabled": True,
    "perceptual_hash_distance": 6,
    "jpeg_quality": 95,
    "normalization": "rgb_0_1",
}


def perceptual_hash(rgb: np.ndarray) -> str:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    transformed = cv2.dct(resized)
    block = transformed[:8, :8]
    median = float(np.median(block.flatten()[1:]))
    bits = (block > median).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def _hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


class DuplicateIndex:
    """Disk-backed exact/perceptual index so dataset-sized hashes are not kept in RAM."""

    def __init__(self, path: Path, max_distance: int) -> None:
        refuse_existing(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE images (sample_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, phash TEXT, "
            "b0 TEXT, b1 TEXT, b2 TEXT, b3 TEXT, relative_path TEXT NOT NULL)"
        )
        self.connection.execute("CREATE INDEX images_sha_idx ON images(sha256)")
        for band in range(4):
            self.connection.execute(f"CREATE INDEX images_b{band}_idx ON images(b{band})")
        self.max_distance = max_distance
        self.pending = 0

    def find(self, sha256: str, phash: str | None) -> tuple[str | None, str | None]:
        row = self.connection.execute(
            "SELECT sample_id FROM images WHERE sha256=? LIMIT 1", (sha256,)
        ).fetchone()
        if row:
            return str(row[0]), "sha256"
        if phash:
            bands = [phash[index:index + 4] for index in range(0, 16, 4)]
            query = (
                "SELECT sample_id, phash FROM images WHERE "
                + " OR ".join(f"b{index}=?" for index in range(4))
                + " LIMIT 500"
            )
            for candidate_id, candidate_hash in self.connection.execute(query, bands):
                if candidate_hash and _hamming(phash, str(candidate_hash)) <= self.max_distance:
                    return str(candidate_id), "perceptual"
        return None, None

    def add(self, identifier: str, sha256: str, phash: str | None, relative_path: str) -> None:
        bands = [phash[index:index + 4] for index in range(0, 16, 4)] if phash else [None] * 4
        self.connection.execute(
            "INSERT INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (identifier, sha256, phash, *bands, relative_path),
        )
        self.pending += 1
        if self.pending >= 1000:
            self.connection.commit()
            self.pending = 0

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Return RGB float32 in [0,1], with no train-only augmentation."""
    return rgb.astype(np.float32) / 255.0


def _decode_rgb(path: Path) -> tuple[np.ndarray, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.size == 0:
        raise ValueError("unreadable_image")
    if image.ndim == 2:
        channels = 1
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 3:
        channels = 3
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 3 and image.shape[2] == 4:
        channels = 4
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError("unsupported_color_channels")
    return rgb, channels


def process_image_directory(
    input_root: Path,
    output_dir: Path | None,
    manifest_path: Path,
    *,
    dataset_name: str,
    dataset_version: str | None = None,
    source_url: str | None = None,
    dataset_license: str | None = None,
    citation: str | None = None,
    default_label: str = "UNKNOWN",
    annotations_path: Path | None = None,
    config_path: Path | None = None,
    validate_only: bool = False,
) -> dict[str, int]:
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Image input directory does not exist: {root}")
    if not validate_only and output_dir is None:
        raise ValueError("output_dir is required when validate_only is false")
    manifest = Path(manifest_path)
    refuse_existing(manifest)
    config = json_config(config_path, DEFAULT_CONFIG)
    config_digest = config_hash(config)
    annotation_store_path = manifest.with_suffix(".annotations.sqlite3")
    annotations = load_annotations(annotations_path, annotation_store_path if annotations_path else None)
    detector = None
    if config.get("face_detection_enabled", True):
        detector = FaceDetector(
            min_size=config["face_min_size"],
            scale_factor=config["face_scale_factor"],
            min_neighbors=config["face_min_neighbors"],
        )

    index_path = manifest.with_suffix(".duplicates.sqlite3")
    duplicate_index = DuplicateIndex(index_path, int(config.get("perceptual_hash_distance", 6)))
    stats = {"processed": 0, "corrupt": 0, "faces_found": 0, "no_face": 0, "duplicates": 0}
    output_root = Path(output_dir) if output_dir is not None else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    try:
        with exclusive_jsonl(manifest) as out:
            for path in iter_media_files(root, config["image_extensions"]):
                relative = relative_posix(path, root)
                identifier = sample_id(dataset_name, relative)
                label = annotation_for(annotations, relative, default_label)
                record: dict[str, Any] = {
                    "sample_id": identifier,
                    "dataset_name": dataset_name,
                    "dataset_version": dataset_version,
                    "source_url": source_url,
                    "dataset_license": dataset_license,
                    "citation": citation,
                    "task": label.get("task") or "image_manipulation",
                    "relative_path": relative,
                    "file_hash": None,
                    "label": label["label"],
                    "original_label": label["original_label"],
                    "source_split": label.get("source_split"),
                    "manipulation_type": label.get("manipulation_type"),
                    "identity_id": label.get("identity_id"),
                    "image_width": None,
                    "image_height": None,
                    "color_channels": None,
                    "face_detected": None if detector is None else False,
                    "face_detection_status": "not_requested" if detector is None else "not_processed",
                    "face_boxes": [],
                    "alignment_status": "not_requested" if detector is None else "not_aligned",
                    "perceptual_hash": None,
                    "duplicate_of": None,
                    "duplicate_kind": None,
                    "output_files": [],
                    "preprocessing_config_hash": config_digest,
                    "preprocessing_status": "pending",
                    "error_code": None,
                }
                try:
                    file_hash = sha256_file(path)
                    rgb, channels = _decode_rgb(path)
                    height, width = rgb.shape[:2]
                    phash = perceptual_hash(rgb) if config.get("perceptual_hash_enabled", True) else None
                    duplicate_of, duplicate_kind = duplicate_index.find(file_hash, phash)
                    faces = detector.detect(rgb, float(config.get("face_margin_ratio", 0.15))) if detector else []
                    record.update({
                        "file_hash": file_hash,
                        "image_width": int(width),
                        "image_height": int(height),
                        "color_channels": int(channels),
                        "face_detected": bool(faces) if detector else None,
                        "face_detection_status": "faces_found" if faces else ("no_face" if detector else "not_requested"),
                        "face_boxes": [face.as_dict() for face in faces],
                        "alignment_status": "eyes_aligned" if any(f.alignment_status == "eyes_aligned" for f in faces) else (
                            "crop_only" if faces else ("no_face" if detector else "not_requested")
                        ),
                        "perceptual_hash": phash,
                        "duplicate_of": duplicate_of,
                        "duplicate_kind": duplicate_kind,
                        "preprocessing_status": "validated",
                    })
                    if duplicate_of:
                        stats["duplicates"] += 1
                    if faces:
                        stats["faces_found"] += 1
                    elif detector:
                        stats["no_face"] += 1

                    if not validate_only and output_root is not None:
                        if faces and config.get("crop_faces", False):
                            candidates = [(face.crop_rgb, f"_face{index:02d}") for index, face in enumerate(faces)]
                        else:
                            candidates = [(rgb, "")]
                        target_width, target_height = map(int, config["target_size"])
                        for candidate, suffix in candidates:
                            resized = cv2.resize(candidate, (target_width, target_height), interpolation=cv2.INTER_AREA)
                            filename = safe_output_name(identifier + suffix, ".jpg")
                            output_path = output_root / "images" / filename
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            refuse_existing(output_path)
                            bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                            ok = cv2.imwrite(
                                str(output_path),
                                bgr,
                                [cv2.IMWRITE_JPEG_QUALITY, int(config.get("jpeg_quality", 95))],
                            )
                            if not ok:
                                raise IOError("image_write_failed")
                            record["output_files"].append(output_path.relative_to(output_root).as_posix())
                        record["preprocessing_status"] = "processed"
                    duplicate_index.add(identifier, file_hash, phash, relative)
                    stats["processed"] += 1
                except Exception as exc:
                    if isinstance(exc, FileExistsError):
                        raise
                    record["preprocessing_status"] = "corrupt" if isinstance(exc, (ValueError, cv2.error)) else "error"
                    record["error_code"] = str(exc)[:160]
                    stats["corrupt"] += 1
                    log_file_error(LOGGER, "image", relative, exc)
                append_jsonl(out, record)
    finally:
        duplicate_index.close()
        if hasattr(annotations, "close"):
            annotations.close()
    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version")
    parser.add_argument("--source-url")
    parser.add_argument("--dataset-license")
    parser.add_argument("--citation")
    parser.add_argument("--label", default="UNKNOWN")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(process_image_directory(
        args.input, args.output, args.manifest, dataset_name=args.dataset_name,
        dataset_version=args.dataset_version, source_url=args.source_url,
        dataset_license=args.dataset_license, citation=args.citation,
        default_label=args.label, annotations_path=args.annotations,
        config_path=args.config, validate_only=args.validate_only,
    ))


if __name__ == "__main__":
    main()
