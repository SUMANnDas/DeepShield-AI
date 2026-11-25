"""Bounded video inspection, frame sampling, face flags, timestamps, and manifests."""

from __future__ import annotations

import logging
import random
import sqlite3
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scripts.datasets.face_utils import FaceDetector
from scripts.datasets.manifest_utils import (
    annotation_for,
    append_jsonl,
    config_hash,
    exclusive_jsonl,
    iter_media_files,
    json_config,
    load_annotations,
    log_file_error,
    relative_posix,
    refuse_existing,
    safe_output_name,
    sample_id,
    sha256_file,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_CONFIG: dict[str, Any] = {
    "video_extensions": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
    "sampling_strategy": "uniform",
    "max_frames": 24,
    "scene_probe_stride": 5,
    "scene_change_threshold": 0.18,
    "face_probe_stride": 5,
    "face_detection_enabled": True,
    "face_min_size": 32,
    "face_scale_factor": 1.1,
    "face_min_neighbors": 5,
    "jpeg_quality": 95,
}


def _fourcc(value: float) -> str | None:
    try:
        number = int(value)
        result = "".join(chr((number >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00 ")
        return result or None
    except (TypeError, ValueError):
        return None


def _uniform_indices(frame_count: int, max_frames: int) -> list[int]:
    if frame_count <= 0 or max_frames <= 0:
        return []
    count = min(frame_count, max_frames)
    if count == 1:
        return [0]
    return sorted({int(round(index * (frame_count - 1) / (count - 1))) for index in range(count)})


def _reservoir_add(
    selected: list[int],
    candidate: int,
    seen: int,
    limit: int,
    rng: random.Random,
) -> int:
    seen += 1
    if len(selected) < limit:
        selected.append(candidate)
    elif limit > 0:
        replacement = rng.randrange(seen)
        if replacement < limit:
            selected[replacement] = candidate
    return seen


def _select_indices(
    path: Path,
    frame_count: int,
    limit: int,
    strategy: str,
    config: dict[str, Any],
    detector: FaceDetector | None,
    seed: int,
) -> tuple[list[int], str]:
    if strategy == "uniform":
        return _uniform_indices(frame_count, limit), "uniform"
    if strategy not in {"scene", "face"}:
        raise ValueError("sampling_strategy must be uniform, scene, or face")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("video_open_failed")
    rng = random.Random(seed)
    candidates: list[int] = []
    seen = 0
    previous: np.ndarray | None = None
    stride_key = "scene_probe_stride" if strategy == "scene" else "face_probe_stride"
    stride = max(1, int(config.get(stride_key, 5)))
    scene_threshold = float(config.get("scene_change_threshold", 0.18))
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % stride == 0:
                if strategy == "scene":
                    thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32) / 255.0
                    if previous is not None and float(np.mean(np.abs(thumb - previous))) >= scene_threshold:
                        seen = _reservoir_add(candidates, frame_number, seen, limit, rng)
                    previous = thumb
                elif detector is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if detector.detect(rgb):
                        seen = _reservoir_add(candidates, frame_number, seen, limit, rng)
            frame_number += 1
    finally:
        capture.release()
    if candidates:
        return sorted(candidates), strategy
    return _uniform_indices(frame_count, limit), f"uniform_fallback_from_{strategy}"


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union else 0.0


def _track_faces(
    boxes: list[dict[str, Any]],
    previous: list[tuple[tuple[int, int, int, int], int]],
    next_track: int,
) -> tuple[list[int], list[tuple[tuple[int, int, int, int], int]], int]:
    used: set[int] = set()
    ids: list[int] = []
    current: list[tuple[tuple[int, int, int, int], int]] = []
    for box in boxes:
        rect = (int(box["x"]), int(box["y"]), int(box["width"]), int(box["height"]))
        matches = sorted(
            (( _iou(rect, old_rect), old_id) for old_rect, old_id in previous if old_id not in used),
            reverse=True,
        )
        if matches and matches[0][0] >= 0.30:
            track_id = matches[0][1]
            used.add(track_id)
        else:
            track_id = next_track
            next_track += 1
        ids.append(track_id)
        current.append((rect, track_id))
    return ids, current, next_track


def process_video_directory(
    input_root: Path,
    output_dir: Path,
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
) -> dict[str, int]:
    root = Path(input_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Video input directory does not exist: {root}")
    output_root = Path(output_dir)
    manifest = Path(manifest_path)
    frame_manifest = manifest.with_name(f"{manifest.stem}_frames.jsonl")
    refuse_existing(manifest)
    refuse_existing(frame_manifest)
    config = json_config(config_path, DEFAULT_CONFIG)
    config_digest = config_hash(config)
    annotation_store_path = manifest.with_suffix(".annotations.sqlite3")
    annotations = load_annotations(annotations_path, annotation_store_path if annotations_path else None)
    strategy = str(config.get("sampling_strategy", "uniform")).lower()
    max_frames = max(1, int(config.get("max_frames", 24)))
    detector = None
    if config.get("face_detection_enabled", True):
        detector = FaceDetector(
            min_size=int(config.get("face_min_size", 32)),
            scale_factor=float(config.get("face_scale_factor", 1.1)),
            min_neighbors=int(config.get("face_min_neighbors", 5)),
        )
    output_root.mkdir(parents=True, exist_ok=True)
    frame_output = output_root / "frames"
    stats = {"videos": 0, "frames": 0, "video_errors": 0, "frame_errors": 0}

    with exclusive_jsonl(manifest) as video_out, exclusive_jsonl(frame_manifest) as frames_out:
        for path in iter_media_files(root, config["video_extensions"]):
            relative = relative_posix(path, root)
            identifier = sample_id(dataset_name, relative)
            labels = annotation_for(annotations, relative, default_label)
            record: dict[str, Any] = {
                "sample_id": identifier,
                "dataset_name": dataset_name,
                "dataset_version": dataset_version,
                "source_url": source_url,
                "dataset_license": dataset_license,
                "citation": citation,
                "task": labels.get("task") or "face_manipulation_video",
                "original_video_id": labels.get("source_video_id") or identifier,
                "source_identity_id": labels.get("identity_id"),
                "relative_path": relative,
                "file_hash": None,
                "label": labels["label"],
                "original_label": labels["original_label"],
                "source_split": labels.get("source_split"),
                "manipulation_type": labels.get("manipulation_type"),
                "duration_seconds": None,
                "frame_rate": None,
                "frame_count": None,
                "width": None,
                "height": None,
                "fourcc": None,
                "sampling_strategy": strategy,
                "extracted_frame_count": 0,
                "decode_failure_count": 0,
                "preprocessing_config_hash": config_digest,
                "preprocessing_status": "pending",
                "error_code": None,
            }
            try:
                record["file_hash"] = sha256_file(path)
                capture = cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    capture.release()
                    raise ValueError("video_open_failed")
                frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                width = max(0, int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
                height = max(0, int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                fourcc = _fourcc(capture.get(cv2.CAP_PROP_FOURCC))
                capture.release()
                if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
                    raise ValueError("invalid_video_metadata")
                record.update({
                    "duration_seconds": frame_count / fps,
                    "frame_rate": fps,
                    "frame_count": frame_count,
                    "width": width,
                    "height": height,
                    "fourcc": fourcc,
                })
                chosen, effective_strategy = _select_indices(
                    path,
                    frame_count,
                    max_frames,
                    strategy,
                    config,
                    detector,
                    int(identifier[:8], 16),
                )
                record["sampling_strategy"] = effective_strategy
                capture = cv2.VideoCapture(str(path))
                wanted = set(chosen)
                current_frame = 0
                previous_tracks: list[tuple[tuple[int, int, int, int], int]] = []
                next_track = 1
                try:
                    while current_frame <= max(chosen, default=-1):
                        ok, frame = capture.read()
                        if current_frame in wanted:
                            timestamp = current_frame / fps
                            frame_record: dict[str, Any] = {
                                "sample_id": f"{identifier}_f{current_frame:09d}",
                                "video_sample_id": identifier,
                                "dataset_name": dataset_name,
                                "original_video_id": record["original_video_id"],
                                "source_identity_id": record["source_identity_id"],
                                "source_relative_path": relative,
                                "frame_number": current_frame,
                                "timestamp_seconds": timestamp,
                                "label": record["label"],
                                "original_label": record["original_label"],
                                "source_split": record["source_split"],
                                "manipulation_type": record["manipulation_type"],
                                "face_detected": None if detector is None else False,
                                "face_detection_status": "not_requested" if detector is None else "no_face",
                                "face_boxes": [],
                                "track_ids": [],
                                "output_path": None,
                                "preprocessing_config_hash": config_digest,
                                "preprocessing_status": "pending",
                                "error_code": None,
                            }
                            if not ok:
                                frame_record["preprocessing_status"] = "decode_error"
                                frame_record["error_code"] = "frame_decode_failed"
                                record["decode_failure_count"] += 1
                                stats["frame_errors"] += 1
                            else:
                                try:
                                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    faces = detector.detect(rgb) if detector else []
                                    boxes = [face.as_dict() for face in faces]
                                    track_ids, previous_tracks, next_track = _track_faces(
                                        boxes, previous_tracks, next_track
                                    )
                                    frame_record.update({
                                        "face_detected": bool(faces) if detector else None,
                                        "face_detection_status": "faces_found" if faces else (
                                            "no_face" if detector else "not_requested"
                                        ),
                                        "face_boxes": boxes,
                                        "track_ids": track_ids,
                                    })
                                    output_path = frame_output / f"{identifier}_frame{current_frame:09d}.jpg"
                                    output_path.parent.mkdir(parents=True, exist_ok=True)
                                    refuse_existing(output_path)
                                    ok_write = cv2.imwrite(
                                        str(output_path),
                                        frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, int(config.get("jpeg_quality", 95))],
                                    )
                                    if not ok_write:
                                        raise IOError("frame_write_failed")
                                    frame_record["output_path"] = output_path.relative_to(output_root).as_posix()
                                    frame_record["preprocessing_status"] = "processed"
                                    record["extracted_frame_count"] += 1
                                    stats["frames"] += 1
                                except FileExistsError:
                                    raise
                                except Exception as exc:
                                    frame_record["preprocessing_status"] = "processing_error"
                                    frame_record["error_code"] = str(exc)[:160]
                                    record["decode_failure_count"] += 1
                                    stats["frame_errors"] += 1
                                    log_file_error(LOGGER, "frame", relative, exc)
                            append_jsonl(frames_out, frame_record)
                        if not ok and current_frame not in wanted:
                            # Preserve a failure marker for every requested frame after decode stops.
                            missing = [index for index in chosen if index >= current_frame]
                            for missing_index in missing:
                                append_jsonl(frames_out, {
                                    "sample_id": f"{identifier}_f{missing_index:09d}",
                                    "video_sample_id": identifier,
                                    "dataset_name": dataset_name,
                                    "original_video_id": record["original_video_id"],
                                    "source_relative_path": relative,
                                    "frame_number": missing_index,
                                    "timestamp_seconds": missing_index / fps,
                                    "label": record["label"],
                                    "original_label": record["original_label"],
                                    "source_split": record["source_split"],
                                    "face_detected": None,
                                    "face_detection_status": "not_processed",
                                    "face_boxes": [],
                                    "track_ids": [],
                                    "output_path": None,
                                    "preprocessing_status": "decode_error",
                                    "error_code": "stream_ended_before_sample",
                                })
                                record["decode_failure_count"] += 1
                                stats["frame_errors"] += 1
                            break
                        current_frame += 1
                finally:
                    capture.release()
                record["preprocessing_status"] = (
                    "processed" if record["extracted_frame_count"] else "error"
                )
                if record["decode_failure_count"]:
                    record["error_code"] = "one_or_more_frame_failures"
                stats["videos"] += 1
            except FileExistsError:
                raise
            except Exception as exc:
                record["preprocessing_status"] = "corrupt" if isinstance(exc, (ValueError, cv2.error)) else "error"
                record["error_code"] = str(exc)[:160]
                stats["video_errors"] += 1
                log_file_error(LOGGER, "video", relative, exc)
            append_jsonl(video_out, record)
    if hasattr(annotations, "close"):
        annotations.close()
    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-version")
    parser.add_argument("--source-url")
    parser.add_argument("--dataset-license")
    parser.add_argument("--citation")
    parser.add_argument("--label", default="UNKNOWN")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print(process_video_directory(
        args.input, args.output, args.manifest, dataset_name=args.dataset_name,
        dataset_version=args.dataset_version, source_url=args.source_url,
        dataset_license=args.dataset_license, citation=args.citation,
        default_label=args.label, annotations_path=args.annotations, config_path=args.config,
    ))


if __name__ == "__main__":
    main()
