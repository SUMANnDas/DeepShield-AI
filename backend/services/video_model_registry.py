from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from scripts.training.video.experiment import config_sha256
from scripts.training.video.storage_budget import DEFAULT_VIDEO_DATA_ROOT

ARCHITECTURE_NAMES = {
    "efficientnet-b0-frame": "EfficientNet-B0 Frame Baseline",
    "efficientnet-b0-bigru-attention": "EfficientNet-B0 + BiGRU (Attention)",
    "efficientnet-b0-temporal-transformer": "EfficientNet-B0 + Temporal Transformer",
}


def _video_root(data_root: Path | None = None) -> Path:
    if data_root is not None:
        return Path(data_root).expanduser().resolve()
    env_root = os.environ.get("DEEPSHIELD_VIDEO_DATA_ROOT")
    return Path(env_root).expanduser().resolve() if env_root else DEFAULT_VIDEO_DATA_ROOT.resolve()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_best_history(path: Path) -> dict[str, Any] | None:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    if not rows or not isinstance(rows[-1], dict):
        return None
    return rows[-1]


def _validated_run(run_dir: Path, root: Path) -> dict[str, Any] | None:
    run_id = run_dir.name
    ckpt_dir = root / "checkpoints" / run_id
    metrics = _read_json(run_dir / "test-metrics.json")
    snapshot = _read_json(run_dir / "config-snapshot.json")
    metadata = _read_json(run_dir / "run-metadata.json")
    normalization = _read_json(run_dir / "normalization.json")
    best = _read_best_history(run_dir / "best-checkpoint-history.jsonl")
    if not (metrics and snapshot and metadata and normalization and best and ckpt_dir.is_dir()):
        return None
    dataset_name = str(metrics.get("dataset") or snapshot.get("dataset") or "")
    if not dataset_name or dataset_name.lower().startswith("synthetic"):
        return None
    if (
        metrics.get("run_id") != run_id
        or metrics.get("architecture") != snapshot.get("architecture")
        or metrics.get("split") != "test"
        or metrics.get("test_set_used_for_selection") is not False
        or metrics.get("source_split_grouped") is not True
        or metadata.get("configuration_sha256") != config_sha256(snapshot)
    ):
        return None
    try:
        metric_ckpt = (root / str(metrics["checkpoint"])).resolve()
        best_ckpt = (root / str(best["checkpoint"])).resolve()
        expected_dir = ckpt_dir.resolve()
        metric_epoch = int(metrics["checkpoint_epoch"])
        best_epoch = int(best["epoch"])
    except (KeyError, TypeError, ValueError, OSError):
        return None
    if (
        metric_ckpt.parent != expected_dir
        or best_ckpt.parent != expected_dir
        or metric_ckpt != best_ckpt
        or metric_epoch != best_epoch
        or not metric_ckpt.is_file()
    ):
        return None

    architecture = str(snapshot["architecture"])
    input_size = int(snapshot.get("input_size", 224))
    max_frames = int(snapshot.get("max_frames", snapshot.get("max_frames_per_video", 12)))
    return {
        "id": run_id,
        "name": ARCHITECTURE_NAMES.get(architecture, architecture),
        "architecture": architecture,
        "dataset": dataset_name,
        "task": "Video facial manipulation detection",
        "status": "available",
        "operational_status": "research_only_available",
        "validation_status": "video_grouped_test_evaluated",
        "run_id": run_id,
        "checkpoint": {
            "id": f"{run_id}:epoch-{metric_epoch}",
            "epoch": metric_epoch,
            "size_bytes": metric_ckpt.stat().st_size,
        },
        "validation_metrics": (
            {"roc_auc": float(best["validation_roc_auc"])}
            if best.get("validation_roc_auc") is not None
            else None
        ),
        "metrics": metrics.get("metrics"),
        "manipulation_strata": metrics.get("manipulation_strata") or {},
        "evaluated_videos": int(metrics.get("evaluated_videos") or 0),
        "test_manifest_sha256": (metrics.get("dataset_manifest") or {}).get("sha256"),
        "input_dimensions": [input_size, input_size],
        "max_frames": max_frames,
        "class_mapping": {"REAL": 0, "MANIPULATED": 1},
        "preprocessing": (
            f"Uniform up to {max_frames} frames, Haar face track, {input_size}x{input_size} RGB face crop"
            + (", ImageNet normalized" if normalization.get("normalize_faces") else "")
        ),
        "limitations": (
            f"Evaluated on grouped {dataset_name} test videos only. "
            "Haar fallback detection, compression, and dataset domain limit generalization."
        ),
        "_run_dir": run_dir,
        "_checkpoint_path": metric_ckpt,
        "_root": root,
    }


def list_video_models(data_root: Path | None = None) -> list[dict[str, Any]]:
    root = _video_root(data_root)
    exp_root = root / "experiments"
    if not exp_root.is_dir():
        return []
    entries: list[dict[str, Any]] = []
    for candidate in sorted(exp_root.iterdir(), reverse=True):
        if not candidate.is_dir():
            continue
        parsed = _validated_run(candidate, root)
        if parsed is None:
            continue
        entries.append({key: value for key, value in parsed.items() if not key.startswith("_")})
    entries.sort(
        key=lambda item: (
            -float(((item.get("metrics") or {}).get("roc_auc")) or -1.0),
            item["run_id"],
        )
    )
    return entries


def get_video_model(model_id: str, data_root: Path | None = None) -> dict[str, Any] | None:
    return next((item for item in list_video_models(data_root) if item["id"] == model_id), None)


def video_model_paths(model_id: str, data_root: Path | None = None) -> tuple[Path, Path, Path]:
    root = _video_root(data_root)
    run_dir = (root / "experiments" / model_id).resolve()
    try:
        run_dir.relative_to((root / "experiments").resolve())
    except ValueError as exc:
        raise KeyError(model_id) from exc
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Video model run directory not found: {model_id}")
    parsed = _validated_run(run_dir, root)
    if parsed is None:
        raise FileNotFoundError(f"Validated video model artifacts unavailable for: {model_id}")
    return parsed["_run_dir"], parsed["_checkpoint_path"], parsed["_root"]
