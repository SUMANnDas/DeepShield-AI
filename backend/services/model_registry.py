from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.training.image.experiment import CHECKPOINT_ROOT, EXPERIMENT_ROOT, PROJECT_ROOT

MODELS = {
    "resnet50": {"name": "ResNet50", "architecture": "resnet50", "run_id": "resnet50-20261124T152708Z-d924cf0d", "config": "configs/training/image/cifake-resnet50.json"},
    "efficientnet-b0": {"name": "EfficientNet-B0", "architecture": "efficientnet-b0", "run_id": "efficientnet-b0-20261124T164138Z-3fb11f07", "config": "configs/training/image/cifake-efficientnet-b0.json"},
    "vit-small": {"name": "Small Vision Transformer", "architecture": "vit-small", "run_id": "vit-small-20261125T052835Z-66479d18", "config": "configs/training/image/cifake-vit-small.json"},
}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _run_records(model_id: str) -> tuple[dict[str, Any], Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    definition = MODELS[model_id]
    run_dir = EXPERIMENT_ROOT / definition["run_id"]
    checkpoint_dir = CHECKPOINT_ROOT / definition["run_id"]
    metrics = _read_json(run_dir / "test-metrics.json")
    snapshot = _read_json(run_dir / "config-snapshot.json")
    config = _read_json(PROJECT_ROOT / definition["config"])
    history_path = run_dir / "best-checkpoint-history.jsonl"
    if not metrics or not snapshot or not config or not history_path.is_file():
        return None
    try:
        history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError):
        return None
    if not history or not isinstance(history[-1], dict):
        return None
    return definition, run_dir, checkpoint_dir, metrics, snapshot, history[-1]


def _selected_checkpoint_matches(
    model_id: str,
    records: tuple[dict[str, Any], Path, Path, dict[str, Any], dict[str, Any], dict[str, Any]] | None,
) -> tuple[Path, int] | None:
    if records is None:
        return None
    definition, _, checkpoint_dir, metrics, snapshot, best = records
    if (
        metrics.get("run_id") != definition["run_id"]
        or metrics.get("architecture") != definition["architecture"]
        or metrics.get("split") != "test"
        or metrics.get("test_set_used_for_selection") is not False
        or snapshot.get("architecture") != definition["architecture"]
    ):
        return None
    try:
        metric_path = (PROJECT_ROOT / str(metrics["checkpoint"])).resolve()
        history_path = (PROJECT_ROOT / str(best["checkpoint"])).resolve()
        expected_dir = checkpoint_dir.resolve()
        epoch = int(metrics["checkpoint_epoch"])
        history_epoch = int(best["epoch"])
    except (KeyError, TypeError, ValueError, OSError):
        return None
    if (
        metric_path.parent != expected_dir
        or history_path.parent != expected_dir
        or metric_path != history_path
        or epoch != history_epoch
        or not metric_path.is_file()
    ):
        return None
    return metric_path, epoch


def list_models() -> list[dict[str, Any]]:
    result = []
    for model_id, definition in MODELS.items():
        records = _run_records(model_id)
        selected = _selected_checkpoint_matches(model_id, records)
        metrics = records[3] if records else None
        best = records[5] if records else None
        checkpoint_path, checkpoint_epoch = selected if selected else (None, None)
        checkpoint = None
        if checkpoint_path is not None:
            checkpoint = {
                "id": f"{definition['run_id']}:epoch-{checkpoint_epoch}",
                "epoch": checkpoint_epoch,
                "size_bytes": checkpoint_path.stat().st_size,
            }
        available = checkpoint is not None
        support = (metrics or {}).get("metrics", {}).get("support", {})
        test_samples = int(support.get("REAL", 0)) + int(support.get("FAKE", 0))
        result.append({
            "id": model_id,
            "name": definition["name"],
            "architecture": definition["architecture"],
            "dataset": "CIFAKE v3",
            "dataset_version": "3",
            "task": "AI-generated image classification",
            "status": "available" if available else "unavailable",
            "operational_status": "research_only_available" if available else "unavailable",
            "validation_status": "cifake_v3_held_out_test_evaluated" if available else "unvalidated_or_unavailable",
            "run_id": definition["run_id"],
            "checkpoint": checkpoint,
            "validation_metrics": ({"roc_auc": float(best["validation_roc_auc"])} if best and best.get("validation_roc_auc") is not None else None),
            "metrics": (metrics or {}).get("metrics"),
            "test_samples": test_samples,
            "test_manifest_sha256": (metrics or {}).get("dataset_manifest", {}).get("sha256"),
            "input_dimensions": [32, 32],
            "class_mapping": {"REAL": 0, "FAKE": 1},
            "preprocessing": "RGB; native 32x32 CIFAKE input; aspect-preserving bilinear fit and run-mean padding for other sizes",
            "limitations": "CIFAKE consists of 32×32 images; results do not establish performance on real-world or facial deepfakes.",
        })
    return result


def get_model(model_id: str) -> dict[str, Any] | None:
    return next((item for item in list_models() if item["id"] == model_id), None)


def model_paths(model_id: str) -> tuple[Path, Path, Path, Path]:
    if model_id not in MODELS:
        raise KeyError(model_id)
    definition = MODELS[model_id]
    run_dir = EXPERIMENT_ROOT / definition["run_id"]
    selected = _selected_checkpoint_matches(model_id, _run_records(model_id))
    if selected is None:
        raise FileNotFoundError("Saved evaluation checkpoint is unavailable")
    checkpoint, _ = selected
    return PROJECT_ROOT / definition["config"], run_dir, checkpoint, PROJECT_ROOT
