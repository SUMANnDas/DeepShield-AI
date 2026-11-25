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


def list_models() -> list[dict[str, Any]]:
    result = []
    for model_id, definition in MODELS.items():
        run_dir = EXPERIMENT_ROOT / definition["run_id"]
        checkpoint_dir = CHECKPOINT_ROOT / definition["run_id"]
        metrics_path = run_dir / "test-metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
        checkpoint = None
        if metrics:
            candidate = PROJECT_ROOT / metrics["checkpoint"]
            if candidate.is_file() and candidate.parent.resolve() == checkpoint_dir.resolve():
                checkpoint = {"path": candidate.relative_to(PROJECT_ROOT).as_posix(), "epoch": metrics["checkpoint_epoch"], "bytes": candidate.stat().st_size}
        config_path = PROJECT_ROOT / definition["config"]
        available = bool(checkpoint and config_path.is_file())
        result.append({
            "id": model_id,
            "name": definition["name"],
            "architecture": definition["architecture"],
            "dataset": "CIFAKE v3",
            "task": "AI-generated image classification",
            "status": "available" if available else "unavailable",
            "checkpoint": checkpoint,
            "metrics": (metrics or {}).get("metrics"),
            "test_samples": (metrics or {}).get("metrics", {}).get("support", {}).get("REAL", 0) + (metrics or {}).get("metrics", {}).get("support", {}).get("FAKE", 0),
            "test_manifest_sha256": (metrics or {}).get("dataset_manifest", {}).get("sha256"),
            "limitations": "CIFAKE consists of 32×32 images; results do not establish performance on real-world or facial deepfakes.",
        })
    return result


def get_model(model_id: str) -> dict[str, Any] | None:
    return next((item for item in list_models() if item["id"] == model_id), None)


def model_paths(model_id: str) -> tuple[Path, Path, Path, Path]:
    definition = MODELS[model_id]
    run_dir = EXPERIMENT_ROOT / definition["run_id"]
    evaluation = json.loads((run_dir / "test-metrics.json").read_text(encoding="utf-8"))
    checkpoint = (PROJECT_ROOT / evaluation["checkpoint"]).resolve()
    expected_dir = (CHECKPOINT_ROOT / definition["run_id"]).resolve()
    if checkpoint.parent != expected_dir or not checkpoint.is_file():
        raise FileNotFoundError("Saved evaluation checkpoint is unavailable")
    return PROJECT_ROOT / definition["config"], run_dir, checkpoint, PROJECT_ROOT
