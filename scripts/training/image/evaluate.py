"""Locked validation/test evaluation and report generation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from scripts.training.image.checkpoint import load_checkpoint
from scripts.training.image.data import build_test_loader, build_train_validation_loaders
from scripts.training.image.experiment import (
    CHECKPOINT_ROOT,
    PROJECT_ROOT,
    append_jsonl,
    manifest_identifier,
    write_json_exclusive,
)
from scripts.training.image.metrics import (
    compute_binary_metrics,
    save_confusion_matrix,
    save_roc_curve,
)
from scripts.training.image.models import build_model
from scripts.training.image.train import _load_config, _read_jsonl
from scripts.training.image.utils import select_device, software_versions


@torch.inference_mode()
def collect_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float], float]:
    model.eval()
    labels_all: list[int] = []
    fake_probabilities: list[float] = []
    total_loss = 0.0
    total_samples = 0
    for images, labels in tqdm(loader, desc="evaluation", leave=False, mininterval=1.0):
        images = images.to(device, non_blocking=True)
        labels_device = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels_device, reduction="sum")
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        labels_all.extend(int(value) for value in labels.tolist())
        fake_probabilities.extend(float(value) for value in probabilities.cpu().tolist())
        total_loss += float(loss.item())
        total_samples += int(labels.shape[0])
    if total_samples == 0:
        raise ValueError("Evaluation loader produced no samples")
    return labels_all, fake_probabilities, total_loss / total_samples


def _best_checkpoint(run_dir: Path, checkpoint_dir: Path) -> Path:
    entries = _read_jsonl(run_dir / "best-checkpoint-history.jsonl")
    if not entries:
        raise FileNotFoundError("No validation-selected checkpoint is recorded")
    path = PROJECT_ROOT / entries[-1]["checkpoint"]
    if not path.is_file() or path.parent.resolve() != checkpoint_dir.resolve():
        raise FileNotFoundError(f"Selected checkpoint is missing or outside its run directory: {path}")
    return path


def _assert_training_complete(run_dir: Path) -> None:
    segments = _read_jsonl(run_dir / "training-segments.jsonl")
    if not segments:
        raise RuntimeError("No training segment is recorded")
    last = segments[-1]
    if not last.get("early_stopped") and int(last["segment_finished_epoch"]) < int(last["configured_max_epochs"]):
        raise RuntimeError("Test evaluation is blocked until training is complete")


def evaluate_run(
    config_path: Path,
    run_dir: Path,
    *,
    split: str = "test",
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    config = _load_config(Path(config_path))
    run_dir = Path(run_dir).resolve()
    run_id = run_dir.name
    checkpoint_dir = CHECKPOINT_ROOT / run_id
    if not run_dir.is_dir() or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Run or checkpoint directory does not exist: {run_id}")
    if split == "test":
        _assert_training_complete(run_dir)
    if checkpoint_path is None:
        checkpoint_path = _best_checkpoint(run_dir, checkpoint_dir)
    else:
        checkpoint_path = Path(checkpoint_path).resolve()
        if checkpoint_path.parent != checkpoint_dir.resolve():
            raise ValueError("Evaluation checkpoint must be inside the matching run checkpoint directory")

    with (run_dir / "normalization.json").open("r", encoding="utf-8") as handle:
        normalization = json.load(handle)
    device = select_device(str(config.get("device", "auto")))
    model = build_model(str(config["architecture"])).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    dataset_root = PROJECT_ROOT / config["dataset_root"]
    if split == "test":
        manifest = PROJECT_ROOT / config["test_manifest"]
        loader = build_test_loader(
            dataset_root,
            manifest,
            mean=normalization["mean"],
            std=normalization["std"],
            batch_size=int(config["batch_size"]),
            seed=int(config["seed"]),
            num_workers=int(config["num_workers"]),
            pin_memory=device.type == "cuda",
        )
    else:
        manifest = PROJECT_ROOT / config["validation_manifest"]
        _, loader, _ = build_train_validation_loaders(
            dataset_root,
            PROJECT_ROOT / config["train_manifest"],
            manifest,
            mean=normalization["mean"],
            std=normalization["std"],
            batch_size=int(config["batch_size"]),
            seed=int(config["seed"]),
            num_workers=int(config["num_workers"]),
            pin_memory=device.type == "cuda",
            horizontal_flip_probability=0.0,
            crop_padding=0,
        )

    labels, probabilities, mean_loss = collect_predictions(model, loader, device)
    metrics = compute_binary_metrics(labels, probabilities, threshold=float(config.get("evaluation_threshold", 0.5)))
    metrics["loss"] = mean_loss
    record = {
        "run_id": run_id,
        "architecture": config["architecture"],
        "split": split,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_epoch": checkpoint["epoch"],
        "seed": int(config["seed"]),
        "dataset_manifest": manifest_identifier(manifest),
        "metrics": metrics,
        "software_versions": software_versions(),
        "device": str(device),
        "test_set_used_for_selection": False if split == "test" else None,
    }
    metrics_path = run_dir / f"{split}-metrics.json"
    write_json_exclusive(metrics_path, record)
    predictions_path = run_dir / f"{split}-predictions.csv"
    with predictions_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "fake_probability", "predicted_label"])
        for target, probability in zip(labels, probabilities, strict=True):
            writer.writerow([target, f"{probability:.10f}", int(probability >= float(config.get("evaluation_threshold", 0.5)))])
    save_confusion_matrix(metrics, run_dir / f"{split}-confusion-matrix.png", title=f"{config['architecture']} — {split}")
    save_roc_curve(labels, probabilities, run_dir / f"{split}-roc-curve.png", title=f"{config['architecture']} — {split}")
    append_jsonl(run_dir / "evaluation-events.jsonl", {
        "split": split,
        "metrics_path": str(metrics_path.relative_to(PROJECT_ROOT)),
        "manifest_sha256": record["dataset_manifest"]["sha256"],
    })
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate_run(args.config, args.run_dir, split=args.split, checkpoint_path=args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
