"""Deterministic, resumable CIFAKE training with validation-only model selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from scripts.training.image.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_exclusive,
)
from scripts.training.image.data import build_train_validation_loaders
from scripts.training.image.experiment import (
    CHECKPOINT_ROOT,
    EXPERIMENT_ROOT,
    append_jsonl,
    create_run_directories,
    environment_snapshot,
    manifest_identifier,
    write_json_exclusive,
)
from scripts.training.image.metrics import compute_binary_metrics, save_training_curves
from scripts.training.image.models import build_model
from scripts.training.image.utils import PROJECT_ROOT, select_device, set_reproducible_seed, software_versions


def _load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Training configuration must be a JSON object")
    return config


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_csv_row(path: Path, row: dict[str, Any], *, new_run: bool) -> None:
    if new_run:
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return
    if not path.is_file():
        raise FileNotFoundError(f"Resume requested but training history is missing: {path}")
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=list(row)).writerow(row)


def _forward_loss(
    model: torch.nn.Module,
    batch_images: torch.Tensor,
    batch_labels: torch.Tensor,
    device: torch.device,
    *,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch_images.to(device, non_blocking=True)
    labels = batch_labels.to(device, non_blocking=True)
    with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
    return logits, loss


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    gradient_clip_norm: float | None = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    amp_enabled = amp and device.type == "cuda"
    for images, labels in tqdm(loader, desc="train", leave=False, mininterval=1.0):
        optimizer.zero_grad(set_to_none=True)
        logits, loss = _forward_loss(model, images, labels, device, amp=amp_enabled)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.detach().item()}")
        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
        count = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * count
        total_samples += count
    if total_samples == 0:
        raise ValueError("Training loader produced no batches")
    return total_loss / total_samples


@torch.inference_mode()
def validate_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    amp: bool = False,
) -> tuple[float, dict[str, object]]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    labels_all: list[int] = []
    probabilities_all: list[float] = []
    for images, labels in tqdm(loader, desc="validation", leave=False, mininterval=1.0):
        logits, loss = _forward_loss(model, images, labels, device, amp=amp)
        count = int(labels.shape[0])
        total_loss += float(loss.item()) * count
        total_samples += count
        labels_all.extend(int(x) for x in labels.tolist())
        probabilities_all.extend(torch.softmax(logits.float(), dim=1)[:, 1].cpu().tolist())
    if total_samples == 0:
        raise ValueError("Validation loader produced no batches")
    return total_loss / total_samples, compute_binary_metrics(labels_all, probabilities_all)


def _checkpoint_files(checkpoint_dir: Path) -> list[Path]:
    return sorted(checkpoint_dir.glob("checkpoint-epoch-*.pt"))


def _run_paths(run_dir: Path) -> tuple[str, Path, Path]:
    run_dir = run_dir.resolve()
    try:
        run_dir.relative_to(EXPERIMENT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("Run directory must be inside the CIFAKE experiment root") from exc
    run_id = run_dir.name
    checkpoint_dir = CHECKPOINT_ROOT / run_id
    if not run_dir.is_dir() or not checkpoint_dir.is_dir():
        raise FileNotFoundError("Both experiment and checkpoint directories must exist for resume")
    return run_id, run_dir, checkpoint_dir


def run_training(
    config_path: Path,
    *,
    run_dir: Path | None = None,
    resume: bool = False,
    until_epoch: int | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    architecture = str(config["architecture"])
    seed = int(config["seed"])
    max_epochs = int(config["max_epochs"])
    stop_epoch = max_epochs if until_epoch is None else int(until_epoch)
    if stop_epoch < 1 or stop_epoch > max_epochs:
        raise ValueError(f"until_epoch must be in [1, {max_epochs}]")
    set_reproducible_seed(seed, deterministic=bool(config.get("deterministic", True)))
    device = select_device(str(config.get("device", "auto")))

    if run_dir is None:
        if resume:
            raise ValueError("--resume requires --run-dir")
        run_id, run_dir, checkpoint_dir = create_run_directories(architecture)
    else:
        if resume:
            run_id, run_dir, checkpoint_dir = _run_paths(Path(run_dir))
        else:
            raise ValueError("A supplied --run-dir is only valid with --resume; new runs choose collision-free IDs")

    config_hash = __import__("hashlib").sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config_snapshot_path = run_dir / "config-snapshot.json"
    if resume:
        saved_config = _load_config(config_snapshot_path)
        saved_hash = __import__("hashlib").sha256(
            json.dumps(saved_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if saved_hash != config_hash:
            raise ValueError("Resume configuration differs from the immutable run configuration")
    else:
        write_json_exclusive(config_snapshot_path, config)

    dataset_root = PROJECT_ROOT / config["dataset_root"]
    train_manifest = PROJECT_ROOT / config["train_manifest"]
    validation_manifest = PROJECT_ROOT / config["validation_manifest"]
    test_manifest = PROJECT_ROOT / config["test_manifest"]
    if not test_manifest.is_file():
        raise FileNotFoundError(f"Approved test manifest missing: {test_manifest}")
    normalization_path = run_dir / "normalization.json"
    if resume:
        with normalization_path.open("r", encoding="utf-8") as handle:
            norm_record = json.load(handle)
        mean, std = tuple(norm_record["mean"]), tuple(norm_record["std"])
    else:
        train_preview, validation_preview, _ = build_train_validation_loaders(
            dataset_root, train_manifest, validation_manifest,
            batch_size=int(config["batch_size"]), seed=seed,
            num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda",
            horizontal_flip_probability=float(config["horizontal_flip_probability"]),
            crop_padding=int(config["crop_padding"]),
        )
        # The loader construction computed statistics from the train manifest only.
        mean, std = train_preview.dataset.transform.mean.view(-1).tolist(), train_preview.dataset.transform.std.view(-1).tolist()
        del train_preview, validation_preview
        write_json_exclusive(normalization_path, {"mean": mean, "std": std, "source": "train manifest only"})

    train_loader, validation_loader, _ = build_train_validation_loaders(
        dataset_root, train_manifest, validation_manifest,
        batch_size=int(config["batch_size"]), seed=seed,
        num_workers=int(config["num_workers"]), pin_memory=device.type == "cuda",
        mean=mean, std=std,
        horizontal_flip_probability=float(config["horizontal_flip_probability"]),
        crop_padding=int(config["crop_padding"]),
    )
    model = build_model(architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    amp_enabled = bool(config.get("mixed_precision", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history_path = run_dir / "history.jsonl"
    history_csv_path = run_dir / "history.csv"
    metadata_path = run_dir / "run-metadata.json"
    if resume:
        if not history_path.is_file() or not history_csv_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Resume run is missing its append-only history or metadata")
    else:
        metadata = {
            "run_id": run_id,
            "architecture": architecture,
            "seed": seed,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "configuration_sha256": config_hash,
            "config_path": config_path.relative_to(PROJECT_ROOT).as_posix(),
            "manifest_identifiers": {
                "train": manifest_identifier(train_manifest),
                "validation": manifest_identifier(validation_manifest),
                "test": manifest_identifier(test_manifest),
            },
            "software_versions": software_versions(),
            "runtime": environment_snapshot(),
            "device": str(device),
        }
        write_json_exclusive(metadata_path, metadata)

    checkpoints = _checkpoint_files(checkpoint_dir)
    start_epoch = 1
    best_auc = -math.inf
    best_epoch = 0
    bad_epochs = 0
    if resume:
        if not checkpoints:
            raise FileNotFoundError(f"No resumable checkpoint in {checkpoint_dir}")
        latest_path = checkpoints[-1]
        state = load_checkpoint(latest_path, map_location=device)
        if state.get("configuration_sha256") != config_hash:
            raise ValueError("Checkpoint configuration hash does not match the requested configuration")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state.get("scaler_state", {}))
        restore_rng_state(state.get("rng_state", {}))
        start_epoch = int(state["epoch"]) + 1
        best_auc = float(state.get("best_validation_roc_auc", -math.inf))
        best_epoch = int(state.get("best_epoch", 0))
        bad_epochs = int(state.get("bad_epochs", 0))
    elif checkpoints or history_path.exists() or history_csv_path.exists():
        raise FileExistsError(f"New run directory contains training artifacts: {run_dir}")

    if start_epoch > max_epochs:
        return {"run_id": run_id, "run_dir": str(run_dir), "status": "already_complete", "last_epoch": start_epoch - 1}

    if not resume:
        with history_path.open("x", encoding="utf-8", newline="\n"):
            pass
        with history_csv_path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "epoch", "train_loss", "validation_loss", "validation_accuracy",
                "validation_precision_fake", "validation_recall_fake", "validation_f1_fake",
                "validation_macro_f1", "validation_weighted_f1", "validation_roc_auc",
                "validation_specificity_real", "validation_false_positive_rate",
                "validation_false_negative_rate", "learning_rate", "epoch_seconds",
                "peak_gpu_memory_bytes",
            ])
            writer.writeheader()

    previous_records = _read_jsonl(history_path)
    history_rows = list(previous_records)
    segment_start = start_epoch
    patience = int(config["early_stopping_patience"])
    minimum_delta = float(config.get("early_stopping_min_delta", 0.0))
    stopped_early = False
    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, stop_epoch + 1):
        # Deriving the sampler/worker seed from the epoch makes resumed runs
        # reproduce the same order and augmentation stream as uninterrupted runs.
        train_loader.generator.manual_seed(seed + epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start_time = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            amp=amp_enabled, scaler=scaler,
            gradient_clip_norm=float(config["gradient_clip_norm"]) if config.get("gradient_clip_norm") is not None else None,
        )
        validation_loss, validation_metrics = validate_one_epoch(
            model, validation_loader, device, amp=amp_enabled,
        )
        scheduler.step()
        epoch_seconds = time.perf_counter() - start_time
        current_auc = float(validation_metrics["roc_auc"])
        improved = current_auc > best_auc + minimum_delta
        if improved:
            best_auc = current_auc
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
        lr = float(optimizer.param_groups[0]["lr"])
        peak_memory = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        checkpoint_path = checkpoint_dir / f"checkpoint-epoch-{epoch:04d}.pt"
        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_validation_roc_auc": best_auc,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
            "configuration_sha256": config_hash,
            "seed": seed,
            "software_versions": software_versions(),
            "rng_state": capture_rng_state(),
            "normalization": {"mean": mean, "std": std},
            "manifest_identifiers": {
                "train": manifest_identifier(train_manifest),
                "validation": manifest_identifier(validation_manifest),
                "test": manifest_identifier(test_manifest),
            },
        }
        save_checkpoint_exclusive(checkpoint_path, checkpoint_payload)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            **{f"validation_{key}": value for key, value in validation_metrics.items() if key not in {"confusion_matrix", "confusion_matrix_layout", "support", "class_order", "positive_class", "threshold"}},
            "validation_support": validation_metrics["support"],
            "validation_confusion_matrix": validation_metrics["confusion_matrix"],
            "validation_threshold": validation_metrics["threshold"],
            "learning_rate": lr,
            "epoch_seconds": epoch_seconds,
            "peak_gpu_memory_bytes": peak_memory,
            "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "best_checkpoint": bool(improved),
        }
        append_jsonl(history_path, record)
        csv_row = {key: record.get(key, "") for key in (
            "epoch", "train_loss", "validation_loss", "validation_accuracy",
            "validation_precision_fake", "validation_recall_fake", "validation_f1_fake",
            "validation_macro_f1", "validation_weighted_f1", "validation_roc_auc",
            "validation_specificity_real", "validation_false_positive_rate",
            "validation_false_negative_rate", "learning_rate", "epoch_seconds",
            "peak_gpu_memory_bytes",
        )}
        _append_csv_row(history_csv_path, csv_row, new_run=False)
        if improved:
            append_jsonl(run_dir / "best-checkpoint-history.jsonl", {
                "epoch": epoch,
                "validation_roc_auc": current_auc,
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            })
        history_rows.append(record)
        final_epoch = epoch
        print(
            f"epoch={epoch}/{max_epochs} train_loss={train_loss:.5f} "
            f"val_loss={validation_loss:.5f} val_auc={current_auc:.5f} "
            f"val_f1={float(validation_metrics['f1_fake']):.5f} "
            f"epoch_seconds={epoch_seconds:.1f} peak_gpu_mib={peak_memory / 1024**2:.1f}"
        )
        if bad_epochs >= patience:
            stopped_early = True
            print(f"Early stopping after epoch {epoch}; best validation ROC-AUC={best_auc:.5f} at epoch {best_epoch}")
            break

    segment = {
        "segment_started_epoch": segment_start,
        "segment_finished_epoch": final_epoch,
        "target_epoch": stop_epoch,
        "configured_max_epochs": max_epochs,
        "resumed": resume,
        "early_stopped": stopped_early,
        "best_epoch_so_far": best_epoch,
        "best_validation_roc_auc": best_auc,
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    append_jsonl(run_dir / "training-segments.jsonl", segment)
    completed = stopped_early or final_epoch >= max_epochs
    if completed:
        save_training_curves(history_rows, run_dir / "plots")
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "last_epoch": final_epoch,
        "best_epoch": best_epoch,
        "best_validation_roc_auc": best_auc,
        "early_stopped": stopped_early,
        "training_complete": completed,
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path, help="Existing run directory; required together with --resume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--until-epoch", type=int, help="Run through this epoch, allowing a short pilot segment")
    args = parser.parse_args()
    result = run_training(args.config, run_dir=args.run_dir, resume=args.resume, until_epoch=args.until_epoch)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
