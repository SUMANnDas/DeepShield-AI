from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .dataset import VideoSequenceDataset, collate_sequences
from .experiment import (
    append_csv_budgeted,
    append_jsonl_budgeted,
    config_sha256,
    manifest_identifier,
    write_json_budgeted,
)
from .metrics import binary_metrics, save_video_training_curves
from .models import build_video_model
from .storage_budget import VideoStorageBudget
from .transforms import IMAGENET_MEAN, IMAGENET_STD
from .utils import seed_everything

HISTORY_CSV_FIELDS = [
    "epoch",
    "train_loss",
    "validation_loss",
    "validation_accuracy",
    "validation_precision",
    "validation_recall",
    "validation_f1",
    "validation_macro_f1",
    "validation_roc_auc",
    "validation_pr_auc",
    "learning_rate",
    "epoch_seconds",
    "peak_gpu_memory_bytes",
    "valid_validation_videos",
    "skipped_validation_videos",
]


def autocast_dtype(device, mixed_precision="off"):
    mode = str(mixed_precision).lower()
    if mode == "off":
        return None
    if mode not in {"fp16", "bf16"}:
        raise ValueError("mixed_precision must be off, fp16, or bf16")
    device_type = torch.device(device).type
    if device_type != "cuda":
        raise ValueError(f"mixed_precision={mode} requires a CUDA device")
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support bfloat16")
        return torch.bfloat16
    return torch.float16


def _autocast(device, dtype):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype)


def _prepare_batch(batch, device):
    frames = batch["frames"].to(device)
    mask = batch["mask"].to(device)
    labels = batch["label"].to(device)
    if frames.ndim in (3, 5):
        valid_sequences = mask.any(dim=1)
        frames = frames[valid_sequences]
        mask = mask[valid_sequences]
        labels = labels[valid_sequences]
    return frames, mask, labels


def _forward_batch(model, frames, mask, labels):
    if frames.ndim == 5 and hasattr(model, "encode_frames"):
        return model(frames, mask), labels
    if frames.ndim == 5:
        valid_frames = mask.reshape(-1)
        flattened = frames.reshape(-1, *frames.shape[2:])[valid_frames]
        frame_labels = labels[:, None].expand(-1, mask.shape[1]).reshape(-1)[valid_frames]
        return {"logits": model(flattened)}, frame_labels
    if frames.ndim == 3:
        return model(frames, mask), labels
    return {"logits": model(frames)}, labels


def train_epoch(model, loader, optimizer, device="cpu", mixed_precision="off", scaler=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    dtype = autocast_dtype(device, mixed_precision)
    if scaler is None:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=dtype is torch.float16 and torch.device(device).type == "cuda",
        )
    total_loss = 0.0
    total_targets = 0
    for batch in loader:
        frames, mask, labels = _prepare_batch(batch, device)
        if labels.numel() == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, dtype):
            output, targets = _forward_batch(model, frames, mask, labels)
            if targets.numel() == 0:
                continue
            loss = loss_fn(output["logits"], targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite video training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * targets.numel()
        total_targets += targets.numel()
    if not total_targets:
        raise ValueError("training split has no samples with a valid face sequence")
    return total_loss / total_targets


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    epochs,
    device,
    seed=42,
    checkpoint_dir=None,
    budget=None,
    config=None,
):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    seed_everything(seed)
    mixed_precision = str((config or {}).get("mixed_precision", "off")).lower()
    dtype = autocast_dtype(device, mixed_precision)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=dtype is torch.float16 and torch.device(device).type == "cuda",
    )
    history = []
    best = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            mixed_precision=mixed_precision, scaler=scaler,
        )
        model.eval()
        validation_loss = 0.0
        validation_targets = 0
        with torch.no_grad():
            for batch in val_loader:
                frames, mask, labels = _prepare_batch(batch, device)
                if labels.numel() == 0:
                    continue
                with _autocast(device, dtype):
                    output, targets = _forward_batch(model, frames, mask, labels)
                    if targets.numel() == 0:
                        continue
                    validation_loss += nn.functional.cross_entropy(
                        output["logits"], targets, reduction="sum"
                    ).item()
                validation_targets += targets.numel()
        if not validation_targets:
            raise ValueError("validation split has no samples with a valid face sequence")
        validation_loss /= validation_targets
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })
        if checkpoint_dir and validation_loss < best:
            best = validation_loss
            save_checkpoint(
                Path(checkpoint_dir) / f"best-{epoch:03}.pt",
                model,
                optimizer,
                epoch,
                config=config or {"seed": seed},
                budget=budget or VideoStorageBudget(),
            )
    return history


@torch.no_grad()
def _validate_epoch(
    model,
    val_loader,
    device,
    *,
    mixed_precision: str = "off",
    threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    dtype = autocast_dtype(device, mixed_precision)
    total_loss = 0.0
    total_targets = 0
    labels_all: list[int] = []
    probabilities_all: list[float] = []
    skipped = 0
    for batch in val_loader:
        frames = batch["frames"].to(device)
        mask = batch["mask"].to(device)
        batch_labels = batch["label"]
        if frames.ndim in (3, 5):
            valid_sequences = mask.any(dim=1)
            skipped += int((~valid_sequences).sum().item())
            frames = frames[valid_sequences]
            mask = mask[valid_sequences]
            batch_labels = batch_labels[valid_sequences.cpu()]
        if batch_labels.numel() == 0:
            continue
        labels_dev = batch_labels.to(device)
        with _autocast(device, dtype):
            output, targets = _forward_batch(model, frames, mask, labels_dev)
            if targets.numel() == 0:
                continue
            loss = nn.functional.cross_entropy(output["logits"], targets, reduction="sum")
        total_loss += float(loss.item())
        total_targets += int(targets.numel())
        if hasattr(model, "encode_frames") or frames.ndim != 5:
            probs = torch.softmax(output["logits"].float(), dim=1)[:, 1].cpu().tolist()
        else:
            frame_probs = torch.softmax(output["logits"].float(), dim=1)[:, 1]
            probs = []
            offset = 0
            for row_mask in mask:
                count = int(row_mask.sum().item())
                probs.append(float(frame_probs[offset:offset + count].mean().item()))
                offset += count
        labels_all.extend(int(v) for v in batch_labels.tolist())
        probabilities_all.extend(float(v) for v in probs)
    if not total_targets or not labels_all:
        raise ValueError("validation split has no samples with a valid face sequence")
    metrics = binary_metrics(labels_all, probabilities_all, threshold=threshold)
    return {
        "validation_loss": total_loss / total_targets,
        "metrics": metrics,
        "valid_videos": len(labels_all),
        "skipped_videos": skipped,
    }


def _load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _select_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _allocate_run_dirs(architecture: str, data_root: Path, budget: VideoStorageBudget) -> tuple[str, Path, Path]:
    slug = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in architecture.lower()).strip("-")
    exp_root = budget.validate_location(data_root / "experiments")
    ckpt_root = budget.validate_location(data_root / "checkpoints")
    exp_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_id = f"{slug}-{stamp}-{uuid4().hex[:8]}"
        run_dir = exp_root / run_id
        ckpt_dir = ckpt_root / run_id
        if run_dir.exists() or ckpt_dir.exists():
            continue
        budget.validate_destination(run_dir)
        budget.validate_destination(ckpt_dir)
        run_dir.mkdir(exist_ok=False)
        ckpt_dir.mkdir(exist_ok=False)
        return run_id, run_dir, ckpt_dir
    raise RuntimeError("Could not allocate a unique video run ID")


def run_video_training(
    config_path: Path,
    *,
    train_manifest: Path,
    val_manifest: Path,
    test_manifest: Path,
    dataset_root: Path,
    data_root: Path | None = None,
    budget: VideoStorageBudget | None = None,
    run_dir: Path | None = None,
    resume: bool = False,
    until_epoch: int | None = None,
    detector: Any = None,
    crop_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Run bounded, resumable, storage-budgeted video model training."""
    active_budget = budget or (VideoStorageBudget(root=data_root) if data_root else VideoStorageBudget())
    root_dir = active_budget.root
    config_path = Path(config_path).resolve()
    config = _load_json(config_path)
    architecture = str(config["architecture"])
    seed = int(config.get("seed", 42))
    max_epochs = int(config.get("epochs", 20))
    stop_epoch = max_epochs if until_epoch is None else int(until_epoch)
    if stop_epoch < 1 or stop_epoch > max_epochs:
        raise ValueError(f"until_epoch must be in [1, {max_epochs}]")

    train_manifest = active_budget.validate_location(train_manifest)
    val_manifest = active_budget.validate_location(val_manifest)
    test_manifest = active_budget.validate_location(test_manifest)
    dataset_root = active_budget.validate_location(dataset_root)
    for required in (train_manifest, val_manifest, test_manifest):
        if not required.is_file():
            raise FileNotFoundError(f"Missing required manifest: {required}")

    seed_everything(seed)
    device = _select_device(str(config.get("device", "auto")))
    mixed_precision = str(config.get("mixed_precision", "off")).lower()
    if device.type != "cuda" and mixed_precision != "off":
        mixed_precision = "off"

    if run_dir is None:
        if resume:
            raise ValueError("--resume requires --run-dir")
        run_id, run_dir_path, ckpt_dir = _allocate_run_dirs(architecture, root_dir, active_budget)
    else:
        if not resume:
            raise ValueError("--run-dir may only be provided with --resume")
        run_dir_path = active_budget.validate_location(run_dir)
        run_id = run_dir_path.name
        ckpt_dir = active_budget.validate_location(root_dir / "checkpoints" / run_id)
        if not run_dir_path.is_dir() or not ckpt_dir.is_dir():
            raise FileNotFoundError(f"Missing run or checkpoint directory for {run_id}")

    cfg_hash = config_sha256(config)
    snapshot_path = run_dir_path / "config-snapshot.json"
    norm_path = run_dir_path / "normalization.json"
    metadata_path = run_dir_path / "run-metadata.json"
    history_path = run_dir_path / "history.jsonl"
    history_csv_path = run_dir_path / "history.csv"
    best_history_path = run_dir_path / "best-checkpoint-history.jsonl"

    normalize = bool(config.get("normalize_faces", True))
    mean = tuple(float(v) for v in config.get("normalization_mean", IMAGENET_MEAN))
    std = tuple(float(v) for v in config.get("normalization_std", IMAGENET_STD))

    if resume:
        saved_cfg = _load_json(snapshot_path)
        if config_sha256(saved_cfg) != cfg_hash:
            raise ValueError("Resume configuration differs from config-snapshot.json")
        if not norm_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("Resume run is missing normalization.json or run-metadata.json")
        norm_record = _load_json(norm_path)
        mean = tuple(float(v) for v in norm_record["mean"])
        std = tuple(float(v) for v in norm_record["std"])
        normalize = bool(norm_record.get("normalize_faces", normalize))
    else:
        write_json_budgeted(snapshot_path, config, active_budget)
        write_json_budgeted(
            norm_path,
            {
                "normalize_faces": normalize,
                "mean": list(mean),
                "std": list(std),
                "source": "imagenet_standard" if normalize else "raw_0_1",
            },
            active_budget,
        )
        write_json_budgeted(
            metadata_path,
            {
                "run_id": run_id,
                "architecture": architecture,
                "dataset": config.get("dataset", "FaceForensics++ c23"),
                "dataset_root": dataset_root.relative_to(root_dir).as_posix() if dataset_root != root_dir else ".",
                "seed": seed,
                "device": str(device),
                "mixed_precision": mixed_precision,
                "configuration_sha256": cfg_hash,
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "manifest_identifiers": {
                    "train": manifest_identifier(train_manifest, root_dir),
                    "validation": manifest_identifier(val_manifest, root_dir),
                    "test": manifest_identifier(test_manifest, root_dir),
                },
            },
            active_budget,
        )

    max_frames = int(config.get("max_frames", config.get("max_frames_per_video", 12)))
    input_size = int(config.get("input_size", 224))
    max_pixels = int(config.get("max_frame_pixels", 16_000_000))
    batch_size = int(config.get("batch_size", 1))
    num_workers = int(config.get("num_workers", 0))

    cache_path = active_budget.validate_location(crop_cache_dir) if crop_cache_dir is not None else None
    flip_prob = float(config.get("horizontal_flip_probability", 0.0))
    train_dataset = VideoSequenceDataset(
        train_manifest,
        dataset_root,
        max_frames=max_frames,
        size=input_size,
        detector=detector,
        max_frame_pixels=max_pixels,
        normalize=normalize,
        mean=mean,
        std=std,
        skip_errors=True,
        crop_cache_dir=cache_path,
        horizontal_flip_prob=flip_prob,
    )
    val_dataset = VideoSequenceDataset(
        val_manifest,
        dataset_root,
        max_frames=max_frames,
        size=input_size,
        detector=detector,
        max_frame_pixels=max_pixels,
        normalize=normalize,
        mean=mean,
        std=std,
        skip_errors=True,
        crop_cache_dir=cache_path,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sequences,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_sequences,
    )

    model = build_video_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-4)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    dtype = autocast_dtype(device, mixed_precision)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=dtype is torch.float16 and device.type == "cuda",
    )

    checkpoints = sorted(ckpt_dir.glob("checkpoint-epoch-*.pt"))
    start_epoch = 1
    best_score = -math.inf
    best_loss = math.inf
    best_epoch = 0
    bad_epochs = 0

    if resume:
        if not checkpoints:
            raise FileNotFoundError(f"No resumable checkpoint in {ckpt_dir}")
        state = load_checkpoint(checkpoints[-1], model, optimizer, map_location=device)
        if state.get("configuration_sha256") != cfg_hash:
            raise ValueError("Checkpoint configuration hash mismatch on resume")
        if state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        if state.get("scaler_state"):
            scaler.load_state_dict(state["scaler_state"])
        start_epoch = int(state["epoch"]) + 1
        best_score = float(state.get("best_validation_score", -math.inf))
        best_loss = float(state.get("best_validation_loss", math.inf))
        best_epoch = int(state.get("best_epoch", 0))
        bad_epochs = int(state.get("bad_epochs", 0))

    if start_epoch > max_epochs:
        return {
            "run_id": run_id,
            "run_dir": str(run_dir_path),
            "checkpoint_dir": str(ckpt_dir),
            "status": "already_complete",
            "last_epoch": start_epoch - 1,
            "best_epoch": best_epoch,
        }

    patience = int(config.get("early_stopping_patience", 5))
    min_delta = float(config.get("early_stopping_min_delta", 0.0005))
    threshold = float(config.get("evaluation_threshold", 0.5))
    segment_start = start_epoch
    stopped_early = False
    final_epoch = start_epoch - 1

    for epoch in range(start_epoch, stop_epoch + 1):
        generator.manual_seed(seed + epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            mixed_precision=mixed_precision, scaler=scaler,
        )
        val_result = _validate_epoch(
            model, val_loader, device, mixed_precision=mixed_precision, threshold=threshold,
        )
        scheduler.step()
        elapsed = time.perf_counter() - t0

        val_loss = float(val_result["validation_loss"])
        val_metrics = val_result["metrics"]
        val_auc = val_metrics.get("roc_auc")
        if val_auc is not None:
            improved = float(val_auc) > best_score + min_delta
            if improved:
                best_score = float(val_auc)
                best_loss = min(best_loss, val_loss)
        else:
            improved = val_loss < best_loss - min_delta
            if improved:
                best_loss = val_loss

        if improved:
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

        peak_mem = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ckpt_path = ckpt_dir / f"checkpoint-epoch-{epoch:04d}.pt"
        save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            epoch,
            config=config,
            budget=active_budget,
            extra={
                "configuration_sha256": cfg_hash,
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict(),
                "best_validation_score": best_score,
                "best_validation_loss": best_loss,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "normalization": {"normalize_faces": normalize, "mean": list(mean), "std": list(std)},
            },
        )
        rel_ckpt = ckpt_path.relative_to(root_dir).as_posix()
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": val_metrics["accuracy"],
            "validation_precision": val_metrics["precision"],
            "validation_recall": val_metrics["recall"],
            "validation_f1": val_metrics["f1"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_roc_auc": val_metrics["roc_auc"],
            "validation_pr_auc": val_metrics["pr_auc"],
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": elapsed,
            "peak_gpu_memory_bytes": peak_mem,
            "valid_validation_videos": val_result["valid_videos"],
            "skipped_validation_videos": val_result["skipped_videos"],
            "checkpoint": rel_ckpt,
            "best_checkpoint": bool(improved),
        }
        append_jsonl_budgeted(history_path, record, active_budget)
        append_csv_budgeted(
            history_csv_path,
            record,
            HISTORY_CSV_FIELDS,
            active_budget,
            write_header=(epoch == 1 and not resume and not history_csv_path.exists()),
        )
        if improved:
            append_jsonl_budgeted(
                best_history_path,
                {
                    "epoch": epoch,
                    "validation_roc_auc": val_metrics["roc_auc"],
                    "validation_loss": val_loss,
                    "checkpoint": rel_ckpt,
                },
                active_budget,
            )
        final_epoch = epoch
        auc_str = "None" if val_metrics["roc_auc"] is None else f"{float(val_metrics['roc_auc']):.5f}"
        print(
            f"epoch={epoch}/{max_epochs} train_loss={train_loss:.5f} "
            f"val_loss={val_loss:.5f} val_auc={auc_str} "
            f"val_acc={float(val_metrics['accuracy']):.5f} "
            f"val_f1={float(val_metrics['f1']):.5f} "
            f"epoch_seconds={elapsed:.1f} peak_gpu_mib={peak_mem / 1024**2:.1f}",
            flush=True,
        )
        if bad_epochs >= patience:
            stopped_early = True
            break

    append_jsonl_budgeted(
        run_dir_path / "training-segments.jsonl",
        {
            "segment_started_epoch": segment_start,
            "segment_finished_epoch": final_epoch,
            "target_epoch": stop_epoch,
            "configured_max_epochs": max_epochs,
            "resumed": resume,
            "early_stopped": stopped_early,
            "best_epoch_so_far": best_epoch,
            "best_validation_roc_auc": None if math.isinf(best_score) else best_score,
            "best_validation_loss": None if math.isinf(best_loss) else best_loss,
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        active_budget,
    )
    completed = stopped_early or final_epoch >= max_epochs
    if completed:
        plots_dir = run_dir_path / "plots"
        if not plots_dir.exists():
            save_video_training_curves(_read_jsonl(history_path), plots_dir, budget=active_budget)

    return {
        "run_id": run_id,
        "run_dir": str(run_dir_path),
        "checkpoint_dir": str(ckpt_dir),
        "last_epoch": final_epoch,
        "best_epoch": best_epoch,
        "best_validation_roc_auc": None if math.isinf(best_score) else best_score,
        "early_stopped": stopped_early,
        "training_complete": completed,
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a bounded DeepShield video model under the video storage budget.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--val-manifest", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, help="Video storage budget root (defaults to DEEPSHIELD_VIDEO_DATA_ROOT)")
    parser.add_argument("--crop-cache-dir", type=Path, help="Optional pre-extracted lossless crop cache directory inside data_root")
    parser.add_argument("--run-dir", type=Path, help="Existing run directory when using --resume")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--until-epoch", type=int)
    args = parser.parse_args()
    result = run_video_training(
        args.config,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        test_manifest=args.test_manifest,
        dataset_root=args.dataset_root,
        data_root=args.data_root,
        run_dir=args.run_dir,
        resume=args.resume,
        until_epoch=args.until_epoch,
        crop_cache_dir=args.crop_cache_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
