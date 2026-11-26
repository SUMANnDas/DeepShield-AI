from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .dataset import VideoSequenceDataset, collate_sequences
from .experiment import append_jsonl_budgeted, config_sha256, manifest_identifier, write_json_budgeted
from .metrics import (
    binary_metrics,
    compute_manipulation_strata,
    save_video_confusion_matrix,
    save_video_roc_curve,
)
from .models import build_video_model
from .storage_budget import VideoStorageBudget
from .train import _autocast, autocast_dtype
from .transforms import IMAGENET_MEAN, IMAGENET_STD


@torch.no_grad()
def evaluate(model, loader, device="cpu", mixed_precision="off", threshold: float = 0.5):
    model.eval()
    dtype = autocast_dtype(device, mixed_precision)
    labels = []
    probabilities = []
    rows = []
    skipped = []
    for batch in loader:
        frames = batch["frames"].to(device)
        mask = batch["mask"].to(device)
        targets = batch["label"].tolist()
        video_ids = batch.get("video_id", [str(len(rows) + i) for i in range(len(targets))])
        methods = batch.get("manipulation_method", ["unknown"] * len(targets))
        if frames.ndim in (3, 5):
            valid_sequences = mask.any(dim=1)
            for index, valid in enumerate(valid_sequences.tolist()):
                if not valid:
                    skipped.append({"video_id": video_ids[index], "reason": "no_valid_face_sequence"})
            keep = valid_sequences.nonzero(as_tuple=False).flatten()
            if not len(keep):
                continue
            frames = frames[keep]
            mask = mask[keep]
            targets = [targets[index] for index in keep.tolist()]
            video_ids = [video_ids[index] for index in keep.tolist()]
            methods = [methods[index] for index in keep.tolist()]
        if frames.ndim == 5:
            if hasattr(model, "encode_frames"):
                with _autocast(device, dtype):
                    output = model(frames, mask)
                batch_probabilities = torch.softmax(output["logits"].float(), dim=1)[:, 1].cpu().tolist()
                frame_scores = output.get("frame_scores")
            else:
                valid_frames = mask.reshape(-1)
                flat = frames.reshape(-1, *frames.shape[2:])[valid_frames]
                with _autocast(device, dtype):
                    logits = model(flat)
                frame_probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
                batch_probabilities = []
                offset = 0
                for row_mask in mask:
                    count = int(row_mask.sum())
                    batch_probabilities.append(frame_probabilities[offset:offset + count].mean().item())
                    offset += count
                frame_scores = None
        elif frames.ndim == 3:
            with _autocast(device, dtype):
                output = model(frames, mask)
            batch_probabilities = torch.softmax(output["logits"].float(), dim=1)[:, 1].cpu().tolist()
            frame_scores = output.get("frame_scores")
        else:
            with _autocast(device, dtype):
                output = {"logits": model(frames)}
            batch_probabilities = torch.softmax(output["logits"].float(), dim=1)[:, 1].cpu().tolist()
            frame_scores = None

        probabilities.extend(batch_probabilities)
        labels.extend(targets)
        for index, probability in enumerate(batch_probabilities):
            row = {
                "video_id": video_ids[index],
                "label": int(targets[index]),
                "manipulation_method": methods[index],
                "probability_manipulated": float(probability),
            }
            if frame_scores is not None:
                row["frame_scores"] = frame_scores[index][mask[index]].detach().cpu().tolist()
            rows.append(row)

    if not labels:
        return {
            "status": "no_valid_face_sequences",
            "metrics": None,
            "manipulation_strata": {},
            "videos": [],
            "skipped": skipped,
        }
    return {
        "status": "evaluated",
        "metrics": binary_metrics(labels, probabilities, threshold=threshold),
        "manipulation_strata": compute_manipulation_strata(rows, threshold=threshold),
        "videos": rows,
        "skipped": skipped,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_video_run(
    config_path: Path,
    run_dir: Path,
    test_manifest: Path,
    dataset_root: Path,
    *,
    data_root: Path | None = None,
    budget: VideoStorageBudget | None = None,
    split: str = "test",
    checkpoint_path: Path | None = None,
    allow_incomplete: bool = False,
    detector: Any = None,
    crop_cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate a trained video run on a grouped split and record budgeted artifacts."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    active_budget = budget or (VideoStorageBudget(root=data_root) if data_root else VideoStorageBudget())
    root_dir = active_budget.root
    run_dir_path = active_budget.validate_location(run_dir)
    run_id = run_dir_path.name
    ckpt_dir = active_budget.validate_location(root_dir / "checkpoints" / run_id)
    if not run_dir_path.is_dir() or not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Run or checkpoint directory does not exist for {run_id}")

    if split == "test" and not allow_incomplete:
        segments = _read_jsonl(run_dir_path / "training-segments.jsonl")
        if not segments:
            raise RuntimeError("No training segment is recorded")
        last = segments[-1]
        if not last.get("early_stopped") and int(last["segment_finished_epoch"]) < int(last["configured_max_epochs"]):
            raise RuntimeError("Test evaluation is blocked until training is complete")

    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    snapshot = json.loads((run_dir_path / "config-snapshot.json").read_text(encoding="utf-8"))
    if config_sha256(config) != config_sha256(snapshot):
        raise ValueError("Evaluation config differs from run config-snapshot.json")

    best_entries = _read_jsonl(run_dir_path / "best-checkpoint-history.jsonl")
    if checkpoint_path is None:
        if not best_entries:
            raise FileNotFoundError("No validation-selected checkpoint recorded")
        selected_ckpt = active_budget.validate_location(root_dir / best_entries[-1]["checkpoint"])
    else:
        selected_ckpt = active_budget.validate_location(checkpoint_path)
    if selected_ckpt.parent != ckpt_dir or not selected_ckpt.is_file():
        raise FileNotFoundError(f"Selected checkpoint invalid or missing: {selected_ckpt}")

    norm = json.loads((run_dir_path / "normalization.json").read_text(encoding="utf-8"))
    normalize = bool(norm.get("normalize_faces", True))
    mean = tuple(float(v) for v in norm.get("mean", IMAGENET_MEAN))
    std = tuple(float(v) for v in norm.get("std", IMAGENET_STD))

    manifest_path = active_budget.validate_location(test_manifest)
    ds_root = active_budget.validate_location(dataset_root)
    max_frames = int(snapshot.get("max_frames", snapshot.get("max_frames_per_video", 12)))
    input_size = int(snapshot.get("input_size", 224))
    max_pixels = int(snapshot.get("max_frame_pixels", 16_000_000))
    batch_size = int(snapshot.get("batch_size", 1))
    threshold = float(snapshot.get("evaluation_threshold", 0.5))

    cache_path = active_budget.validate_location(crop_cache_dir) if crop_cache_dir is not None else None
    dataset = VideoSequenceDataset(
        manifest_path,
        ds_root,
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_sequences)

    requested_device = str(snapshot.get("device", "auto"))
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else ("cpu" if requested_device == "auto" else requested_device))
    mixed_precision = str(snapshot.get("mixed_precision", "off")).lower()
    if device.type != "cuda":
        mixed_precision = "off"

    model = build_video_model(snapshot).to(device)
    state = load_checkpoint(selected_ckpt, model, map_location=device)
    if state.get("configuration_sha256") and state["configuration_sha256"] != config_sha256(snapshot):
        raise ValueError("Checkpoint configuration SHA-256 does not match run snapshot")

    eval_result = evaluate(model, loader, device=device, mixed_precision=mixed_precision, threshold=threshold)
    if eval_result["status"] != "evaluated" or eval_result["metrics"] is None:
        raise RuntimeError("No valid face sequences were evaluated in the target manifest")

    record = {
        "run_id": run_id,
        "architecture": snapshot["architecture"],
        "dataset": snapshot.get("dataset", "FaceForensics++ c23"),
        "split": split,
        "source_split_grouped": True,
        "checkpoint": selected_ckpt.relative_to(root_dir).as_posix(),
        "checkpoint_epoch": int(state["epoch"]),
        "seed": int(snapshot.get("seed", 42)),
        "dataset_manifest": manifest_identifier(manifest_path, root_dir),
        "metrics": eval_result["metrics"],
        "manipulation_strata": eval_result["manipulation_strata"],
        "evaluated_videos": len(eval_result["videos"]),
        "skipped_videos": eval_result["skipped"],
        "device": str(device),
        "test_set_used_for_selection": False if split == "test" else None,
    }
    metrics_path = write_json_budgeted(run_dir_path / f"{split}-metrics.json", record, active_budget)

    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(["video_id", "target", "manipulation_method", "probability_manipulated", "predicted_label"])
    for item in eval_result["videos"]:
        prob = float(item["probability_manipulated"])
        writer.writerow([
            item["video_id"],
            item["label"],
            item.get("manipulation_method", "unknown"),
            f"{prob:.10f}",
            int(prob >= threshold),
        ])
    csv_bytes = csv_buffer.getvalue().encode("utf-8")
    csv_target = active_budget.validate_destination(run_dir_path / f"{split}-predictions.csv")
    active_budget.require_capacity(len(csv_bytes), purpose=f"writing {csv_target.name}")
    with csv_target.open("xb") as handle:
        handle.write(csv_bytes)

    save_video_confusion_matrix(
        eval_result["metrics"],
        run_dir_path / f"{split}-confusion-matrix.png",
        title=f"{snapshot['architecture']} — {split}",
        budget=active_budget,
    )
    save_video_roc_curve(
        [int(item["label"]) for item in eval_result["videos"]],
        [float(item["probability_manipulated"]) for item in eval_result["videos"]],
        run_dir_path / f"{split}-roc-curve.png",
        title=f"{snapshot['architecture']} — {split}",
        budget=active_budget,
    )
    append_jsonl_budgeted(
        run_dir_path / "evaluation-events.jsonl",
        {
            "split": split,
            "metrics_path": metrics_path.relative_to(root_dir).as_posix(),
            "manifest_sha256": record["dataset_manifest"]["sha256"],
            "checkpoint_epoch": int(state["epoch"]),
        },
        active_budget,
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained video model checkpoint on a grouped manifest.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--crop-cache-dir", type=Path)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = evaluate_video_run(
        args.config,
        args.run_dir,
        args.manifest,
        args.dataset_root,
        data_root=args.data_root,
        split=args.split,
        checkpoint_path=args.checkpoint,
        allow_incomplete=args.allow_incomplete,
        crop_cache_dir=args.crop_cache_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
