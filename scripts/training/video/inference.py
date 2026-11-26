from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import cv2
import torch

from .checkpoint import load_checkpoint
from .explain import frame_heatmap
from .experiment import config_sha256
from .models import build_video_model
from .preprocess_video import preprocess_video
from .report import forensic_report
from .storage_budget import VideoStorageBudget
from .train import _autocast, autocast_dtype
from .transforms import IMAGENET_MEAN, IMAGENET_STD, normalize_masked_sequence


def _encode_heatmap_jpeg_base64(crop_rgb, score: float, *, preview_size: int = 160) -> str | None:
    overlay = frame_heatmap(crop_rgb, score)
    resized = cv2.resize(overlay, (preview_size, preview_size), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(resized, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def load_video_for_inference(
    run_dir: Path,
    *,
    data_root: Path | None = None,
    budget: VideoStorageBudget | None = None,
    checkpoint_path: Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], torch.device, str]:
    active_budget = budget or (VideoStorageBudget(root=data_root) if data_root else VideoStorageBudget())
    root_dir = active_budget.root
    run_path = active_budget.validate_location(run_dir)
    run_id = run_path.name
    snapshot_path = run_path / "config-snapshot.json"
    norm_path = run_path / "normalization.json"
    best_history_path = run_path / "best-checkpoint-history.jsonl"
    if not snapshot_path.is_file() or not norm_path.is_file() or not best_history_path.is_file():
        raise FileNotFoundError(f"Video run {run_id} is missing snapshot, normalization, or best checkpoint history")
    config = json.loads(snapshot_path.read_text(encoding="utf-8"))
    norm = json.loads(norm_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in best_history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise FileNotFoundError(f"No best validation checkpoint recorded for {run_id}")
    ckpt_dir = active_budget.validate_location(root_dir / "checkpoints" / run_id)
    target_ckpt = active_budget.validate_location(
        checkpoint_path if checkpoint_path is not None else root_dir / rows[-1]["checkpoint"]
    )
    if target_ckpt.parent != ckpt_dir or not target_ckpt.is_file():
        raise ValueError(f"Video checkpoint is missing or outside run directory: {target_ckpt}")

    state = load_checkpoint(target_ckpt, map_location="cpu")
    if int(state["epoch"]) != int(rows[-1]["epoch"]):
        raise ValueError("Selected checkpoint epoch does not match best-checkpoint-history.jsonl")
    if state.get("configuration_sha256") and state["configuration_sha256"] != config_sha256(config):
        raise ValueError("Checkpoint configuration SHA-256 does not match config-snapshot.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_video_model(config)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    runtime = {
        "config": config,
        "normalization": norm,
        "max_frames": int(config.get("max_frames", config.get("max_frames_per_video", 12))),
        "input_size": int(config.get("input_size", 224)),
        "mixed_precision": str(config.get("mixed_precision", "off")).lower() if device.type == "cuda" else "off",
        "epoch": int(state["epoch"]),
    }
    identifier = f"{config['architecture']}:{run_id}:epoch-{state['epoch']}"
    return model, runtime, device, identifier


@torch.no_grad()
def predict_video(
    path,
    model,
    device="cpu",
    max_frames=12,
    size=224,
    detector=None,
    max_frame_pixels=16_000_000,
    *,
    normalize: bool = False,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    mixed_precision: str = "off",
    include_heatmaps: bool = False,
):
    prepared = preprocess_video(path, detector, max_frames, size=size, max_frame_pixels=max_frame_pixels)
    if prepared.get("error"):
        return {"status": "error", "error": prepared["error"], "metadata": prepared["metadata"]}
    if not prepared["frames"]:
        return {
            "status": "inconclusive",
            "reason": "no_reliable_face_track",
            "metadata": prepared["metadata"],
        }

    sequence = torch.zeros(1, max_frames, 3, size, size, dtype=torch.float32)
    mask = torch.zeros(1, max_frames, dtype=torch.bool)
    count = min(max_frames, len(prepared["frames"]))
    for index, item in enumerate(prepared["frames"][:count]):
        sequence[0, index] = torch.from_numpy(item["crop"].transpose(2, 0, 1).copy()).float() / 255.0
        mask[0, index] = True
    if normalize and mask.any():
        sequence[0] = normalize_masked_sequence(sequence[0], mask[0], mean=mean, std=std)

    model.eval()
    dtype = autocast_dtype(device, mixed_precision)
    if hasattr(model, "encode_frames"):
        with _autocast(device, dtype):
            output = model(sequence.to(device), mask.to(device))
        probability = torch.softmax(output["logits"].float(), dim=1)[0, 1].item()
        frame_scores = output["frame_scores"][0, :count].detach().cpu().tolist()
    else:
        with _autocast(device, dtype):
            logits = model(sequence[0, :count].to(device))
        frame_probabilities = torch.softmax(logits.float(), dim=1)[:, 1]
        probability = frame_probabilities.mean().item()
        frame_scores = frame_probabilities.detach().cpu().tolist()

    evidence = []
    for item, score in zip(prepared["frames"][:count], frame_scores):
        row: dict[str, Any] = {
            "frame_index": item["frame_index"],
            "timestamp_seconds": item["timestamp_seconds"],
            "score": float(score),
        }
        if include_heatmaps:
            heatmap_b64 = _encode_heatmap_jpeg_base64(item["crop"], float(score))
            if heatmap_b64 is not None:
                row["heatmap_jpeg_base64"] = heatmap_b64
        evidence.append(row)

    return {
        "status": "ok",
        "report": forensic_report(
            prepared["metadata"].get("file_hash"),
            probability,
            evidence,
            prepared["metadata"],
            type(model).__name__,
            {
                "size": size,
                "sampling": "uniform",
                "max_frames": max_frames,
                "face_detector": prepared["face_detector"],
                "normalized": bool(normalize),
            },
            prepared.get("errors"),
        ),
    }
