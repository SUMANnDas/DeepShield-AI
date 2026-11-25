"""Integration-ready single and batch image inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import hashlib
import numpy as np
import torch
from PIL import Image

from scripts.training.image.checkpoint import load_checkpoint
from scripts.training.image.data import INDEX_TO_LABEL
from scripts.training.image.experiment import CHECKPOINT_ROOT, PROJECT_ROOT
from scripts.training.image.models import build_model
from scripts.training.image.preprocessing import prepare_model_rgb
from scripts.training.image.train import _load_config
from scripts.training.image.transforms import ImageTransform
from scripts.training.image.utils import select_device


def _image_to_rgb(image: str | Path | Image.Image, transform: ImageTransform) -> tuple[np.ndarray, dict[str, object]]:
    if isinstance(image, Image.Image):
        source = image
        prepared, metadata = prepare_model_rgb(
            source,
            size=32,
            fill_rgb=tuple(round(float(value) * 255) for value in transform.mean.flatten().tolist()),
        )
    else:
        path = Path(image)
        with Image.open(path) as source:
            prepared, metadata = prepare_model_rgb(
                source,
                size=32,
                fill_rgb=tuple(round(float(value) * 255) for value in transform.mean.flatten().tolist()),
            )
    return prepared, metadata


def predict_images(
    model: torch.nn.Module,
    images: Sequence[str | Path | Image.Image],
    *,
    transform: ImageTransform,
    model_identifier: str,
    device: torch.device,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    model.eval()
    for image in images:
        rgb, preparation = _image_to_rgb(image, transform)
        tensor = transform(rgb).unsqueeze(0).to(device)
        start = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(model(tensor).float(), dim=1)[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        duration_ms = (time.perf_counter() - start) * 1000.0
        real_probability = float(probabilities[0].item())
        fake_probability = float(probabilities[1].item())
        predicted_index = int(probabilities.argmax().item())
        predicted_label = INDEX_TO_LABEL[predicted_index]
        native_resolution = preparation["source_width"] == 32 and preparation["source_height"] == 32
        results.append({
            "predicted_label": predicted_label,
            "real_probability": real_probability,
            "fake_probability": fake_probability,
            "confidence": max(real_probability, fake_probability),
            "assessment": {
                "status": "model_indication" if native_resolution else "inconclusive",
                "decision": predicted_label if native_resolution else "INCONCLUSIVE",
                "raw_model_label": predicted_label,
                "reason": (
                    "The source image is 32x32, matching CIFAKE's training resolution. This remains a CIFAKE-specific model indication."
                    if native_resolution
                    else "The source image is not 32x32. Its resized model score is outside the validated CIFAKE resolution domain."
                ),
            },
            "preprocessing": preparation,
            "model_identifier": model_identifier,
            "inference_duration_ms": duration_ms,
            "calibrated_forensic_certainty": False,
        })
    return results


def load_for_inference(config_path: Path, run_dir: Path) -> tuple[torch.nn.Module, ImageTransform, torch.device, str]:
    run_dir = Path(run_dir).resolve()
    run_id = run_dir.name
    checkpoint_dir = CHECKPOINT_ROOT / run_id
    snapshot_path = run_dir / "config-snapshot.json"
    if not snapshot_path.is_file():
        raise FileNotFoundError("The checkpoint run is missing its configuration snapshot.")
    config = _load_config(snapshot_path)
    with (run_dir / "best-checkpoint-history.jsonl").open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise FileNotFoundError("No best validation checkpoint has been recorded")
    checkpoint_path = (PROJECT_ROOT / rows[-1]["checkpoint"]).resolve()
    if checkpoint_path.parent != checkpoint_dir.resolve():
        raise ValueError("Checkpoint is outside the run-specific checkpoint directory")
    state = load_checkpoint(checkpoint_path, map_location="cpu")
    if int(state["epoch"]) != int(rows[-1]["epoch"]):
        raise ValueError("Selected checkpoint epoch does not match best validation history")
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected_hash = hashlib.sha256(canonical).hexdigest()
    if state.get("configuration_sha256") != expected_hash:
        raise ValueError("Checkpoint configuration does not match its run snapshot")
    if config.get("architecture") not in {"resnet50", "efficientnet-b0", "vit-small"}:
        raise ValueError("Unsupported architecture in checkpoint run snapshot")
    device = select_device(str(config.get("device", "auto")))
    model = build_model(str(config["architecture"]))
    model.load_state_dict(state["model_state"])
    model.to(device).eval()
    with (run_dir / "normalization.json").open("r", encoding="utf-8") as handle:
        norm = json.load(handle)
    transform = ImageTransform(norm["mean"], norm["std"], training=False)
    return model, transform, device, f"{config['architecture']}:{run_id}:epoch-{state['epoch']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()
    model, transform, device, model_id = load_for_inference(args.config, args.run_dir)
    result = predict_images(model, args.images, transform=transform, model_identifier=model_id, device=device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
