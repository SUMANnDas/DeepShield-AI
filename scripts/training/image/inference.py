"""Integration-ready single and batch image inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from PIL import Image

from scripts.training.image.checkpoint import load_checkpoint
from scripts.training.image.data import INDEX_TO_LABEL, EXPECTED_SIZE
from scripts.training.image.experiment import CHECKPOINT_ROOT, PROJECT_ROOT
from scripts.training.image.models import build_model
from scripts.training.image.train import _load_config
from scripts.training.image.transforms import ImageTransform
from scripts.training.image.utils import select_device


def _image_to_rgb(image: str | Path | Image.Image) -> np.ndarray:
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        path = Path(image)
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != (EXPECTED_SIZE[1], EXPECTED_SIZE[0]):
        raise ValueError(f"Expected a 32x32 input image, got {rgb.shape[1]}x{rgb.shape[0]}")
    return np.ascontiguousarray(rgb)


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
        rgb = _image_to_rgb(image)
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
        results.append({
            "predicted_label": INDEX_TO_LABEL[predicted_index],
            "real_probability": real_probability,
            "fake_probability": fake_probability,
            "confidence": max(real_probability, fake_probability),
            "model_identifier": model_identifier,
            "inference_duration_ms": duration_ms,
            "calibrated_forensic_certainty": False,
        })
    return results


def load_for_inference(config_path: Path, run_dir: Path) -> tuple[torch.nn.Module, ImageTransform, torch.device, str]:
    config = _load_config(Path(config_path))
    run_dir = Path(run_dir).resolve()
    run_id = run_dir.name
    checkpoint_dir = CHECKPOINT_ROOT / run_id
    with (run_dir / "best-checkpoint-history.jsonl").open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise FileNotFoundError("No best validation checkpoint has been recorded")
    checkpoint_path = PROJECT_ROOT / rows[-1]["checkpoint"]
    if checkpoint_path.parent.resolve() != checkpoint_dir.resolve():
        raise ValueError("Checkpoint is outside the run-specific checkpoint directory")
    state = load_checkpoint(checkpoint_path, map_location="cpu")
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
