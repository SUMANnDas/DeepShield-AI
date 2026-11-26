"""CNN Grad-CAM and ViT patch-level gradient attribution."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from scripts.training.image.checkpoint import load_checkpoint
from scripts.training.image.data import build_train_validation_loaders
from scripts.training.image.experiment import CHECKPOINT_ROOT, PROJECT_ROOT, write_json_exclusive
from scripts.training.image.models import build_model
from scripts.training.image.train import _load_config
from scripts.training.image.utils import select_device


def _normalize_map(values: torch.Tensor) -> torch.Tensor:
    values = values - values.min()
    maximum = values.max()
    if float(maximum.item()) > 0:
        values = values / maximum
    return values


def gradcam(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_layer: torch.nn.Module,
    *,
    class_index: int | None = None,
) -> np.ndarray:
    """Create a 32x32 Grad-CAM map from a final convolutional layer."""
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(image)
        selected = int(logits.argmax(dim=1).item()) if class_index is None else int(class_index)
        logits[0, selected].backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")
        activation = activations[-1]
        gradient = gradients[-1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        heatmap = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        heatmap = F.interpolate(heatmap, size=(32, 32), mode="bilinear", align_corners=False)[0, 0]
        heatmap = _normalize_map(heatmap)
        return heatmap.detach().cpu().numpy().astype(np.float32)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def vit_patch_attribution(
    model: torch.nn.Module,
    image: torch.Tensor,
    *,
    class_index: int | None = None,
    patch_size: int = 4,
) -> np.ndarray:
    """Aggregate absolute gradient-times-input into a patch-grid attribution map."""
    if image.shape[-2:] != (32, 32) or 32 % patch_size:
        raise ValueError("ViT attribution expects a 32x32 image and a divisor patch size")
    model.eval()
    model.zero_grad(set_to_none=True)
    source = image.detach().clone().requires_grad_(True)
    logits = model(source)
    selected = int(logits.argmax(dim=1).item()) if class_index is None else int(class_index)
    logits[0, selected].backward()
    saliency = (source.grad * source).abs().mean(dim=1, keepdim=True)
    grid = 32 // patch_size
    patch_values = saliency.reshape(1, 1, grid, patch_size, grid, patch_size).mean(dim=(3, 5))[0, 0]
    patch_values = _normalize_map(patch_values)
    model.zero_grad(set_to_none=True)
    return patch_values.detach().cpu().numpy().astype(np.float32)


def make_overlay(image_rgb: np.ndarray, attribution: np.ndarray) -> np.ndarray:
    heat = np.clip(attribution * 255.0, 0, 255).astype(np.uint8)
    color_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 0.58, color_rgb, 0.42, 0)


def _gradcam_layer(model: torch.nn.Module, architecture: str) -> torch.nn.Module:
    key = architecture.lower()
    if key == "resnet50":
        return model.layer4[-1]
    if key == "efficientnet-b0":
        return model.features[-1]
    raise ValueError(f"No CNN Grad-CAM target configured for {architecture}")


def _encode_rgb_png_base64(image_rgb: np.ndarray, *, preview_size: int = 192) -> str:
    upscaled = cv2.resize(image_rgb, (preview_size, preview_size), interpolation=cv2.INTER_NEAREST)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode explainability preview PNG")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def explain_prepared_image(
    model: torch.nn.Module,
    architecture: str,
    rgb_32x32: np.ndarray,
    tensor: torch.Tensor,
    *,
    class_index: int | None = None,
    preview_size: int = 192,
) -> dict[str, Any]:
    """Generate native 32x32 Grad-CAM or ViT patch attribution for a single input."""
    key = architecture.strip().lower().replace("_", "-")
    if tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[-2:] != (32, 32):
        raise ValueError("Expected tensor shape (1, C, 32, 32)")
    if rgb_32x32.shape != (32, 32, 3):
        raise ValueError("Expected rgb_32x32 shape (32, 32, 3)")
    if class_index is None:
        with torch.inference_mode():
            selected = int(model(tensor).argmax(dim=1).item())
    else:
        selected = int(class_index)

    if key in {"vit-small", "vit"}:
        attribution = vit_patch_attribution(model, tensor, class_index=selected)
        expanded = cv2.resize(attribution, (32, 32), interpolation=cv2.INTER_CUBIC)
        overlay = make_overlay(rgb_32x32, expanded)
        kind = "vit_patch_gradient_input"
        method_label = "ViT patch gradient × input (8×8 patches)"
        grid_shape = [int(attribution.shape[0]), int(attribution.shape[1])]
    else:
        attribution = gradcam(model, tensor, _gradcam_layer(model, key), class_index=selected)
        overlay = make_overlay(rgb_32x32, attribution)
        kind = "gradcam"
        method_label = "Grad-CAM (final convolutional block)"
        grid_shape = [int(attribution.shape[0]), int(attribution.shape[1])]

    return {
        "method": kind,
        "method_label": method_label,
        "target_class": "FAKE" if selected == 1 else "REAL",
        "resolution": [32, 32],
        "grid_shape": grid_shape,
        "model_input_png_base64": _encode_rgb_png_base64(rgb_32x32, preview_size=preview_size),
        "overlay_png_base64": _encode_rgb_png_base64(overlay, preview_size=preview_size),
        "limitation": "Coarse 32x32 visual attribution highlights regions influencing the raw score; it is not causal forensic proof.",
    }


def explain_validation_examples(config_path: Path, run_dir: Path, *, examples_per_class: int = 2) -> Path:
    config = _load_config(Path(config_path))
    run_dir = Path(run_dir).resolve()
    run_id = run_dir.name
    checkpoint_dir = CHECKPOINT_ROOT / run_id
    best_records = []
    with (run_dir / "best-checkpoint-history.jsonl").open("r", encoding="utf-8") as handle:
        best_records = [json.loads(line) for line in handle if line.strip()]
    if not best_records:
        raise FileNotFoundError("No validation-selected checkpoint is available for explainability")
    checkpoint_path = PROJECT_ROOT / best_records[-1]["checkpoint"]
    if checkpoint_path.parent.resolve() != checkpoint_dir.resolve():
        raise ValueError("Explainability checkpoint is outside the matching run directory")
    with (run_dir / "normalization.json").open("r", encoding="utf-8") as handle:
        norm = json.load(handle)
    device = select_device(str(config.get("device", "auto")))
    model = build_model(str(config["architecture"])).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    _, validation_loader, _ = build_train_validation_loaders(
        PROJECT_ROOT / config["dataset_root"],
        PROJECT_ROOT / config["train_manifest"],
        PROJECT_ROOT / config["validation_manifest"],
        mean=norm["mean"], std=norm["std"],
        batch_size=int(config["batch_size"]), seed=int(config["seed"]),
        num_workers=0, pin_memory=False,
        horizontal_flip_probability=0.0, crop_padding=0,
    )
    dataset = validation_loader.dataset
    output_dir = run_dir / "explainability"
    output_dir.mkdir(exist_ok=False)
    chosen: list[int] = []
    per_class = {0: 0, 1: 0}
    for index, sample in enumerate(dataset.samples):
        if per_class[sample.target] < examples_per_class:
            chosen.append(index)
            per_class[sample.target] += 1
        if all(value >= examples_per_class for value in per_class.values()):
            break
    if any(value == 0 for value in per_class.values()):
        raise ValueError("Validation manifest does not contain both classes")

    records: list[dict[str, Any]] = []
    architecture = str(config["architecture"])
    for index in chosen:
        sample = dataset.samples[index]
        rgb = dataset.get_rgb(index)
        tensor, label = dataset[index]
        batch = tensor.unsqueeze(0).to(device)
        if architecture == "vit-small":
            attribution = vit_patch_attribution(model, batch)
            expanded = cv2.resize(attribution, (32, 32), interpolation=cv2.INTER_CUBIC)
            overlay = make_overlay(rgb, expanded)
            kind = "vit_patch_gradient_input"
        else:
            attribution = gradcam(model, batch, _gradcam_layer(model, architecture))
            overlay = make_overlay(rgb, attribution)
            kind = "gradcam"
        token = f"{sample.sample_id}-{kind}.png"
        output_path = output_dir / token
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite explainability output: {output_path}")
        if not cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not write explainability image: {output_path}")
        with torch.inference_mode():
            prediction = int(torch.softmax(model(batch), dim=1).argmax(1).item())
        records.append({
            "sample_id": sample.sample_id,
            "split": "validation",
            "true_label": sample.label_name,
            "predicted_label": "FAKE" if prediction == 1 else "REAL",
            "method": kind,
            "output": str(output_path.relative_to(PROJECT_ROOT)),
            "input_resolution": [32, 32],
        })
    metadata_path = output_dir / "manifest.json"
    write_json_exclusive(metadata_path, {
        "architecture": architecture,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "source_split": "validation",
        "records": records,
        "limitation": "Attribution is coarse at 32x32 and is not forensic proof.",
    })
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--examples-per-class", type=int, default=2)
    args = parser.parse_args()
    print(explain_validation_examples(args.config, args.run_dir, examples_per_class=args.examples_per_class))


if __name__ == "__main__":
    main()
