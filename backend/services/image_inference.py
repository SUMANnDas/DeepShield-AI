from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from scripts.training.image.checkpoint import load_checkpoint
from scripts.training.image.data import INDEX_TO_LABEL
from scripts.training.image.explain import explain_prepared_image
from scripts.training.image.models import build_model
from scripts.training.image.preprocessing import prepare_model_rgb
from scripts.training.image.transforms import ImageTransform
from scripts.training.image.utils import select_device
from scripts.training.image.train import _load_config

from backend.services.model_registry import MODELS, list_models, model_paths

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageInputError(ValueError):
    pass


class ImageInferenceService:
    """Loads one verified CIFAKE checkpoint at a time to stay within VRAM limits."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._loaded_id: str | None = None
        self._architecture: str | None = None
        self._model: torch.nn.Module | None = None
        self._transform: ImageTransform | None = None
        self._device = torch.device("cpu")
        self._identifier: str | None = None

    @staticmethod
    def validate_image(data: bytes, filename: str, declared_mime: str | None) -> tuple[Image.Image, dict[str, Any]]:
        if not data:
            raise ImageInputError("The uploaded file is empty.")
        if len(data) > MAX_IMAGE_BYTES:
            raise ImageInputError("Image exceeds the 10 MB upload limit.")
        suffix = Path(filename or "").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ImageInputError("Use a JPEG, PNG, or WebP image.")
        if declared_mime and declared_mime.lower() not in {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}:
            raise ImageInputError("The declared content type is not an accepted image type.")
        expected_formats = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP"}
        try:
            with Image.open(__import__("io").BytesIO(data)) as probe:
                actual_format = (probe.format or "").upper()
                original_width, original_height = probe.size
                if actual_format not in SUPPORTED_FORMATS:
                    raise ImageInputError("The file content is not a supported image format.")
                if expected_formats[suffix] != actual_format:
                    raise ImageInputError("The filename extension does not match the image content.")
                expected_mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[actual_format]
                if declared_mime and declared_mime.lower() not in {expected_mime, "application/octet-stream"}:
                    raise ImageInputError("The declared content type does not match the image content.")
                if original_width < 1 or original_height < 1 or original_width * original_height > MAX_IMAGE_PIXELS:
                    raise ImageInputError("Image dimensions are invalid or exceed the 40 megapixel limit.")
                probe.verify()
            with Image.open(__import__("io").BytesIO(data)) as image:
                image.load()
                orientation = image.getexif().get(274)
                rgb = ImageOps.exif_transpose(image).convert("RGB")
        except ImageInputError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise ImageInputError("The uploaded file is corrupt or cannot be decoded as an image.") from exc
        width, height = rgb.size
        return rgb, {
            "format": actual_format,
            "width": width,
            "height": height,
            "original_width": original_width,
            "original_height": original_height,
            "exif_orientation": int(orientation) if orientation is not None else None,
            "exif_orientation_applied": orientation not in (None, 1),
            "bytes": len(data),
        }

    def _load(self, model_id: str) -> None:
        if model_id not in MODELS:
            raise KeyError(model_id)
        if self._loaded_id == model_id and self._model is not None:
            return
        self._model = None
        self._transform = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        config_path, run_dir, checkpoint_path, _ = model_paths(model_id)
        # The run snapshot is immutable and authoritative. The top-level training
        # config may have changed after this checkpoint was produced.
        snapshot_path = run_dir / "config-snapshot.json"
        if not snapshot_path.is_file():
            raise RuntimeError("The checkpoint run is missing its configuration snapshot.")
        config = _load_config(snapshot_path)
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        model_record = next((entry for entry in list_models() if entry["id"] == model_id), None)
        if model_record is None or model_record.get("status") != "available":
            raise RuntimeError("The selected model registry entry is unavailable.")
        if int(checkpoint["epoch"]) != int(model_record["checkpoint"]["epoch"]):
            raise RuntimeError("Checkpoint epoch does not match its saved test evaluation record.")
        canonical_config = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        expected_hash = hashlib.sha256(canonical_config).hexdigest()
        if checkpoint.get("configuration_sha256") != expected_hash:
            raise RuntimeError("Checkpoint configuration does not match its run snapshot.")
        if config.get("architecture") != MODELS[model_id]["architecture"]:
            raise RuntimeError("The run snapshot architecture does not match the selected model.")
        model = build_model(str(config["architecture"]))
        model.load_state_dict(checkpoint["model_state"], strict=True)
        norm = json.loads((run_dir / "normalization.json").read_text(encoding="utf-8"))
        device = select_device("auto")
        model.to(device).eval()
        self._model = model
        self._transform = ImageTransform(norm["mean"], norm["std"], training=False)
        self._device = device
        self._loaded_id = model_id
        self._architecture = str(config["architecture"])
        self._identifier = f"{config['architecture']}:{MODELS[model_id]['run_id']}:epoch-{checkpoint['epoch']}"

    def predict(self, model_id: str, image: Image.Image, *, include_explainability: bool = True) -> dict[str, Any]:
        with self._lock:
            self._load(model_id)
            assert self._model is not None and self._transform is not None and self._architecture is not None
            width, height = image.size
            assert self._transform is not None
            mean_rgb = tuple(round(float(value) * 255) for value in self._transform.mean.flatten().tolist())
            array, resize_metadata = prepare_model_rgb(image, size=32, fill_rgb=mean_rgb)
            tensor = self._transform(array).unsqueeze(0).to(self._device)
            start = time.perf_counter()
            with torch.inference_mode():
                probs = torch.softmax(self._model(tensor).float(), dim=1)[0]
                if self._device.type == "cuda":
                    torch.cuda.synchronize(self._device)
            elapsed = (time.perf_counter() - start) * 1000
            predicted_index = int(probs.argmax().item())
            predicted_label = INDEX_TO_LABEL[predicted_index]
            explainability = (
                explain_prepared_image(
                    self._model,
                    self._architecture,
                    array,
                    tensor,
                    class_index=predicted_index,
                )
                if include_explainability
                else None
            )
            native_resolution = (width, height) == (32, 32)
            assessment = {
                "status": "model_indication" if native_resolution else "inconclusive",
                "decision": predicted_label if native_resolution else "INCONCLUSIVE",
                "raw_model_label": predicted_label,
                "reason": (
                    "The source image is 32x32, matching CIFAKE's training resolution. "
                    "This remains a CIFAKE-specific model indication, not forensic certainty."
                    if native_resolution
                    else "The source image is not 32x32. Its resized model score is outside the validated CIFAKE resolution domain."
                ),
            }
            payload = {
                "predicted_label": predicted_label,
                "real_probability": float(probs[0].item()),
                "fake_probability": float(probs[1].item()),
                "confidence_score": float(probs.max().item()),
                "confidence_calibrated": False,
                "assessment": assessment,
                "model_id": model_id,
                "model_identifier": self._identifier,
                "device": str(self._device),
                "inference_duration_ms": round(elapsed, 3),
                "preprocessing": {
                    **resize_metadata,
                    "original_width": width,
                    "original_height": height,
                    "model_input": "32x32 RGB; aspect-preserving fit with training-mean padding",
                    "resolution_matches_training": native_resolution,
                },
            }
            if explainability is not None:
                payload["explainability"] = explainability
            return payload

