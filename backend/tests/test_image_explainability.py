"""Explainability shape and finite-value tests on a synthetic image."""

from __future__ import annotations

import base64
import unittest

import numpy as np
import torch
from PIL import Image

from backend.services.image_inference import ImageInferenceService
from scripts.training.image.explain import explain_prepared_image, gradcam, vit_patch_attribution
from scripts.training.image.models import build_model


class ImageExplainabilityTests(unittest.TestCase):
    def test_resnet_gradcam_returns_native_resolution_map(self) -> None:
        torch.set_num_threads(2)
        model = build_model("resnet50")
        image = torch.rand(1, 3, 32, 32)
        values = gradcam(model, image, model.layer4[-1])
        self.assertEqual(values.shape, (32, 32))
        self.assertTrue(np.isfinite(values).all())
        self.assertGreaterEqual(float(values.min()), 0.0)
        self.assertLessEqual(float(values.max()), 1.0)

    def test_vit_patch_attribution_returns_8_by_8_grid(self) -> None:
        model = build_model("vit-small")
        values = vit_patch_attribution(model, torch.rand(1, 3, 32, 32))
        self.assertEqual(values.shape, (8, 8))
        self.assertTrue(np.isfinite(values).all())

    def test_explain_prepared_image_supports_all_registered_architectures(self) -> None:
        rgb = np.full((32, 32, 3), 128, dtype=np.uint8)
        tensor = torch.rand(1, 3, 32, 32)
        for architecture, expected_method, expected_grid in (
            ("resnet50", "gradcam", [32, 32]),
            ("efficientnet-b0", "gradcam", [32, 32]),
            ("vit-small", "vit_patch_gradient_input", [8, 8]),
        ):
            model = build_model(architecture)
            payload = explain_prepared_image(model, architecture, rgb, tensor, class_index=1, preview_size=64)
            self.assertEqual(payload["method"], expected_method)
            self.assertEqual(payload["target_class"], "FAKE")
            self.assertEqual(payload["grid_shape"], expected_grid)
            self.assertTrue(base64.b64decode(payload["overlay_png_base64"]).startswith(b"\x89PNG"))
            self.assertTrue(base64.b64decode(payload["model_input_png_base64"]).startswith(b"\x89PNG"))

    def test_image_inference_service_returns_live_explainability(self) -> None:
        service = ImageInferenceService()
        image = Image.fromarray(np.full((32, 32, 3), 160, dtype=np.uint8), mode="RGB")
        result = service.predict("efficientnet-b0", image)
        self.assertIn("explainability", result)
        self.assertEqual(result["explainability"]["method"], "gradcam")
        self.assertTrue(base64.b64decode(result["explainability"]["overlay_png_base64"]).startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
