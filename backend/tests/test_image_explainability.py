"""Explainability shape and finite-value tests on a synthetic image."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.training.image.explain import gradcam, vit_patch_attribution
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


if __name__ == "__main__":
    unittest.main()
