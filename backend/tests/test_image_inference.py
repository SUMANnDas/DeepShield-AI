"""Structured inference output tests with a synthetic PIL image."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from PIL import Image

from scripts.training.image.inference import predict_images
from scripts.training.image.transforms import ImageTransform


class ImageInferenceTests(unittest.TestCase):
    def test_pil_input_returns_probabilities_and_model_identity(self) -> None:
        class TinyModel(torch.nn.Module):
            def forward(self, value: torch.Tensor) -> torch.Tensor:
                mean = value.mean(dim=(1, 2, 3))
                return torch.stack((-mean, mean), dim=1)

        image = Image.fromarray(np.full((32, 32, 3), 200, dtype=np.uint8), mode="RGB")
        result = predict_images(
            TinyModel(), [image],
            transform=ImageTransform((0.5,) * 3, (0.5,) * 3),
            model_identifier="synthetic:test",
            device=torch.device("cpu"),
        )[0]
        self.assertIn(result["predicted_label"], {"REAL", "FAKE"})
        self.assertAlmostEqual(result["real_probability"] + result["fake_probability"], 1.0, places=6)
        self.assertEqual(result["model_identifier"], "synthetic:test")
        self.assertGreaterEqual(result["inference_duration_ms"], 0.0)

    def test_non_32_input_is_rejected(self) -> None:
        image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8), mode="RGB")
        with self.assertRaisesRegex(ValueError, "Expected a 32x32"):
            predict_images(
                torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 2)), [image],
                transform=ImageTransform((0.5,) * 3, (0.5,) * 3),
                model_identifier="synthetic:test", device=torch.device("cpu"),
            )


if __name__ == "__main__":
    unittest.main()
