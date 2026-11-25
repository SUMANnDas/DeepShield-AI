"""Architecture shape tests using randomly initialized small-image models."""

from __future__ import annotations

import unittest

import torch

from scripts.training.image.models import build_model


class ImageModelTests(unittest.TestCase):
    def test_all_architectures_accept_32x32_rgb(self) -> None:
        torch.set_num_threads(2)
        for architecture in ("resnet50", "efficientnet-b0", "vit-small"):
            with self.subTest(architecture=architecture):
                model = build_model(architecture).eval()
                with torch.inference_mode():
                    output = model(torch.zeros(2, 3, 32, 32))
                self.assertEqual(tuple(output.shape), (2, 2))

    def test_resnet_small_image_stem_is_preserved(self) -> None:
        model = build_model("resnet50")
        self.assertEqual(model.conv1.kernel_size, (3, 3))
        self.assertEqual(model.conv1.stride, (1, 1))
        self.assertIsInstance(model.maxpool, torch.nn.Identity)

    def test_vit_has_64_four_pixel_patches(self) -> None:
        model = build_model("vit-small")
        self.assertEqual(model.conv_proj.kernel_size, (4, 4))
        self.assertEqual(model.conv_proj.stride, (4, 4))


if __name__ == "__main__":
    unittest.main()
