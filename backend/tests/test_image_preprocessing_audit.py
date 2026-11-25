from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from backend.services.image_inference import ImageInferenceService
from scripts.training.image.preprocessing import prepare_model_rgb


class ImagePreprocessingAuditTests(unittest.TestCase):
    def test_native_cifake_pixels_pass_through_unchanged(self) -> None:
        values = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        output, details = prepare_model_rgb(Image.fromarray(values, mode="RGB"))
        np.testing.assert_array_equal(output, values)
        self.assertEqual(details["method"], "native_size")
        self.assertEqual(output.shape, (32, 32, 3))

    def test_non_square_upload_is_letterboxed_without_stretching(self) -> None:
        image = Image.new("RGB", (64, 32), (200, 20, 40))
        output, details = prepare_model_rgb(image, fill_rgb=(100, 110, 120))
        self.assertEqual(output.shape, (32, 32, 3))
        self.assertEqual(details["method"], "aspect_preserving_bilinear_letterbox")
        self.assertEqual(details["resized_width"], 32)
        self.assertEqual(details["resized_height"], 16)
        self.assertEqual(details["padding"], {"top": 8, "bottom": 8, "left": 0, "right": 0})
        np.testing.assert_array_equal(output[0, 0], np.array([100, 110, 120], dtype=np.uint8))
        np.testing.assert_array_equal(output[15, 0], np.array([200, 20, 40], dtype=np.uint8))

    def test_upload_validation_applies_exif_orientation(self) -> None:
        image = Image.new("RGB", (3, 2), (30, 100, 220))
        exif = Image.Exif()
        exif[274] = 6
        stream = io.BytesIO()
        image.save(stream, format="JPEG", exif=exif)
        oriented, metadata = ImageInferenceService.validate_image(stream.getvalue(), "rotated.jpg", "image/jpeg")
        self.assertEqual(oriented.size, (2, 3))
        self.assertEqual((metadata["original_width"], metadata["original_height"]), (3, 2))
        self.assertEqual((metadata["width"], metadata["height"]), (2, 3))
        self.assertEqual(metadata["exif_orientation"], 6)
        self.assertTrue(metadata["exif_orientation_applied"])


if __name__ == "__main__":
    unittest.main()
