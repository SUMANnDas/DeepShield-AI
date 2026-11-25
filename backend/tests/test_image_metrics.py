"""Known-answer metric and plot tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.training.image.metrics import compute_binary_metrics, save_confusion_matrix, save_roc_curve


class ImageMetricTests(unittest.TestCase):
    def test_binary_metrics_and_roc_auc(self) -> None:
        result = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.7, 0.8, 0.4], threshold=0.5)
        self.assertEqual(result["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["precision_fake"], 0.5)
        self.assertEqual(result["recall_fake"], 0.5)
        self.assertEqual(result["f1_fake"], 0.5)
        self.assertEqual(result["roc_auc"], 0.75)
        self.assertEqual(result["specificity_real"], 0.5)
        self.assertEqual(result["sensitivity_fake"], 0.5)

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            compute_binary_metrics([0, 1], [-0.1, 0.9])

    def test_plots_are_created_once_and_collision_is_rejected(self) -> None:
        result = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9])
        with tempfile.TemporaryDirectory(prefix="cifake-metrics-") as temp:
            root = Path(temp)
            matrix = save_confusion_matrix(result, root / "cm.png", title="test")
            roc = save_roc_curve([0, 0, 1, 1], [0.1, 0.3, 0.7, 0.9], root / "roc.png", title="test")
            self.assertTrue(matrix.is_file())
            self.assertTrue(roc.is_file())
            with self.assertRaises(FileExistsError):
                save_confusion_matrix(result, matrix, title="test")


if __name__ == "__main__":
    unittest.main()
