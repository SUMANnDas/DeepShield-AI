"""Synthetic tests for CIFAKE manifest parsing and safe image loading."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.training.image.data import (
    CIFAKEDataset,
    LABEL_TO_INDEX,
    assert_group_disjoint,
    read_manifest,
    resolve_image_path,
)
from scripts.training.image.transforms import ImageTransform


class ImageManifestLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="cifake-loader-")
        self.root = Path(self.temp.name)
        self.dataset_root = self.root / "dataset"
        self.dataset_root.mkdir()
        self.image_rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        self.image_rgb[:, :, 0] = 255
        bgr = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2BGR)
        self.image_path = self.dataset_root / "train" / "REAL" / "sample.png"
        self.image_path.parent.mkdir(parents=True)
        self.assertTrue(cv2.imwrite(str(self.image_path), bgr))
        self.row = {
            "dataset_name": "CIFAKE", "dataset_version": "3", "task": "ai_generated_image",
            "relative_path": "train/REAL/sample.png", "label": "REAL", "split": "train",
            "sample_id": "sample-001", "file_hash": "hash-real", "split_group_id": "group-real",
            "preprocessing_status": "validated", "image_width": 32, "image_height": 32,
        }
        self.manifest = self.root / "train.jsonl"
        self.manifest.write_text(json.dumps(self.row) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_label_mapping_and_loader_output_shape(self) -> None:
        rows = read_manifest(self.manifest, self.dataset_root, "train")
        dataset = CIFAKEDataset(rows, ImageTransform((0.5,) * 3, (0.5,) * 3))
        image, label = dataset[0]
        self.assertEqual(LABEL_TO_INDEX, {"REAL": 0, "FAKE": 1})
        self.assertEqual(tuple(image.shape), (3, 32, 32))
        self.assertEqual(label, 0)
        self.assertGreater(float(image[0].mean()), 0.9)
        self.assertLess(float(image[1].mean()), -0.9)

    def test_rejects_split_mismatch_and_unexpected_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 'validation'"):
            read_manifest(self.manifest, self.dataset_root, "validation")
        row = dict(self.row, label="UNKNOWN")
        manifest = self.root / "unknown.jsonl"
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unexpected CIFAKE label"):
            read_manifest(manifest, self.dataset_root, "train")

    def test_rejects_path_traversal_and_missing_files(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsafe manifest"):
            resolve_image_path(self.dataset_root, "../outside.png")
        row = dict(self.row, relative_path="train/REAL/missing.png")
        manifest = self.root / "missing.jsonl"
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            read_manifest(manifest, self.dataset_root, "train")

    def test_detects_cross_split_group_reuse(self) -> None:
        train = read_manifest(self.manifest, self.dataset_root, "train")
        validation_sample = train[0].__class__(
            relative_path=train[0].relative_path,
            absolute_path=train[0].absolute_path,
            label_name="REAL", target=0, split="validation", sample_id="validation-1",
            file_hash="other-hash", split_group_id=train[0].split_group_id,
        )
        with self.assertRaisesRegex(ValueError, "Leakage group"):
            assert_group_disjoint(train, [validation_sample])

    def test_unreadable_image_is_not_silently_skipped(self) -> None:
        self.image_path.write_bytes(b"not an image")
        rows = read_manifest(self.manifest, self.dataset_root, "train")
        dataset = CIFAKEDataset(rows, ImageTransform((0.5,) * 3, (0.5,) * 3))
        with self.assertRaisesRegex(ValueError, "Unreadable image"):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
