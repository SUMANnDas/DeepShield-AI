from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.training.video.dataset import VideoSequenceDataset, collate_sequences


class VideoDatasetContractTests(unittest.TestCase):
    def test_preprocessing_manifest_labels_and_source_ids_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "clip.mp4").write_bytes(b"synthetic fixture")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({
                    "sample_id": "sample-a",
                    "relative_path": "clip.mp4",
                    "label": "FAKE",
                    "original_video_id": "source-a",
                }) + "\n" + json.dumps({
                    "video_id": "sample-b",
                    "relative_path": "clip.mp4",
                    "label": 0,
                    "source_video_id": "source-b",
                }) + "\n",
                encoding="utf-8",
            )
            prepared = {
                "frames": [{
                    "crop": np.full((8, 8, 3), 127, dtype=np.uint8),
                    "frame_index": 3,
                    "timestamp_seconds": 0.1,
                }],
                "metadata": {"decode_status": "ok"},
                "face_detector": "fixture",
            }
            with patch("scripts.training.video.dataset.preprocess_video", return_value=prepared):
                dataset = VideoSequenceDataset(
                    manifest, root, max_frames=2, size=8, detector=object()
                )
                fake = dataset[0]
                real = dataset[1]

            self.assertEqual(fake["label"].item(), 1)
            self.assertEqual(fake["video_id"], "sample-a")
            self.assertEqual(fake["source_video_id"], "source-a")
            self.assertEqual(real["label"].item(), 0)
            self.assertEqual(real["source_video_id"], "source-b")
            self.assertEqual(fake["frames"].shape, (2, 3, 8, 8))
            self.assertEqual(fake["mask"].tolist(), [True, False])
            self.assertTrue(0.49 < fake["frames"][0].mean().item() < 0.51)
            batch = collate_sequences([fake, real])
            self.assertEqual(batch["frames"].shape, (2, 2, 3, 8, 8))

    def test_unknown_labels_and_path_traversal_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unknown = root / "unknown.jsonl"
            unknown.write_text(json.dumps({"relative_path": "clip.mp4", "label": "UNKNOWN"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unsupported video label"):
                VideoSequenceDataset(unknown, root, detector=object())

            (root / "valid.mp4").write_bytes(b"synthetic fixture")
            escaping = root / "escaping.jsonl"
            escaping.write_text(
                json.dumps({"relative_path": "../outside.mp4", "label": "REAL"}),
                encoding="utf-8",
            )
            dataset = VideoSequenceDataset(escaping, root, detector=object())
            with self.assertRaisesRegex(ValueError, "escapes dataset root"):
                dataset[0]


if __name__ == "__main__":
    unittest.main()
