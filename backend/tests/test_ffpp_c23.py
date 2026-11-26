from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.datasets.ffpp_c23 import (
    download_ffpp_c23,
    prepare_ffpp_dataset,
    run_preflight,
    validate_subsets,
)
from scripts.training.video.storage_budget import VideoStorageBudget


def _write_mp4(path: Path, seed_value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64))
    assert writer.isOpened()
    for idx in range(4):
        frame = np.full((64, 64, 3), (seed_value * 29) % 255, dtype=np.uint8)
        cv2.circle(frame, (20 + (seed_value + idx) % 24, 32), 10, (200, 180, 140), -1)
        writer.write(frame)
    writer.release()


class FaceForensicsC23ToolingTests(unittest.TestCase):
    def test_preflight_blocks_excluded_subsets_and_requires_approval(self) -> None:
        with self.assertRaisesRegex(ValueError, "excluded"):
            validate_subsets(["original", "DeepFakeDetection"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "video-data"
            script = Path(temporary) / "download-FaceForensics.py"
            script.write_text("# dummy script\n", encoding="utf-8")
            budget = VideoStorageBudget(
                root=root,
                hard_limit_bytes=100_000_000_000,
                target_bytes=90_000_000_000,
                minimum_free_bytes=0,
                project_root=None,
            )
            pre = run_preflight(budget=budget, num_videos=100, download_script=script)
            self.assertTrue(pre["ready_for_approved_download"])
            self.assertTrue(pre["within_target_90gb"])
            with self.assertRaises(PermissionError):
                download_ffpp_c23(script, budget=budget, approved=False)

    def test_prepare_ffpp_dataset_prevents_pair_id_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "video-data"
            raw = root / "raw" / "FaceForensics++-c23"
            for i, idx in enumerate(("000", "001", "002", "003", "004", "005"), start=1):
                _write_mp4(raw / "original_sequences" / "youtube" / "c23" / "videos" / f"{idx}.mp4", i)
            _write_mp4(raw / "manipulated_sequences" / "Deepfakes" / "c23" / "videos" / "000_001.mp4", 10)
            _write_mp4(raw / "manipulated_sequences" / "FaceSwap" / "c23" / "videos" / "002_003.mp4", 11)
            _write_mp4(raw / "manipulated_sequences" / "NeuralTextures" / "c23" / "videos" / "004_005.mp4", 12)

            budget = VideoStorageBudget(
                root=root,
                hard_limit_bytes=500_000_000,
                target_bytes=400_000_000,
                minimum_free_bytes=0,
                project_root=None,
            )
            manifest_dir = root / "manifests" / "pilot"
            summary = prepare_ffpp_dataset(raw, manifest_dir, budget=budget)
            self.assertTrue(summary["leakage_passed"])
            self.assertEqual(summary["total_valid_videos"], 9)
            self.assertEqual(sum(summary["split_counts"].values()), 9)
            self.assertTrue((manifest_dir / "leakage-report.json").is_file())
            leakage = json.loads((manifest_dir / "leakage-report.json").read_text(encoding="utf-8"))
            self.assertTrue(leakage["passed"])


if __name__ == "__main__":
    unittest.main()
