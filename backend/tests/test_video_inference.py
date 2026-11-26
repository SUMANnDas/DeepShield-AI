from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts.training.video.inference import predict_video


class TinyFrameClassifier(nn.Module):
    def forward(self, frames):
        score = (frames.mean(dim=(1, 2, 3)) - 0.5) * 4.0
        return torch.stack((-score, score), dim=1)


class VideoInferenceTests(unittest.TestCase):
    def test_frame_classifier_inference_aggregates_frame_scores(self):
        prepared = {
            "metadata": {"file_hash": "fixture-hash", "decode_status": "ok"},
            "frames": [
                {"crop": np.zeros((8, 8, 3), dtype=np.uint8), "frame_index": 0, "timestamp_seconds": 0.0},
                {"crop": np.full((8, 8, 3), 255, dtype=np.uint8), "frame_index": 8, "timestamp_seconds": 0.5},
            ],
            "face_detector": "fixture",
            "errors": [],
        }
        with patch("scripts.training.video.inference.preprocess_video", return_value=prepared):
            result = predict_video("fixture.mp4", TinyFrameClassifier(), max_frames=2, size=8)
        self.assertEqual(result["status"], "ok")
        report = result["report"]
        self.assertEqual(len(report["frame_evidence"]), 2)
        self.assertEqual([row["frame_index"] for row in report["frame_evidence"]], [0, 8])
        self.assertAlmostEqual(report["probability_manipulated"], 0.5, places=5)

    def test_no_reliable_face_track_returns_inconclusive(self):
        prepared = {
            "metadata": {"decode_status": "ok"},
            "frames": [],
            "face_detector": "fixture",
        }
        with patch("scripts.training.video.inference.preprocess_video", return_value=prepared):
            result = predict_video("fixture.mp4", TinyFrameClassifier())
        self.assertEqual(result["status"], "inconclusive")
        self.assertEqual(result["reason"], "no_reliable_face_track")


if __name__ == "__main__":
    unittest.main()
