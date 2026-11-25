from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from scripts.training.video.frame_sampling import sample_video
from scripts.training.video.video_metadata import inspect_video


class _FakeCapture:
    def __init__(self, frame_count: int = 3, frame_shape: tuple[int, int, int] = (64, 128, 3)) -> None:
        self.frame_count = frame_count
        self.frame = np.zeros(frame_shape, dtype=np.uint8)
        self.read_count = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        values = {
            cv2.CAP_PROP_FRAME_COUNT: self.frame_count,
            cv2.CAP_PROP_FPS: 10,
            cv2.CAP_PROP_FRAME_WIDTH: self.frame.shape[1],
            cv2.CAP_PROP_FRAME_HEIGHT: self.frame.shape[0],
            cv2.CAP_PROP_FOURCC: 0,
        }
        return float(values.get(property_id, 0))

    def set(self, _property_id: int, _value: float) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        self.read_count += 1
        return True, self.frame.copy()

    def release(self) -> None:
        self.released = True


class VideoBoundedProcessingTests(unittest.TestCase):
    def test_sampler_downsizes_before_retaining_each_sample(self) -> None:
        capture = _FakeCapture(frame_count=7)
        with patch("scripts.training.video.frame_sampling.cv2.VideoCapture", return_value=capture):
            frames = sample_video("synthetic.avi", max_frames=4, max_dimension=32)

        self.assertEqual(len(frames), 4)
        self.assertTrue(all(item["decode_ok"] for item in frames))
        self.assertTrue(all(max(item["frame"].shape[:2]) <= 32 for item in frames))
        self.assertTrue(capture.released)

    def test_metadata_rejects_extreme_resolution_before_decoding_frame(self) -> None:
        with tempfile.TemporaryDirectory(prefix="deepshield-video-metadata-") as directory:
            fixture = Path(directory) / "metadata.avi"
            fixture.write_bytes(b"synthetic")
            capture = _FakeCapture(frame_shape=(4096, 4096, 3))
            with patch("scripts.training.video.video_metadata.cv2.VideoCapture", return_value=capture):
                metadata = inspect_video(fixture, max_frame_pixels=16_000_000 - 1)

        self.assertEqual(metadata["decode_status"], "metadata_only")
        self.assertEqual(metadata["error"], "resolution_exceeds_processing_limit")
        self.assertEqual(capture.read_count, 0)
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
