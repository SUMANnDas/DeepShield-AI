from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.services.analysis_store import AnalysisStore
from backend.app.api.routes import application as application_routes


class ApplicationIntegrationHardeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="deepshield-api-test-")
        self.app = create_app()
        self.app.state.analysis_store = AnalysisStore(Path(self.temp_dir.name) / "history.sqlite3")
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.client.close()
        self.temp_dir.cleanup()

    def test_empty_and_oversized_image_uploads_are_rejected(self) -> None:
        empty = self.client.post(
            "/api/v1/analysis/images",
            data={"model_id": "resnet50"},
            files={"file": ("empty.png", b"", "image/png")},
        )
        self.assertEqual(empty.status_code, 400)
        self.assertIn("empty", empty.json()["detail"].lower())

        with patch.object(application_routes, "MAX_IMAGE_BYTES", 16):
            oversized = self.client.post(
                "/api/v1/analysis/images",
                data={"model_id": "resnet50"},
                files={"file": ("large.png", b"x" * 17, "image/png")},
            )
        self.assertEqual(oversized.status_code, 413)

    def test_video_upload_is_inspected_and_persisted_without_prediction(self) -> None:
        video_path = Path(self.temp_dir.name) / "synthetic.avi"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
        self.assertTrue(writer.isOpened(), "MJPG video writer is required for this local integration test")
        for index in range(9):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.circle(frame, (12 + index * 4, 32), 8, (220, 180, 40), -1)
            writer.write(frame)
        writer.release()
        video_bytes = video_path.read_bytes()

        response = self.client.post(
            "/api/v1/analysis/videos",
            files={"file": ("synthetic.avi", video_bytes, "video/x-msvideo")},
        )

        self.assertEqual(response.status_code, 200, response.text)
        record = response.json()
        self.assertFalse(record["prediction_available"])
        self.assertEqual(record["status"], "experimental_metadata_only")
        self.assertNotIn("file_path", record["metadata"])
        self.assertEqual(record["sha256"], hashlib.sha256(video_bytes).hexdigest())
        self.assertGreaterEqual(record["metadata"]["frame_count"], 1)
        self.assertLessEqual(len(record["sampled_frames"]), 6)
        self.assertTrue(record["sampled_frames"])
        history = self.client.get("/api/v1/history?media_type=video").json()
        self.assertEqual(history["total"], 1)
        self.assertEqual(history["items"][0]["sha256"], record["sha256"])

    def test_video_duration_limit_rejects_before_frame_sampling(self) -> None:
        metadata = {
            "decode_status": "ok",
            "duration_seconds": 601,
            "file_path": "temporary.avi",
        }
        with patch.object(application_routes, "inspect_video", return_value=metadata), patch.object(
            application_routes, "sample_video"
        ) as sample_video:
            response = self.client.post(
                "/api/v1/analysis/videos",
                files={"file": ("long.avi", b"synthetic payload", "video/x-msvideo")},
            )

        self.assertEqual(response.status_code, 415)
        self.assertIn("10-minute", response.json()["detail"])
        sample_video.assert_not_called()

    def test_unsupported_and_empty_video_uploads_are_rejected(self) -> None:
        unsupported = self.client.post(
            "/api/v1/analysis/videos",
            files={"file": ("clip.exe", b"data", "application/octet-stream")},
        )
        empty = self.client.post(
            "/api/v1/analysis/videos",
            files={"file": ("clip.mp4", b"", "video/mp4")},
        )
        self.assertEqual(unsupported.status_code, 415)
        self.assertEqual(empty.status_code, 400)

    def test_oversized_video_upload_is_rejected(self) -> None:
        with patch.object(application_routes, "MAX_VIDEO_BYTES", 8):
            response = self.client.post(
                "/api/v1/analysis/videos",
                files={"file": ("large.avi", b"x" * 9, "video/x-msvideo")},
            )

        self.assertEqual(response.status_code, 413)
        self.assertIn("200 MB", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
