from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.app.api.routes.application import performance, models as api_models, report_html
from backend.services.analysis_store import AnalysisStore
from backend.services.system_status import system_status
from backend.services.video_detection.detector import VideoDetectionService
from backend.services.video_model_registry import get_video_model, list_video_models, video_model_paths
from scripts.training.video.evaluate import evaluate_video_run
from scripts.training.video.storage_budget import VideoStorageBudget
from scripts.training.video.train import run_video_training


class _DummyAppState:
    def __init__(self, store: AnalysisStore, budget: VideoStorageBudget):
        self.analysis_store = store
        self.video_storage_budget = budget


class _DummyApp:
    def __init__(self, store: AnalysisStore, budget: VideoStorageBudget):
        self.state = _DummyAppState(store, budget)


class _DummyRequest:
    def __init__(self, store: AnalysisStore, budget: VideoStorageBudget):
        self.app = _DummyApp(store, budget)


class VideoRegistryAndServiceTests(unittest.TestCase):
    def test_missing_video_root_reports_no_available_models_without_creating_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            missing_root = Path(temporary) / "nonexistent-video-data"
            self.assertEqual(list_video_models(missing_root), [])
            self.assertFalse(missing_root.exists())
            status = system_status(video_data_root=missing_root)
            self.assertEqual(status["video_models_available"], 0)
            self.assertEqual(status["video_prediction_status"], "unavailable_no_validated_video_checkpoint")

    def test_synthetic_dataset_runs_are_never_registered_for_production(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "video-data"
            ds_root = root / "raw"
            ds_root.mkdir(parents=True)
            for name in ("r1.mp4", "f1.mp4", "r2.mp4", "f2.mp4"):
                (ds_root / name).write_bytes(b"fixture")
            manifests = root / "manifests"
            manifests.mkdir()
            train_m = manifests / "train.jsonl"
            val_m = manifests / "val.jsonl"
            test_m = manifests / "test.jsonl"
            train_m.write_text(
                json.dumps({"sample_id": "r1", "relative_path": "r1.mp4", "label": "REAL", "source_video_id": "s1"}) + "\n"
                + json.dumps({"sample_id": "f1", "relative_path": "f1.mp4", "label": "MANIPULATED", "source_video_id": "s2"}) + "\n",
                encoding="utf-8",
            )
            val_rows = (
                json.dumps({"sample_id": "r2", "relative_path": "r2.mp4", "label": "REAL", "source_video_id": "s3"}) + "\n"
                + json.dumps({"sample_id": "f2", "relative_path": "f2.mp4", "label": "MANIPULATED", "source_video_id": "s4"}) + "\n"
            )
            val_m.write_text(val_rows, encoding="utf-8")
            test_m.write_text(val_rows, encoding="utf-8")
            cfg = Path(temporary) / "cfg.json"
            cfg.write_text(json.dumps({
                "dataset": "SyntheticFixture",
                "architecture": "efficientnet-b0-frame",
                "input_size": 32,
                "max_frames": 2,
                "weights": None,
                "batch_size": 2,
                "epochs": 1,
                "mixed_precision": "off",
                "device": "cpu",
            }), encoding="utf-8")

            def fake_preprocess(path, *_args, **_kwargs):
                return {
                    "metadata": {"file_hash": "h", "decode_status": "ok"},
                    "frames": [{"crop": np.full((32, 32, 3), 120, dtype=np.uint8), "frame_index": 0, "timestamp_seconds": 0.0}],
                    "face_detector": "fixture",
                    "errors": [],
                }

            budget = VideoStorageBudget(root=root, hard_limit_bytes=500_000_000, target_bytes=400_000_000, minimum_free_bytes=0, project_root=None)
            with patch("scripts.training.video.dataset.preprocess_video", side_effect=fake_preprocess):
                run = run_video_training(cfg, train_manifest=train_m, val_manifest=val_m, test_manifest=test_m, dataset_root=ds_root, budget=budget)
                evaluate_video_run(cfg, Path(run["run_dir"]), test_m, ds_root, budget=budget)
            self.assertEqual(list_video_models(root), [])

    def test_validated_real_video_run_registers_and_runs_detection_service(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "video-data"
            ds_root = root / "raw"
            ds_root.mkdir(parents=True)
            for name in ("r1.mp4", "f1.mp4", "r2.mp4", "f2.mp4"):
                (ds_root / name).write_bytes(b"fixture")
            manifests = root / "manifests"
            manifests.mkdir()
            train_m = manifests / "train.jsonl"
            val_m = manifests / "val.jsonl"
            test_m = manifests / "test.jsonl"
            train_m.write_text(
                json.dumps({"sample_id": "r1", "relative_path": "r1.mp4", "label": "REAL", "source_video_id": "s1", "manipulation_method": "original"}) + "\n"
                + json.dumps({"sample_id": "f1", "relative_path": "f1.mp4", "label": "MANIPULATED", "source_video_id": "s2", "manipulation_method": "Deepfakes"}) + "\n",
                encoding="utf-8",
            )
            val_rows = (
                json.dumps({"sample_id": "r2", "relative_path": "r2.mp4", "label": "REAL", "source_video_id": "s3", "manipulation_method": "original"}) + "\n"
                + json.dumps({"sample_id": "f2", "relative_path": "f2.mp4", "label": "MANIPULATED", "source_video_id": "s4", "manipulation_method": "NeuralTextures"}) + "\n"
            )
            val_m.write_text(val_rows, encoding="utf-8")
            test_m.write_text(val_rows, encoding="utf-8")
            cfg = Path(temporary) / "cfg.json"
            cfg.write_text(json.dumps({
                "dataset": "FaceForensics++ c23",
                "architecture": "efficientnet-b0-frame",
                "input_size": 32,
                "max_frames": 2,
                "weights": None,
                "normalize_faces": True,
                "batch_size": 2,
                "epochs": 1,
                "mixed_precision": "off",
                "device": "cpu",
            }), encoding="utf-8")

            def fake_preprocess(path, *_args, **_kwargs):
                val = 210 if "f" in Path(path).name else 30
                return {
                    "metadata": {"file_hash": "hash-" + Path(path).name, "decode_status": "ok"},
                    "frames": [{"crop": np.full((32, 32, 3), val, dtype=np.uint8), "frame_index": 0, "timestamp_seconds": 0.0}],
                    "face_detector": "fixture",
                    "errors": [],
                }

            budget = VideoStorageBudget(root=root, hard_limit_bytes=500_000_000, target_bytes=400_000_000, minimum_free_bytes=0, project_root=None)
            with patch("scripts.training.video.dataset.preprocess_video", side_effect=fake_preprocess), patch(
                "scripts.training.video.inference.preprocess_video", side_effect=fake_preprocess
            ):
                run = run_video_training(cfg, train_manifest=train_m, val_manifest=val_m, test_manifest=test_m, dataset_root=ds_root, budget=budget)
                evaluate_video_run(cfg, Path(run["run_dir"]), test_m, ds_root, budget=budget)

                items = list_video_models(root)
                self.assertEqual(len(items), 1)
                model_id = items[0]["id"]
                self.assertEqual(model_id, run["run_id"])
                self.assertIn("NeuralTextures", items[0]["manipulation_strata"])
                self.assertIsNotNone(get_video_model(model_id, root))
                _, ckpt_path, _ = video_model_paths(model_id, root)
                self.assertTrue(ckpt_path.is_file())

                service = VideoDetectionService()
                detection = service.detect(ds_root / "f2.mp4", model_id=model_id, data_root=root, include_heatmaps=True)
                self.assertEqual(detection["status"], "ok")
                self.assertEqual(detection["model_id"], model_id)
                self.assertIn(detection["report"]["prediction"], {"REAL", "MANIPULATED"})
                self.assertIn("heatmap_jpeg_base64", detection["report"]["frame_evidence"][0])
                self.assertNotIn("file_path", detection["report"]["metadata"])

                store = AnalysisStore(Path(temporary) / "app.sqlite3")
                req = _DummyRequest(store, budget)
                self.assertEqual(len(api_models(req)["items"]), 3)
                self.assertEqual(len(api_models(req)["video_items"]), 1)
                self.assertEqual(len(performance(req)["video_items"]), 1)
                self.assertEqual(system_status(video_data_root=root)["video_prediction_status"], "available")

                saved = store.add(
                    media_type="video",
                    filename="f2.mp4",
                    sha256="abc",
                    metadata={"width": 32, "height": 32},
                    result={
                        "prediction_available": True,
                        "prediction": detection["report"]["prediction"],
                        "frame_evidence": detection["report"]["frame_evidence"],
                    },
                )
                html_resp = report_html(req, saved["id"])
                self.assertIn("Video frame evidence", html_resp.body.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
