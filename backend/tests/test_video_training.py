from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts.training.video.dataset import VideoSequenceDataset, build_crop_cache
from scripts.training.video.evaluate import evaluate, evaluate_video_run
from scripts.training.video.inference import load_video_for_inference, predict_video
from scripts.training.video.metrics import binary_metrics, compute_manipulation_strata
from scripts.training.video.models import TemporalTransformer, build_video_model
from scripts.training.video.storage_budget import VideoStorageBudget
from scripts.training.video.train import autocast_dtype, run_video_training, train_epoch


class TinyFrameClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(4.0))

    def forward(self, frames):
        mean = frames.mean(dim=(1, 2, 3))
        fake_logit = (mean - 0.5) * self.scale
        return torch.stack((-fake_logit, fake_logit), dim=1)


def make_batch():
    frames = torch.zeros(3, 2, 3, 8, 8)
    frames[1, 0] = 1.0
    mask = torch.tensor([[True, True], [True, False], [False, False]])
    return {
        "frames": frames,
        "mask": mask,
        "label": torch.tensor([0, 1, 1]),
        "video_id": ["real-video", "fake-video", "empty-video"],
        "source_video_id": ["real-source", "fake-source", "empty-source"],
    }


class VideoTrainingTests(unittest.TestCase):
    def test_train_and_evaluation_skip_empty_face_sequences(self):
        model = TinyFrameClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss = train_epoch(model, [make_batch()], optimizer)
        result = evaluate(model, [make_batch()])
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(result["metrics"]["accuracy"], 1.0)
        self.assertEqual([row["video_id"] for row in result["videos"]], ["real-video", "fake-video"])
        self.assertEqual(result["skipped"], [{"video_id": "empty-video", "reason": "no_valid_face_sequence"}])

    def test_all_empty_sequences_are_not_scored_or_trained(self):
        model = TinyFrameClassifier()
        batch = make_batch()
        batch["mask"].zero_()
        result = evaluate(model, [batch])
        self.assertEqual(result["status"], "no_valid_face_sequences")
        self.assertIsNone(result["metrics"])
        with self.assertRaisesRegex(ValueError, "no samples with a valid face sequence"):
            train_epoch(model, [batch], torch.optim.SGD(model.parameters(), lr=0.01))

    def test_feature_sequence_evaluation_skips_empty_rows(self):
        model = TemporalTransformer(dim=4, max_frames=2, layers=1, heads=2)
        batch = {
            "frames": torch.zeros(1, 2, 4),
            "mask": torch.zeros(1, 2, dtype=torch.bool),
            "label": torch.tensor([1]),
            "video_id": ["missing-face"],
        }
        result = evaluate(model, [batch])
        self.assertEqual(result["status"], "no_valid_face_sequences")
        self.assertEqual(result["skipped"][0]["video_id"], "missing-face")

    def test_metrics_reject_empty_or_invalid_probabilities(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            binary_metrics([], [])
        with self.assertRaisesRegex(ValueError, "finite values"):
            binary_metrics([0, 1], [float("nan"), 0.9])

    def test_mixed_precision_requires_cuda(self):
        self.assertIsNone(autocast_dtype("cpu", "off"))
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            autocast_dtype("cpu", "bf16")

    def test_build_video_model_and_strata_metrics(self):
        model = build_video_model({"architecture": "efficientnet-b0-bigru-attention", "hidden_dim": 16, "gru_layers": 1})
        self.assertTrue(hasattr(model, "encode_frames"))
        with self.assertRaisesRegex(ValueError, "weights must be null"):
            build_video_model({"architecture": "efficientnet-b0-frame", "weights": "DEFAULT"})
        strata = compute_manipulation_strata([
            {"label": 0, "manipulation_method": "original", "probability_manipulated": 0.1},
            {"label": 1, "manipulation_method": "Deepfakes", "probability_manipulated": 0.9},
            {"label": 1, "manipulation_method": "FaceSwap", "probability_manipulated": 0.4},
        ])
        self.assertEqual(strata["Deepfakes"]["detection_rate"], 1.0)
        self.assertEqual(strata["FaceSwap"]["detection_rate"], 0.0)
        self.assertIn("vs_all_real", strata["Deepfakes"])

    def test_end_to_end_video_training_resume_eval_and_inference_cli_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "video-data"
            ds_root = root / "raw"
            ds_root.mkdir(parents=True)
            for name in ("real1.mp4", "real2.mp4", "fake1.mp4", "fake2.mp4"):
                (ds_root / name).write_bytes(b"fixture")
            manifests = root / "manifests"
            manifests.mkdir()
            rows = [
                {"sample_id": "r1", "relative_path": "real1.mp4", "label": "REAL", "source_video_id": "s1", "manipulation_method": "original"},
                {"sample_id": "f1", "relative_path": "fake1.mp4", "label": "MANIPULATED", "source_video_id": "s2", "manipulation_method": "Deepfakes"},
            ]
            val_rows = [
                {"sample_id": "r2", "relative_path": "real2.mp4", "label": "REAL", "source_video_id": "s3", "manipulation_method": "original"},
                {"sample_id": "f2", "relative_path": "fake2.mp4", "label": "MANIPULATED", "source_video_id": "s4", "manipulation_method": "FaceSwap"},
            ]
            train_m = manifests / "train.jsonl"
            val_m = manifests / "val.jsonl"
            test_m = manifests / "test.jsonl"
            train_m.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            val_m.write_text("\n".join(json.dumps(r) for r in val_rows) + "\n", encoding="utf-8")
            test_m.write_text("\n".join(json.dumps(r) for r in val_rows) + "\n", encoding="utf-8")
            cfg_path = Path(temporary) / "video-cfg.json"
            cfg_path.write_text(json.dumps({
                "dataset": "SyntheticFixture",
                "architecture": "efficientnet-b0-frame",
                "input_size": 32,
                "max_frames": 2,
                "weights": None,
                "normalize_faces": True,
                "seed": 42,
                "batch_size": 2,
                "epochs": 2,
                "learning_rate": 0.001,
                "early_stopping_patience": 5,
                "early_stopping_min_delta": 0.0,
                "mixed_precision": "off",
                "device": "cpu",
            }), encoding="utf-8")

            def fake_preprocess(path, *_args, **_kwargs):
                val = 230 if "fake" in str(path) else 25
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
                part1 = run_video_training(
                    cfg_path,
                    train_manifest=train_m,
                    val_manifest=val_m,
                    test_manifest=test_m,
                    dataset_root=ds_root,
                    budget=budget,
                    until_epoch=1,
                )
                self.assertEqual(part1["last_epoch"], 1)
                self.assertFalse(part1["training_complete"])

                part2 = run_video_training(
                    cfg_path,
                    train_manifest=train_m,
                    val_manifest=val_m,
                    test_manifest=test_m,
                    dataset_root=ds_root,
                    budget=budget,
                    run_dir=Path(part1["run_dir"]),
                    resume=True,
                )
                self.assertEqual(part2["last_epoch"], 2)
                self.assertTrue(part2["training_complete"])

                eval_rec = evaluate_video_run(
                    cfg_path,
                    Path(part2["run_dir"]),
                    test_m,
                    ds_root,
                    budget=budget,
                )
                self.assertFalse(eval_rec["test_set_used_for_selection"])
                self.assertIn("FaceSwap", eval_rec["manipulation_strata"])
                self.assertTrue((Path(part2["run_dir"]) / "test-metrics.json").is_file())
                self.assertTrue((Path(part2["run_dir"]) / "test-confusion-matrix.png").is_file())

                model, runtime, device, identifier = load_video_for_inference(Path(part2["run_dir"]), budget=budget)
                self.assertIn(part2["run_id"], identifier)
                pred = predict_video(
                    ds_root / "fake2.mp4",
                    model,
                    device=device,
                    max_frames=runtime["max_frames"],
                    size=runtime["input_size"],
                    normalize=runtime["normalization"]["normalize_faces"],
                    include_heatmaps=True,
                )
                self.assertEqual(pred["status"], "ok")
                self.assertIn("heatmap_jpeg_base64", pred["report"]["frame_evidence"][0])

    def test_crop_cache_and_horizontal_flip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "video-data"
            ds_root = root / "raw"
            ds_root.mkdir(parents=True)
            (ds_root / "clip.mp4").write_bytes(b"fixture")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps({"sample_id": "c1", "relative_path": "clip.mp4", "label": "REAL", "source_video_id": "s1"}) + "\n",
                encoding="utf-8",
            )
            crop = np.zeros((16, 16, 3), dtype=np.uint8)
            crop[:, :8, :] = 200
            prepared = {
                "metadata": {"decode_status": "ok"},
                "frames": [{"crop": crop, "frame_index": 0, "timestamp_seconds": 0.0}],
                "face_detector": "fixture",
                "errors": [],
            }
            budget = VideoStorageBudget(root=root, hard_limit_bytes=50_000_000, target_bytes=40_000_000, minimum_free_bytes=0, project_root=None)
            cache_dir = root / "cache"
            with patch("scripts.training.video.dataset.preprocess_video", return_value=prepared):
                stats = build_crop_cache([manifest], ds_root, cache_dir, budget=budget, max_frames=2, size=16, workers=1)
            self.assertEqual(stats["cached"], 1)
            self.assertEqual(stats["empty_face_tracks"], 0)
            ds = VideoSequenceDataset(
                manifest, ds_root, max_frames=2, size=16, detector=object(), crop_cache_dir=cache_dir, horizontal_flip_prob=1.0
            )
            item = ds[0]
            self.assertTrue(item["mask"].tolist() == [True, False])
            # Flipping swaps bright left half [0:8] to right half [8:16].
            self.assertGreater(item["frames"][0, :, :, 8:].mean().item(), 0.7)
            self.assertLess(item["frames"][0, :, :, :8].mean().item(), 0.1)


if __name__ == "__main__":
    unittest.main()
