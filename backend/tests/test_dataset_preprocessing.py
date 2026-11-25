"""Local generated-fixture tests; fixtures contain no external or personal media."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import unittest
import uuid
import wave
from pathlib import Path

import cv2
import numpy as np

from scripts.datasets.audio_pipeline import process_audio_directory
from scripts.datasets.image_pipeline import process_image_directory
from scripts.datasets.leakage_pipeline import check_split_leakage
from scripts.datasets.manifest_utils import read_jsonl
from scripts.datasets.report_pipeline import generate_dataset_report
from scripts.datasets.split_pipeline import create_group_splits
from scripts.datasets.video_pipeline import process_video_directory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = Path(__file__).resolve().parent / "generated"


class DatasetPreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = FIXTURE_ROOT / f"run-{uuid.uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=False)

    def write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2)

    def write_annotations(self, path: Path, rows: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_image_pipeline_labels_hashes_corrupt_file_and_reports(self) -> None:
        source = self.root / "images"
        source.mkdir()
        real = np.zeros((80, 96, 3), dtype=np.uint8)
        real[10:60, 20:70] = (30, 150, 230)
        fake = np.zeros((80, 96, 3), dtype=np.uint8)
        fake[::2, ::2] = 255
        self.assertTrue(cv2.imwrite(str(source / "real.png"), real))
        shutil.copyfile(source / "real.png", source / "real-copy.png")
        self.assertTrue(cv2.imwrite(str(source / "fake.png"), fake))
        (source / "broken.png").write_bytes(b"not an image")
        annotations = self.root / "image-labels.csv"
        self.write_annotations(annotations, [
            {"relative_path": "real.png", "label": "authentic"},
            {"relative_path": "real-copy.png", "label": "real"},
            {"relative_path": "fake.png", "label": "synthetic"},
            {"relative_path": "broken.png", "label": "fake"},
        ])
        config = self.root / "image-config.json"
        self.write_json(config, {
            "target_size": [32, 32], "face_detection_enabled": False,
            "perceptual_hash_enabled": True, "perceptual_hash_distance": 6,
        })
        manifest = self.root / "images.jsonl"
        stats = process_image_directory(
            source, self.root / "image-output", manifest, dataset_name="fixture-images",
            annotations_path=annotations, config_path=config,
        )
        self.assertEqual(stats["processed"], 3)
        self.assertEqual(stats["corrupt"], 1)
        self.assertEqual(stats["duplicates"], 1)
        rows = list(read_jsonl(manifest))
        labels = {row["relative_path"]: row["label"] for row in rows}
        self.assertEqual(labels["real.png"], "REAL")
        self.assertEqual(labels["fake.png"], "FAKE")
        self.assertEqual(next(row for row in rows if row["relative_path"] == "broken.png")["preprocessing_status"], "corrupt")
        self.assertEqual(len(list((self.root / "image-output" / "images").glob("*.jpg"))), 3)
        report = generate_dataset_report(manifest, self.root / "image-quality", source)
        self.assertEqual(report["records"], 4)
        self.assertEqual(report["quality_counts"].get("unlabeled", 0), 0)
        self.assertTrue((self.root / "image-quality.svg").is_file())

    def test_existing_manifest_is_refused_without_modification(self) -> None:
        source = self.root / "empty-images"
        source.mkdir()
        manifest = self.root / "protected.jsonl"
        manifest.write_text("keep this existing content\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            process_image_directory(source, self.root / "image-output", manifest, dataset_name="fixture")
        self.assertEqual(manifest.read_text(encoding="utf-8"), "keep this existing content\n")

    def test_audio_pipeline_resamples_segments_and_writes_log_mel(self) -> None:
        source = self.root / "audio"
        source.mkdir()
        sample_rate = 8000
        values = (np.sin(2.0 * math.pi * 440.0 * np.arange(sample_rate) / sample_rate) * 0.25)
        pcm = np.rint(values * 32767.0).astype("<i2")
        with wave.open(str(source / "tone.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
        annotations = self.root / "audio-labels.csv"
        self.write_annotations(annotations, [{"relative_path": "tone.wav", "label": "spoof", "speaker_id": "fixture-speaker"}])
        config = self.root / "audio-config.json"
        self.write_json(config, {
            "target_sample_rate": 16000, "segment_seconds": 0.4, "overlap_seconds": 0.1,
            "mel_bins": 24, "fft_size": 512,
        })
        manifest = self.root / "audio.jsonl"
        stats = process_audio_directory(
            source, self.root / "audio-output", manifest, dataset_name="fixture-audio",
            annotations_path=annotations, config_path=config,
        )
        self.assertEqual(stats["audio_files"], 1)
        self.assertGreater(stats["segments"], 1)
        rows = list(read_jsonl(manifest))
        self.assertEqual(rows[0]["label"], "FAKE")
        self.assertEqual(rows[0]["sample_rate"], 8000)
        self.assertEqual(rows[0]["target_sample_rate"], 16000)
        feature = np.load(self.root / "audio-output" / rows[0]["feature_path"], allow_pickle=False)
        self.assertEqual(feature.shape[0], 24)
        with wave.open(str(self.root / "audio-output" / rows[0]["audio_path"]), "rb") as handle:
            self.assertEqual(handle.getframerate(), 16000)
            self.assertEqual(handle.getnchannels(), 1)

    def test_audio_pipeline_decodes_flac_when_soundfile_is_available(self) -> None:
        try:
            import soundfile as sf
        except ImportError:
            self.skipTest("Optional SoundFile dependency is unavailable")
        source = self.root / "flac-audio"
        source.mkdir()
        sample_rate = 16000
        values = (np.sin(2.0 * math.pi * 220.0 * np.arange(sample_rate) / sample_rate) * 0.2).astype(np.float32)
        sf.write(source / "tone.flac", values, sample_rate, format="FLAC")
        config = self.root / "flac-config.json"
        self.write_json(config, {
            "segment_seconds": 0.5, "overlap_seconds": 0.1, "mel_bins": 16, "fft_size": 512,
        })
        manifest = self.root / "flac.jsonl"
        stats = process_audio_directory(
            source, self.root / "flac-output", manifest, dataset_name="fixture-flac", config_path=config,
        )
        self.assertEqual(stats["audio_files"], 1)
        self.assertEqual(stats["errors"], 0)
        rows = list(read_jsonl(manifest))
        self.assertGreater(stats["segments"], 1)
        self.assertTrue((self.root / "flac-output" / rows[0]["feature_path"]).is_file())

    def test_video_pipeline_samples_bounded_frames_with_timestamps(self) -> None:
        source = self.root / "video"
        source.mkdir()
        video_path = source / "clip.avi"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 8.0, (96, 64))
        if not writer.isOpened():
            writer.release()
            self.skipTest("OpenCV MJPG video writer is unavailable in this runtime")
        for index in range(8):
            frame = np.zeros((64, 96, 3), dtype=np.uint8)
            cv2.rectangle(frame, (index * 4, 10), (index * 4 + 20, 35), (40, 180, 230), -1)
            writer.write(frame)
        writer.release()
        config = self.root / "video-config.json"
        self.write_json(config, {"sampling_strategy": "uniform", "max_frames": 3, "face_detection_enabled": False})
        manifest = self.root / "videos.jsonl"
        stats = process_video_directory(
            source, self.root / "video-output", manifest, dataset_name="fixture-video", config_path=config,
        )
        self.assertEqual(stats["videos"], 1)
        self.assertEqual(stats["video_errors"], 0)
        rows = list(read_jsonl(self.root / "videos_frames.jsonl"))
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["frame_number"] for row in rows], [0, 4, 7])
        self.assertTrue(all(row["preprocessing_status"] == "processed" for row in rows))

    def test_group_splits_keep_speakers_together_and_leakage_checker_passes(self) -> None:
        source = self.root / "source.jsonl"
        with source.open("x", encoding="utf-8") as handle:
            for speaker in range(8):
                for sample in range(2):
                    row = {
                        "sample_id": f"sample-{speaker}-{sample}", "speaker_id": f"speaker-{speaker}",
                        "label": "REAL" if speaker % 2 == 0 else "FAKE",
                        "file_hash": hashlib.sha256(f"sample-{speaker}-{sample}".encode()).hexdigest(),
                    }
                    handle.write(json.dumps(row) + "\n")
            duplicate_hash = hashlib.sha256(b"exact duplicate image fixture").hexdigest()
            for sample_id in ("duplicate-image-a", "duplicate-image-b"):
                handle.write(json.dumps({
                    "sample_id": sample_id, "label": "REAL", "file_hash": duplicate_hash,
                }) + "\n")
        counts = create_group_splits(source, self.root / "split", config_path=PROJECT_ROOT / "configs/datasets/splits.json")
        self.assertEqual(sum(counts.values()), 18)
        split_manifests = [self.root / f"split_{name}.jsonl" for name in ("train", "validation", "test")]
        speaker_splits: dict[str, set[str]] = {}
        for path in split_manifests:
            for row in read_jsonl(path):
                if row.get("speaker_id"):
                    speaker_splits.setdefault(row["speaker_id"], set()).add(row["split"])
        self.assertTrue(all(len(values) == 1 for values in speaker_splits.values()))
        duplicate_splits = {
            row["split"] for path in split_manifests for row in read_jsonl(path)
            if row["sample_id"].startswith("duplicate-image-")
        }
        self.assertEqual(len(duplicate_splits), 1)
        result = check_split_leakage(split_manifests, self.root / "clean-leakage.json")
        self.assertTrue(result["passed"], result["issues"])

    def test_splitter_preserves_published_source_partitions(self) -> None:
        source = self.root / "official-splits.jsonl"
        with source.open("x", encoding="utf-8") as handle:
            for index in range(6):
                split = "train" if index < 4 else "test"
                handle.write(json.dumps({
                    "sample_id": f"official-{index}", "source_split": split,
                    "label": "REAL" if index % 2 == 0 else "FAKE",
                    "file_hash": hashlib.sha256(f"official-{index}".encode()).hexdigest(),
                }) + "\n")
        counts = create_group_splits(source, self.root / "official-output")
        self.assertEqual(counts, {"train": 4, "validation": 0, "test": 2})
        for name in ("train", "validation", "test"):
            for row in read_jsonl(self.root / f"official-output_{name}.jsonl"):
                self.assertEqual(row["split"], row["source_split"])

    def test_splitter_honors_unequal_configured_ratios(self) -> None:
        source = self.root / "ratio-source.jsonl"
        with source.open("x", encoding="utf-8") as handle:
            for label in ("REAL", "FAKE"):
                for index in range(100):
                    handle.write(json.dumps({
                        "sample_id": f"{label.lower()}-{index}",
                        "label": label,
                        "file_hash": hashlib.sha256(f"{label}-{index}".encode()).hexdigest(),
                    }) + "\n")
        config = self.root / "ratio-config.json"
        self.write_json(config, {
            "seed": "ratio-regression",
            "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "group_field": "auto",
            "label_field": "label",
            "source_split_field": "",
            "source_split_map": {},
        })
        counts = create_group_splits(source, self.root / "ratio-output", config_path=config)
        self.assertEqual(counts, {"train": 160, "validation": 20, "test": 20})

    def test_leakage_checker_flags_cross_split_exact_and_identity_duplicates(self) -> None:
        train = self.root / "manual_train.jsonl"
        validation = self.root / "manual_validation.jsonl"
        common_hash = hashlib.sha256(b"same generated fixture bytes").hexdigest()
        rows = [
            {"sample_id": "train-1", "split": "train", "split_group_id": "identity:person-1",
             "identity_id": "person-1", "file_hash": common_hash, "perceptual_hash": "0123456789abcdef"},
            {"sample_id": "validation-1", "split": "validation", "split_group_id": "identity:person-1",
             "identity_id": "person-1", "file_hash": common_hash, "perceptual_hash": "0123456789abcdee"},
        ]
        for path, row in zip((train, validation), rows):
            with path.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        result = check_split_leakage([train, validation], self.root / "leaky.json")
        kinds = {item["type"] for item in result["issues"]}
        self.assertIn("exact_file_duplicate_cross_split", kinds)
        self.assertIn("group_cross_split", kinds)
        self.assertIn("perceptual_duplicate_cross_split", kinds)
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
