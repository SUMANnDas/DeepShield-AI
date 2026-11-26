from __future__ import annotations

import unittest

import torch
from torch import nn

from scripts.training.video.evaluate import evaluate
from scripts.training.video.metrics import binary_metrics
from scripts.training.video.models import TemporalTransformer
from scripts.training.video.train import autocast_dtype, train_epoch


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


if __name__ == "__main__":
    unittest.main()
