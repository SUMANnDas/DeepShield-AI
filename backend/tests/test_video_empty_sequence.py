from __future__ import annotations

import unittest

import torch

from scripts.training.video.models import TemporalTransformer


class VideoEmptySequenceTests(unittest.TestCase):
    def test_transformer_handles_all_padded_sequences_with_finite_neutral_output(self) -> None:
        model = TemporalTransformer(dim=32, max_frames=4, layers=1, heads=4, dropout=0.0)
        features = torch.randn(2, 4, 32, requires_grad=True)
        mask = torch.tensor([[False, False, False, False], [True, True, False, False]])
        output = model(features, mask)
        for value in output.values():
            self.assertTrue(torch.isfinite(value).all())
        torch.testing.assert_close(output["logits"][0], torch.zeros(2))
        torch.testing.assert_close(output["attention"][0], torch.zeros(4))
        torch.testing.assert_close(output["frame_scores"][0], torch.zeros(4))
        output["logits"].sum().backward()
        self.assertIsNotNone(features.grad)
        self.assertTrue(torch.isfinite(features.grad).all())


if __name__ == "__main__":
    unittest.main()
