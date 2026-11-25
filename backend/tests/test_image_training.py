"""Small synthetic training-loop and split-isolation checks."""

from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.training.image.train import train_one_epoch, validate_one_epoch


class ImageTrainingTests(unittest.TestCase):
    def test_one_epoch_updates_weights_and_loss_is_finite(self) -> None:
        torch.manual_seed(7)
        images = torch.randn(8, 3, 32, 32)
        labels = torch.tensor([0, 1] * 4)
        loader = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model[1].weight.detach().clone()
        loss = train_one_epoch(model, loader, optimizer, torch.device("cpu"), gradient_clip_norm=1.0)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertFalse(torch.equal(before, model[1].weight.detach()))

    def test_validation_metrics_have_expected_support(self) -> None:
        images = torch.zeros(4, 3, 32, 32)
        labels = torch.tensor([0, 0, 1, 1])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 2))
        loss, metrics = validate_one_epoch(model, loader, torch.device("cpu"))
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(metrics["support"], {"REAL": 2, "FAKE": 2})
        self.assertEqual(np_shape(metrics["confusion_matrix"]), (2, 2))


def np_shape(value: object) -> tuple[int, ...]:
    return tuple(torch.as_tensor(value).shape)


if __name__ == "__main__":
    unittest.main()
