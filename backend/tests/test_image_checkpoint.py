"""Append-only checkpoint persistence and collision handling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.training.image.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_exclusive,
)


class ImageCheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_and_collision_protection(self) -> None:
        model = torch.nn.Linear(3, 2)
        payload = {"epoch": 4, "model_state": model.state_dict(), "configuration_sha256": "abc"}
        with tempfile.TemporaryDirectory(prefix="cifake-checkpoint-") as temp:
            path = Path(temp) / "checkpoint.pt"
            save_checkpoint_exclusive(path, payload)
            loaded = load_checkpoint(path)
            self.assertEqual(loaded["epoch"], 4)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded["model_state"][key]))
            with self.assertRaises(FileExistsError):
                save_checkpoint_exclusive(path, payload)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CUDA RNG recovery")
    def test_cuda_rng_state_restores_after_device_mapped_checkpoint_load(self) -> None:
        model = torch.nn.Linear(3, 2)
        payload = {"epoch": 1, "model_state": model.state_dict(), "rng_state": capture_rng_state()}
        with tempfile.TemporaryDirectory(prefix="cifake-cuda-rng-") as temp:
            path = Path(temp) / "checkpoint.pt"
            save_checkpoint_exclusive(path, payload)
            loaded = load_checkpoint(path, map_location=torch.device("cuda"))
            restore_rng_state(loaded["rng_state"])


if __name__ == "__main__":
    unittest.main()
