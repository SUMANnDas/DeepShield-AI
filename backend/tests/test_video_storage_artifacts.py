from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.datasets.manifest_utils import load_annotations
from scripts.training.video.checkpoint import load_checkpoint, save_checkpoint
from scripts.training.video.experiment import create_run
from scripts.training.video.storage_budget import (
    StorageBudgetExceeded,
    VideoStorageBudget,
    directory_size,
)


class VideoStorageArtifactTests(unittest.TestCase):
    def make_budget(self, root: Path, *, hard_limit: int = 1_000_000) -> VideoStorageBudget:
        return VideoStorageBudget(
            root=root,
            hard_limit_bytes=hard_limit,
            target_bytes=min(hard_limit, 800_000),
            minimum_free_bytes=0,
            project_root=None,
        )

    def test_checkpoint_round_trip_stays_under_budget_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "video-data"
            root.mkdir()
            budget = self.make_budget(root)
            model = torch.nn.Linear(3, 2)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            checkpoint = root / "runs" / "run-1" / "best.pt"
            save_checkpoint(checkpoint, model, optimizer, epoch=4, config={"seed": 9}, budget=budget)

            recovered = torch.nn.Linear(3, 2)
            state = load_checkpoint(checkpoint, recovered, map_location="cpu")
            self.assertEqual(state["epoch"], 4)
            self.assertEqual(state["config"], {"seed": 9})
            self.assertTrue(torch.equal(model.weight, recovered.weight))
            self.assertLessEqual(directory_size(root), budget.hard_limit_bytes)

            with self.assertRaises(FileExistsError):
                save_checkpoint(checkpoint, model, budget=budget)

    def test_checkpoint_write_stops_at_quota_without_exceeding_it(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "video-data"
            root.mkdir()
            budget = self.make_budget(root, hard_limit=256)
            model = torch.nn.Linear(8, 8)
            with self.assertRaises(StorageBudgetExceeded):
                save_checkpoint(root / "runs" / "too-large.pt", model, budget=budget)
            self.assertLessEqual(directory_size(root), budget.hard_limit_bytes)

    def test_experiment_config_is_exclusive_and_quota_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "video-data"
            root.mkdir()
            budget = self.make_budget(root)
            run_id, run_path = create_run(root / "runs", {"seed": 11, "frames": 12}, budget=budget)
            self.assertTrue(run_id in run_path.name)
            self.assertEqual(json.loads((run_path / "config.json").read_text(encoding="utf-8")),
                             {"seed": 11, "frames": 12})
            with self.assertRaises(ValueError):
                create_run(Path(temporary) / "outside", {}, budget=budget)

    def test_experiment_config_refuses_quota_before_creating_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "video-data"
            root.mkdir()
            budget = self.make_budget(root, hard_limit=8)
            with self.assertRaises(StorageBudgetExceeded):
                create_run(root / "runs", {"large": "x" * 256}, budget=budget)
            self.assertEqual(directory_size(root), 0)

    def test_annotation_index_obeys_its_page_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            annotations = root / "annotations.csv"
            annotations.write_text(
                "filename,label\n" + "".join(f"clip-{index}.mp4,FAKE\n" for index in range(500)),
                encoding="utf-8",
            )
            index = root / "annotations.sqlite3"
            with self.assertRaises(sqlite3.DatabaseError):
                load_annotations(annotations, index, index_max_bytes=8192)
            self.assertLessEqual(index.stat().st_size, 8192)


if __name__ == "__main__":
    unittest.main()
