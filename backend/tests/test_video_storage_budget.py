from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import shutil

from scripts.training.video.storage_budget import (
    GB,
    StorageBudgetExceeded,
    VideoStorageBudget,
    directory_size,
)


class VideoStorageBudgetTests(unittest.TestCase):
    def test_directory_size_counts_regular_files_without_following_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a.bin").write_bytes(b"1234")
            nested = root / "nested"
            nested.mkdir()
            (nested / "b.bin").write_bytes(b"56789")
            self.assertEqual(directory_size(root), 9)

    def test_budget_refuses_write_above_hard_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "video-data"
            root.mkdir()
            (root / "retained-archive.zip").write_bytes(b"x" * 12)
            budget = VideoStorageBudget(
                root=root,
                hard_limit_bytes=20,
                target_bytes=16,
                minimum_free_bytes=0,
            )
            with self.assertRaises(StorageBudgetExceeded):
                budget.require_capacity(9, purpose="extracting archive")

    def test_budget_refuses_output_when_drive_reserve_would_be_breached(self):
        with tempfile.TemporaryDirectory() as temporary:
            budget = VideoStorageBudget(
                root=Path(temporary) / "video-data",
                hard_limit_bytes=2 * GB,
                target_bytes=GB,
                minimum_free_bytes=50,
            )
            usage = shutil._ntuple_diskusage(total=100, used=50, free=50)
            with patch("scripts.training.video.storage_budget.shutil.disk_usage", return_value=usage):
                with self.assertRaises(StorageBudgetExceeded):
                    budget.require_capacity(1, purpose="writing a cache shard")

    def test_destination_must_be_new_and_inside_budget_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "video-data"
            root.mkdir()
            budget = VideoStorageBudget(root=root, hard_limit_bytes=100, target_bytes=80)
            self.assertEqual(budget.validate_destination(root / "raw" / "sample.mp4"), root / "raw" / "sample.mp4")
            with self.assertRaises(ValueError):
                budget.validate_destination(Path(temporary) / "outside.bin")
            existing = root / "existing.bin"
            existing.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                budget.validate_destination(existing)
            self.assertEqual(existing.read_text(encoding="utf-8"), "preserve")

    def test_data_root_cannot_be_inside_project(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            inside = project / "datasets" / "video"
            with self.assertRaises(ValueError):
                VideoStorageBudget(root=inside, project_root=project)


if __name__ == "__main__":
    unittest.main()
