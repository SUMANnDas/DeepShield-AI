"""Non-destructive storage checks for local video datasets and experiments.

All video archives, extracted media, optional caches, and run artifacts should
live under one dedicated root so the configured quota covers retained copies as
well as outputs. This module checks capacity; it never creates or removes files.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

GB = 1_000_000_000
DEFAULT_VIDEO_DATA_ROOT = Path(
    os.environ.get("DEEPSHIELD_VIDEO_DATA_ROOT", r"D:\DeepShield-Data\VideoDetection")
)
DEFAULT_HARD_LIMIT_BYTES = 40 * GB
DEFAULT_TARGET_BYTES = 30 * GB
DEFAULT_MINIMUM_FREE_BYTES = 5 * GB
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class StorageBudgetExceeded(RuntimeError):
    """Raised when a planned write could exceed a storage limit."""


def directory_size(path: str | Path) -> int:
    """Return the byte size of regular files below *path*, without following links."""
    root = Path(path)
    if not root.exists():
        return 0
    if not root.is_dir():
        raise NotADirectoryError(root)

    total = 0

    def fail(error: OSError) -> None:
        raise error

    for current, directories, filenames in os.walk(root, followlinks=False, onerror=fail):
        current_path = Path(current)
        directories[:] = [name for name in directories if not (current_path / name).is_symlink()]
        for filename in filenames:
            file_path = current_path / filename
            if file_path.is_symlink():
                continue
            total += file_path.stat().st_size
    return total


def _existing_volume_path(path: Path) -> Path:
    current = path
    while not current.exists():
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(f"No existing parent for storage root: {path}")
        current = parent
    return current


@dataclass(frozen=True)
class StorageReport:
    root: Path
    current_bytes: int
    planned_bytes: int
    hard_limit_bytes: int
    free_bytes: int
    minimum_free_bytes: int

    @property
    def after_write_bytes(self) -> int:
        return self.current_bytes + self.planned_bytes

    @property
    def within_budget(self) -> bool:
        return (
            self.after_write_bytes <= self.hard_limit_bytes
            and self.free_bytes - self.planned_bytes >= self.minimum_free_bytes
        )


@dataclass(frozen=True)
class VideoStorageBudget:
    """A hard quota for all newly acquired video data and generated artifacts."""

    root: Path = DEFAULT_VIDEO_DATA_ROOT
    hard_limit_bytes: int = DEFAULT_HARD_LIMIT_BYTES
    target_bytes: int = DEFAULT_TARGET_BYTES
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES
    project_root: Path | None = DEFAULT_PROJECT_ROOT

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        object.__setattr__(self, "root", root)
        if self.hard_limit_bytes <= 0:
            raise ValueError("hard_limit_bytes must be positive")
        if self.target_bytes <= 0 or self.target_bytes > self.hard_limit_bytes:
            raise ValueError("target_bytes must be positive and no larger than the hard limit")
        if self.minimum_free_bytes < 0:
            raise ValueError("minimum_free_bytes cannot be negative")
        if self.project_root is not None:
            project = Path(self.project_root).expanduser().resolve()
            try:
                root.relative_to(project)
            except ValueError:
                pass
            else:
                raise ValueError("video data root must be outside the Git project")
        if os.name == "nt" and root.drive.upper() not in {"C:", "D:"}:
            raise ValueError("video data must be stored on C: or D:")

    def report(self, planned_bytes: int = 0, *, current_bytes: int | None = None) -> StorageReport:
        if planned_bytes < 0:
            raise ValueError("planned_bytes cannot be negative")
        if current_bytes is not None and current_bytes < 0:
            raise ValueError("current_bytes cannot be negative")
        anchor = _existing_volume_path(self.root)
        return StorageReport(
            root=self.root,
            current_bytes=directory_size(self.root) if current_bytes is None else current_bytes,
            planned_bytes=planned_bytes,
            hard_limit_bytes=self.hard_limit_bytes,
            free_bytes=shutil.disk_usage(anchor).free,
            minimum_free_bytes=self.minimum_free_bytes,
        )

    def require_capacity(
        self,
        planned_bytes: int,
        *,
        purpose: str,
        current_bytes: int | None = None,
    ) -> StorageReport:
        """Fail before a write if quota or the drive free-space reserve is exceeded."""
        report = self.report(planned_bytes, current_bytes=current_bytes)
        if report.after_write_bytes > self.hard_limit_bytes:
            raise StorageBudgetExceeded(
                f"{purpose} would use {report.after_write_bytes / GB:.2f} GB, "
                f"above the {self.hard_limit_bytes / GB:.2f} GB video-data limit."
            )
        if report.free_bytes - planned_bytes < self.minimum_free_bytes:
            raise StorageBudgetExceeded(
                f"{purpose} would leave less than "
                f"{self.minimum_free_bytes / GB:.2f} GB free on the target drive."
            )
        return report

    def validate_location(self, location: str | Path) -> Path:
        """Require a path, existing or new, to remain under the budgeted root."""
        path = Path(location).expanduser().resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError("video data and experiment outputs must stay under the budget root") from error
        return path

    def validate_destination(self, destination: str | Path) -> Path:
        """Require a new output path to remain under the budgeted root."""
        path = self.validate_location(destination)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing video artifact: {path}")
        return path
