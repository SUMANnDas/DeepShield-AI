"""Shared dataset manifest, label, path, hashing, and configuration utilities."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

logger = logging.getLogger(__name__)

STANDARD_LABELS = {"REAL", "FAKE", "UNKNOWN"}
_REAL_ALIASES = {"real", "authentic", "bona_fide", "bonafide", "genuine", "human"}
_FAKE_ALIASES = {"fake", "synthetic", "spoof", "manipulated", "forged", "deepfake"}


def canonical_label(value: Any) -> str:
    """Map a known binary source label to the standard schema; preserve detail elsewhere."""
    if value is None:
        return "UNKNOWN"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _REAL_ALIASES:
        return "REAL"
    if normalized in _FAKE_ALIASES:
        return "FAKE"
    if normalized in {"real", "fake", "unknown"}:
        return normalized.upper()
    return "UNKNOWN"


class AnnotationStore:
    """Disk-backed annotation lookup so large source label files stay out of RAM."""

    def __init__(self, path: Path, max_bytes: int | None = None) -> None:
        if max_bytes is not None and max_bytes < 4096:
            raise ValueError("max_bytes must allow at least one 4096-byte SQLite page")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        try:
            if max_bytes is not None:
                self.connection.execute("PRAGMA page_size=4096")
                self.connection.execute("PRAGMA journal_mode=MEMORY")
                max_pages = max_bytes // 4096
                self.connection.execute(f"PRAGMA max_page_count={max_pages}")
            self.connection.execute(
                "CREATE TABLE annotations (key TEXT PRIMARY KEY, basename TEXT NOT NULL, row_json TEXT NOT NULL)"
            )
            self.connection.execute("CREATE INDEX annotations_basename_idx ON annotations(basename)")
        except BaseException:
            self.connection.close()
            raise

    def __enter__(self) -> "AnnotationStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def add(self, row: Mapping[str, Any]) -> None:
        key_value = row.get("relative_path") or row.get("original_filename") or row.get("filename")
        if not key_value:
            raise ValueError("Each annotation row needs relative_path or original_filename")
        key = Path(str(key_value).replace("\\", "/")).as_posix()
        try:
            self.connection.execute(
                "INSERT INTO annotations VALUES (?, ?, ?)",
                (key, Path(key).name, json.dumps(dict(row), ensure_ascii=False, default=str)),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Duplicate annotation key: {key}") from exc

    def finish(self) -> None:
        self.connection.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        key = Path(str(key).replace("\\", "/")).as_posix()
        row = self.connection.execute("SELECT row_json FROM annotations WHERE key=?", (key,)).fetchone()
        if row is None:
            row = self.connection.execute(
                "SELECT MIN(row_json), COUNT(*) FROM annotations WHERE basename=? GROUP BY basename HAVING COUNT(*)=1",
                (Path(key).name,),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        self.connection.close()


def load_annotations(
    path: Path | None,
    index_path: Path | None = None,
    *,
    index_max_bytes: int | None = None,
) -> AnnotationStore | dict[str, dict[str, Any]]:
    """Index CSV/JSONL annotations on disk; JSON arrays are accepted for small sidecars."""
    if path is None:
        return {}
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    records: Iterator[dict[str, Any]]
    if path.suffix.lower() == ".csv":
        def csv_records() -> Iterator[dict[str, Any]]:
            with path.open("r", newline="", encoding="utf-8-sig") as handle:
                yield from csv.DictReader(handle)
        records = csv_records()
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        def jsonl_records() -> Iterator[dict[str, Any]]:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if line.strip():
                        value = json.loads(line)
                        if not isinstance(value, dict):
                            raise ValueError(f"Annotation line {line_number} must be an object")
                        yield value
        records = jsonl_records()
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
            raise ValueError("JSON annotations must be a list of objects")
        records = iter(value)
    else:
        raise ValueError("Annotations must be CSV, JSON, or JSONL")

    if index_path is None:
        index: dict[str, dict[str, Any]] = {}
        for row in records:
            key = row.get("relative_path") or row.get("original_filename") or row.get("filename")
            if not key:
                raise ValueError("Each annotation row needs relative_path or original_filename")
            key = Path(str(key).replace("\\", "/")).as_posix()
            if key in index:
                raise ValueError(f"Duplicate annotation key: {key}")
            original = row.get("original_label", row.get("label", row.get("class", "UNKNOWN")))
            index[key] = {**row, "relative_path": key, "label": canonical_label(row.get("label", original)), "original_label": str(original)}
        return index

    refuse_existing(index_path)
    store = AnnotationStore(index_path, max_bytes=index_max_bytes)
    try:
        for row in records:
            store.add(row)
        store.finish()
        return store
    except BaseException:
        store.close()
        raise


def annotation_for(index: Mapping[str, Mapping[str, Any]], relative_path: str, default_label: str) -> dict[str, Any]:
    row = index.get(relative_path)
    if row is None:
        # Basename fallback is accepted only when exactly one annotation uses that name.
        if hasattr(index, "get"):
            row = index.get(relative_path) or {}
        else:
            matches = [value for key, value in index.items() if Path(key).name == Path(relative_path).name]
            row = matches[0] if len(matches) == 1 else {}
    original = str(row.get("original_label", row.get("label", default_label)))
    label = canonical_label(row.get("label", original))
    return {
        "label": label if label != "UNKNOWN" or canonical_label(default_label) == "UNKNOWN" else canonical_label(default_label),
        "original_label": original,
        "manipulation_type": row.get("manipulation_type") or row.get("attack_type"),
        "source_video_id": row.get("source_video_id") or row.get("original_video_id") or row.get("video_id"),
        "identity_id": row.get("identity_id") or row.get("speaker_id") or row.get("subject_id"),
        "speaker_id": row.get("speaker_id"),
        "source_split": row.get("source_split") or row.get("dataset_split"),
        "task": row.get("task"),
    }


def iter_media_files(root: Path, extensions: Iterable[str]) -> Iterator[Path]:
    root = Path(root)
    allowed = {str(ext).lower() if str(ext).startswith(".") else f".{str(ext).lower()}" for ext in extensions}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            yield path


def relative_posix(path: Path, root: Path) -> str:
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_id(dataset_name: str, relative_path: str) -> str:
    token = f"{dataset_name}\0{Path(relative_path).as_posix()}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()[:24]


def json_config(path: Path | None, defaults: Mapping[str, Any]) -> dict[str, Any]:
    config = dict(defaults)
    if path is None:
        return config
    loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("Configuration JSON must contain an object")
    config.update(loaded)
    return config


def config_hash(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def exclusive_jsonl(path: Path) -> Any:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("x", encoding="utf-8", newline="\n")


def append_jsonl(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str))
    handle.write("\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Manifest line {line_number} must be an object")
            yield value


def refuse_existing(path: Path) -> None:
    if Path(path).exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def safe_output_name(identifier: str, suffix: str) -> str:
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return f"{identifier}{suffix.lower()}"


def log_file_error(logger_: logging.Logger, modality: str, relative_path: str, exc: BaseException) -> None:
    logger_.warning("%s processing failed for %s (%s)", modality, relative_path, type(exc).__name__)
