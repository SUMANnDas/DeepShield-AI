"""Exact, perceptual, source-video, speaker, and identity split-leakage checks."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from scripts.datasets.image_pipeline import _hamming
from scripts.datasets.manifest_utils import read_jsonl, refuse_existing


def check_split_leakage(
    manifests: Iterable[Path],
    report_path: Path,
    *,
    perceptual_distance: int = 6,
    max_reported_issues: int = 10000,
) -> dict[str, Any]:
    sources = [Path(item) for item in manifests]
    if not sources:
        raise ValueError("At least one split manifest is required")
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
    report_path = Path(report_path)
    refuse_existing(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    db_path = report_path.with_suffix(".leakage-index.sqlite3")
    refuse_existing(db_path)
    connection = sqlite3.connect(db_path)
    issues: list[dict[str, Any]] = []
    issue_count = 0
    records_seen = 0

    def issue(value: dict[str, Any]) -> None:
        nonlocal issue_count
        issue_count += 1
        if len(issues) < max_reported_issues:
            issues.append(value)

    try:
        connection.execute("CREATE TABLE samples (sample_id TEXT NOT NULL, split TEXT NOT NULL, file_hash TEXT, phash TEXT, group_id TEXT)")
        connection.execute("CREATE INDEX samples_hash_idx ON samples(file_hash)")
        for band in range(4):
            connection.execute(f"CREATE INDEX samples_b{band}_idx ON samples(substr(phash,{band * 4 + 1},4))")
        connection.execute("CREATE TABLE groups (kind TEXT NOT NULL, group_value TEXT NOT NULL, split TEXT NOT NULL, sample_id TEXT NOT NULL)")
        connection.execute("CREATE INDEX groups_lookup_idx ON groups(kind,group_value)")
        for source in sources:
            for row in read_jsonl(source):
                records_seen += 1
                sample_id = str(row.get("sample_id") or f"line:{records_seen}")
                split = str(row.get("split") or "")
                if not split:
                    raise ValueError(f"Split field missing on row {records_seen} of {source}")
                group_id = str(row.get("split_group_id") or "")
                for field in ("source_video_id", "original_video_id", "audio_sample_id", "speaker_id", "source_identity_id", "identity_id"):
                    value = row.get(field)
                    if value is None or not str(value).strip():
                        continue
                    normalized = str(value)
                    previous = connection.execute(
                        "SELECT split, sample_id FROM groups WHERE kind=? AND group_value=? AND split<>? LIMIT 1",
                        (field, normalized, split),
                    ).fetchone()
                    if previous:
                        issue({"type": "group_cross_split", "field": field, "group_id": normalized,
                               "first_sample_id": previous[1], "first_split": previous[0],
                               "sample_id": sample_id, "split": split})
                    connection.execute("INSERT INTO groups VALUES (?, ?, ?, ?)", (field, normalized, split, sample_id))
                if group_id:
                    previous = connection.execute(
                        "SELECT split, sample_id FROM groups WHERE kind='split_group_id' AND group_value=? AND split<>? LIMIT 1",
                        (group_id, split),
                    ).fetchone()
                    if previous:
                        issue({"type": "split_group_cross_split", "group_id": group_id,
                               "first_sample_id": previous[1], "first_split": previous[0],
                               "sample_id": sample_id, "split": split})
                    connection.execute("INSERT INTO groups VALUES ('split_group_id', ?, ?, ?)", (group_id, split, sample_id))

                file_hash = row.get("file_hash")
                if file_hash:
                    previous = connection.execute(
                        "SELECT sample_id, split FROM samples WHERE file_hash=? AND split<>? LIMIT 1",
                        (str(file_hash), split),
                    ).fetchone()
                    if previous:
                        issue({"type": "exact_file_duplicate_cross_split", "hash": str(file_hash),
                               "first_sample_id": previous[0], "first_split": previous[1],
                               "sample_id": sample_id, "split": split})
                phash = str(row.get("perceptual_hash") or "").lower()
                if len(phash) == 16:
                    candidates = connection.execute(
                        "SELECT sample_id, split, phash FROM samples WHERE "
                        + " OR ".join(f"substr(phash,{band * 4 + 1},4)=?" for band in range(4)),
                        tuple(phash[band * 4:(band + 1) * 4] for band in range(4)),
                    )
                    for candidate_id, candidate_split, candidate_hash in candidates:
                        if candidate_split != split and _hamming(phash, str(candidate_hash)) <= perceptual_distance:
                            issue({"type": "perceptual_duplicate_cross_split", "first_sample_id": candidate_id,
                                   "first_split": candidate_split, "sample_id": sample_id, "split": split,
                                   "distance": _hamming(phash, str(candidate_hash))})
                connection.execute(
                    "INSERT INTO samples VALUES (?, ?, ?, ?, ?)",
                    (sample_id, split, str(file_hash) if file_hash else None, phash or None, group_id or None),
                )
        connection.commit()
        result = {
            "records_seen": records_seen,
            "issue_count": issue_count,
            "issues_truncated": issue_count > len(issues),
            "issues": issues,
            "passed": issue_count == 0,
            "source_manifests": [str(path) for path in sources],
            "perceptual_distance_threshold": perceptual_distance,
        }
        with report_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return result
    finally:
        connection.close()
