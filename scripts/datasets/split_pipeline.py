"""Deterministic group-aware train/validation/test splits using SQLite-backed indexing."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from scripts.datasets.manifest_utils import json_config, read_jsonl, refuse_existing

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": "deepshield-phase2",
    "ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
    "group_field": "auto",
    "label_field": "label",
    "source_split_field": "source_split",
    "source_split_map": {
        "train": "train", "training": "train",
        "dev": "validation", "valid": "validation", "validation": "validation",
        "eval": "test", "evaluation": "test", "test": "test",
    },
}
AUTO_GROUP_FIELDS = (
    "source_identity_id", "identity_id", "speaker_id", "source_video_id",
    "original_video_id", "video_sample_id", "audio_sample_id", "group_id", "file_hash",
)


def _validate_ratios(ratios: dict[str, float]) -> dict[str, float]:
    if set(ratios) != {"train", "validation", "test"}:
        raise ValueError("ratios must provide train, validation, and test")
    values = {key: float(value) for key, value in ratios.items()}
    if any(value <= 0 for value in values.values()) or abs(sum(values.values()) - 1.0) > 1e-8:
        raise ValueError("split ratios must be positive and sum to 1")
    return values


def _group_key(record: dict[str, Any], group_field: str) -> str:
    if group_field == "auto":
        for field in AUTO_GROUP_FIELDS:
            value = record.get(field)
            if value is not None and str(value).strip():
                return f"{field}:{value}"
        return f"sample_id:{record.get('sample_id', '')}"
    value = record.get(group_field)
    if value is None or not str(value).strip():
        return f"sample_id:{record.get('sample_id', '')}"
    return f"{group_field}:{value}"


def create_group_splits(
    manifest_path: Path,
    output_prefix: Path,
    *,
    config_path: Path | None = None,
) -> dict[str, int]:
    source = Path(manifest_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {name: output_prefix.with_name(f"{output_prefix.name}_{name}.jsonl") for name in ("train", "validation", "test")}
    for path in outputs.values():
        refuse_existing(path)
    config = json_config(config_path, DEFAULT_CONFIG)
    ratios = _validate_ratios(config["ratios"])
    group_field = str(config.get("group_field", "auto"))
    label_field = str(config.get("label_field", "label"))
    source_split_field = str(config.get("source_split_field", "source_split"))
    source_split_map = {str(key).strip().lower(): str(value).strip().lower() for key, value in config.get("source_split_map", {}).items()}
    if any(value not in {"train", "validation", "test"} for value in source_split_map.values()):
        raise ValueError("source_split_map values must be train, validation, or test")
    database = output_prefix.with_name(f"{output_prefix.name}.split-index.sqlite3")
    refuse_existing(database)
    connection = sqlite3.connect(database)
    counts = {name: 0 for name in outputs}
    try:
        connection.execute("CREATE TABLE samples (sample_id TEXT PRIMARY KEY, group_id TEXT NOT NULL, label TEXT NOT NULL, source_split TEXT NOT NULL, line TEXT NOT NULL)")
        connection.execute("CREATE INDEX samples_group_idx ON samples(group_id)")
        connection.execute("CREATE TABLE assignments (group_id TEXT PRIMARY KEY, split TEXT NOT NULL)")
        for row in read_jsonl(source):
            identifier = str(row.get("sample_id") or "")
            if not identifier:
                raise ValueError("Each manifest row needs sample_id")
            group = _group_key(row, group_field)
            label = str(row.get(label_field, "UNKNOWN"))
            source_value = str(row.get(source_split_field) or "").strip().lower()
            fixed_split = source_split_map.get(source_value, "") if source_value else ""
            if source_value and not fixed_split:
                raise ValueError(f"Unrecognized source split {source_value!r} for sample {identifier}")
            try:
                connection.execute(
                    "INSERT INTO samples VALUES (?, ?, ?, ?, ?)",
                    (identifier, group, label, fixed_split, json.dumps(row, ensure_ascii=False, separators=(",", ":"))),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Duplicate sample_id in source manifest: {identifier}") from exc
        connection.commit()
        totals = {str(label): int(count) for label, count in connection.execute("SELECT label, COUNT(*) FROM samples GROUP BY label")}
        target = {split: {label: count * ratios[split] for label, count in totals.items()} for split in outputs}
        assigned = {split: {label: 0 for label in totals} for split in outputs}
        total_rows = sum(totals.values())
        assigned_total = {split: 0 for split in outputs}
        ordered_groups = connection.execute(
            "SELECT group_id, COUNT(*) AS group_size FROM samples GROUP BY group_id ORDER BY group_size DESC, group_id ASC"
        )
        for group, _size in ordered_groups:
            fixed = {str(value) for (value,) in connection.execute(
                "SELECT DISTINCT source_split FROM samples WHERE group_id=? AND source_split<>''", (group,)
            )}
            if len(fixed) > 1:
                raise ValueError(f"Group {group!r} spans conflicting provider splits: {sorted(fixed)}")
            per_label = {str(label): int(count) for label, count in connection.execute(
                "SELECT label, COUNT(*) FROM samples WHERE group_id=? GROUP BY label", (group,)
            )}
            if fixed:
                fixed_split = next(iter(fixed))
                existing = connection.execute("SELECT split FROM assignments WHERE group_id=?", (group,)).fetchone()
                if existing and existing[0] != fixed_split:
                    raise ValueError(f"Group {group!r} has inconsistent split assignments")
                if not existing:
                    connection.execute("INSERT INTO assignments VALUES (?, ?)", (group, fixed_split))
                    assigned_total[fixed_split] += sum(per_label.values())
                    for label, count in per_label.items():
                        assigned[fixed_split][label] += count
                continue
            best_split: str | None = None
            best_score: float | None = None
            for split in outputs:
                score = 0.0
                # Score the whole assignment after placing this group. Looking only at
                # the candidate split's own deficit biases unequal ratios toward an
                # approximately even partition.
                for candidate_split in outputs:
                    for label in totals:
                        goal = target[candidate_split][label]
                        addition = per_label.get(label, 0) if candidate_split == split else 0
                        after = assigned[candidate_split][label] + addition
                        score += ((after - goal) / max(1.0, goal)) ** 2
                    goal_total = total_rows * ratios[candidate_split]
                    addition_total = sum(per_label.values()) if candidate_split == split else 0
                    after_total = assigned_total[candidate_split] + addition_total
                    score += ((after_total - goal_total) / max(1.0, goal_total)) ** 2
                tie = hashlib.sha256(f"{config['seed']}\0{group}\0{split}".encode()).hexdigest()
                candidate = (score, tie)
                if best_score is None or candidate < (best_score, best_tie):
                    best_score, best_tie, best_split = score, tie, split
            assert best_split is not None
            connection.execute("INSERT INTO assignments VALUES (?, ?)", (group, best_split))
            assigned_total[best_split] += sum(per_label.values())
            for label, count in per_label.items():
                assigned[best_split][label] += count
        connection.commit()
        with ExitStack() as stack:
            handles = {split: stack.enter_context(path.open("x", encoding="utf-8", newline="\n")) for split, path in outputs.items()}
            cursor = connection.execute(
                "SELECT samples.line, assignments.split FROM samples JOIN assignments USING(group_id) ORDER BY samples.sample_id"
            )
            for serialized, split in cursor:
                row = json.loads(serialized)
                row["split"] = split
                row["split_group_id"] = _group_key(row, group_field)
                handles[split].write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[split] += 1
    finally:
        connection.close()
    return counts
