"""Stream JSONL manifests into factual quality summaries, CSV tables, and SVG charts."""

from __future__ import annotations

import csv
import html
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.datasets.manifest_utils import read_jsonl, refuse_existing


def _svg_bars(values: Counter[str], title: str) -> str:
    width, height = 900, 460
    margin_left, margin_bottom, chart_width, chart_height = 220, 75, 620, 320
    maximum = max(values.values(), default=1)
    row_height = min(42, chart_height / max(1, len(values)))
    bars = [
        f'<text x="{margin_left - 10}" y="{margin_bottom + index * row_height + row_height * 0.72}" text-anchor="end" font-size="13">{html.escape(label[:32])}</text>'
        f'<rect x="{margin_left}" y="{margin_bottom + index * row_height}" width="{chart_width * count / maximum:.2f}" height="{max(2, row_height - 6):.2f}" fill="#2878b5"/>'
        f'<text x="{margin_left + chart_width * count / maximum + 8:.2f}" y="{margin_bottom + index * row_height + row_height * 0.72}" font-size="12">{count}</text>'
        for index, (label, count) in enumerate(values.most_common())
    ]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{margin_left}" y="34" font-size="21" font-weight="bold">{html.escape(title)}</text>'
        + "".join(bars)
        + "</svg>\n"
    )


def generate_dataset_report(manifest_path: Path, output_prefix: Path, media_root: Path | None = None) -> dict[str, Any]:
    source = Path(manifest_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = {suffix: prefix.with_suffix(suffix) for suffix in (".json", ".csv", ".svg")}
    for output in outputs.values():
        refuse_existing(output)
    path_index = prefix.with_suffix(".report-index.sqlite3") if media_root is not None else None
    size_index = None
    if path_index is not None:
        refuse_existing(path_index)
        path_index.parent.mkdir(parents=True, exist_ok=True)
        size_index = sqlite3.connect(path_index)
        size_index.execute("CREATE TABLE source_paths (relative_path TEXT PRIMARY KEY, size_bytes INTEGER NOT NULL)")
    counters = {key: Counter() for key in ("dataset_name", "task", "label", "preprocessing_status")}
    quality = Counter()
    numeric: dict[str, dict[str, float]] = {}
    total = 0
    media_root_resolved = Path(media_root).resolve() if media_root is not None else None
    for row in read_jsonl(source):
        total += 1
        for key in counters:
            counters[key][str(row.get(key) or "UNKNOWN")] += 1
        status = str(row.get("preprocessing_status") or "UNKNOWN")
        label = str(row.get("label") or "UNKNOWN")
        quality[f"status:{status}"] += 1
        quality[f"label:{label}"] += 1
        if label == "UNKNOWN":
            quality["unlabeled"] += 1
        if row.get("duplicate_of"):
            quality["duplicates"] += 1
        if row.get("face_detected") is False:
            quality["no_face"] += 1
        if row.get("silent") is True:
            quality["silent_audio_segments"] += 1
        if row.get("error_code"):
            quality["records_with_errors"] += 1
        for field in ("image_width", "image_height", "duration_seconds", "sample_rate", "target_sample_rate"):
            value = row.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                accumulator = numeric.setdefault(field, {"count": 0.0, "sum": 0.0, "min": float(value), "max": float(value)})
                accumulator["count"] += 1
                accumulator["sum"] += float(value)
                accumulator["min"] = min(accumulator["min"], float(value))
                accumulator["max"] = max(accumulator["max"], float(value))
        relative = row.get("relative_path")
        if media_root_resolved is not None and relative:
            candidate = media_root_resolved / Path(str(relative))
            try:
                candidate_resolved = candidate.resolve()
                relative_safe = candidate_resolved.relative_to(media_root_resolved).as_posix()
                if candidate_resolved.is_file() and size_index is not None:
                    size_index.execute(
                        "INSERT OR IGNORE INTO source_paths VALUES (?, ?)",
                        (relative_safe, candidate_resolved.stat().st_size),
                    )
            except OSError:
                quality["unavailable_source_paths"] += 1
            except ValueError:
                quality["unsafe_source_paths"] += 1
    total_bytes = None
    if size_index is not None:
        size_index.commit()
        total_bytes = int(size_index.execute("SELECT COALESCE(SUM(size_bytes), 0) FROM source_paths").fetchone()[0])
        size_index.close()
    numeric_summary = {
        field: {
            **{key: int(value) if key == "count" else value for key, value in values.items()},
            "mean": values["sum"] / values["count"] if values["count"] else None,
        }
        for field, values in numeric.items()
    }
    result = {
        "source_manifest": str(source),
        "records": total,
        "source_media_bytes_found": total_bytes,
        "distributions": {key: dict(value.most_common()) for key, value in counters.items()},
        "quality_counts": dict(quality.most_common()),
        "numeric_fields": numeric_summary,
    }
    with outputs[".json"].open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with outputs[".csv"].open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "value", "count"))
        for metric, values in counters.items():
            for value, count in values.most_common():
                writer.writerow((metric, value, count))
        for metric, count in quality.most_common():
            writer.writerow(("quality", metric, count))
    with outputs[".svg"].open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_svg_bars(counters["label"], "Manifest label distribution"))
    return result
