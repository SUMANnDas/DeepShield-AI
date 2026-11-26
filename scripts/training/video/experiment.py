from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .storage_budget import VideoStorageBudget


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha256(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def manifest_identifier(path: Path, root: Path | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    display = str(resolved)
    if root is not None:
        try:
            display = resolved.relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            display = resolved.name
    return {"path": display, "sha256": sha256_file(resolved), "size_bytes": resolved.stat().st_size}


def write_json_budgeted(path: Path, value: Any, budget: VideoStorageBudget) -> Path:
    target = budget.validate_destination(Path(path))
    payload = (json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n").encode("utf-8")
    budget.require_capacity(len(payload), purpose=f"writing {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)
    return target


def append_jsonl_budgeted(path: Path, value: dict[str, Any], budget: VideoStorageBudget) -> Path:
    target = budget.validate_location(Path(path))
    payload = (json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str) + "\n").encode("utf-8")
    budget.require_capacity(len(payload), purpose=f"appending to {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("ab") as handle:
        handle.write(payload)
    return target


def append_csv_budgeted(
    path: Path,
    row: dict[str, Any],
    fieldnames: list[str],
    budget: VideoStorageBudget,
    *,
    write_header: bool = False,
) -> Path:
    target = budget.validate_location(Path(path))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    writer.writerow({field: row.get(field, "") for field in fieldnames})
    payload = buffer.getvalue().encode("utf-8")
    budget.require_capacity(len(payload), purpose=f"appending to {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "xb" if write_header else "ab"
    with target.open(mode) as handle:
        handle.write(payload)
    return target


def create_run(root, config, budget=None):
    active_budget = budget or VideoStorageBudget()
    run_root = active_budget.validate_location(Path(root))
    rid = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    path = active_budget.validate_destination(run_root / rid)
    config_path = path / "config.json"
    payload = json.dumps(config, indent=2) + "\n"
    active_budget.require_capacity(
        len(payload.encode("utf-8")),
        purpose="writing a video experiment configuration",
    )
    path.mkdir(parents=True, exist_ok=False)
    active_budget.validate_destination(config_path)
    with config_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    return rid, path
