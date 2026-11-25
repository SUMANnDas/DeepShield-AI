"""Local JSON experiment metadata and unique path management."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from scripts.training.image.utils import PROJECT_ROOT, gpu_metadata, software_versions


EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "runs" / "image_detection" / "cifake-v3"
CHECKPOINT_ROOT = PROJECT_ROOT / "models" / "checkpoints" / "image_detection" / "cifake-v3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_identifier(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", value.lower()).strip("-")


def create_run_directories(architecture: str) -> tuple[str, Path, Path]:
    """Create new run/checkpoint directories and fail closed on every collision."""
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    model_slug = _slug(architecture)
    for _ in range(10):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{model_slug}-{stamp}-{uuid.uuid4().hex[:8]}"
        run_dir = EXPERIMENT_ROOT / run_id
        checkpoint_dir = CHECKPOINT_ROOT / run_id
        if run_dir.exists() or checkpoint_dir.exists():
            continue
        run_dir.mkdir(exist_ok=False)
        try:
            checkpoint_dir.mkdir(exist_ok=False)
        except BaseException:
            raise
        return run_id, run_dir, checkpoint_dir
    raise RuntimeError("Could not allocate a collision-free experiment run ID")


def write_json_exclusive(path: Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
    return path


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str))
        handle.write("\n")


def environment_snapshot() -> dict[str, Any]:
    memory = None
    if torch.cuda.is_available():
        memory = {
            "allocated_bytes": int(torch.cuda.memory_allocated(0)),
            "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        }
    installed_packages = sorted(
        (
            {"name": distribution.metadata.get("Name", ""), "version": distribution.version}
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        ),
        key=lambda item: item["name"].lower(),
    )
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        nvidia_report = {"return_code": nvidia_smi.returncode, "stdout": nvidia_smi.stdout.strip(), "stderr": nvidia_smi.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        nvidia_report = {"error": str(exc)}
    disk_usage = shutil.disk_usage(PROJECT_ROOT)
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "software_versions": software_versions(),
        "installed_packages": installed_packages,
        "gpu": gpu_metadata(),
        "nvidia_smi": nvidia_report,
        "disk_space_bytes": {"total": disk_usage.total, "used": disk_usage.used, "free": disk_usage.free},
        "cuda_memory_at_capture": memory,
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }


def save_environment_report() -> Path:
    report_dir = EXPERIMENT_ROOT / "environment"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return write_json_exclusive(report_dir / f"phase3-environment-{stamp}-{uuid.uuid4().hex[:6]}.json", environment_snapshot())
