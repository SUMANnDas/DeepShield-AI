"""FaceForensics++ c23 preflight, authorized download wrapper, and leak-free split preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.training.video.leakage_check import check_leakage
from scripts.training.video.storage_budget import GB, VideoStorageBudget
from scripts.training.video.video_metadata import inspect_video

ALLOWED_SUBSETS = ("original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")
DEFAULT_SUBSETS = ("original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
EXCLUDED_SUBSETS = {"DeepFakeDetection", "DeepFakeDetection_original", "actors", "all"}
DEFAULT_ESTIMATED_C23_BYTES = 10 * GB


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_subsets(subsets: list[str] | tuple[str, ...]) -> list[str]:
    cleaned: list[str] = []
    for raw in subsets:
        name = raw.strip()
        if not name:
            continue
        if name in EXCLUDED_SUBSETS:
            raise ValueError(f"Subset {name!r} is excluded by the C: storage-budget policy")
        if name not in ALLOWED_SUBSETS:
            raise ValueError(f"Unsupported FaceForensics++ subset {name!r}; allowed: {ALLOWED_SUBSETS}")
        if name not in cleaned:
            cleaned.append(name)
    if not cleaned:
        raise ValueError("At least one allowed c23 subset must be selected")
    return cleaned


def run_preflight(
    *,
    data_root: Path | None = None,
    budget: VideoStorageBudget | None = None,
    subsets: list[str] | tuple[str, ...] = DEFAULT_SUBSETS,
    num_videos: int | None = 100,
    estimated_bytes: int | None = None,
    download_script: Path | None = None,
) -> dict[str, Any]:
    """Verify destination, subset policy, C: free reserve, and 90/100 GB video budget."""
    active_budget = budget or (VideoStorageBudget(root=data_root) if data_root else VideoStorageBudget())
    selected_subsets = validate_subsets(subsets)
    if num_videos is not None and num_videos < 1:
        raise ValueError("num_videos must be positive when specified")

    if estimated_bytes is not None:
        planned_bytes = int(estimated_bytes)
    elif num_videos is None:
        planned_bytes = DEFAULT_ESTIMATED_C23_BYTES
    else:
        planned_bytes = max(1, round(DEFAULT_ESTIMATED_C23_BYTES * min(num_videos, 1000) / 1000))

    report = active_budget.report(planned_bytes)
    raw_destination = active_budget.validate_location(active_budget.root / "raw" / "FaceForensics++-c23")
    script_info = None
    if download_script is not None:
        script_path = Path(download_script).expanduser().resolve()
        if not script_path.is_file():
            raise FileNotFoundError(f"FaceForensics++ download script not found: {script_path}")
        script_info = {
            "path": str(script_path),
            "size_bytes": script_path.stat().st_size,
            "sha256": _sha256_file(script_path),
        }

    within_target = report.after_write_bytes <= active_budget.target_bytes
    within_hard_limit = report.within_budget
    return {
        "video_data_root": str(active_budget.root),
        "raw_destination": str(raw_destination),
        "destination_exists": raw_destination.exists(),
        "subsets": selected_subsets,
        "compression": "c23",
        "file_type": "videos",
        "num_videos": num_videos,
        "current_used_bytes": report.current_bytes,
        "planned_bytes": planned_bytes,
        "projected_total_bytes": report.after_write_bytes,
        "target_limit_bytes": active_budget.target_bytes,
        "hard_limit_bytes": active_budget.hard_limit_bytes,
        "volume_free_bytes": report.free_bytes,
        "minimum_free_bytes": active_budget.minimum_free_bytes,
        "within_target_90gb": within_target,
        "within_hard_cap_100gb": within_hard_limit,
        "size_source": "user_supplied_bytes" if estimated_bytes is not None else "official_readme_c23_estimate",
        "download_script": script_info,
        "ready_for_approved_download": bool(within_hard_limit and script_info is not None),
    }


def download_ffpp_c23(
    download_script: Path,
    *,
    data_root: Path | None = None,
    budget: VideoStorageBudget | None = None,
    subsets: list[str] | tuple[str, ...] = DEFAULT_SUBSETS,
    num_videos: int | None = 100,
    estimated_bytes: int | None = None,
    approved: bool = False,
    server: str | None = None,
) -> dict[str, Any]:
    """Execute the user-supplied official FaceForensics++ download script under budget checks."""
    if not approved:
        raise PermissionError("Separate explicit download approval (--approved) is required before downloading FaceForensics++")
    preflight = run_preflight(
        data_root=data_root,
        budget=budget,
        subsets=subsets,
        num_videos=num_videos,
        estimated_bytes=estimated_bytes,
        download_script=download_script,
    )
    if not preflight["within_hard_cap_100gb"]:
        raise RuntimeError("Preflight exceeds the 100 GB hard limit or 10 GB free-space reserve")
    if not preflight["within_target_90gb"]:
        raise RuntimeError("Preflight exceeds the 90 GB target; select a smaller --num-videos subset")

    active_budget = budget or (VideoStorageBudget(root=data_root) if data_root else VideoStorageBudget())
    destination = active_budget.validate_location(Path(preflight["raw_destination"]))
    destination.mkdir(parents=True, exist_ok=True)

    commands_run: list[list[str]] = []
    for subset in preflight["subsets"]:
        cmd = [
            sys.executable,
            str(Path(download_script).expanduser().resolve()),
            str(destination),
            "-d",
            subset,
            "-c",
            "c23",
            "-t",
            "videos",
        ]
        if num_videos is not None:
            cmd.extend(["--num_videos", str(num_videos)])
        if server:
            cmd.extend(["--server", server])
        commands_run.append(cmd)
        proc = subprocess.run(cmd, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"Official FaceForensics++ script failed for subset {subset} with code {proc.returncode}")
        # Re-check hard storage budget after each subset completes.
        active_budget.require_capacity(0, purpose=f"verifying storage after downloading {subset}")

    post_report = active_budget.report()
    record_path = active_budget.validate_destination(destination / "acquisition-record.json")
    record = {
        "dataset": "FaceForensics++",
        "compression": "c23",
        "acquired_utc": datetime.now(timezone.utc).isoformat(),
        "subsets": preflight["subsets"],
        "num_videos": num_videos,
        "download_script": preflight["download_script"],
        "actual_root_bytes": post_report.current_bytes,
        "volume_free_bytes": post_report.free_bytes,
        "commands_executed": commands_run,
    }
    payload = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    active_budget.require_capacity(len(payload), purpose="writing FF++ acquisition record")
    with record_path.open("xb") as handle:
        handle.write(payload)
    return record


def _parse_ffpp_video_ids(stem: str, manipulation_method: str) -> tuple[str, tuple[str, ...]]:
    if manipulation_method == "original":
        token = stem.strip()
        return token, (token,)
    match = re.match(r"^([A-Za-z0-9]+)_([A-Za-z0-9]+)$", stem.strip())
    if match:
        target_id, source_id = match.group(1), match.group(2)
        return target_id, tuple(sorted({target_id, source_id}))
    token = stem.strip()
    return token, (token,)


def _discover_ffpp_videos(raw_root: Path) -> list[tuple[Path, str, str]]:
    """Discover c23 videos under original_sequences and manipulated_sequences."""
    discovered: list[tuple[Path, str, str]] = []
    orig_dir = raw_root / "original_sequences" / "youtube" / "c23" / "videos"
    if orig_dir.is_dir():
        for video in sorted(orig_dir.glob("*.mp4")):
            discovered.append((video, "REAL", "original"))
    manip_base = raw_root / "manipulated_sequences"
    if manip_base.is_dir():
        for method_dir in sorted(manip_base.iterdir()):
            if not method_dir.is_dir() or method_dir.name == "DeepFakeDetection":
                continue
            videos_dir = method_dir / "c23" / "videos"
            if videos_dir.is_dir():
                for video in sorted(videos_dir.glob("*.mp4")):
                    discovered.append((video, "MANIPULATED", method_dir.name))
    return discovered


def _load_official_split_map(splits_dir: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split_name, filename in (("train", "train.json"), ("validation", "val.json"), ("test", "test.json")):
        file_path = Path(splits_dir) / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing official FF++ split file: {file_path}")
        pairs = json.loads(file_path.read_text(encoding="utf-8"))
        for pair in pairs:
            for item in pair:
                vid = str(item)
                if vid in mapping and mapping[vid] != split_name:
                    raise ValueError(f"Official FF++ split conflict for video ID {vid}")
                mapping[vid] = split_name
    return mapping


def _connected_component_splits(records: list[dict[str, Any]], ratios=(0.72, 0.14, 0.14), seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    for rec in records:
        ids = rec["involved_ids"]
        for other in ids[1:]:
            union(ids[0], other)
        if rec.get("file_hash"):
            union(f"hash:{rec['file_hash']}", ids[0])

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        group_root = find(rec["involved_ids"][0])
        rec["split_group_id"] = f"ffpp_component:{group_root}"
        groups[group_root].append(rec)

    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    if n < 3:
        raise ValueError("At least 3 disjoint video groups are required to form train/validation/test splits")
    n_val = max(1, round(n * ratios[1]))
    n_test = max(1, round(n * ratios[2]))
    n_train = max(1, n - n_val - n_test)
    train_keys = keys[:n_train]
    val_keys = keys[n_train:n_train + n_val]
    test_keys = keys[n_train + n_val:]
    return {
        "train": [r for k in train_keys for r in groups[k]],
        "validation": [r for k in val_keys for r in groups[k]],
        "test": [r for k in test_keys for r in groups[k]],
    }


def _check_ffpp_leakage(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    base = check_leakage(splits)
    violations = list(base["violations"])
    seen_ids: dict[str, str] = {}
    for split_name, records in splits.items():
        for rec in records:
            for vid in rec.get("involved_ids") or ():
                if vid in seen_ids and seen_ids[vid] != split_name:
                    violations.append({"field": "involved_id", "value": vid, "splits": [seen_ids[vid], split_name]})
                else:
                    seen_ids[vid] = split_name
    return {"passed": not violations, "violations": violations}


def prepare_ffpp_dataset(
    raw_root: Path,
    manifest_dir: Path,
    *,
    data_root: Path | None = None,
    budget: VideoStorageBudget | None = None,
    official_splits_dir: Path | None = None,
    seed: int = 42,
    max_frame_pixels: int = 16_000_000,
) -> dict[str, Any]:
    """Inspect FF++ c23 videos and write leak-free train/validation/test manifests."""
    active_budget = budget or (VideoStorageBudget(root=data_root) if data_root else VideoStorageBudget())
    raw_path = active_budget.validate_location(raw_root)
    out_dir = active_budget.validate_destination(manifest_dir)
    discovered = _discover_ffpp_videos(raw_path)
    if not discovered:
        raise FileNotFoundError(f"No FaceForensics++ c23 MP4 videos found under {raw_path}")

    official_map = _load_official_split_map(official_splits_dir) if official_splits_dir else None
    all_records: list[dict[str, Any]] = []
    decode_errors = 0

    for video_path, label_name, method in discovered:
        rel = video_path.relative_to(raw_path).as_posix()
        source_video_id, involved_ids = _parse_ffpp_video_ids(video_path.stem, method)
        meta = inspect_video(video_path, max_frame_pixels=max_frame_pixels)
        if meta.get("decode_status") != "ok":
            decode_errors += 1
            continue
        all_records.append({
            "sample_id": f"{method}:{video_path.stem}",
            "video_id": f"{method}:{video_path.stem}",
            "source_video_id": source_video_id,
            "involved_ids": list(involved_ids),
            "relative_path": rel,
            "label": 0 if label_name == "REAL" else 1,
            "label_name": label_name,
            "manipulation_method": method,
            "compression": "c23",
            "file_hash": meta["file_hash"],
            "duration_seconds": meta["duration_seconds"],
            "fps": meta["fps"],
            "width": meta["width"],
            "height": meta["height"],
            "frame_count": meta["frame_count"],
            "codec": meta["codec"],
            "decode_status": meta["decode_status"],
        })

    if not all_records:
        raise RuntimeError("No readable FF++ c23 videos passed metadata inspection")

    if official_map is not None:
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
        for rec in all_records:
            assigned = {official_map.get(vid) for vid in rec["involved_ids"]}
            if None in assigned or len(assigned) != 1:
                raise ValueError(f"Video {rec['relative_path']} cannot map cleanly to one official FF++ split: {assigned}")
            split_name = next(iter(assigned))
            rec["split"] = split_name
            rec["split_group_id"] = f"ffpp_official:{split_name}:{rec['source_video_id']}"
            splits[split_name].append(rec)
    else:
        splits = _connected_component_splits(all_records, seed=seed)
        for split_name, rows in splits.items():
            for row in rows:
                row["split"] = split_name

    leakage = _check_ffpp_leakage(splits)
    if not leakage["passed"]:
        raise RuntimeError(f"Cross-split leakage detected in FF++ split assignment: {leakage['violations'][:5]}")

    out_dir.mkdir(parents=True, exist_ok=False)

    def write_jsonl(name: str, rows: list[dict[str, Any]]) -> Path:
        target = active_budget.validate_destination(out_dir / name)
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
        active_budget.require_capacity(len(payload), purpose=f"writing FF++ manifest {name}")
        with target.open("xb") as handle:
            handle.write(payload)
        return target

    write_jsonl("all.jsonl", all_records)
    for split_name in ("train", "validation", "test"):
        write_jsonl(f"{split_name}.jsonl", splits[split_name])

    leakage_path = active_budget.validate_destination(out_dir / "leakage-report.json")
    leakage_bytes = (json.dumps(leakage, indent=2) + "\n").encode("utf-8")
    active_budget.require_capacity(len(leakage_bytes), purpose="writing FF++ leakage report")
    with leakage_path.open("xb") as handle:
        handle.write(leakage_bytes)

    summary = {
        "dataset": "FaceForensics++ c23",
        "raw_root": str(raw_path),
        "manifest_dir": str(out_dir),
        "total_valid_videos": len(all_records),
        "decode_errors_skipped": decode_errors,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "split_class_counts": {
            k: dict(Counter(row["label_name"] for row in v)) for k, v in splits.items()
        },
        "split_method_counts": {
            k: dict(Counter(row["manipulation_method"] for row in v)) for k, v in splits.items()
        },
        "leakage_passed": leakage["passed"],
        "split_strategy": "official_ffpp_json" if official_map is not None else "connected_component_grouped",
    }
    summary_path = active_budget.validate_destination(out_dir / "summary.json")
    summary_bytes = (json.dumps(summary, indent=2) + "\n").encode("utf-8")
    active_budget.require_capacity(len(summary_bytes), purpose="writing FF++ summary report")
    with summary_path.open("xb") as handle:
        handle.write(summary_bytes)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pre = subparsers.add_parser("preflight", help="Inspect storage budget and subset policy before FF++ download.")
    pre.add_argument("--data-root", type=Path)
    pre.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    pre.add_argument("--num-videos", type=int, default=100)
    pre.add_argument("--all-videos", action="store_true", help="Do not cap --num-videos (full c23 subset)")
    pre.add_argument("--estimated-bytes", type=int)
    pre.add_argument("--script", type=Path, help="Path to official download-FaceForensics.py script")

    dl = subparsers.add_parser("download", help="Download c23 videos via user-supplied official script after approval.")
    dl.add_argument("--script", required=True, type=Path)
    dl.add_argument("--data-root", type=Path)
    dl.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    dl.add_argument("--num-videos", type=int, default=100)
    dl.add_argument("--all-videos", action="store_true")
    dl.add_argument("--estimated-bytes", type=int)
    dl.add_argument("--server", help="Optional official FaceForensics mirror server flag (e.g. EU, EU2, CA)")
    dl.add_argument("--approved", action="store_true", help="Confirm explicit user approval to download")

    prep = subparsers.add_parser("prepare", help="Build leak-free manifests and splits from extracted FF++ c23 videos.")
    prep.add_argument("--raw-root", required=True, type=Path)
    prep.add_argument("--manifest-dir", required=True, type=Path)
    prep.add_argument("--data-root", type=Path)
    prep.add_argument("--official-splits-dir", type=Path)
    prep.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if args.command == "preflight":
        print(json.dumps(run_preflight(
            data_root=args.data_root,
            subsets=args.subsets,
            num_videos=None if args.all_videos else args.num_videos,
            estimated_bytes=args.estimated_bytes,
            download_script=args.script,
        ), indent=2))
    elif args.command == "download":
        print(json.dumps(download_ffpp_c23(
            args.script,
            data_root=args.data_root,
            subsets=args.subsets,
            num_videos=None if args.all_videos else args.num_videos,
            estimated_bytes=args.estimated_bytes,
            approved=args.approved,
            server=args.server,
        ), indent=2))
    elif args.command == "prepare":
        print(json.dumps(prepare_ffpp_dataset(
            args.raw_root,
            args.manifest_dir,
            data_root=args.data_root,
            official_splits_dir=args.official_splits_dir,
            seed=args.seed,
        ), indent=2))


if __name__ == "__main__":
    main()
