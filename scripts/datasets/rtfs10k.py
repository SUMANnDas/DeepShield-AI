"""Open Kaggle API acquisition and double-disjoint split preparation for RealTimeFaceSwap-10k (CC BY-SA 4.0)."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import json
import random
import struct
import sys
import threading
import time
import zlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote
import zipfile

import httpx

from scripts.training.video.leakage_check import check_leakage
from scripts.training.video.storage_budget import GB, VideoStorageBudget
from scripts.training.video.video_metadata import inspect_video

DATASET_SLUG = "stplusplus/rtfs-10k"
DATASET_NAME = "RealTimeFaceSwap-10k"
DATASET_URL = f"https://www.kaggle.com/datasets/{DATASET_SLUG}"
METADATA_API_URL = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET_SLUG}"
DOWNLOAD_API_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET_SLUG}"
REQUIRED_LICENSE = "CC BY-SA 4.0"


class HttpRangeFile(io.RawIOBase):
    """Read-only seekable file wrapper over an HTTP Range-capable URL."""

    def __init__(self, url: str, size: int, timeout: float = 60.0) -> None:
        self.url = url
        self.size = int(size)
        self.pos = 0
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        self.pos = max(0, min(self.size, self.pos))
        return self.pos

    def readinto(self, b: bytearray | memoryview) -> int:
        if self.pos >= self.size:
            return 0
        count = len(b)
        if count == 0:
            return 0
        end = min(self.size - 1, self.pos + count - 1)
        resp = self._client.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        resp.raise_for_status()
        data = resp.content
        n = min(len(data), count)
        b[:n] = data[:n]
        self.pos += n
        return n

    def close(self) -> None:
        self._client.close()
        super().close()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_kaggle_snapshot_and_url() -> tuple[dict[str, Any], str, int]:
    """Fetch official Kaggle metadata snapshot and GCS signed bundle URL + size."""
    meta_resp = httpx.get(METADATA_API_URL, timeout=30.0, follow_redirects=True)
    meta_resp.raise_for_status()
    meta = meta_resp.json()
    dl_resp = httpx.get(DOWNLOAD_API_URL, follow_redirects=False, timeout=30.0)
    if dl_resp.status_code not in (301, 302, 303, 307, 308):
        raise RuntimeError(f"Expected redirect from Kaggle download API, got {dl_resp.status_code}")
    signed_url = dl_resp.headers["location"]
    head_resp = httpx.head(signed_url, timeout=30.0, follow_redirects=True)
    head_resp.raise_for_status()
    archive_size = int(head_resp.headers["content-length"])
    snapshot = {
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "record_url": DATASET_URL,
        "metadata_api_url": METADATA_API_URL,
        "title": meta.get("title"),
        "creator_name": meta.get("creatorName"),
        "owner_name": meta.get("ownerName"),
        "license_name": meta.get("licenseName"),
        "last_updated_utc": meta.get("lastUpdated"),
        "listed_total_bytes": meta.get("totalBytes"),
        "archive_zip_bytes": archive_size,
    }
    return snapshot, signed_url, archive_size


def _extract_zip_entry_from_range(
    client: httpx.Client,
    url: str,
    info: zipfile.ZipInfo,
    *,
    retries: int = 6,
) -> bytes:
    """Read a single entry from a remote ZIP archive using its header_offset and verify CRC-32."""
    last_error: BaseException | None = None
    for attempt in range(retries):
        try:
            header_offset = info.header_offset
            probe_end = header_offset + 512
            probe = client.get(url, headers={"Range": f"bytes={header_offset}-{probe_end}"})
            probe.raise_for_status()
            hdr = probe.content
            if hdr[:4] != b"PK\x03\x04":
                raise ValueError(f"Invalid ZIP local file header for {info.filename}")
            fn_len, ex_len = struct.unpack("<HH", hdr[26:30])
            data_start = header_offset + 30 + fn_len + ex_len
            comp_size = info.compress_size
            if comp_size == 0:
                raw = b""
            else:
                data_end = data_start + comp_size - 1
                already_offset = data_start - header_offset
                if already_offset + comp_size <= len(hdr):
                    comp_bytes = hdr[already_offset:already_offset + comp_size]
                else:
                    resp = client.get(url, headers={"Range": f"bytes={data_start}-{data_end}"})
                    resp.raise_for_status()
                    comp_bytes = resp.content
                if len(comp_bytes) != comp_size:
                    raise IOError(f"Short range read for {info.filename}: got {len(comp_bytes)}, expected {comp_size}")
                if info.compress_type == zipfile.ZIP_STORED:
                    raw = comp_bytes
                elif info.compress_type == zipfile.ZIP_DEFLATED:
                    raw = zlib.decompress(comp_bytes, -zlib.MAX_WBITS)
                else:
                    raise ValueError(f"Unsupported ZIP compression method {info.compress_type} for {info.filename}")
            if len(raw) != info.file_size:
                raise IOError(f"Decompressed size mismatch for {info.filename}: got {len(raw)}, expected {info.file_size}")
            if (zlib.crc32(raw) & 0xFFFFFFFF) != info.CRC:
                raise IOError(f"CRC-32 mismatch for {info.filename}")
            return raw
        except (httpx.HTTPError, IOError, zlib.error) as exc:
            last_error = exc
            if attempt + 1 >= retries:
                break
            time.sleep(min(20.0, 1.5 * (2 ** attempt)))
    assert last_error is not None
    raise last_error


def _parse_csv_bytes(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def select_double_disjoint_subset(
    *,
    videos_rows: list[dict[str, str]],
    inswapper_rows: list[dict[str, str]],
    uniface_rows: list[dict[str, str]],
    entry_sizes: dict[str, int],
    target_real_per_split: tuple[int, int, int] = (420, 90, 90),
    target_fake_per_split: tuple[int, int, int] = (420, 90, 90),
    max_clip_bytes: int = 12_000_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Select a balanced, double-disjoint subset (disjoint YouTube IDs AND source photos across splits)."""
    rng = random.Random(seed)
    video_by_filename: dict[str, dict[str, str]] = {}
    yt_to_clips: dict[str, list[dict[str, Any]]] = {}

    for row in videos_rows:
        fname = (row.get("filename") or "").strip()
        yt_id = (row.get("id") or "").strip()
        rel = f"original_videos/{fname}"
        size = entry_sizes.get(rel)
        if not fname or not yt_id or size is None or size <= 0 or size > max_clip_bytes:
            continue
        item = {
            "sample_id": f"orig:{fname}",
            "video_id": f"orig:{fname}",
            "source_video_id": yt_id,
            "source_photo_id": None,
            "relative_path": rel,
            "filename": fname,
            "label": 0,
            "label_name": "REAL",
            "manipulation_method": "original",
            "additional_features": "",
            "tags": row.get("tags", ""),
            "original_link": row.get("original_link", ""),
            "file_size_bytes": size,
        }
        video_by_filename[fname] = item
        yt_to_clips.setdefault(yt_id, []).append(item)

    yt_ids = sorted(yt_to_clips)
    rng.shuffle(yt_ids)
    n_yt = len(yt_ids)
    if n_yt < 10:
        raise ValueError("Not enough valid YouTube source videos to create splits")
    n_train_yt = round(n_yt * 0.70)
    n_val_yt = round(n_yt * 0.15)
    yt_split: dict[str, str] = {}
    for idx, yid in enumerate(yt_ids):
        if idx < n_train_yt:
            yt_split[yid] = "train"
        elif idx < n_train_yt + n_val_yt:
            yt_split[yid] = "validation"
        else:
            yt_split[yid] = "test"

    all_photos = sorted({
        (r.get("photo") or "").strip()
        for r in (*inswapper_rows, *uniface_rows)
        if (r.get("photo") or "").strip()
    })
    rng.shuffle(all_photos)
    n_ph = len(all_photos)
    n_train_ph = round(n_ph * 0.70)
    n_val_ph = round(n_ph * 0.15)
    photo_split: dict[str, str] = {}
    for idx, ph in enumerate(all_photos):
        if idx < n_train_ph:
            photo_split[ph] = "train"
        elif idx < n_train_ph + n_val_ph:
            photo_split[ph] = "validation"
        else:
            photo_split[ph] = "test"

    selected_real: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for yid in yt_ids:
        sp = yt_split[yid]
        clips = list(yt_to_clips[yid])
        rng.shuffle(clips)
        # Prefer at most 2 clips per YouTube ID so speaker/scene diversity stays high.
        selected_real[sp].extend(clips[:2])

    split_names = ("train", "validation", "test")
    final_real: dict[str, list[dict[str, Any]]] = {}
    for sp, cap in zip(split_names, target_real_per_split):
        pool = selected_real[sp]
        rng.shuffle(pool)
        final_real[sp] = [{**item, "split": sp, "split_group_id": f"yt:{item['source_video_id']}"} for item in pool[:cap]]

    # Build double-disjoint manipulated candidates (both target YouTube ID AND source photo belong to the same split).
    fake_candidates: dict[str, dict[str, list[dict[str, Any]]]] = {
        sp: {"inswapper": [], "uniface": []} for sp in split_names
    }
    for method, rows in (("inswapper", inswapper_rows), ("uniface", uniface_rows)):
        for row in rows:
            if str(row.get("success", "")).strip().lower() != "true":
                continue
            target_fname = (row.get("video") or "").strip()
            photo = (row.get("photo") or "").strip()
            gen_fname = (row.get("filename") or "").strip()
            rel = f"{method}/{gen_fname}"
            size = entry_sizes.get(rel)
            orig = video_by_filename.get(target_fname)
            if not orig or not photo or not gen_fname or size is None or size <= 0 or size > max_clip_bytes:
                continue
            sp_v = yt_split.get(orig["source_video_id"])
            sp_p = photo_split.get(photo)
            if sp_v is None or sp_v != sp_p:
                continue
            fake_candidates[sp_v][method].append({
                "sample_id": f"{method}:{gen_fname}",
                "video_id": f"{method}:{gen_fname}",
                "source_video_id": orig["source_video_id"],
                "source_photo_id": photo,
                "relative_path": rel,
                "filename": gen_fname,
                "label": 1,
                "label_name": "MANIPULATED",
                "manipulation_method": method,
                "additional_features": (row.get("additional_features") or "").strip(),
                "tags": row.get("video_tags", ""),
                "file_size_bytes": size,
                "split": sp_v,
                "split_group_id": f"yt:{orig['source_video_id']}+ph:{photo}",
            })

    final_fake: dict[str, list[dict[str, Any]]] = {}
    for sp, cap in zip(split_names, target_fake_per_split):
        per_method = cap // 2
        chosen: list[dict[str, Any]] = []
        for method in ("inswapper", "uniface"):
            pool = fake_candidates[sp][method]
            rng.shuffle(pool)
            chosen.extend(pool[:per_method])
        final_fake[sp] = chosen

    splits = {sp: final_real[sp] + final_fake[sp] for sp in split_names}
    for sp in split_names:
        rng.shuffle(splits[sp])

    all_selected = [row for sp in split_names for row in splits[sp]]
    total_bytes = sum(int(row["file_size_bytes"]) for row in all_selected)
    return {
        "splits": splits,
        "selected_records": all_selected,
        "total_videos": len(all_selected),
        "total_bytes": total_bytes,
        "split_counts": {sp: len(rows) for sp, rows in splits.items()},
        "split_class_counts": {sp: dict(Counter(r["label_name"] for r in rows)) for sp, rows in splits.items()},
        "split_method_counts": {sp: dict(Counter(r["manipulation_method"] for r in rows)) for sp, rows in splits.items()},
    }


def run_preflight(
    *,
    data_root: Path | None = None,
    budget: VideoStorageBudget | None = None,
    target_real_per_split: tuple[int, int, int] = (420, 90, 90),
    target_fake_per_split: tuple[int, int, int] = (420, 90, 90),
    max_clip_bytes: int = 12_000_000,
    seed: int = 42,
) -> dict[str, Any]:
    active_budget = budget or (VideoStorageBudget(root=data_root) if data_root else VideoStorageBudget())
    snapshot, signed_url, archive_size = fetch_kaggle_snapshot_and_url()
    with HttpRangeFile(signed_url, archive_size) as raw_reader:
        with zipfile.ZipFile(io.BufferedReader(raw_reader, buffer_size=2 * 1024 * 1024)) as zf:
            infos = {info.filename: info for info in zf.infolist()}
            entry_sizes = {name: info.file_size for name, info in infos.items()}
            with httpx.Client(timeout=60.0) as client:
                videos_rows = _parse_csv_bytes(_extract_zip_entry_from_range(client, signed_url, infos["videos.csv"]))
                inswapper_rows = _parse_csv_bytes(_extract_zip_entry_from_range(client, signed_url, infos["inswapper.csv"]))
                uniface_rows = _parse_csv_bytes(_extract_zip_entry_from_range(client, signed_url, infos["uniface.csv"]))

    plan = select_double_disjoint_subset(
        videos_rows=videos_rows,
        inswapper_rows=inswapper_rows,
        uniface_rows=uniface_rows,
        entry_sizes=entry_sizes,
        target_real_per_split=target_real_per_split,
        target_fake_per_split=target_fake_per_split,
        max_clip_bytes=max_clip_bytes,
        seed=seed,
    )
    planned_bytes = int(plan["total_bytes"]) + 20_000_000  # metadata + manifests overhead
    report = active_budget.report(planned_bytes)
    raw_dest = active_budget.validate_location(active_budget.root / "raw" / DATASET_NAME)
    return {
        "dataset": DATASET_NAME,
        "kaggle_slug": DATASET_SLUG,
        "license": snapshot["license_name"],
        "raw_destination": str(raw_dest),
        "total_archive_entries": len(entry_sizes),
        "selected_videos": plan["total_videos"],
        "selected_bytes": plan["total_bytes"],
        "split_counts": plan["split_counts"],
        "split_class_counts": plan["split_class_counts"],
        "split_method_counts": plan["split_method_counts"],
        "current_used_bytes": report.current_bytes,
        "projected_total_bytes": report.after_write_bytes,
        "target_limit_bytes": active_budget.target_bytes,
        "hard_limit_bytes": active_budget.hard_limit_bytes,
        "volume_free_bytes": report.free_bytes,
        "within_target_90gb": report.after_write_bytes <= active_budget.target_bytes,
        "within_hard_cap_100gb": report.within_budget,
        "snapshot": snapshot,
    }


def acquire_and_prepare(
    *,
    data_root: Path | None = None,
    budget: VideoStorageBudget | None = None,
    manifest_name: str = "rtfs10k-v1",
    target_real_per_split: tuple[int, int, int] = (420, 90, 90),
    target_fake_per_split: tuple[int, int, int] = (420, 90, 90),
    max_clip_bytes: int = 12_000_000,
    seed: int = 42,
    workers: int = 6,
    resume: bool = False,
) -> dict[str, Any]:
    """Download the verified CC BY-SA 4.0 subset via Kaggle GCS Range requests and write leak-free manifests."""
    active_budget = budget or (VideoStorageBudget(root=data_root) if data_root else VideoStorageBudget())
    if resume:
        raw_dest = active_budget.validate_location(active_budget.root / "raw" / DATASET_NAME)
    else:
        raw_dest = active_budget.validate_destination(active_budget.root / "raw" / DATASET_NAME)
    manifest_dir = active_budget.validate_destination(active_budget.root / "manifests" / manifest_name)

    snapshot, signed_url, archive_size = fetch_kaggle_snapshot_and_url()
    with HttpRangeFile(signed_url, archive_size) as raw_reader:
        with zipfile.ZipFile(io.BufferedReader(raw_reader, buffer_size=2 * 1024 * 1024)) as zf:
            infos = {info.filename: info for info in zf.infolist()}

    entry_sizes = {name: info.file_size for name, info in infos.items()}
    if resume and all((raw_dest / name).is_file() for name in ("videos.csv", "photos.csv", "inswapper.csv", "uniface.csv")):
        csv_payloads = {
            name: (raw_dest / name).read_bytes()
            for name in ("videos.csv", "photos.csv", "inswapper.csv", "uniface.csv")
        }
    else:
        with httpx.Client(timeout=60.0) as client:
            csv_payloads = {
                name: _extract_zip_entry_from_range(client, signed_url, infos[name])
                for name in ("videos.csv", "photos.csv", "inswapper.csv", "uniface.csv")
            }

    plan = select_double_disjoint_subset(
        videos_rows=_parse_csv_bytes(csv_payloads["videos.csv"]),
        inswapper_rows=_parse_csv_bytes(csv_payloads["inswapper.csv"]),
        uniface_rows=_parse_csv_bytes(csv_payloads["uniface.csv"]),
        entry_sizes=entry_sizes,
        target_real_per_split=target_real_per_split,
        target_fake_per_split=target_fake_per_split,
        max_clip_bytes=max_clip_bytes,
        seed=seed,
    )
    active_budget.require_capacity(int(plan["total_bytes"]) + 50_000_000, purpose="acquiring RealTimeFaceSwap-10k subset")

    raw_dest.mkdir(parents=True, exist_ok=resume)
    for csv_name, csv_bytes in csv_payloads.items():
        target = active_budget.validate_location(raw_dest / csv_name)
        if not target.exists():
            with target.open("xb") as handle:
                handle.write(csv_bytes)

    snap_path = active_budget.validate_location(raw_dest / "kaggle-record-snapshot.json")
    if not snap_path.exists():
        with snap_path.open("xb") as handle:
            handle.write((json.dumps(snapshot, indent=2) + "\n").encode("utf-8"))

    # Download selected MP4 clips in parallel with CRC-32 verification and budget tracking.
    lock = threading.Lock()
    local_state = threading.local()

    def _client() -> httpx.Client:
        cli = getattr(local_state, "client", None)
        if cli is None:
            cli = httpx.Client(timeout=90.0)
            local_state.client = cli
        return cli

    def _download_one(rec: dict[str, Any]) -> dict[str, Any]:
        rel = rec["relative_path"]
        info = infos[rel]
        out_path = active_budget.validate_location(raw_dest / rel)
        if resume and out_path.is_file() and out_path.stat().st_size == info.file_size:
            meta = inspect_video(out_path, max_frame_pixels=16_000_000)
            return {**rec, **meta, "relative_path": rel, "file_hash": meta.get("file_hash") or _sha256_file(out_path)}
        raw_bytes = _extract_zip_entry_from_range(_client(), signed_url, info)
        with lock:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "wb" if (resume and out_path.exists()) else "xb"
            with out_path.open(mode) as handle:
                handle.write(raw_bytes)
        meta = inspect_video(out_path, max_frame_pixels=16_000_000)
        return {**rec, **meta, "relative_path": rel, "file_hash": meta.get("file_hash") or _sha256_bytes(raw_bytes)}

    records_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    decode_errors = 0
    total_planned = len(plan["selected_records"])
    completed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_one, rec): rec for rec in plan["selected_records"]}
        for future in concurrent.futures.as_completed(futures):
            enriched = future.result()
            completed_count += 1
            if completed_count % 100 == 0 or completed_count == total_planned:
                print(f"acquired {completed_count}/{total_planned} videos", file=sys.stderr, flush=True)
            if enriched.get("decode_status") != "ok":
                decode_errors += 1
                continue
            enriched.pop("file_path", None)
            records_by_split[enriched["split"]].append(enriched)

    # Sort deterministically after parallel download.
    for sp in records_by_split:
        records_by_split[sp].sort(key=lambda item: item["sample_id"])

    # Verify zero cross-split leakage across video_id, source_video_id (YouTube ID), source_photo_id, and file_hash.
    base_leak = check_leakage(records_by_split)
    violations = list(base_leak["violations"])
    seen_photos: dict[str, str] = {}
    for sp, rows in records_by_split.items():
        for row in rows:
            ph = row.get("source_photo_id")
            if not ph:
                continue
            if ph in seen_photos and seen_photos[ph] != sp:
                violations.append({"field": "source_photo_id", "value": ph, "splits": [seen_photos[ph], sp]})
            else:
                seen_photos[ph] = sp
    if violations:
        raise RuntimeError(f"Cross-split leakage detected in RTFS-10k: {violations[:5]}")

    manifest_dir.mkdir(parents=True, exist_ok=False)
    all_rows = [row for sp in ("train", "validation", "test") for row in records_by_split[sp]]

    def _write_jsonl(name: str, rows: list[dict[str, Any]]) -> Path:
        target = active_budget.validate_destination(manifest_dir / name)
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode("utf-8")
        active_budget.require_capacity(len(payload), purpose=f"writing {name}")
        with target.open("xb") as handle:
            handle.write(payload)
        return target

    _write_jsonl("all.jsonl", all_rows)
    for sp in ("train", "validation", "test"):
        _write_jsonl(f"{sp}.jsonl", records_by_split[sp])

    leakage_report = {"passed": True, "violations": []}
    leak_path = active_budget.validate_destination(manifest_dir / "leakage-report.json")
    with leak_path.open("xb") as handle:
        handle.write((json.dumps(leakage_report, indent=2) + "\n").encode("utf-8"))

    post = active_budget.report()
    summary = {
        "dataset": DATASET_NAME,
        "kaggle_slug": DATASET_SLUG,
        "license": REQUIRED_LICENSE,
        "acquired_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_dest),
        "manifest_dir": str(manifest_dir),
        "total_valid_videos": len(all_rows),
        "decode_errors_skipped": decode_errors,
        "split_counts": {sp: len(rows) for sp, rows in records_by_split.items()},
        "split_class_counts": {sp: dict(Counter(r["label_name"] for r in rows)) for sp, rows in records_by_split.items()},
        "split_method_counts": {sp: dict(Counter(r["manipulation_method"] for r in rows)) for sp, rows in records_by_split.items()},
        "leakage_passed": True,
        "split_strategy": "double_disjoint_youtube_id_and_source_photo",
        "actual_video_root_bytes": post.current_bytes,
        "volume_free_bytes": post.free_bytes,
    }
    sum_path = active_budget.validate_destination(manifest_dir / "summary.json")
    with sum_path.open("xb") as handle:
        handle.write((json.dumps(summary, indent=2) + "\n").encode("utf-8"))
    acq_path = active_budget.validate_destination(raw_dest / "acquisition-record.json")
    with acq_path.open("xb") as handle:
        handle.write((json.dumps(summary, indent=2) + "\n").encode("utf-8"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "acquire"):
        p = sub.add_parser(name)
        p.add_argument("--data-root", type=Path)
        p.add_argument("--train-real", type=int, default=420)
        p.add_argument("--val-real", type=int, default=90)
        p.add_argument("--test-real", type=int, default=90)
        p.add_argument("--train-fake", type=int, default=420)
        p.add_argument("--val-fake", type=int, default=90)
        p.add_argument("--test-fake", type=int, default=90)
        p.add_argument("--max-clip-bytes", type=int, default=12_000_000)
        p.add_argument("--seed", type=int, default=42)
        if name == "acquire":
            p.add_argument("--manifest-name", default="rtfs10k-v1")
            p.add_argument("--workers", type=int, default=6)
            p.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    real_t = (args.train_real, args.val_real, args.test_real)
    fake_t = (args.train_fake, args.val_fake, args.test_fake)
    if args.command == "preflight":
        print(json.dumps(run_preflight(
            data_root=args.data_root,
            target_real_per_split=real_t,
            target_fake_per_split=fake_t,
            max_clip_bytes=args.max_clip_bytes,
            seed=args.seed,
        ), indent=2))
    else:
        print(json.dumps(acquire_and_prepare(
            data_root=args.data_root,
            manifest_name=args.manifest_name,
            target_real_per_split=real_t,
            target_fake_per_split=fake_t,
            max_clip_bytes=args.max_clip_bytes,
            seed=args.seed,
            workers=args.workers,
            resume=args.resume,
        ), indent=2))


if __name__ == "__main__":
    main()
