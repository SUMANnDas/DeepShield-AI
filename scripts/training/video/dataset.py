from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .face_detection import HaarFaceDetector
from .preprocess_video import preprocess_video

LABEL_TO_INDEX = {
    "REAL": 0,
    "AUTHENTIC": 0,
    "FAKE": 1,
    "MANIPULATED": 1,
    "DEEPFAKE": 1,
    "SYNTHETIC": 1,
}


def _label_index(record: dict) -> int:
    value = record.get("label")
    if isinstance(value, int) and not isinstance(value, bool) and value in (0, 1):
        return value
    normalized = str(value or record.get("label_name") or "").strip().upper()
    try:
        return LABEL_TO_INDEX[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported video label {value!r} for sample {record.get('sample_id')!r}") from exc

class VideoSequenceDataset(Dataset):
    def __init__(self,manifest_path,root,max_frames=12,size=224,detector=None,max_frame_pixels=16_000_000):
        if max_frames < 1 or size < 1 or max_frame_pixels < 1:
            raise ValueError("max_frames, size, and max_frame_pixels must be positive")
        self.records = []
        with Path(manifest_path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Manifest line {line_number} must contain an object")
                if not record.get("relative_path"):
                    raise ValueError(f"Manifest line {line_number} has no relative_path")
                record["_label_index"] = _label_index(record)
                self.records.append(record)
        self.root = Path(root).resolve()
        self.max_frames = max_frames
        self.size = size
        self.detector = detector or HaarFaceDetector()
        self.max_frame_pixels = max_frame_pixels

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        record = self.records[index]
        path = (self.root / record["relative_path"]).resolve()
        if self.root not in path.parents:
            raise ValueError("manifest path escapes dataset root")
        if not path.is_file():
            raise FileNotFoundError(f"Video sample is missing: {record['relative_path']}")
        prepared = preprocess_video(
            path,
            detector=self.detector,
            max_frames=self.max_frames,
            max_crops=self.max_frames,
            size=self.size,
            max_frame_pixels=self.max_frame_pixels,
        )
        if prepared.get("error"):
            raise ValueError(f"Video preprocessing failed: {prepared['error']}")

        frames = torch.zeros(self.max_frames, 3, self.size, self.size, dtype=torch.float32)
        mask = torch.zeros(self.max_frames, dtype=torch.bool)
        for frame_index, item in enumerate(prepared["frames"][: self.max_frames]):
            crop = np.ascontiguousarray(item["crop"].transpose(2, 0, 1))
            frames[frame_index] = torch.from_numpy(crop).float() / 255.0
            mask[frame_index] = True

        video_id = str(record.get("video_id") or record.get("sample_id") or record["relative_path"])
        source_video_id = str(
            record.get("source_video_id")
            or record.get("original_video_id")
            or record.get("video_id")
            or record.get("sample_id")
            or video_id
        )
        return {
            "frames": frames,
            "mask": mask,
            "label": torch.tensor(record["_label_index"], dtype=torch.long),
            "video_id": video_id,
            "source_video_id": source_video_id,
        }

def collate_sequences(batch):
    return {
        "frames": torch.stack([item["frames"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
        "video_id": [item["video_id"] for item in batch],
        "source_video_id": [item["source_video_id"] for item in batch],
    }
