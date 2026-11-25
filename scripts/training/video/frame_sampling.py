from __future__ import annotations

import cv2
import numpy as np

def uniform_frame_indices(frame_count:int, max_frames:int=12)->list[int]:
    if frame_count<0 or max_frames<1: raise ValueError('frame_count must be >=0 and max_frames >=1')
    if frame_count==0:return []
    k=min(frame_count,max_frames)
    return np.unique(np.rint(np.linspace(0,frame_count-1,k)).astype(int)).tolist()

def sample_video(path: str, max_frames: int = 12, *, max_dimension: int | None = None) -> list[dict]:
    """Read uniformly sampled frames, optionally shrinking them before retention."""
    if max_dimension is not None and max_dimension < 1:
        raise ValueError("max_dimension must be >=1 when provided")
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    out = []
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
        for index in uniform_frame_indices(frame_count, max_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok and frame is not None and max_dimension is not None:
                height, width = frame.shape[:2]
                longest_side = max(height, width)
                if longest_side > max_dimension:
                    scale = max_dimension / longest_side
                    target_size = (max(1, round(width * scale)), max(1, round(height * scale)))
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ok and frame is not None else None
            out.append({
                "frame_index": index,
                "timestamp_seconds": index / fps if fps > 0 else None,
                "frame": rgb,
                "decode_ok": bool(ok and frame is not None),
            })
    finally: cap.release()
    return out
