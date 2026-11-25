from __future__ import annotations

from pathlib import Path

import cv2

from .utils import sha256_file


def inspect_video(path: str | Path, *, max_frame_pixels: int | None = None) -> dict:
    """Read video metadata and verify a frame can be decoded.

    Callers handling untrusted uploads can set ``max_frame_pixels`` so an
    extreme frame size is rejected before OpenCV allocates a decoded frame.
    """
    video_path = Path(path)
    result = {
        "file_path": str(video_path),
        "file_hash": None,
        "duration_seconds": None,
        "fps": None,
        "width": None,
        "height": None,
        "frame_count": None,
        "codec": None,
        "audio_present": None,
        "decode_status": "failed",
        "error": None,
    }
    if not video_path.is_file() or video_path.stat().st_size == 0:
        result["error"] = "missing_or_empty_file"
        return result
    result["file_hash"] = sha256_file(video_path)
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    try:
        if not capture.isOpened():
            result["error"] = "decoder_open_failed"
            return result

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        result.update(
            fps=float(fps) if fps > 0 else None,
            frame_count=frame_count or None,
            width=width or None,
            height=height or None,
            duration_seconds=(frame_count / fps if fps > 0 and frame_count > 0 else None),
            decode_status="metadata_only",
        )

        if max_frame_pixels is not None and width > 0 and height > 0 and width * height > max_frame_pixels:
            result["error"] = "resolution_exceeds_processing_limit"
            return result

        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        result["codec"] = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00") if fourcc else None
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = capture.read()
        result["decode_status"] = "ok" if ok and frame is not None else "failed"
        if result["decode_status"] == "failed":
            result["error"] = "first_frame_decode_failed"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        capture.release()
    return result
