from __future__ import annotations

from datetime import datetime, timezone


def _evidence_entry(item: dict) -> dict:
    entry = {
        "frame_index": int(item["frame_index"]),
        "timestamp_seconds": item.get("timestamp_seconds"),
        "probability_manipulated": float(item["score"]),
        "suspicious": bool(item["score"] >= 0.5),
    }
    if item.get("heatmap_jpeg_base64"):
        entry["heatmap_jpeg_base64"] = item["heatmap_jpeg_base64"]
    return entry


def forensic_report(video_id, p_video, frame_scores, metadata, model_name, preprocessing, errors=None):
    return {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "video_id": str(video_id),
        "prediction": "MANIPULATED" if p_video >= 0.5 else "REAL",
        "probability_manipulated": float(p_video),
        "threshold": 0.5,
        "certainty_claim": False,
        "interpretation": "Model-based forensic indicator; not proof of authenticity or manipulation.",
        "frame_evidence": [_evidence_entry(x) for x in frame_scores],
        "metadata": metadata,
        "model": model_name,
        "preprocessing": preprocessing,
        "errors": errors or [],
    }
