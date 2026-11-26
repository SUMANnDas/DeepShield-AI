from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.app.schemas.application import HistoryResponse, ImageAnalysisResponse, VideoAnalysisResponse
from backend.services.analysis_store import AnalysisStore
from backend.services.image_inference import ImageInferenceService, ImageInputError, MAX_IMAGE_BYTES
from backend.services.model_registry import get_model, list_models
from backend.services.system_status import system_status
from backend.services.video_detection.detector import VideoDetectionService
from backend.services.video_model_registry import get_video_model, list_video_models
from scripts.training.video.frame_sampling import sample_video
from scripts.training.video.storage_budget import (
    DEFAULT_VIDEO_DATA_ROOT,
    StorageBudgetExceeded,
    VideoStorageBudget,
)
from scripts.training.video.video_metadata import inspect_video

router = APIRouter(tags=["application"])
logger = logging.getLogger(__name__)
image_service = ImageInferenceService()
video_service = VideoDetectionService()
MAX_VIDEO_BYTES = 200 * 1024 * 1024
MAX_VIDEO_DURATION_SECONDS = 10 * 60
MAX_VIDEO_FRAME_PIXELS = 16_000_000
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _store(request: Request) -> AnalysisStore:
    store = getattr(request.app.state, "analysis_store", None)
    if store is None:
        store = AnalysisStore()
        request.app.state.analysis_store = store
    return store


def _safe_filename(value: str | None) -> str:
    name = Path(value or "upload").name.replace("\x00", "")
    return name[:180] or "upload"


def _video_budget(request: Request) -> VideoStorageBudget:
    configured = getattr(request.app.state, "video_storage_budget", None)
    return configured if configured is not None else VideoStorageBudget()


def _video_root(request: Request) -> Path:
    configured = getattr(request.app.state, "video_storage_budget", None)
    return configured.root if configured is not None else DEFAULT_VIDEO_DATA_ROOT


@router.get("/system")
def get_system_status(request: Request):
    return system_status(video_data_root=_video_root(request))


@router.get("/models")
def models(request: Request):
    return {"items": list_models(), "video_items": list_video_models(_video_root(request))}


@router.get("/performance")
def performance(request: Request):
    video_items = list_video_models(_video_root(request))
    video_dataset = video_items[0].get("dataset") if video_items else "Source-grouped held-out video test split"
    return {
        "dataset": "CIFAKE v3 held-out test benchmark — 12,000 images",
        "scope_warning": "32×32 AI-generated image classification; not a facial or universal deepfake benchmark.",
        "items": list_models(),
        "video_dataset": f"{video_dataset} — source-grouped held-out test split",
        "video_scope_warning": "Facial manipulation detection in video; separate from CIFAKE image detection.",
        "video_items": video_items,
    }


@router.get("/dashboard")
def dashboard(request: Request):
    store = _store(request)
    items = store.list(limit=8)
    video_root = _video_root(request)
    return {
        "counts": store.counts(),
        "recent": items,
        "models": list_models(),
        "video_models": list_video_models(video_root),
        "system": system_status(video_data_root=video_root),
    }


@router.post("/analysis/images", response_model=ImageAnalysisResponse)
async def analyze_image(request: Request, model_id: str = Form(...), file: UploadFile = File(...)):
    model = get_model(model_id)
    if model is None or model["status"] != "available":
        raise HTTPException(status_code=404, detail="The selected model is unavailable.")
    data = await file.read(MAX_IMAGE_BYTES + 1)
    filename = _safe_filename(file.filename)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit.")
    try:
        image, dimensions = await run_in_threadpool(ImageInferenceService.validate_image, data, filename, file.content_type)
        result = await run_in_threadpool(image_service.predict, model_id, image)
    except ImageInputError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("image inference failed model_id=%s", model_id)
        raise HTTPException(status_code=503, detail="The selected model could not complete inference.") from exc
    metadata = {**dimensions, "preprocessing": result.pop("preprocessing"), "retention": "Uploaded media is processed in memory and is not retained."}
    record = _store(request).add(media_type="image", filename=filename, sha256=hashlib.sha256(data).hexdigest(), metadata=metadata, result=result)
    return record


def _video_process(path: str) -> tuple[dict, list[dict]]:
    metadata = inspect_video(path, max_frame_pixels=MAX_VIDEO_FRAME_PIXELS)
    if metadata.get("decode_status") != "ok":
        if metadata.get("error") == "resolution_exceeds_processing_limit":
            raise ValueError("Video resolution exceeds the local inspection limit of 16 megapixels per frame.")
        raise ValueError("The video could not be decoded.")
    duration = metadata.get("duration_seconds")
    if duration is not None and duration > MAX_VIDEO_DURATION_SECONDS:
        raise ValueError("Video exceeds the 10-minute local inspection limit.")
    # Shrink each sampled frame as it is read so the API retains only small
    # preview images instead of six potentially full-resolution frames.
    frames = sample_video(path, max_frames=6, max_dimension=320)
    sampled = []
    for item in frames:
        if not item["decode_ok"] or item["frame"] is None:
            continue
        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(item["frame"], cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 75],
        )
        if encoded_ok:
            sampled.append({
                "frame_index": item["frame_index"],
                "timestamp_seconds": item["timestamp_seconds"],
                "preview_jpeg_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            })
    metadata.pop("file_path", None)
    return metadata, sampled


@router.post("/analysis/videos", response_model=VideoAnalysisResponse)
async def analyze_video(
    request: Request,
    file: UploadFile = File(...),
    model_id: str | None = Form(default=None),
):
    filename = _safe_filename(file.filename)
    if Path(filename).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Use an MP4, AVI, MOV, MKV, or WebM video.")
    available_models = list_video_models(_video_root(request))
    selected_model_id: str | None = None
    if model_id:
        match = get_video_model(model_id, _video_root(request))
        if match is None or match.get("status") != "available":
            raise HTTPException(status_code=404, detail="The selected video model is unavailable.")
        selected_model_id = match["id"]
    elif available_models:
        selected_model_id = available_models[0]["id"]
    return await _process_video_upload(request, file, filename, selected_model_id=selected_model_id)


async def _process_video_upload(
    request: Request,
    file: UploadFile,
    filename: str,
    *,
    selected_model_id: str | None = None,
):
    temp_path = None
    file_size = 0
    digest = hashlib.sha256()
    budget = _video_budget(request)
    detection_payload: dict[str, Any] | None = None
    try:
        try:
            budget.require_capacity(0, purpose="starting video upload")
        except StorageBudgetExceeded as exc:
            raise HTTPException(status_code=507, detail="Video storage budget or free-space reserve is exhausted.") from exc
        temp_dir = budget.validate_location(budget.root / "inference-tmp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        used_bytes = budget.report().current_bytes
        with tempfile.NamedTemporaryFile(
            prefix="deepshield-video-",
            suffix=Path(filename).suffix.lower(),
            delete=False,
            dir=temp_dir,
        ) as handle:
            temp_path = handle.name
            budget.validate_location(temp_path)
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                file_size += len(chunk)
                if file_size > MAX_VIDEO_BYTES:
                    raise HTTPException(status_code=413, detail="Video exceeds the 200 MB upload limit.")
                try:
                    budget.require_capacity(
                        len(chunk),
                        purpose="staging a video upload",
                        current_bytes=used_bytes,
                    )
                except StorageBudgetExceeded as exc:
                    raise HTTPException(
                        status_code=507,
                        detail="Video upload exceeds the configured storage budget or free-space reserve.",
                    ) from exc
                digest.update(chunk)
                handle.write(chunk)
                used_bytes += len(chunk)

        if file_size == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        metadata, sampled = await run_in_threadpool(_video_process, temp_path)
        if selected_model_id is not None:
            detection_payload = await run_in_threadpool(
                video_service.detect,
                temp_path,
                model_id=selected_model_id,
                data_root=budget.root,
                include_heatmaps=True,
            )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("video inspection failed")
        raise HTTPException(status_code=503, detail="The video could not be safely inspected.") from exc
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.exception("failed to remove temporary video upload")
                raise HTTPException(
                    status_code=503,
                    detail="Temporary video cleanup failed; no analysis was saved.",
                ) from exc
        await file.close()

    if detection_payload is None:
        prediction_available = False
        status = "experimental_metadata_only"
        limitation = "Video metadata and frame inspection are available. Validated video deepfake classification is not currently available."
        result: dict[str, Any] = {
            "prediction_available": False,
            "limitation": limitation,
        }
    elif detection_payload.get("status") == "ok":
        report = detection_payload["report"]
        prediction_available = True
        status = "complete"
        limitation = "Model-based facial manipulation indicator only; not forensic certainty."
        probability_manipulated = float(report["probability_manipulated"])
        result = {
            "prediction_available": True,
            "prediction": report["prediction"],
            "probability_manipulated": probability_manipulated,
            "probability_real": float(1.0 - probability_manipulated),
            "confidence_score": float(max(probability_manipulated, 1.0 - probability_manipulated)),
            "confidence_calibrated": False,
            "frame_evidence": report["frame_evidence"],
            "model_id": detection_payload.get("model_id"),
            "model_identifier": detection_payload.get("model_identifier"),
            "device": detection_payload.get("device"),
            "inference_duration_ms": detection_payload.get("inference_duration_ms"),
            "preprocessing": report.get("preprocessing"),
            "limitation": limitation,
        }
    else:
        prediction_available = False
        status = "inconclusive"
        reason = detection_payload.get("reason") or detection_payload.get("error") or "no_reliable_face_track"
        limitation = "No reliable face track was detected in the sampled frames; video prediction is inconclusive."
        result = {
            "prediction_available": False,
            "prediction": "INCONCLUSIVE",
            "reason": reason,
            "model_id": detection_payload.get("model_id"),
            "model_identifier": detection_payload.get("model_identifier"),
            "limitation": limitation,
        }

    record = _store(request).add(
        media_type="video",
        filename=filename,
        sha256=digest.hexdigest(),
        metadata={
            **metadata,
            "sampled_frame_count": len(sampled),
            "retention": "Uploaded media is removed after temporary processing.",
        },
        result=result,
        status=status,
    )
    return {
        "id": record["id"],
        "filename": filename,
        "sha256": record["sha256"],
        "status": record["status"],
        "prediction_available": prediction_available,
        "limitation": limitation,
        "result": result,
        "metadata": record["metadata"],
        "sampled_frames": sampled,
    }


@router.get("/history", response_model=HistoryResponse)
def history(
    request: Request,
    query: str = "",
    media_type: str = Query(default="", pattern="^(|image|video)$"),
    limit: int = Query(default=100, ge=1, le=500),
):
    items = _store(request).list(query=query, media_type=media_type, limit=limit)
    return {"items": items, "total": len(items)}


@router.get("/history/{analysis_id}")
def history_detail(request: Request, analysis_id: str):
    item = _store(request).get(analysis_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return item


@router.get("/reports/{analysis_id}.json")
def report_json(request: Request, analysis_id: str):
    item = _store(request).get(analysis_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return JSONResponse(item, headers={"Content-Disposition": f'attachment; filename="deepshield-{analysis_id}.json"'})


def _redact_base64_for_text(value: Any) -> Any:
    if isinstance(value, list):
        return [_redact_base64_for_text(entry) for entry in value]
    if isinstance(value, dict):
        return {
            key: (
                f"[base64 image omitted · {len(val)} chars]"
                if key.endswith("_base64") and isinstance(val, str)
                else _redact_base64_for_text(val)
            )
            for key, val in value.items()
        }
    return value


@router.get("/reports/{analysis_id}.html", response_class=HTMLResponse)
def report_html(request: Request, analysis_id: str):
    item = _store(request).get(analysis_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    visual_sections: list[str] = []
    explainability = (item.get("result") or {}).get("explainability")
    if isinstance(explainability, dict) and explainability.get("overlay_png_base64"):
        method = html.escape(str(explainability.get("method_label") or explainability.get("method") or "Attribution"))
        target = html.escape(str(explainability.get("target_class") or ""))
        limitation = html.escape(str(explainability.get("limitation") or ""))
        input_img = explainability.get("model_input_png_base64")
        input_block = (
            f'<figure><img src="data:image/png;base64,{html.escape(input_img)}" alt="32x32 model input"/><figcaption>Prepared 32x32 input</figcaption></figure>'
            if isinstance(input_img, str)
            else ""
        )
        overlay_block = (
            f'<figure><img src="data:image/png;base64,{html.escape(str(explainability["overlay_png_base64"]))}" alt="Attribution overlay"/>'
            f"<figcaption>{method} (target {target})</figcaption></figure>"
        )
        visual_sections.append(
            f"<section><h2>Visual attribution</h2><div class='figs'>{input_block}{overlay_block}</div><p class='note'>{limitation}</p></section>"
        )
    frame_evidence = (item.get("result") or {}).get("frame_evidence")
    if isinstance(frame_evidence, list) and any(isinstance(f, dict) and f.get("heatmap_jpeg_base64") for f in frame_evidence):
        cards = []
        for frame in frame_evidence:
            if not isinstance(frame, dict) or not frame.get("heatmap_jpeg_base64"):
                continue
            idx = html.escape(str(frame.get("frame_index")))
            score = float(frame.get("probability_manipulated", 0.0)) * 100.0
            cards.append(
                f'<figure><img src="data:image/jpeg;base64,{html.escape(str(frame["heatmap_jpeg_base64"]))}" alt="Frame {idx}"/>'
                f"<figcaption>Frame {idx} · {score:.1f}% manipulated</figcaption></figure>"
            )
        if cards:
            visual_sections.append(f"<section><h2>Video frame evidence</h2><div class='figs'>{''.join(cards)}</div></section>")
    safe = json.dumps(_redact_base64_for_text(item), indent=2, ensure_ascii=False)
    return HTMLResponse(
        "<!doctype html><meta charset='utf-8'><title>DeepShield Forensic Report</title>"
        "<style>body{font:15px system-ui;background:#0b1220;color:#e5eefc;max-width:960px;margin:40px auto;padding:24px}"
        "pre{white-space:pre-wrap;background:#111c2e;padding:24px;border-radius:12px}"
        "button{padding:10px 16px}.figs{display:flex;flex-wrap:wrap;gap:16px;margin:12px 0}"
        "figure{margin:0;background:#111c2e;padding:10px;border-radius:10px;text-align:center}"
        "img{max-width:192px;border-radius:6px;image-rendering:pixelated}figcaption,.note{font-size:12px;color:#b8c7e0;margin-top:6px}</style>"
        "<h1>DeepShield AI · Forensic Analysis Report</h1>"
        "<p>Model-based indicator only. This is not proof of authenticity or manipulation.</p>"
        "<button onclick='window.print()'>Print / Save as PDF</button>"
        + "".join(visual_sections)
        + f"<pre>{html.escape(safe)}</pre>"
    )
