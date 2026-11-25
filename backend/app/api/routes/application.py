from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import json
import logging
import os
import tempfile
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from backend.app.schemas.application import HistoryResponse, ImageAnalysisResponse, VideoAnalysisResponse
from backend.services.analysis_store import AnalysisStore
from backend.services.image_inference import ImageInferenceService, ImageInputError, MAX_IMAGE_BYTES
from backend.services.model_registry import get_model, list_models
from backend.services.system_status import system_status
from scripts.training.video.frame_sampling import sample_video
from scripts.training.video.video_metadata import inspect_video

router = APIRouter(tags=["application"])
logger = logging.getLogger(__name__)
image_service = ImageInferenceService()
MAX_VIDEO_BYTES = 200 * 1024 * 1024
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

@router.get("/system")
def get_system_status():
    return system_status()

@router.get("/models")
def models():
    return {"items": list_models()}

@router.get("/performance")
def performance():
    return {"dataset": "CIFAKE v3 held-out test benchmark — 12,000 images", "scope_warning": "32×32 AI-generated image classification; not a facial or universal deepfake benchmark.", "items": list_models()}

@router.get("/dashboard")
def dashboard(request: Request):
    store = _store(request)
    items = store.list(limit=8)
    return {"counts": store.counts(), "recent": items, "models": list_models(), "system": system_status()}

@router.post("/analysis/images", response_model=ImageAnalysisResponse)
async def analyze_image(request: Request, model_id: str = Form(...), file: UploadFile = File(...)):
    model = get_model(model_id)
    if model is None or model["status"] != "available":
        raise HTTPException(status_code=404, detail="The selected model is unavailable.")
    data = await file.read(MAX_IMAGE_BYTES + 1)
    filename = _safe_filename(file.filename)
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
    metadata = inspect_video(path)
    if metadata.get("decode_status") != "ok":
        raise ValueError("The video could not be decoded.")
    frames = sample_video(path, max_frames=6)
    sampled=[]
    for item in frames:
        if not item["decode_ok"] or item["frame"] is None: continue
        frame=item["frame"]
        scale=min(1.0,320/max(frame.shape[0],frame.shape[1]))
        thumb=cv2.resize(frame,(max(1,int(frame.shape[1]*scale)),max(1,int(frame.shape[0]*scale))))
        ok,encoded=cv2.imencode('.jpg',cv2.cvtColor(thumb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,75])
        if ok: sampled.append({"frame_index":item["frame_index"],"timestamp_seconds":item["timestamp_seconds"],"preview_jpeg_base64":base64.b64encode(encoded.tobytes()).decode("ascii")})
    metadata.pop("file_path",None)
    return metadata,sampled

@router.post("/analysis/videos", response_model=VideoAnalysisResponse)
async def analyze_video(request: Request, file: UploadFile = File(...)):
    filename=_safe_filename(file.filename)
    if Path(filename).suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=415,detail="Use an MP4, AVI, MOV, MKV, or WebM video.")
    data=await file.read(MAX_VIDEO_BYTES+1)
    if not data: raise HTTPException(status_code=400,detail="The uploaded file is empty.")
    if len(data)>MAX_VIDEO_BYTES: raise HTTPException(status_code=413,detail="Video exceeds the 200 MB upload limit.")
    temp_path=None
    try:
        handle=tempfile.NamedTemporaryFile(prefix="deepshield-video-",suffix=Path(filename).suffix.lower(),delete=False)
        temp_path=handle.name
        try: handle.write(data)
        finally: handle.close()
        metadata,sampled=await run_in_threadpool(_video_process,temp_path)
    except ValueError as exc:
        raise HTTPException(status_code=415,detail=str(exc)) from exc
    finally:
        if temp_path:
            try: os.unlink(temp_path)
            except FileNotFoundError: pass
    digest=hashlib.sha256(data).hexdigest()
    result={"prediction_available":False,"limitation":"Video prediction is unavailable: no validated trained video checkpoint or real-video evaluation dataset is available."}
    record=_store(request).add(media_type="video",filename=filename,sha256=digest,metadata={**metadata,"sampled_frame_count":len(sampled),"retention":"Uploaded media is removed after temporary processing."},result=result,status="experimental_metadata_only")
    return {"id":record["id"],"filename":filename,"sha256":digest,"status":record["status"],**result,"metadata":record["metadata"],"sampled_frames":sampled}

@router.get("/history", response_model=HistoryResponse)
def history(request: Request, query: str = "", media_type: str = Query(default="", pattern="^(|image|video)$"), limit: int = Query(default=100, ge=1, le=500)):
    items=_store(request).list(query=query,media_type=media_type,limit=limit)
    return {"items":items,"total":len(items)}

@router.get("/history/{analysis_id}")
def history_detail(request: Request, analysis_id: str):
    item=_store(request).get(analysis_id)
    if item is None: raise HTTPException(status_code=404,detail="Analysis not found.")
    return item

@router.get("/reports/{analysis_id}.json")
def report_json(request: Request, analysis_id: str):
    item=_store(request).get(analysis_id)
    if item is None: raise HTTPException(status_code=404,detail="Report not found.")
    return JSONResponse(item,headers={"Content-Disposition":f'attachment; filename="deepshield-{analysis_id}.json"'})

@router.get("/reports/{analysis_id}.html",response_class=HTMLResponse)
def report_html(request: Request, analysis_id: str):
    item=_store(request).get(analysis_id)
    if item is None: raise HTTPException(status_code=404,detail="Report not found.")
    safe=json.dumps(item,indent=2,ensure_ascii=False)
    return HTMLResponse(f"<!doctype html><meta charset='utf-8'><title>DeepShield Forensic Report</title><style>body{{font:15px system-ui;background:#0b1220;color:#e5eefc;max-width:900px;margin:40px auto;padding:24px}}pre{{white-space:pre-wrap;background:#111c2e;padding:24px;border-radius:12px}}button{{padding:10px 16px}}</style><h1>DeepShield AI · Forensic Analysis Report</h1><p>Model-based indicator only. This is not proof of authenticity or manipulation.</p><button onclick='window.print()'>Print / Save as PDF</button><pre>{html.escape(safe)}</pre>")
