from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field

class ImageAnalysisResponse(BaseModel):
    id: str
    created_at: str
    filename: str
    sha256: str
    media_type: str = "image"
    metadata: dict[str, Any]
    result: dict[str, Any]
    status: str = "complete"

class VideoAnalysisResponse(BaseModel):
    id: str
    filename: str
    sha256: str
    status: str
    prediction_available: bool = False
    limitation: str
    result: dict[str, Any] | None = None
    metadata: dict[str, Any]
    sampled_frames: list[dict[str, Any]] = Field(default_factory=list)

class HistoryResponse(BaseModel):
    items: list[dict[str, Any]]
    total: int

