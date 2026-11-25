"""Process liveness and initial readiness routes."""

from fastapi import APIRouter, Request

from backend.app import __version__
from backend.app.core.config import Settings
from backend.app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


def _response(request: Request, status: str) -> HealthResponse:
    settings: Settings = request.app.state.settings
    return HealthResponse(status=status, service=settings.app_name, version=__version__)


@router.get("/health/live", response_model=HealthResponse, summary="Check process liveness")
def liveness(request: Request) -> HealthResponse:
    """Return success when the API process is able to serve requests."""
    return _response(request, "ok")


@router.get("/health/ready", response_model=HealthResponse, summary="Check API readiness")
def readiness(request: Request) -> HealthResponse:
    """Report that the Phase 1 API foundation has loaded.

    Database, worker, storage, and model checks will be added when those
    subsystems are implemented.
    """
    return _response(request, "ready")
