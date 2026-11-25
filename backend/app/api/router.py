"""API router composition."""

from fastapi import APIRouter

from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.application import router as application_router


def build_api_router(prefix: str) -> APIRouter:
    router = APIRouter(prefix=prefix)
    router.include_router(health_router)
    router.include_router(application_router)
    return router
