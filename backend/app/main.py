"""FastAPI application factory and ASGI entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app import __version__
from backend.app.api.router import build_api_router
from backend.app.core.config import Settings, get_settings
from backend.app.core.logging_config import configure_logging

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "service_starting service=%s environment=%s",
            app_settings.app_name,
            app_settings.environment,
        )
        yield
        logger.info("service_stopped service=%s", app_settings.app_name)

    docs_url = "/docs" if app_settings.docs_enabled else None
    redoc_url = "/redoc" if app_settings.docs_enabled else None
    openapi_url = "/openapi.json" if app_settings.docs_enabled else None

    application = FastAPI(
        title=app_settings.app_name,
        version=__version__,
        description=(
            "Backend foundation for a probabilistic media-forensics research prototype. "
            "Automated outputs are not certified forensic conclusions."
        ),
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )
    application.state.settings = app_settings
    application.include_router(build_api_router(app_settings.api_prefix))
    origins = [value.strip() for value in os.getenv("DEEPSHIELD_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if value.strip()]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @application.get("/", include_in_schema=False)
    def root() -> dict[str, str]:
        return {"service": app_settings.app_name, "status": "ok"}

    return application


app = create_app()
