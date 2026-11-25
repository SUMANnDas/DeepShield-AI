"""Tests for Phase 1 configuration, health, and documentation routes."""

import asyncio
import os
import unittest
from unittest.mock import patch

import httpx

from backend.app.core.config import Settings
from backend.app.main import create_app


class HealthRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = create_app(
            Settings(
                app_name="DeepShield Test",
                environment="test",
                log_level="WARNING",
                api_prefix="/api/v1",
                docs_enabled=True,
            )
        )

    def get(self, path: str, app=None) -> httpx.Response:
        async def request() -> httpx.Response:
            transport = httpx.ASGITransport(app=app or self.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://testserver",
            ) as client:
                return await client.get(path)

        return asyncio.run(request())

    def test_liveness_reports_running_service(self) -> None:
        response = self.get("/api/v1/health/live")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "service": "DeepShield Test", "version": "0.1.0"},
        )

    def test_readiness_reports_initial_api_ready(self) -> None:
        response = self.get("/api/v1/health/ready")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")

    def test_root_route_is_available(self) -> None:
        response = self.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"service": "DeepShield Test", "status": "ok"})

    def test_openapi_documentation_is_available(self) -> None:
        response = self.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        self.assertIn("/api/v1/health/live", response.json()["paths"])

    def test_documentation_can_be_disabled(self) -> None:
        app = create_app(
            Settings(
                environment="production",
                log_level="WARNING",
                docs_enabled=False,
            )
        )

        self.assertEqual(self.get("/docs", app=app).status_code, 404)
        self.assertEqual(self.get("/openapi.json", app=app).status_code, 404)

    def test_production_defaults_to_disabled_documentation(self) -> None:
        with patch.dict(
            os.environ,
            {"DEEPSHIELD_ENVIRONMENT": "production"},
            clear=True,
        ):
            settings = Settings.from_environment()

        self.assertFalse(settings.docs_enabled)

    def test_invalid_api_prefix_is_rejected(self) -> None:
        with patch.dict(
            os.environ,
            {"DEEPSHIELD_API_PREFIX": "api/v1"},
            clear=True,
        ):
            with self.assertRaises(ValueError):
                Settings.from_environment()


if __name__ == "__main__":
    unittest.main()
