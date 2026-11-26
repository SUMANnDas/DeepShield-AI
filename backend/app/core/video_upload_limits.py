"""Admission and request-size limits applied before multipart file spooling."""

from __future__ import annotations

import json
from typing import Any

MAX_VIDEO_REQUEST_BYTES = 201 * 1024 * 1024


class _RequestBodyTooLarge(Exception):
    pass


class VideoUploadLimitMiddleware:
    """Limit raw video upload bodies and allow one in-flight request per worker."""

    def __init__(self, app, max_bytes: int = MAX_VIDEO_REQUEST_BYTES) -> None:
        self.app = app
        self.max_bytes = max_bytes
        self.video_request_active = False

    async def __call__(self, scope: dict[str, Any], receive, send) -> None:
        is_video_upload = (
            scope.get("type") == "http"
            and scope.get("method") == "POST"
            and str(scope.get("path", "")).endswith("/analysis/videos")
        )
        if not is_video_upload:
            await self.app(scope, receive, send)
            return

        if self.video_request_active:
            await self._respond(send, 429, "Video inspection is busy; retry after the current upload finishes.")
            return

        self.video_request_active = True
        response_started = False
        received_bytes = 0

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        async def limited_receive() -> dict[str, Any]:
            nonlocal received_bytes
            message = await receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > self.max_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            content_length = next(
                (value for key, value in scope.get("headers", []) if key.lower() == b"content-length"),
                None,
            )
            if content_length is not None:
                try:
                    if int(content_length) > self.max_bytes:
                        await self._respond(send, 413, "Video upload request exceeds the 200 MB limit.")
                        return
                except ValueError:
                    pass
            try:
                await self.app(scope, limited_receive, tracked_send)
            except _RequestBodyTooLarge:
                if not response_started:
                    await self._respond(send, 413, "Video upload request exceeds the 200 MB limit.")
        finally:
            self.video_request_active = False

    @staticmethod
    async def _respond(send, status_code: int, detail: str) -> None:
        body = json.dumps({"detail": detail}).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        })
        await send({"type": "http.response.body", "body": body, "more_body": False})
