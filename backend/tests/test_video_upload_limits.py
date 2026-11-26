from __future__ import annotations

import asyncio
import unittest

from backend.app.core.video_upload_limits import VideoUploadLimitMiddleware


def scope(content_length: bytes | None = None):
    headers = [] if content_length is None else [(b"content-length", content_length)]
    return {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/analysis/videos",
        "headers": headers,
    }


class VideoUploadLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_raw_body_limit_stops_multipart_parser_before_spooling_more(self):
        async def app(_scope, receive, send):
            while True:
                message = await receive()
                if not message.get("more_body"):
                    break
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        messages = iter([
            {"type": "http.request", "body": b"1234", "more_body": True},
            {"type": "http.request", "body": b"5678", "more_body": False},
        ])
        sent = []
        middleware = VideoUploadLimitMiddleware(app, max_bytes=6)

        async def send(message):
            sent.append(message)

        await middleware(scope(), lambda: asyncio.sleep(0, result=next(messages)), send)
        self.assertEqual(sent[0]["status"], 413)

    async def test_content_length_is_rejected_before_calling_app(self):
        called = False

        async def app(*_args):
            nonlocal called
            called = True

        sent = []
        middleware = VideoUploadLimitMiddleware(app, max_bytes=10)

        async def send(message):
            sent.append(message)

        await middleware(scope(b"11"), lambda: asyncio.sleep(0), send)
        self.assertFalse(called)
        self.assertEqual(sent[0]["status"], 413)

    async def test_only_one_video_request_is_admitted_at_a_time(self):
        entered = asyncio.Event()
        release = asyncio.Event()

        async def app(_scope, _receive, send):
            entered.set()
            await release.wait()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        first_sent = []
        second_sent = []

        async def send_first(message):
            first_sent.append(message)

        async def send_second(message):
            second_sent.append(message)

        middleware = VideoUploadLimitMiddleware(app, max_bytes=10)
        first = asyncio.create_task(middleware(scope(), receive, send_first))
        await entered.wait()
        await middleware(scope(), receive, send_second)
        release.set()
        await first
        self.assertEqual(second_sent[0]["status"], 429)
        self.assertEqual(first_sent[0]["status"], 200)


if __name__ == "__main__":
    unittest.main()
