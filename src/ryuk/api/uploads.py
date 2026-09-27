"""Refuse a photo upload whose request body is too large before it is read.

The multipart parser spools a file part to a temporary file as it arrives, so a limit checked in
the route only applies once the whole body is on disk: a multi-GB POST would fill the temporary
directory first. This middleware answers `413 photo_too_large` from the Content-Length header
without reading the body, and for a chunked body (no Content-Length) stops reading it as soon as
more than the ceiling has arrived. The route's own check on the photo stays as a second line.
"""

import re
from typing import Final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ryuk.api.problems import problem_for_status, problem_response
from ryuk.watchlist.photos import MAX_PHOTO_BYTES

MAX_UPLOAD_BYTES: Final = MAX_PHOTO_BYTES + 64 * 1024
"""The largest photo plus room for the multipart framing and the other form fields."""

# POST /api/persons (enroll) and POST /api/persons/{id}/photos (add a photo).
_UPLOAD_PATH: Final = re.compile(r"/api/persons(?:/[^/]+/photos)?")


class PhotoUploadLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or not _UPLOAD_PATH.fullmatch(scope["path"])
        ):
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > MAX_UPLOAD_BYTES:
            await _too_large(scope, receive, send)
            return

        # A body without Content-Length (chunked) is counted as it is read. Once it passes the
        # ceiling the app is told the client went away, so it stops reading, and whatever it
        # answers to that is replaced by the 413.
        received = 0
        exceeded = False
        started = False

        async def limited_receive() -> Message:
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > MAX_UPLOAD_BYTES:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal started
            if exceeded and not started:
                return
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            # An app that raises on the cut-short body instead of answering: the 413 still holds.
            if not exceeded or started:
                raise
        if exceeded and not started:
            await _too_large(scope, receive, send)


def _content_length(scope: Scope) -> int | None:
    """The request's Content-Length, or None when it has none or it is not a number.

    The server has already refused a request whose Content-Length is malformed or repeated with
    different values, and holds the body to it.
    """
    headers: list[tuple[bytes, bytes]] = scope["headers"]
    value = dict(headers).get(b"content-length")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


async def _too_large(scope: Scope, receive: Receive, send: Send) -> None:
    problem = problem_for_status(
        413,
        f"The photo is over {MAX_PHOTO_BYTES // (1024 * 1024)} MB.",
        code="photo_too_large",
    )
    response = problem_response(problem)
    # The rest of the body is never read, so the connection cannot carry another request.
    response.headers["connection"] = "close"
    await response(scope, receive, send)
