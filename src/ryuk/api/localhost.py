"""Refuse requests addressed to any host but this machine.

Binding to loopback stops other machines reaching the service; this stops a web
page on another origin reaching it through DNS rebinding (its hostname resolved
to 127.0.0.1), which the bind address alone cannot prevent.
"""

from starlette.types import ASGIApp, Receive, Scope, Send

from ryuk.api.problems import problem_for_status, problem_response
from ryuk.loopback import is_loopback_host

# RFC 6455 policy violation: the socket is refused before it is accepted.
_WS_POLICY_VIOLATION = 1008


def is_local_host(host_header: str) -> bool:
    """True when a Host header names this machine, with or without a port."""
    if host_header.startswith("["):
        return is_loopback_host(host_header[1 : host_header.find("]")])
    if host_header.count(":") == 1:
        return is_loopback_host(host_header.rsplit(":", 1)[0])
    return is_loopback_host(host_header)


class LocalhostOnlyMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        if is_local_host(headers.get(b"host", b"").decode("latin-1")):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": _WS_POLICY_VIOLATION})
            return
        problem = problem_for_status(
            400, "Ryuk only answers requests addressed to localhost.", code="invalid_host"
        )
        await problem_response(problem)(scope, receive, send)
