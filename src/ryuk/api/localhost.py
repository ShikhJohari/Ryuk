"""Refuse requests addressed to any host but this machine.

Binding to loopback stops other machines reaching the service; this stops a web
page on another origin reaching it through DNS rebinding (its hostname resolved
to 127.0.0.1), which the bind address alone cannot prevent.
"""

from ipaddress import ip_address

from starlette.types import ASGIApp, Receive, Scope, Send

from ryuk.api.problems import problem_for_status, problem_response

# RFC 6455 policy violation: the socket is refused before it is accepted.
_WS_POLICY_VIOLATION = 1008


def is_local_host(host_header: str) -> bool:
    """True for `localhost` or a loopback IP, with or without a port."""
    if host_header.startswith("["):
        host = host_header[1 : host_header.find("]")]
    else:
        host = host_header.rsplit(":", 1)[0] if host_header.count(":") == 1 else host_header
    if host == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


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
