"""Refuse requests addressed to any host but this machine or sent from another origin.

Binding to loopback stops other machines reaching the service; this stops a web
page on another origin reaching it through DNS rebinding (its hostname resolved
to 127.0.0.1), which the bind address alone cannot prevent. It also stops such a
page making the browser send a change straight to 127.0.0.1 (CSRF): a multipart
POST needs no CORS preflight, so the browser delivers it and only the Origin
header tells it apart from a request the service's own page made.
"""

from urllib.parse import urlsplit

from starlette.types import ASGIApp, Receive, Scope, Send

from ryuk.api.problems import problem_for_status, problem_response
from ryuk.loopback import is_loopback_host

# RFC 6455 policy violation: the socket is refused before it is accepted.
_WS_POLICY_VIOLATION = 1008

# Reads cannot change anything, and without CORS headers a page on another origin
# cannot see their responses, so only these skip the Origin check.
_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


def is_local_host(host_header: str) -> bool:
    """True when a Host header names this machine, with or without a port."""
    if host_header.startswith("["):
        return is_loopback_host(host_header[1 : host_header.find("]")])
    if host_header.count(":") == 1:
        return is_loopback_host(host_header.rsplit(":", 1)[0])
    return is_loopback_host(host_header)


def is_local_origin(origin_header: str) -> bool:
    """True when an Origin header names a page served from this machine, on any port.

    `null` (a sandboxed or opaque origin) and anything unparseable are not local.
    """
    try:
        hostname = urlsplit(origin_header).hostname
    except ValueError:
        return False
    return hostname is not None and is_loopback_host(hostname)


class LocalhostOnlyMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        headers: list[tuple[bytes, bytes]] = scope["headers"]
        host = dict(headers).get(b"host", b"").decode("latin-1")
        if not is_local_host(host):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": _WS_POLICY_VIOLATION})
                return
            problem = problem_for_status(
                400, "Ryuk only answers requests addressed to localhost.", code="invalid_host"
            )
            await problem_response(problem)(scope, receive, send)
            return

        origins = [value.decode("latin-1") for name, value in headers if name == b"origin"]
        if _needs_origin_check(scope) and not _from_this_machine(scope, origins):
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": _WS_POLICY_VIOLATION})
                return
            problem = problem_for_status(
                403,
                "Ryuk refuses changes sent by a web page on another origin.",
                code="cross_origin",
            )
            await problem_response(problem)(scope, receive, send)
            return

        await self.app(scope, receive, send)


def _needs_origin_check(scope: Scope) -> bool:
    """A socket handshake or a request that can change something."""
    return scope["type"] == "websocket" or scope["method"] not in _SAFE_METHODS


def _from_this_machine(scope: Scope, origins: list[str]) -> bool:
    """Whether every Origin header names a local page.

    A request with none passes: curl and the service's own same-origin requests may omit it. A
    socket handshake with none does not: browsers always send one, and a socket opened without it
    would take the live monitor over from the operator's page.
    """
    if scope["type"] == "websocket" and not origins:
        return False
    return all(is_local_origin(origin) for origin in origins)
