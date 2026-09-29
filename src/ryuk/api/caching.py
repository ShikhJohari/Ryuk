"""Keep every response out of the browser's cache.

Responses carry names and face photos. A cached copy is beyond purge's reach (ADR 0004), and
nothing the service answers is worth caching on loopback, so each response is `no-store` unless
its route set a Cache-Control of its own.
"""

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class NoStoreMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_no_store(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers.setdefault("cache-control", "no-store")
            await send(message)

        await self.app(scope, receive, send_no_store)
