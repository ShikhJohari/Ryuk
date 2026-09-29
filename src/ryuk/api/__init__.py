"""The FastAPI service: everything lives under `/api`."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from pydantic.alias_generators import to_camel

from ryuk.api import health, models, monitor, persons, sightings
from ryuk.api.caching import NoStoreMiddleware
from ryuk.api.contract import install_openapi
from ryuk.api.localhost import LocalhostOnlyMiddleware
from ryuk.api.monitor import LiveMonitor
from ryuk.api.problems import install_problem_handlers
from ryuk.api.uploads import PhotoUploadLimitMiddleware
from ryuk.watchlist.service import Watchlist


def create_app(start_watchlist: Callable[[], Watchlist] | None = None) -> FastAPI:
    """The service. `start_watchlist` runs at startup, before any request is served.

    Without it the service runs with no watchlist, and its routes answer 503: enough to write
    the contract or test the shell without touching a database.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        watchlist = None if start_watchlist is None else start_watchlist()
        app.state.watchlist = watchlist
        try:
            yield
        finally:
            if watchlist is not None:
                watchlist.close()

    app = FastAPI(
        title="Ryuk",
        version=version("ryuk"),
        separate_input_output_schemas=False,
        generate_unique_id_function=_operation_id,
        lifespan=lifespan,
    )
    app.state.watchlist = None
    app.state.monitor = LiveMonitor()
    # The last added runs first: requests are checked for host and origin before anything else,
    # and every response, a refusal included, is kept out of the browser's cache.
    app.add_middleware(PhotoUploadLimitMiddleware)
    app.add_middleware(LocalhostOnlyMiddleware)
    app.add_middleware(NoStoreMiddleware)
    install_problem_handlers(app)

    api = APIRouter(prefix="/api")
    api.include_router(health.router)
    api.include_router(models.router)
    api.include_router(monitor.router)
    api.include_router(persons.router)
    api.include_router(sightings.router)
    app.include_router(api)
    install_openapi(app)
    return app


def _operation_id(route: APIRoute) -> str:
    """`get_health` becomes `getHealth`: route function names are the operation IDs."""
    return to_camel(route.name)
