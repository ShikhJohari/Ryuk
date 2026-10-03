"""The FastAPI service: everything lives under `/api`."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from importlib.metadata import version

from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from pydantic.alias_generators import to_camel

from ryuk.api import evaluation as evaluation_routes
from ryuk.api import health, models, monitor, persons, sightings
from ryuk.api.caching import NoStoreMiddleware
from ryuk.api.contract import install_openapi
from ryuk.api.evaluation_models import EvaluationReport
from ryuk.api.localhost import LocalhostOnlyMiddleware
from ryuk.api.monitor import TICK_INTERVAL, LiveMonitor
from ryuk.api.problems import install_problem_handlers
from ryuk.api.uploads import PhotoUploadLimitMiddleware
from ryuk.watchlist.service import Watchlist


def create_app(
    start_watchlist: Callable[[], Watchlist] | None = None,
    *,
    evaluation: EvaluationReport | None = None,
    tick_interval: float = TICK_INTERVAL,
) -> FastAPI:
    """The service. `start_watchlist` runs at startup, before any request is served.
    `evaluation` is what the evaluation page shows, read from the committed outputs.
    `tick_interval` is the seconds between the live monitor's sighting ticks.

    Without a watchlist or an evaluation the service still runs, and the routes that need them
    answer 503: enough to write the contract or test the shell without touching a database.
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
    app.state.evaluation = evaluation
    app.state.monitor = LiveMonitor(tick_interval)
    # The last added runs first: requests are checked for host and origin before anything else,
    # and every response, a refusal included, is kept out of the browser's cache.
    app.add_middleware(PhotoUploadLimitMiddleware)
    app.add_middleware(LocalhostOnlyMiddleware)
    app.add_middleware(NoStoreMiddleware)
    install_problem_handlers(app)

    api = APIRouter(prefix="/api")
    api.include_router(evaluation_routes.router)
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
