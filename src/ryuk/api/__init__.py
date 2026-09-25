"""The FastAPI service: everything lives under `/api`."""

from importlib.metadata import version

from fastapi import APIRouter, FastAPI
from fastapi.routing import APIRoute
from pydantic.alias_generators import to_camel

from ryuk.api import health
from ryuk.api.contract import install_openapi
from ryuk.api.localhost import LocalhostOnlyMiddleware
from ryuk.api.problems import install_problem_handlers
from ryuk.settings import Settings


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(
        title="Ryuk",
        version=version("ryuk"),
        separate_input_output_schemas=False,
        generate_unique_id_function=_operation_id,
    )
    app.state.settings = settings
    app.add_middleware(LocalhostOnlyMiddleware)
    install_problem_handlers(app)

    api = APIRouter(prefix="/api")
    api.include_router(health.router)
    app.include_router(api)
    install_openapi(app)
    return app


def _operation_id(route: APIRoute) -> str:
    """`get_health` becomes `getHealth`: route function names are the operation IDs."""
    return to_camel(route.name)
