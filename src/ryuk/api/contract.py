"""The OpenAPI document committed as `openapi.json`.

The client generates its types from that file, so it has to describe what the
service actually sends: problem responses instead of FastAPI's default
validation errors, and the WebSocket messages, which OpenAPI has no path for.
"""

import json
from collections.abc import Sequence
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from ryuk.api.monitor import MonitorMessage
from ryuk.api.problems import PROBLEM_MEDIA_TYPE, Problem, WarningsProblem

WEBSOCKET_MODELS: Sequence[type[BaseModel]] = (MonitorMessage,)
"""The live monitor's messages, which OpenAPI has no path for."""

_REF_TEMPLATE = "#/components/schemas/{model}"
_FASTAPI_VALIDATION_SCHEMAS = ("HTTPValidationError", "ValidationError")


def openapi_schema(
    app: FastAPI, websocket_models: Sequence[type[BaseModel]] = WEBSOCKET_MODELS
) -> dict[str, Any]:
    schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
        separate_input_output_schemas=False,
    )
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    components.update(_component_schemas([Problem, WarningsProblem, *websocket_models]))

    problem = {
        "content": {PROBLEM_MEDIA_TYPE: {"schema": {"$ref": _REF_TEMPLATE.format(model="Problem")}}}
    }
    for path in schema.get("paths", {}).values():
        for operation in path.values():
            responses = operation["responses"]
            if "422" in responses:
                responses["422"] = {"description": "Invalid request", **problem}
            responses["default"] = {"description": "Problem", **problem}
    for name in _FASTAPI_VALIDATION_SCHEMAS:
        components.pop(name, None)
    return schema


def problem_response_doc(model: type[Problem], description: str) -> dict[str, Any]:
    """A route's `responses` entry for a problem with extension fields, such as `409 warnings`.

    The model's schema is registered by `openapi_schema`; list it there when adding one.
    """
    schema = {"$ref": _REF_TEMPLATE.format(model=model.__name__)}
    return {"description": description, "content": {PROBLEM_MEDIA_TYPE: {"schema": schema}}}


def render_openapi(schema: dict[str, Any]) -> str:
    return json.dumps(schema, indent=2, ensure_ascii=False) + "\n"


def install_openapi(app: FastAPI) -> None:
    """Serve the same document at `/openapi.json` that `ryuk openapi` commits."""

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            app.openapi_schema = openapi_schema(app)
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]


def _component_schemas(models: Sequence[type[BaseModel]]) -> dict[str, Any]:
    _, definitions = models_json_schema(
        [(model, "validation") for model in models], ref_template=_REF_TEMPLATE
    )
    schemas: dict[str, Any] = definitions.get("$defs", {})
    return schemas
