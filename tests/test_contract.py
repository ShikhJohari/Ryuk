"""The OpenAPI document the client's types are generated from."""

from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI

from ryuk.api import create_app
from ryuk.api.contract import openapi_schema, render_openapi
from ryuk.api.schema import ApiModel
from ryuk.settings import Settings

COMMITTED = Path(__file__).parents[1] / "openapi.json"
PROBLEM_REF = {"$ref": "#/components/schemas/Problem"}


class FaceBox(ApiModel):
    x: float
    width: float


class FrameResult(ApiModel):
    type: Literal["result"]
    frame_id: int
    faces: list[FaceBox]


def test_the_committed_contract_is_current(settings: Settings) -> None:
    # Regenerate with `uv run ryuk openapi` after changing any API model.
    assert COMMITTED.read_text() == render_openapi(openapi_schema(create_app(settings)))


def test_health_is_described_with_camel_case_fields(settings: Settings) -> None:
    schema = openapi_schema(create_app(settings))

    operation = schema["paths"]["/api/health"]["get"]
    assert operation["operationId"] == "getHealth"
    assert schema["components"]["schemas"]["Health"]["required"] == ["status", "version"]


def test_every_operation_can_answer_with_a_problem(settings: Settings) -> None:
    schema = openapi_schema(create_app(settings))

    problem = schema["components"]["schemas"]["Problem"]
    assert problem["required"] == ["type", "title", "status", "detail", "code"]
    for operation in _operations(schema):
        default = operation["responses"]["default"]
        assert default["content"] == {"application/problem+json": {"schema": PROBLEM_REF}}


def test_invalid_requests_are_described_as_problems(settings: Settings) -> None:
    app = create_app(settings)

    @app.get("/api/test/items/{item_id}")
    def get_item(item_id: int) -> int:
        return item_id

    schema = openapi_schema(app)

    invalid = schema["paths"]["/api/test/items/{item_id}"]["get"]["responses"]["422"]
    assert invalid["content"] == {"application/problem+json": {"schema": PROBLEM_REF}}
    assert "HTTPValidationError" not in schema["components"]["schemas"]
    assert "ValidationError" not in schema["components"]["schemas"]
    assert "HTTPValidationError" not in render_openapi(schema)


def test_websocket_messages_are_merged_into_the_components(settings: Settings) -> None:
    schema = openapi_schema(create_app(settings), websocket_models=[FrameResult])

    schemas = schema["components"]["schemas"]
    assert schemas["FrameResult"]["required"] == ["type", "frameId", "faces"]
    assert schemas["FrameResult"]["properties"]["faces"]["items"] == {
        "$ref": "#/components/schemas/FaceBox"
    }
    assert schemas["FaceBox"]["required"] == ["x", "width"]


def test_rendering_is_stable_and_ends_with_a_newline(settings: Settings) -> None:
    first = render_openapi(openapi_schema(create_app(settings)))
    second = render_openapi(openapi_schema(create_app(settings)))

    assert first == second
    assert first.endswith("}\n")


def test_the_app_serves_the_same_contract(settings: Settings) -> None:
    app: FastAPI = create_app(settings)

    assert app.openapi() == openapi_schema(create_app(settings))
    assert app.openapi() is app.openapi()


def _operations(schema: dict[str, Any]) -> list[dict[str, Any]]:
    return [operation for path in schema["paths"].values() for operation in path.values()]
