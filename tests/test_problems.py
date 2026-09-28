"""Every error the service returns is an RFC 9457 problem with a stable `code`."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ryuk.api import create_app
from ryuk.api.problems import PROBLEM_MEDIA_TYPE, ProblemError


@pytest.fixture
def app() -> FastAPI:
    app = create_app()

    @app.get("/api/test/refused")
    def refused() -> None:
        raise ProblemError(status=422, code="no_face", detail="No face was found in the photo.")

    @app.get("/api/test/crash")
    def crash() -> None:
        raise RuntimeError("secret internals")

    @app.get("/api/test/items/{item_id}")
    def item(item_id: int) -> int:
        return item_id

    @app.websocket("/api/test/socket")
    async def socket(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("hello")
        await websocket.close()

    return app


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False) as client:
        yield client


def test_a_domain_refusal_carries_its_code(client: TestClient) -> None:
    response = client.get("/api/test/refused")

    assert response.status_code == 422
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.json() == {
        "type": "about:blank",
        "title": "Unprocessable Content",
        "status": 422,
        "detail": "No face was found in the photo.",
        "code": "no_face",
    }


def test_an_unknown_route_is_not_found(client: TestClient) -> None:
    response = client.get("/api/nothing-here")

    assert response.status_code == 404
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.json()["code"] == "not_found"
    assert response.json()["title"] == "Not Found"


def test_a_wrong_method_is_not_allowed(client: TestClient) -> None:
    response = client.post("/api/health")

    assert response.status_code == 405
    assert response.json()["code"] == "method_not_allowed"


def test_an_invalid_request_names_the_offending_field(client: TestClient) -> None:
    response = client.get("/api/test/items/not-a-number")

    assert response.status_code == 422
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    body = response.json()
    assert body["code"] == "invalid_request"
    assert "item_id" in body["detail"]


def test_a_crash_is_an_internal_error_that_leaks_nothing(client: TestClient) -> None:
    response = client.get("/api/test/crash")

    assert response.status_code == 500
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.json()["code"] == "internal_error"
    assert "secret" not in response.text


@pytest.mark.parametrize("host", ["evil.example", "192.168.1.20:8000", ""])
def test_a_request_for_another_host_is_refused(client: TestClient, host: str) -> None:
    # Blocks DNS rebinding: a page on another origin resolving its name to 127.0.0.1.
    response = client.get("/api/health", headers={"host": host})

    assert response.status_code == 400
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.json()["code"] == "invalid_host"


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1:8000", "localhost:5173", "localhost", "LocalHost", "localhost.:8000", "[::1]:8000"],
)
def test_a_request_for_localhost_is_served(client: TestClient, host: str) -> None:
    response = client.get("/api/health", headers={"host": host})

    assert response.status_code == 200


def test_a_socket_for_another_host_is_refused_before_it_opens(client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as refused,
        client.websocket_connect("/api/test/socket", headers={"host": "evil.example"}),
    ):
        pass

    assert refused.value.code == 1008


def test_a_socket_for_localhost_opens(client: TestClient) -> None:
    with client.websocket_connect(
        "ws://127.0.0.1/api/test/socket", headers={"origin": "http://localhost:5173"}
    ) as socket:
        assert socket.receive_text() == "hello"


def test_a_nonstandard_status_keeps_its_code() -> None:
    app = create_app()

    @app.get("/api/test/nonstandard")
    def nonstandard() -> None:
        raise ProblemError(status=499, code="client_closed", detail="The client went away.")

    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/api/test/nonstandard")

    assert response.status_code == 499
    assert response.json()["code"] == "client_closed"
    assert response.json()["title"] == "Error"
