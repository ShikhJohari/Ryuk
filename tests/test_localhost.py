"""A web page on another origin cannot make the operator's browser change the watchlist (CSRF)."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ryuk.api import create_app
from ryuk.api.localhost import is_local_origin
from ryuk.api.problems import PROBLEM_MEDIA_TYPE

# What a hostile page's `fetch(..., {method: "POST", mode: "no-cors", body: formData})` sends.
_ENROLMENT = {"name": (None, "Mallory")}
_PHOTO = {"photos": ("face.jpg", b"\xff\xd8\xff", "image/jpeg")}


@pytest.fixture
def app() -> FastAPI:
    app = create_app()

    @app.post("/api/test/change")
    def change() -> str:
        return "changed"

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


@pytest.mark.parametrize(
    "origin", ["http://evil.example", "https://127.0.0.1.evil.example", "null", "http://[::1"]
)
def test_an_enrolment_from_another_origin_is_refused(client: TestClient, origin: str) -> None:
    response = client.post(
        "/api/persons", headers={"origin": origin}, files={**_ENROLMENT, **_PHOTO}
    )

    assert response.status_code == 403
    assert response.headers["content-type"] == PROBLEM_MEDIA_TYPE
    assert response.json()["code"] == "cross_origin"


@pytest.mark.parametrize("method", ["PUT", "PATCH", "DELETE"])
def test_every_change_from_another_origin_is_refused(client: TestClient, method: str) -> None:
    response = client.request(
        method, "/api/persons/some-id", headers={"origin": "http://evil.example"}
    )

    assert response.status_code == 403
    assert response.json()["code"] == "cross_origin"


def test_a_second_origin_header_cannot_hide_a_hostile_one(client: TestClient) -> None:
    response = client.post(
        "/api/test/change",
        headers=[("origin", "http://evil.example"), ("origin", "http://localhost:5173")],
    )

    assert response.status_code == 403


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:5173",  # the Vite dev page, proxied with Origin unchanged
        "http://localhost:4173",  # the Playwright preview
        "http://127.0.0.1:8000",
        "http://LocalHost",
        "http://[::1]:8000",
    ],
)
def test_an_enrolment_from_this_machine_reaches_the_watchlist(
    client: TestClient, origin: str
) -> None:
    response = client.post(
        "/api/persons", headers={"origin": origin}, files={**_ENROLMENT, **_PHOTO}
    )

    assert response.status_code == 503
    assert response.json()["code"] == "watchlist_unavailable"


def test_a_change_without_an_origin_is_served(client: TestClient) -> None:
    response = client.post("/api/test/change")

    assert response.status_code == 200
    assert response.json() == "changed"


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
def test_a_read_from_another_origin_is_not_refused_here(client: TestClient, method: str) -> None:
    # Without CORS headers the browser keeps the response from the other page.
    response = client.request(method, "/api/health", headers={"origin": "http://evil.example"})

    assert response.status_code != 403


def test_a_socket_from_another_origin_is_refused_before_it_opens(client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as refused,
        client.websocket_connect(
            "ws://127.0.0.1/api/test/socket", headers={"origin": "http://evil.example"}
        ),
    ):
        pass

    assert refused.value.code == 1008


@pytest.mark.parametrize("headers", [{}, {"origin": "http://localhost:5173"}])
def test_a_socket_from_this_machine_opens(client: TestClient, headers: dict[str, str]) -> None:
    with client.websocket_connect("ws://127.0.0.1/api/test/socket", headers=headers) as socket:
        assert socket.receive_text() == "hello"


def test_a_host_refusal_comes_before_the_origin_check(client: TestClient) -> None:
    response = client.post(
        "/api/test/change", headers={"host": "evil.example", "origin": "http://evil.example"}
    )

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_host"


@pytest.mark.parametrize(
    ("origin", "local"),
    [
        ("http://localhost:5173", True),
        ("https://localhost.", True),
        ("http://127.8.9.10", True),
        ("http://[::1]", True),
        ("null", False),
        ("", False),
        ("localhost", False),
        ("http://192.168.1.20:8000", False),
        ("http://[::1", False),
    ],
)
def test_a_local_origin_is_one_served_by_this_machine(origin: str, local: bool) -> None:
    assert is_local_origin(origin) is local
