"""The live monitor through the WebSocket seam: frames in, results out, with the real YuNet
detector and fake recognition models standing in for the networks."""

import struct
import threading
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ryuk.api.frames import MAX_FRAME_BYTES
from ryuk.detector import MIN_USABLE_FACE_SIZE, Image
from ryuk.recognition import AlignedSize, Embedding, ModelKey, RecognitionModel
from synthetic import fake
from watchlist_service import (
    THRESHOLD,
    distant,
    encode,
    evaluated,
    frame,
    portrait,
    serve,
    upload,
)

MONITOR = "ws://127.0.0.1/api/monitor"
BROWSER = {"origin": "http://localhost:5173"}
"""The Origin a browser sends on the handshake from the client's page."""


def serving(tmp_path: Path, model: RecognitionModel) -> AbstractContextManager[TestClient]:
    """The service with `model` as its only recognition model, evaluated and active."""
    return serve(tmp_path / "ryuk.sqlite3", [model], evaluated(model.key, first_active=model.key))


def enroll(client: TestClient, name: str, look: int) -> Any:
    response = client.post("/api/persons", data={"name": name}, files=upload(portrait(look)))
    assert response.status_code == 201, response.text
    return response.json()


def test_a_face_of_a_person_on_the_watchlist_is_a_match_with_their_name_and_score(
    tmp_path: Path,
) -> None:
    sface = fake("sface")
    with serving(tmp_path, sface) as client:
        ada = enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            monitor.send_bytes(frame(portrait(0, shot=1), seq=7, captured_at=1_732_000_000_123))
            result = monitor.receive_json()

    assert result["type"] == "result"
    assert result["seq"] == 7
    assert result["capturedAt"] == 1_732_000_000_123
    assert result["modelKey"] == sface.key.id
    [face] = result["faces"]
    assert face["outcome"] == "match"
    assert face["person"] == {"id": ada["id"], "name": "Ada Lovelace"}
    assert face["score"] >= result["threshold"]


def test_a_face_not_on_the_watchlist_is_a_no_match_with_its_score_and_no_name(
    tmp_path: Path,
) -> None:
    sface = fake("sface")
    with serving(tmp_path, sface) as client:
        enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            monitor.send_bytes(frame(portrait(3)))
            result = monitor.receive_json()

    [face] = result["faces"]
    assert face["outcome"] == "no_match"
    assert face["score"] < result["threshold"]
    assert "person" not in face


def test_a_face_too_small_to_use_is_boxed_but_never_scored(tmp_path: Path) -> None:
    sface = fake("sface")
    with serving(tmp_path, sface) as client:
        enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            monitor.send_bytes(frame(distant(0)))
            result = monitor.receive_json()

    [face] = result["faces"]
    assert face["outcome"] == "too_small"
    assert set(face) == {"outcome", "box"}
    assert face["box"]["width"] < MIN_USABLE_FACE_SIZE


def test_every_face_in_a_frame_is_judged_on_its_own(tmp_path: Path) -> None:
    sface = fake("sface")
    with serving(tmp_path, sface) as client:
        # The fake is sensitive to scale, so the frame's faces are the size of the enrolled one.
        client.post(
            "/api/persons", data={"name": "Ada Lovelace"}, files=upload(portrait(0, size=300))
        )
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            monitor.send_bytes(frame(np.hstack([portrait(0, 1, size=300), portrait(3, size=300)])))
            result = monitor.receive_json()

    by_left = sorted(result["faces"], key=lambda face: face["box"]["x"])
    assert [face["outcome"] for face in by_left] == ["match", "no_match"]
    assert (result["width"], result["height"]) == (600, 300)


def test_a_persons_live_score_is_the_cosine_to_their_best_enrolled_photo(tmp_path: Path) -> None:
    sface = fake("sface")
    with serving(tmp_path, sface) as client:
        ada = enroll(client, "Ada Lovelace", look=0)
        grace = enroll(client, "Grace Hopper", look=3)
        added = client.post(
            f"/api/persons/{ada['id']}/photos",
            data={"acknowledgedWarnings": "may_not_be_same_person"},
            files=upload(portrait(5)),
        )
        assert added.status_code == 201, added.text
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            monitor.send_bytes(frame(portrait(5, shot=1), seq=1))
            second_photo = monitor.receive_json()
            monitor.send_bytes(frame(portrait(3, shot=1), seq=2))
            other_person = monitor.receive_json()

    # A mean over Ada's two very different photos would fall far under the threshold.
    assert second_photo["faces"][0]["person"]["id"] == ada["id"]
    assert other_person["faces"][0]["person"]["id"] == grace["id"]


def test_with_nobody_on_the_watchlist_a_face_is_a_no_match_without_a_score(
    tmp_path: Path,
) -> None:
    sface = fake("sface")
    with (
        serving(tmp_path, sface) as client,
        client.websocket_connect(MONITOR, headers=BROWSER) as monitor,
    ):
        monitor.send_bytes(frame(portrait(0)))
        result = monitor.receive_json()

    assert result["faces"] == [
        {"outcome": "no_match", "box": result["faces"][0]["box"], "score": None}
    ]


def test_the_next_frame_sees_a_person_enrolled_or_renamed_while_the_monitor_runs(
    tmp_path: Path,
) -> None:
    sface = fake("sface")
    with (
        serving(tmp_path, sface) as client,
        client.websocket_connect(MONITOR, headers=BROWSER) as monitor,
    ):
        monitor.send_bytes(frame(portrait(0, shot=1), seq=1))
        before = monitor.receive_json()
        ada = enroll(client, "Ada Lovelace", look=0)
        monitor.send_bytes(frame(portrait(0, shot=1), seq=2))
        enrolled = monitor.receive_json()
        client.patch(f"/api/persons/{ada['id']}", json={"name": "Augusta Ada King"})
        monitor.send_bytes(frame(portrait(0, shot=1), seq=3))
        renamed = monitor.receive_json()

    assert before["faces"][0]["outcome"] == "no_match"
    assert enrolled["faces"][0]["person"]["name"] == "Ada Lovelace"
    assert renamed["faces"][0]["person"] == {"id": ada["id"], "name": "Augusta Ada King"}


class Gated:
    """A fake that, once closed, holds each embedding until the test opens it again, so the test
    can keep inference busy while more frames arrive."""

    def __init__(self, model: RecognitionModel) -> None:
        self._model = model
        self.entered = threading.Event()
        self.open = threading.Event()
        self.open.set()

    @property
    def key(self) -> ModelKey:
        return self._model.key

    @property
    def dimension(self) -> int:
        return self._model.dimension

    @property
    def input_size(self) -> AlignedSize:
        return self._model.input_size

    def embed(self, face: Image) -> Embedding:
        self.entered.set()
        if not self.open.wait(timeout=10):
            raise TimeoutError("the test never let the embedding through")
        return self._model.embed(face)


def test_frames_that_arrive_while_inference_is_busy_are_dropped_but_the_latest(
    tmp_path: Path,
) -> None:
    gated = Gated(fake("sface"))
    with serving(tmp_path, gated) as client:
        enroll(client, "Ada Lovelace", look=0)
        gated.open.clear()
        gated.entered.clear()
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            monitor.send_bytes(frame(portrait(0, shot=1), seq=1))
            assert gated.entered.wait(timeout=10)
            for seq in (2, 3, 4):
                monitor.send_bytes(frame(portrait(0, shot=1), seq=seq))
            # Answered by the receive loop in order, so frames 2 to 4 have all arrived.
            monitor.send_bytes(b"not a frame")
            barrier = monitor.receive_json()
            gated.open.set()
            results = [monitor.receive_json(), monitor.receive_json()]
            monitor.send_bytes(b"not a frame")
            after = monitor.receive_json()

    assert barrier["type"] == "error"
    assert [result["seq"] for result in results] == [1, 4]
    assert after["type"] == "error"  # nothing else was waiting: frames 2 and 3 were dropped


def test_a_new_connection_takes_over_and_the_old_one_is_closed_with_4001(
    tmp_path: Path,
) -> None:
    sface = fake("sface")
    with serving(tmp_path, sface) as client:
        enroll(client, "Ada Lovelace", look=0)
        with (
            client.websocket_connect(MONITOR, headers=BROWSER) as first,
            client.websocket_connect(MONITOR, headers=BROWSER) as second,
        ):
            with pytest.raises(WebSocketDisconnect) as closed:
                first.receive_json()
            second.send_bytes(frame(portrait(0, shot=1)))
            result = second.receive_json()

    assert closed.value.code == 4001
    assert result["faces"][0]["outcome"] == "match"


def answer_without_origin(client: TestClient) -> Any:
    """What the monitor answers a socket opened without an Origin, as curl or a script might."""
    with client.websocket_connect(MONITOR) as socket:
        socket.send_bytes(b"not a frame")
        return socket.receive_json()


def test_a_socket_without_an_origin_cannot_take_over_the_live_monitor(tmp_path: Path) -> None:
    sface = fake("sface")
    with serving(tmp_path, sface) as client:
        enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as operator:
            with pytest.raises(WebSocketDisconnect) as refused:
                answer_without_origin(client)
            operator.send_bytes(frame(portrait(0, shot=1)))
            result = operator.receive_json()

    assert refused.value.code == 1008
    assert result["faces"][0]["outcome"] == "match"


def test_with_no_model_that_can_be_active_the_monitor_closes_with_4002(tmp_path: Path) -> None:
    sface = fake("sface")
    not_evaluated = evaluated()
    with (
        serve(tmp_path / "ryuk.sqlite3", [sface], not_evaluated) as client,
        client.websocket_connect(MONITOR, headers=BROWSER) as monitor,
        pytest.raises(WebSocketDisconnect) as closed,
    ):
        monitor.receive_json()

    assert closed.value.code == 4002
    assert closed.value.reason == "No evaluated recognition model can be active."


def test_without_the_detector_the_monitor_closes_with_4002(tmp_path: Path) -> None:
    sface = fake("sface")
    with (
        serve(
            tmp_path / "ryuk.sqlite3",
            [sface],
            evaluated(sface.key, first_active=sface.key),
            detector=False,
        ) as client,
        client.websocket_connect(MONITOR, headers=BROWSER) as monitor,
        pytest.raises(WebSocketDisconnect) as closed,
    ):
        monitor.receive_json()

    assert closed.value.code == 4002
    assert closed.value.reason == "The face detector's weights are missing."


def test_without_a_watchlist_the_monitor_closes_with_4002(client: TestClient) -> None:
    with (
        client.websocket_connect(MONITOR, headers=BROWSER) as monitor,
        pytest.raises(WebSocketDisconnect) as closed,
    ):
        monitor.receive_json()

    assert closed.value.code == 4002


@pytest.mark.parametrize(
    ("message", "code", "seq"),
    [
        pytest.param(b"\x01\x00", "invalid_frame", None, id="short header"),
        pytest.param(struct.pack(">BIQHH", 2, 5, 0, 10, 10) + b"x", "invalid_frame", 5, id="type"),
        pytest.param(
            struct.pack(">BIQHH", 1, 6, 0, 4000, 10) + b"x", "frame_too_large", 6, id="width"
        ),
        pytest.param(
            struct.pack(">BIQHH", 1, 7, 0, 10, 10) + b"x" * (MAX_FRAME_BYTES + 1),
            "frame_too_large",
            7,
            id="bytes",
        ),
        pytest.param(
            struct.pack(">BIQHH", 1, 8, 0, 10, 10) + b"not a jpeg", "invalid_frame", 8, id="jpeg"
        ),
        pytest.param(
            struct.pack(">BIQHH", 1, 9, 0, 640, 480) + encode(portrait(0)),
            "invalid_frame",
            9,
            id="size",
        ),
    ],
)
def test_a_message_that_is_not_a_usable_frame_is_answered_with_an_error(
    tmp_path: Path, message: bytes, code: str, seq: int | None
) -> None:
    sface = fake("sface")
    with (
        serving(tmp_path, sface) as client,
        client.websocket_connect(MONITOR, headers=BROWSER) as monitor,
    ):
        monitor.send_bytes(message)
        error = monitor.receive_json()
        monitor.send_bytes(frame(portrait(0)))
        still_open = monitor.receive_json()

    assert error["type"] == "error"
    assert error["code"] == code
    assert error["seq"] == seq
    assert still_open["type"] == "result"


def test_a_text_message_is_answered_with_an_error(tmp_path: Path) -> None:
    sface = fake("sface")
    with (
        serving(tmp_path, sface) as client,
        client.websocket_connect(MONITOR, headers=BROWSER) as monitor,
    ):
        monitor.send_text("hello")
        error = monitor.receive_json()

    assert error == {
        "type": "error",
        "seq": None,
        "code": "invalid_frame",
        "detail": "A frame is a binary message.",
    }


class Faulty:
    """A fake that raises while `failing` is set, as a bug anywhere in recognition might."""

    def __init__(self, model: RecognitionModel) -> None:
        self._model = model
        self.failing = threading.Event()

    @property
    def key(self) -> ModelKey:
        return self._model.key

    @property
    def dimension(self) -> int:
        return self._model.dimension

    @property
    def input_size(self) -> AlignedSize:
        return self._model.input_size

    def embed(self, face: Image) -> Embedding:
        if self.failing.is_set():
            raise ValueError("a bug in recognition")
        return self._model.embed(face)


def test_an_unexpected_error_in_recognition_is_answered_and_the_monitor_goes_on(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    faulty = Faulty(fake("sface"))
    with serving(tmp_path, faulty) as client:
        enroll(client, "Ada Lovelace", look=0)
        faulty.failing.set()
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            monitor.send_bytes(frame(portrait(0, shot=1), seq=3))
            error = monitor.receive_json()
            monitor.send_bytes(frame(portrait(0, shot=1), seq=4))
            again = monitor.receive_json()
            faulty.failing.clear()
            monitor.send_bytes(frame(portrait(0, shot=1), seq=5))
            result = monitor.receive_json()

    assert error == {
        "type": "error",
        "seq": 3,
        "code": "internal_error",
        "detail": "The service hit an unexpected error recognising this frame.",
    }
    assert (again["seq"], again["code"]) == (4, "internal_error")
    assert result["type"] == "result"
    assert result["faces"][0]["outcome"] == "match"
    # A fault that persists logs its traceback once per connection, then one line per frame.
    first, repeat = [r for r in caplog.records if r.name == "ryuk.api.monitor"]
    assert (first.levelname, first.exc_info is not None) == ("ERROR", True)
    assert "frame 3" in first.getMessage()
    assert (repeat.levelname, repeat.exc_info) == ("WARNING", None)
    assert "frame 4" in repeat.getMessage()
    assert "ValueError: a bug in recognition" in repeat.getMessage()
    assert "\n" not in repeat.getMessage()


def test_switching_the_active_model_takes_effect_on_the_next_frame_without_re_enrollment(
    tmp_path: Path,
) -> None:
    sface, facenet = fake("sface"), fake("facenet", seed=1)
    evaluation = evaluated(sface.key, facenet.key, first_active=sface.key)
    with serve(tmp_path / "ryuk.sqlite3", [sface, facenet], evaluation) as client:
        ada = enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            monitor.send_bytes(frame(portrait(0, shot=1), seq=1))
            before = monitor.receive_json()
            switched = client.put("/api/active-model", json={"modelKey": facenet.key.id})
            announced = monitor.receive_json()
            monitor.send_bytes(frame(portrait(0, shot=1), seq=2))
            after = monitor.receive_json()

    assert before["modelKey"] == sface.key.id
    assert switched.status_code == 200
    assert switched.json()["id"] == facenet.key.id
    assert switched.json()["state"] == "active"
    assert announced == {
        "type": "active_model_changed",
        "modelKey": facenet.key.id,
        "threshold": THRESHOLD,
    }
    assert after["modelKey"] == facenet.key.id
    assert after["faces"][0]["person"]["id"] == ada["id"]
