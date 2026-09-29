"""Sightings through the live monitor's WebSocket seam: frames in, sighting messages out, and the
sightings the REST API then lists, with the Watchlist's clock driven by the test (#16, #31).

Each frame is sent only once the previous one's result is back, so none is dropped from the
one-slot buffer, and the clock moves between frames. The periodic tick runs once an hour unless a
test asks for it, so every change comes from a frame the test sent, except in the tests of the
tick itself.
"""

import datetime
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import update
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from ryuk.api import create_app
from ryuk.detector import Detector, Image
from ryuk.recognition import RecognitionModel
from ryuk.watchlist.database import open_database
from ryuk.watchlist.monitoring import MonitoringSession, SightingEvent
from ryuk.watchlist.service import Watchlist, start_watchlist
from ryuk.watchlist.tables import SightingRow
from synthetic import YUNET, fake
from watchlist_service import (
    FakeClock,
    blank,
    encode,
    evaluated,
    frame,
    portrait,
    upload,
    writing,
)

MONITOR = "ws://127.0.0.1/api/monitor"
BROWSER = {"origin": "http://localhost:5173"}
T0 = datetime.datetime(2026, 9, 29, 12, 0, tzinfo=datetime.UTC)
HOURLY = 3600.0
"""A tick interval no test waits for."""

ADA = portrait(0, shot=1)
"""Ada, matched at about 0.993 with Grace as runner-up at about 0.26."""
ADA_BETTER = portrait(0, shot=0)
"""Ada matched higher, at about 0.997."""
ADA_WORSE = portrait(0, shot=1, size=380)
"""Ada matched lower, at about 0.94."""
ADA_BELOW = portrait(0, shot=1, size=360)
"""Ada's face scored under the threshold: a no match."""


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(T0)


@contextmanager
def serving(
    tmp_path: Path,
    clock: FakeClock,
    *models: RecognitionModel,
    tick_interval: float = HOURLY,
) -> Iterator[TestClient]:
    """The service with `models` evaluated and the first active, on `clock`, as `serve` starts
    it but with the live monitor ticking every `tick_interval` seconds."""
    models = models or (fake("sface"),)
    evaluation = evaluated(*(model.key for model in models), first_active=models[0].key)

    def start() -> Watchlist:
        return start_watchlist(
            open_database(tmp_path / "ryuk.sqlite3"), Detector(YUNET), models, evaluation, clock
        )

    app = create_app(start, tick_interval=tick_interval)
    with TestClient(app, base_url="http://127.0.0.1", raise_server_exceptions=False) as client:
        yield client


def enroll(client: TestClient, name: str, look: int) -> Any:
    response = client.post("/api/persons", data={"name": name}, files=upload(portrait(look)))
    assert response.status_code == 201, response.text
    return response.json()


def send(monitor: WebSocketTestSession, image: Image, seq: int = 1) -> list[Any]:
    """Every message up to and including the frame's result."""
    monitor.send_bytes(frame(image, seq=seq))
    messages = []
    while True:
        message = monitor.receive_json()
        messages.append(message)
        if message["type"] in {"result", "error"}:
            return messages


def run(
    monitor: WebSocketTestSession, clock: FakeClock, *images: Image, step: float = 0.1
) -> list[Any]:
    """Every message from sending `images` in turn, `step` seconds apart, starting now."""
    messages = []
    for index, image in enumerate(images):
        if index:
            clock.advance(step)
        messages += send(monitor, image, seq=index + 1)
    return messages


def receive_within(monitor: WebSocketTestSession, seconds: float = 10) -> Any:
    """The next message, failing rather than hanging if none comes, as for a tick's."""
    waiting = ThreadPoolExecutor(max_workers=1)
    try:
        return waiting.submit(monitor.receive_json).result(timeout=seconds)
    finally:
        # A receive still blocked ends when the socket closes.
        waiting.shutdown(wait=False)


def of(messages: list[Any], kind: str) -> list[Any]:
    return [message for message in messages if message["type"] == kind]


def confirm(monitor: WebSocketTestSession, clock: FakeClock, image: Image = ADA) -> Any:
    """Ada's sighting, opened by three matched frames 100 ms apart: the opened message."""
    [opened] = of(run(monitor, clock, image, image, image), "sighting_opened")
    return opened


def moment(clock: FakeClock, seconds: float = 0) -> str:
    """The clock's time plus `seconds`, as the API writes a time."""
    later = clock.now + datetime.timedelta(seconds=seconds)
    return later.isoformat().replace("+00:00", "Z")


def keys(value: Any) -> Iterator[str]:
    """Every key anywhere in a JSON value."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys(item)


# Confirmation and the sighting ID


def test_three_matched_frames_open_a_sighting_announced_before_the_result_that_carries_its_id(
    tmp_path: Path, clock: FakeClock
) -> None:
    with serving(tmp_path, clock) as client:
        ada = enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            first = send(monitor, ADA, 1)
            clock.advance(0.1)
            [second] = send(monitor, ADA, 2)
            clock.advance(0.1)
            third = send(monitor, ADA, 3)
            stored = client.get("/api/sightings").json()

    assert [m["type"] for m in [*first, second]] == ["result", "result"]
    assert first[0]["faces"][0]["sightingId"] is None
    assert second["faces"][0]["sightingId"] is None
    opened, result = third
    assert opened["type"] == "sighting_opened"
    sighting = opened["sighting"]
    assert sighting["person"] == {"id": ada["id"], "name": "Ada Lovelace", "status": "on_watchlist"}
    assert sighting["startedAt"] == moment(clock, -0.2)
    assert sighting["lastSeenAt"] == moment(clock)
    assert sighting["endedAt"] is None
    assert sighting["bestScore"] >= sighting["threshold"]
    assert result["faces"][0]["sightingId"] == sighting["id"]
    assert stored["items"] == [sighting]


def test_a_single_fluke_frame_never_opens_a_sighting(tmp_path: Path, clock: FakeClock) -> None:
    with serving(tmp_path, clock) as client:
        enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            messages = run(monitor, clock, ADA, blank(), blank(), blank(), blank())
            clock.advance(5)
            messages += send(monitor, blank())
        stored = client.get("/api/sightings").json()

    assert {m["type"] for m in messages} == {"result"}
    assert stored["items"] == []


def test_matches_in_under_half_the_frames_do_not_confirm(tmp_path: Path, clock: FakeClock) -> None:
    with serving(tmp_path, clock) as client:
        enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            # Three matches among seven frames within 500 ms.
            images = [blank()] * 4 + [ADA] * 3
            messages = run(monitor, clock, *images, step=0.06)

    assert of(messages, "sighting_opened") == []


# Best crop and runner-up


def test_the_runner_up_comes_from_the_best_frame_and_is_shown_only_on_the_sighting_page(
    tmp_path: Path, clock: FakeClock
) -> None:
    with serving(tmp_path, clock) as client:
        enroll(client, "Ada Lovelace", look=0)
        grace = enroll(client, "Grace Hopper", look=3)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            messages = run(monitor, clock, ADA, ADA, ADA)
            clock.advance(3)
            messages += send(monitor, blank())
        [opened] = of(messages, "sighting_opened")
        detail = client.get(f"/api/sightings/{opened['sighting']['id']}").json()

    assert detail["runnerUp"]["person"]["id"] == grace["id"]
    assert 0 < detail["runnerUp"]["score"] < detail["threshold"]
    assert "runnerUp" not in set(keys(messages))


def test_the_crop_is_replaced_only_by_a_strictly_higher_score(
    tmp_path: Path, clock: FakeClock
) -> None:
    with serving(tmp_path, clock) as client:
        enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            opened = confirm(monitor, clock)["sighting"]
            crop = client.get(f"/api/sightings/{opened['id']}/crop")
            clock.advance(0.1)
            send(monitor, ADA_BETTER)
            clock.advance(1)
            [better] = of(send(monitor, ADA_WORSE), "sighting_updated")
            better_crop = client.get(f"/api/sightings/{opened['id']}/crop").content
            clock.advance(1)
            [worse] = of(send(monitor, ADA), "sighting_updated")
            kept_crop = client.get(f"/api/sightings/{opened['id']}/crop").content

    assert crop.status_code == 200
    assert crop.headers["content-type"] == "image/jpeg"
    assert crop.content.startswith(b"\xff\xd8")
    assert better["sighting"]["bestScore"] > opened["bestScore"]
    assert better_crop != crop.content
    assert worse["sighting"]["bestScore"] == better["sighting"]["bestScore"]
    assert kept_crop == better_crop


# Writes


def test_many_frames_within_a_second_give_one_update_then_another_a_second_later(
    tmp_path: Path, clock: FakeClock
) -> None:
    with serving(tmp_path, clock) as client:
        enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            opened = confirm(monitor, clock)["sighting"]
            clock.advance(0.1)
            within = run(monitor, clock, *[ADA] * 9)  # up to 0.9 s after the open
            clock.advance(0.1)
            [first] = of(send(monitor, ADA), "sighting_updated")
            clock.advance(0.1)
            again = run(monitor, clock, *[ADA] * 9)
            clock.advance(0.1)
            [second] = of(send(monitor, ADA), "sighting_updated")
            stored = client.get(f"/api/sightings/{opened['id']}").json()

    assert of(within, "sighting_updated") == []
    assert first["sighting"]["id"] == opened["id"]
    assert first["sighting"]["lastSeenAt"] == moment(clock, -1)
    assert of(again, "sighting_updated") == []
    assert second["sighting"]["lastSeenAt"] == moment(clock)
    assert stored["lastSeenAt"] == moment(clock)


# Every end


def test_a_frame_after_the_3_s_gap_ends_the_sighting_before_its_result(
    tmp_path: Path, clock: FakeClock
) -> None:
    with serving(tmp_path, clock) as client:
        enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            opened = confirm(monitor, clock)["sighting"]
            last_seen = moment(clock)
            clock.advance(2.9)
            before = send(monitor, ADA_BELOW)
            clock.advance(0.1)
            ended, result = send(monitor, blank())
        stored = client.get(f"/api/sightings/{opened['id']}").json()

    assert of(before, "sighting_ended") == []
    assert ended["type"] == "sighting_ended"
    assert result["type"] == "result"
    assert ended["sighting"]["id"] == opened["id"]
    # The frame under the threshold did not extend it.
    assert ended["sighting"]["lastSeenAt"] == last_seen
    assert ended["sighting"]["endedAt"] == last_seen
    assert stored["endedAt"] == last_seen


def test_the_tick_alone_ends_a_sighting_3_s_after_the_last_match_with_no_frame(
    tmp_path: Path, clock: FakeClock
) -> None:
    with serving(tmp_path, clock, tick_interval=0.01) as client:
        enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            opened = confirm(monitor, clock)["sighting"]
            last_seen = moment(clock)
            clock.advance(3)
            ended = receive_within(monitor)
            stored = client.get(f"/api/sightings/{opened['id']}").json()

    assert ended["type"] == "sighting_ended"
    assert ended["sighting"]["id"] == opened["id"]
    assert ended["sighting"]["endedAt"] == last_seen
    assert stored["endedAt"] == last_seen


def test_the_tick_writes_held_changes_once_frames_stop(tmp_path: Path, clock: FakeClock) -> None:
    with serving(tmp_path, clock, tick_interval=0.01) as client:
        enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            opened = confirm(monitor, clock)["sighting"]
            clock.advance(0.1)
            send(monitor, ADA_BETTER)
            clock.advance(1)
            updated = receive_within(monitor)

    assert updated["type"] == "sighting_updated"
    assert updated["sighting"]["id"] == opened["id"]
    assert updated["sighting"]["bestScore"] > opened["bestScore"]
    assert updated["sighting"]["lastSeenAt"] == moment(clock, -1)


def test_closing_the_socket_ends_its_open_sightings_when_last_seen(
    tmp_path: Path, clock: FakeClock
) -> None:
    with serving(tmp_path, clock) as client:
        enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            opened = confirm(monitor, clock)["sighting"]
            last_seen = moment(clock)
        clock.advance(1)
        stored = client.get(f"/api/sightings/{opened['id']}").json()

    assert (stored["lastSeenAt"], stored["endedAt"]) == (last_seen, last_seen)


def test_a_connection_that_takes_over_keeps_its_own_sightings_open(
    tmp_path: Path, clock: FakeClock
) -> None:
    with serving(tmp_path, clock) as client:
        enroll(client, "Ada Lovelace", look=0)
        enroll(client, "Grace Hopper", look=3)
        with client.websocket_connect(MONITOR, headers=BROWSER) as first:
            ada = confirm(first, clock)["sighting"]
            with client.websocket_connect(MONITOR, headers=BROWSER) as second:
                with pytest.raises(WebSocketDisconnect):
                    first.receive_json()
                clock.advance(0.1)
                messages = run(second, clock, portrait(3, 1), portrait(3, 1), portrait(3, 1))
                # The first connection's end reaches the second in its own time; each frame
                # sent is answered, so waiting on frames cannot hang.
                for _ in range(50):
                    if of(messages, "sighting_ended"):
                        break
                    messages += send(second, blank())
                [grace] = of(messages, "sighting_opened")
                stored = {item["id"]: item for item in client.get("/api/sightings").json()["items"]}

    [ended] = of(messages, "sighting_ended")
    assert ended["sighting"]["id"] == ada["id"]
    assert stored[ada["id"]]["endedAt"] == ada["lastSeenAt"]
    assert stored[grace["sighting"]["id"]]["endedAt"] is None


def test_the_teardown_of_a_superseded_session_ends_only_its_own_sightings(
    tmp_path: Path, clock: FakeClock
) -> None:
    sface = fake("sface")
    watchlist = start_watchlist(
        open_database(tmp_path / "ryuk.sqlite3"),
        Detector(YUNET),
        [sface],
        evaluated(sface.key, first_active=sface.key),
        clock,
    )
    try:
        watchlist.enroll("Ada Lovelace", encode(portrait(0)))
        old, new = watchlist.begin_monitoring(), watchlist.begin_monitoring()
        opened = confirm_on(watchlist, new, clock)

        ended = watchlist.end_monitoring(old)
        still = watchlist.sighting(opened)
        closing = watchlist.end_monitoring(new)
    finally:
        watchlist.close()

    assert ended == ()
    assert still.summary.ended_at is None
    assert [(event.type, event.sighting.id) for event in closing] == [("sighting_ended", opened)]


def confirm_on(watchlist: Watchlist, monitoring: MonitoringSession, clock: FakeClock) -> str:
    """Open Ada's sighting in `monitoring` straight through the Watchlist; its ID."""
    events: list[SightingEvent] = []
    for _ in range(3):
        clock.advance(0.1)
        events += watchlist.recognise(ADA, monitoring).sightings
    [opened] = events
    assert opened.type == "sighting_opened"
    return opened.sighting.id


def test_removal_ends_the_persons_sighting_and_announces_it(
    tmp_path: Path, clock: FakeClock
) -> None:
    with serving(tmp_path, clock) as client:
        ada = enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            opened = confirm(monitor, clock)["sighting"]
            last_seen = moment(clock)
            clock.advance(0.5)
            removed = client.patch(f"/api/persons/{ada['id']}", json={"status": "removed"})
            # Announced before the response, so ahead of the next frame's result.
            ended, _ = send(monitor, blank())
        stored = client.get(f"/api/sightings/{opened['id']}").json()

    assert removed.status_code == 200
    assert ended["type"] == "sighting_ended"
    assert ended["sighting"]["id"] == opened["id"]
    assert ended["sighting"]["person"]["status"] == "removed"
    assert ended["sighting"]["endedAt"] == last_seen
    assert stored["endedAt"] == last_seen


def test_a_rename_ends_nothing(tmp_path: Path, clock: FakeClock) -> None:
    with serving(tmp_path, clock) as client:
        ada = enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            opened = confirm(monitor, clock)["sighting"]
            client.patch(f"/api/persons/{ada['id']}", json={"name": "Augusta Ada King"})
            clock.advance(0.1)
            messages = send(monitor, ADA)
        stored = client.get(f"/api/sightings/{opened['id']}").json()

    assert of(messages, "sighting_ended") == []
    assert messages[-1]["faces"][0]["sightingId"] == opened["id"]
    assert stored["person"]["name"] == "Augusta Ada King"


def test_purge_ends_the_persons_sighting_with_its_last_state_and_nothing_is_left(
    tmp_path: Path, clock: FakeClock
) -> None:
    with serving(tmp_path, clock) as client:
        ada = enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            opened = confirm(monitor, clock)["sighting"]
            clock.advance(0.5)
            send(monitor, ADA)  # held, not yet written
            last_seen = moment(clock)
            purged = client.delete(f"/api/persons/{ada['id']}")
            clock.advance(0.1)
            ended, *after = send(monitor, ADA)
        stored = client.get(f"/api/sightings/{opened['id']}")

    assert purged.status_code == 204
    assert ended["type"] == "sighting_ended"
    assert ended["sighting"]["id"] == opened["id"]
    assert ended["sighting"]["person"] == {
        "id": ada["id"],
        "name": "Ada Lovelace",
        "status": "on_watchlist",
    }
    assert ended["sighting"]["lastSeenAt"] == last_seen
    assert ended["sighting"]["endedAt"] == last_seen
    assert [m["type"] for m in after] == ["result"]
    assert after[0]["faces"][0]["outcome"] == "no_match"
    assert stored.status_code == 404


def test_purging_the_runner_up_of_a_held_best_match_clears_them_and_keeps_their_score(
    tmp_path: Path, clock: FakeClock
) -> None:
    with serving(tmp_path, clock) as client:
        enroll(client, "Ada Lovelace", look=0)
        grace = enroll(client, "Grace Hopper", look=3)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            opened = confirm(monitor, clock)["sighting"]
            clock.advance(0.1)
            send(monitor, ADA_BETTER)  # a new best match, Grace its runner-up, held
            held = client.get(f"/api/sightings/{opened['id']}").json()
            purged = client.delete(f"/api/persons/{grace['id']}")
            clock.advance(1)
            messages = send(monitor, ADA)
        after = client.get(f"/api/sightings/{opened['id']}").json()

    assert held["runnerUp"]["person"]["id"] == grace["id"]
    assert purged.status_code == 204
    # A runner-up is never sent live, so the purge of one announces nothing.
    assert [m["type"] for m in messages] == ["sighting_updated", "result"]
    assert after["bestScore"] > opened["bestScore"]
    assert after["runnerUp"]["person"] is None
    assert 0 < after["runnerUp"]["score"] < after["threshold"]


def test_switching_the_active_model_ends_every_sighting_before_announcing_the_switch(
    tmp_path: Path, clock: FakeClock
) -> None:
    sface, facenet = fake("sface"), fake("facenet", seed=1)
    with serving(tmp_path, clock, sface, facenet) as client:
        enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            opened = confirm(monitor, clock)["sighting"]
            last_seen = moment(clock)
            clock.advance(0.5)
            switched = client.put("/api/active-model", json={"modelKey": facenet.key.id})
            ended, changed, _ = send(monitor, blank())
            clock.advance(0.1)
            reopened = confirm(monitor, clock)["sighting"]

    assert switched.status_code == 200
    assert (ended["type"], changed["type"]) == ("sighting_ended", "active_model_changed")
    assert ended["sighting"]["id"] == opened["id"]
    assert ended["sighting"]["endedAt"] == last_seen
    assert reopened["id"] != opened["id"]
    assert reopened["modelKey"] == facenet.key.id


def test_a_sighting_left_open_by_a_run_that_stopped_ends_at_startup_when_last_seen(
    tmp_path: Path, clock: FakeClock
) -> None:
    with serving(tmp_path, clock) as client:
        enroll(client, "Ada Lovelace", look=0)
        with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
            opened = confirm(monitor, clock)["sighting"]
    # A run that stopped without closing the socket, as a crash would, leaves it open.
    with writing(tmp_path / "ryuk.sqlite3") as session:
        session.execute(update(SightingRow).values(ended_at=None))
    clock.advance(60)
    with serving(tmp_path, clock) as client:
        stored = client.get(f"/api/sightings/{opened['id']}").json()

    assert stored["endedAt"] == opened["lastSeenAt"]
