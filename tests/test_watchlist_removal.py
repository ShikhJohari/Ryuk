"""Removal, restore and purge of a person of interest through the HTTP seam (#31)."""

import datetime
import os
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ryuk.detector import Detector
from ryuk.watchlist import sightings
from ryuk.watchlist.database import open_database
from ryuk.watchlist.service import start_watchlist
from ryuk.watchlist.sightings import BestMatch, NewSighting
from synthetic import YUNET, fake
from watchlist_service import (
    FakeClock,
    encode,
    evaluated,
    frame,
    portrait,
    serve,
    upload,
    writing,
)

MONITOR = "ws://127.0.0.1/api/monitor"
BROWSER = {"origin": "http://localhost:5173"}
ENROLLED_AT = datetime.datetime(2026, 9, 29, 9, 0, tzinfo=datetime.UTC)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(ENROLLED_AT)


@pytest.fixture
def database(tmp_path: Path) -> Path:
    return tmp_path / "ryuk.sqlite3"


@pytest.fixture
def client(database: Path, clock: FakeClock) -> Iterator[TestClient]:
    model = fake("sface")
    evaluation = evaluated(model.key, first_active=model.key)
    with serve(database, [model], evaluation, clock=clock) as served:
        yield served


def enroll(client: TestClient, name: str, look: int, acknowledged: str = "") -> dict[str, Any]:
    response = client.post(
        "/api/persons",
        data={"name": name, "acknowledgedWarnings": [acknowledged] if acknowledged else []},
        files=upload(portrait(look)),
    )
    assert response.status_code == 201, response.text
    person: dict[str, Any] = response.json()
    return person


def patch(client: TestClient, person_id: str, **changes: Any) -> Any:
    return client.patch(f"/api/persons/{person_id}", json=changes)


def iso(moment: datetime.datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def record_sighting(
    database: Path,
    person_id: str,
    *,
    crop: bytes = b"\xff\xd8 crop",
    runner_up: str | None = None,
    runner_up_score: float | None = None,
) -> str:
    """A sighting stored as the live monitor's tracker stores one."""
    with writing(database) as session:
        opened = sightings.open_sighting(
            session,
            NewSighting(
                person_id=person_id,
                model_key="sface-cpu-aa",
                threshold=0.9,
                started_at=ENROLLED_AT,
                last_seen_at=ENROLLED_AT,
                best=BestMatch(0.97, crop, runner_up, runner_up_score),
            ),
        )
        return opened.id


def rows(database: Path, query: str, *parameters: object) -> list[tuple[Any, ...]]:
    with closing(sqlite3.connect(database)) as connection:
        return connection.execute(query, parameters).fetchall()


def test_removal_takes_a_person_off_the_watchlist_and_restore_puts_them_back(
    client: TestClient, clock: FakeClock
) -> None:
    ada = enroll(client, "Ada", 0)
    removed_at = clock.advance(60)

    removed = patch(client, ada["id"], status="removed")

    assert removed.status_code == 200
    assert (removed.json()["status"], removed.json()["statusChangedAt"]) == (
        "removed",
        iso(removed_at),
    )
    assert client.get("/api/persons").json() == []
    assert [p["id"] for p in client.get("/api/persons?status=removed").json()] == [ada["id"]]
    # Removal keeps the enrolled photos.
    assert removed.json()["photos"] == ada["photos"]

    restored_at = clock.advance(60)
    restored = patch(client, ada["id"], status="on_watchlist")

    assert (restored.json()["status"], restored.json()["statusChangedAt"]) == (
        "on_watchlist",
        iso(restored_at),
    )
    assert [p["id"] for p in client.get("/api/persons").json()] == [ada["id"]]


def test_the_watchlist_removes_and_restores_by_status(database: Path, clock: FakeClock) -> None:
    model = fake("sface")
    watchlist = start_watchlist(
        open_database(database),
        Detector(YUNET),
        [model],
        evaluated(model.key, first_active=model.key),
        clock,
    )
    try:
        ada = watchlist.enroll("Ada", encode(portrait(0)))
        removed_at = clock.advance(5)

        removed = watchlist.set_status(ada.id, "removed")
        clock.advance(5)
        again = watchlist.set_status(ada.id, "removed")

        assert (removed.status, removed.status_changed_at) == ("removed", removed_at)
        assert again.status_changed_at == removed_at
        assert watchlist.persons("on_watchlist") == []
        assert watchlist.set_status(ada.id, "on_watchlist").status_changed_at == clock.now
    finally:
        watchlist.close()


def test_setting_the_status_a_person_already_has_changes_nothing(
    client: TestClient, clock: FakeClock
) -> None:
    ada = enroll(client, "Ada", 0)
    removed_at = clock.advance(60)
    patch(client, ada["id"], status="removed")
    clock.advance(60)

    again = patch(client, ada["id"], status="removed")
    still_on = patch(client, enroll(client, "Grace", 1)["id"], status="on_watchlist")

    assert again.status_code == 200
    assert again.json()["statusChangedAt"] == iso(removed_at)
    assert still_on.json()["statusChangedAt"] == still_on.json()["createdAt"]


def test_a_removed_person_is_no_longer_matched_live_until_restored(client: TestClient) -> None:
    ada = enroll(client, "Ada", 0)

    with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
        monitor.send_bytes(frame(portrait(0, shot=1), seq=1))
        before = monitor.receive_json()
        patch(client, ada["id"], status="removed")
        monitor.send_bytes(frame(portrait(0, shot=1), seq=2))
        removed = monitor.receive_json()
        patch(client, ada["id"], status="on_watchlist")
        monitor.send_bytes(frame(portrait(0, shot=1), seq=3))
        restored = monitor.receive_json()

    assert before["faces"][0]["outcome"] == "match"
    # Nobody else is on the watchlist, so there is no candidate to score.
    assert (removed["faces"][0]["outcome"], removed["faces"][0]["score"]) == ("no_match", None)
    assert restored["faces"][0]["person"]["id"] == ada["id"]


def test_an_empty_change_returns_the_person_unchanged(client: TestClient, clock: FakeClock) -> None:
    ada = enroll(client, "Ada", 0)
    clock.advance(60)

    response = patch(client, ada["id"])

    assert response.status_code == 200
    assert response.json() == ada
    assert patch(client, "nobody").status_code == 404


def test_the_name_and_the_status_change_together_or_alone(
    client: TestClient, clock: FakeClock
) -> None:
    ada = enroll(client, "Ada", 0)
    changed_at = clock.advance(60)

    renamed = patch(client, ada["id"], name="Ada Lovelace")
    both = patch(client, ada["id"], name="Augusta Ada King", status="removed")

    assert (renamed.json()["name"], renamed.json()["status"]) == ("Ada Lovelace", "on_watchlist")
    assert renamed.json()["statusChangedAt"] == ada["statusChangedAt"]
    assert (both.json()["name"], both.json()["status"], both.json()["statusChangedAt"]) == (
        "Augusta Ada King",
        "removed",
        iso(changed_at),
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"name": "Grace", "photos": []},
        {"createdAt": "2026-01-01T00:00:00Z"},
        {"status": "purged"},
        {"name": None},
        {"status": None},
    ],
    ids=["unknown-field", "read-only-field", "unknown-status", "null-name", "null-status"],
)
def test_a_change_patch_cannot_make_is_refused_and_nothing_changes(
    client: TestClient, changes: dict[str, Any]
) -> None:
    ada = enroll(client, "Ada", 0)

    response = client.patch(f"/api/persons/{ada['id']}", json=changes)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert client.get(f"/api/persons/{ada['id']}").json() == ada


def test_an_invalid_name_changes_neither_the_name_nor_the_status(client: TestClient) -> None:
    ada = enroll(client, "Ada", 0)

    response = patch(client, ada["id"], name=" ", status="removed")

    assert response.json()["code"] == "invalid_name"
    assert client.get(f"/api/persons/{ada['id']}").json() == ada


@pytest.mark.parametrize("status", ["on_watchlist", "removed"])
def test_a_purge_erases_the_person_with_every_photo_embedding_and_sighting(
    client: TestClient, database: Path, status: str
) -> None:
    ada = enroll(client, "Ada", 0)
    grace = enroll(client, "Grace", 1)
    patch(client, ada["id"], status=status)
    [photo] = ada["photos"]
    image = client.get(f"/api/persons/{ada['id']}/photos/{photo['id']}/image").content
    crop = os.urandom(50_000)  # spans overflow pages, and appears nowhere else
    record_sighting(database, ada["id"], crop=crop, runner_up=grace["id"], runner_up_score=0.4)
    graces = record_sighting(database, grace["id"], runner_up=ada["id"], runner_up_score=0.42)

    response = client.delete(f"/api/persons/{ada['id']}")

    assert response.status_code == 204
    assert response.content == b""
    assert client.get(f"/api/persons/{ada['id']}").status_code == 404
    assert client.get(f"/api/persons/{ada['id']}/photos/{photo['id']}/image").status_code == 404
    for table, column in [
        ("person_of_interest", "id"),
        ("enrolled_photo", "person_id"),
        ("sighting", "person_id"),
        ("sighting", "runner_up_person_id"),
    ]:
        assert rows(database, f"SELECT * FROM {table} WHERE {column} = ?", ada["id"]) == []  # noqa: S608
    assert rows(database, "SELECT photo_id FROM embedding WHERE photo_id = ?", photo["id"]) == []
    # Grace's sighting keeps the purged runner-up's score, without naming them.
    assert rows(
        database, "SELECT runner_up_person_id, runner_up_score FROM sighting WHERE id = ?", graces
    ) == [(None, 0.42)]
    # secure_delete overwrote the freed pages: neither the photo nor the crop is in the file.
    stored = b"".join(path.read_bytes() for path in database.parent.glob("ryuk.sqlite3*"))
    assert image[len(image) // 2 : len(image) // 2 + 64] not in stored
    assert crop[25_000:25_064] not in stored


def test_a_purged_person_is_gone_from_every_list_and_from_the_live_monitor(
    client: TestClient,
) -> None:
    ada = enroll(client, "Ada", 0)

    client.delete(f"/api/persons/{ada['id']}")

    assert client.get("/api/persons?status=all").json() == []
    with client.websocket_connect(MONITOR, headers=BROWSER) as monitor:
        monitor.send_bytes(frame(portrait(0, shot=1)))
        result = monitor.receive_json()
    assert result["faces"][0]["outcome"] == "no_match"


def test_purging_nobody_is_not_found(client: TestClient) -> None:
    response = client.delete("/api/persons/nobody")

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_a_photo_like_a_removed_person_warns_and_says_they_were_removed(
    client: TestClient,
) -> None:
    ada = enroll(client, "Ada", 0)
    patch(client, ada["id"], status="removed")

    warned = client.post("/api/persons", data={"name": "Grace"}, files=upload(portrait(0, 3)))

    [warning] = warned.json()["warnings"]
    assert (warning["code"], warning["personId"]) == ("looks_like_other", ada["id"])
    assert "Ada (removed from the watchlist)" in warning["detail"]


def test_a_photo_like_a_person_on_the_watchlist_warns_without_saying_removed(
    client: TestClient,
) -> None:
    ada = enroll(client, "Ada", 0)

    warned = client.post("/api/persons", data={"name": "Grace"}, files=upload(portrait(0, 3)))

    [warning] = warned.json()["warnings"]
    assert (warning["code"], warning["personId"]) == ("looks_like_other", ada["id"])
    assert "removed" not in warning["detail"]


def test_a_photo_like_a_purged_person_raises_no_warning(client: TestClient) -> None:
    ada = enroll(client, "Ada", 0)
    client.delete(f"/api/persons/{ada['id']}")

    response = client.post("/api/persons", data={"name": "Ada"}, files=upload(portrait(0, 3)))

    assert response.status_code == 201
