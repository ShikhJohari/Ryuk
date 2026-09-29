"""The sightings history through the HTTP seam: paging, filtering, one sighting and its crop
(#12, #31). Sightings are stored as the live monitor's tracker stores them."""

import base64
import datetime
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ryuk.watchlist import sightings
from ryuk.watchlist.sightings import BestMatch, NewSighting
from synthetic import fake
from watchlist_service import blank, encode, evaluated, portrait, serve, upload, writing

T0 = datetime.datetime(2026, 9, 29, 12, 0, 0, 125_000, tzinfo=datetime.UTC)
MODEL = "sface-cpu-aa"


@pytest.fixture
def database(tmp_path: Path) -> Path:
    return tmp_path / "ryuk.sqlite3"


@pytest.fixture
def client(database: Path) -> Iterator[TestClient]:
    model = fake("sface")
    with serve(database, [model], evaluated(model.key, first_active=model.key)) as served:
        yield served


def enroll(client: TestClient, name: str, look: int) -> str:
    response = client.post("/api/persons", data={"name": name}, files=upload(portrait(look)))
    assert response.status_code == 201, response.text
    person_id: str = response.json()["id"]
    return person_id


def at(seconds: float) -> datetime.datetime:
    return T0 + datetime.timedelta(seconds=seconds)


def iso(moment: datetime.datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def record(
    database: Path,
    person_id: str,
    *starts: float,
    crop: bytes = b"\xff\xd8 crop",
    runner_up: tuple[str, float] | None = None,
    ended: bool = True,
) -> list[str]:
    """A sighting of `person_id` starting at each of `starts` seconds after T0, each seen for
    two seconds, stored as the tracker stores them."""
    ids = []
    with writing(database) as session:
        for start in starts:
            opened = sightings.open_sighting(
                session,
                NewSighting(
                    id=uuid.uuid4().hex,
                    person_id=person_id,
                    model_key=MODEL,
                    threshold=0.9,
                    started_at=at(start),
                    last_seen_at=at(start + 0.5),
                    best=BestMatch(
                        0.96,
                        crop,
                        None if runner_up is None else runner_up[0],
                        None if runner_up is None else runner_up[1],
                    ),
                ),
            )
            if ended:
                sightings.end_sighting(session, opened.id, last_seen_at=at(start + 2))
            ids.append(opened.id)
    return ids


def page(client: TestClient, **params: Any) -> Any:
    response = client.get("/api/sightings", params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_a_sighting_is_listed_with_its_person_model_times_and_best_score(
    client: TestClient, database: Path
) -> None:
    ada = enroll(client, "Ada Lovelace", 0)
    [ended] = record(database, ada, 0)
    [still_open] = record(database, ada, 10, ended=False)

    listed = page(client)

    assert listed == {
        "items": [
            {
                "id": still_open,
                "person": {"id": ada, "name": "Ada Lovelace", "status": "on_watchlist"},
                "modelKey": MODEL,
                "threshold": 0.9,
                "startedAt": iso(at(10)),
                "lastSeenAt": iso(at(10.5)),
                "endedAt": None,
                "bestScore": 0.96,
            },
            {
                "id": ended,
                "person": {"id": ada, "name": "Ada Lovelace", "status": "on_watchlist"},
                "modelKey": MODEL,
                "threshold": 0.9,
                "startedAt": iso(at(0)),
                "lastSeenAt": iso(at(2)),
                "endedAt": iso(at(2)),
                "bestScore": 0.96,
            },
        ],
        "nextCursor": None,
    }


def test_pages_run_newest_first_and_the_cursor_carries_on_where_the_last_page_ended(
    client: TestClient, database: Path
) -> None:
    ada = enroll(client, "Ada", 0)
    grace = enroll(client, "Grace", 1)
    # Three start at the same moment: the ID breaks the tie, so no page repeats or skips one.
    ids = [*record(database, ada, 0, 30, 30), *record(database, grace, 30, 60)]
    newest_first = [ids[4], *sorted(ids[1:4], reverse=True), ids[0]]

    seen: list[str] = []
    cursors: list[str | None] = []
    cursor = None
    while True:
        listed = page(client, limit=2, **({} if cursor is None else {"cursor": cursor}))
        seen.extend(item["id"] for item in listed["items"])
        cursor = listed["nextCursor"]
        cursors.append(cursor)
        if cursor is None:
            break

    assert seen == newest_first
    assert len(cursors) == 3
    assert page(client, limit=100)["nextCursor"] is None


def test_a_page_holds_fifty_sightings_by_default(client: TestClient, database: Path) -> None:
    ada = enroll(client, "Ada", 0)
    record(database, ada, *range(51))

    first = page(client)

    assert len(first["items"]) == 50
    assert first["items"][0]["startedAt"] == iso(at(50))
    [last] = page(client, cursor=first["nextCursor"])["items"]
    assert last["startedAt"] == iso(at(0))


@pytest.mark.parametrize("limit", [0, -1, 101, "many"])
def test_a_page_size_outside_one_to_a_hundred_is_an_invalid_request(
    client: TestClient, limit: int | str
) -> None:
    response = client.get("/api/sightings", params={"limit": limit})

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def b64(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


@pytest.mark.parametrize(
    "cursor",
    [
        "not a cursor!",
        "",
        b64("no separator"),
        b64("yesterday|abc"),
        b64("2026-09-29T12:00:00|abc"),  # no timezone
        b64("2026-09-29T12:00:00+00:00|"),
        base64.urlsafe_b64encode(b"\xff\xfe|abc").decode(),
        "Adaé",
    ],
    ids=[
        "punctuation",
        "empty",
        "no-separator",
        "no-date",
        "naive",
        "no-id",
        "not-utf8",
        "non-ascii",
    ],
)
def test_a_cursor_the_list_did_not_give_out_is_refused(client: TestClient, cursor: str) -> None:
    response = client.get("/api/sightings", params={"cursor": cursor})

    assert response.status_code == 422
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["code"] == "invalid_cursor"


def test_the_history_is_filtered_by_person_of_interest(client: TestClient, database: Path) -> None:
    ada = enroll(client, "Ada", 0)
    grace = enroll(client, "Grace", 1)
    adas = record(database, ada, 0, 20)
    record(database, grace, 10)

    listed = page(client, personId=ada)

    assert [item["id"] for item in listed["items"]] == adas[::-1]
    assert page(client, personId="nobody") == {"items": [], "nextCursor": None}


def test_a_removed_persons_sightings_are_kept_and_say_they_are_removed(
    client: TestClient, database: Path
) -> None:
    ada = enroll(client, "Ada", 0)
    [sighting] = record(database, ada, 0)

    client.patch(f"/api/persons/{ada}", json={"status": "removed"})

    assert page(client)["items"][0]["person"]["status"] == "removed"
    assert client.get(f"/api/sightings/{sighting}").json()["person"]["status"] == "removed"


def test_a_sighting_carries_its_runner_up_and_their_score(
    client: TestClient, database: Path
) -> None:
    ada = enroll(client, "Ada", 0)
    grace = enroll(client, "Grace", 1)
    [sighting] = record(database, ada, 0, runner_up=(grace, 0.41))
    [alone] = record(database, ada, 10)

    detail = client.get(f"/api/sightings/{sighting}").json()

    assert detail["runnerUp"] == {
        "person": {"id": grace, "name": "Grace", "status": "on_watchlist"},
        "score": 0.41,
    }
    assert {k: v for k, v in detail.items() if k != "runnerUp"} == page(client)["items"][1]
    assert client.get(f"/api/sightings/{alone}").json()["runnerUp"] is None


def test_a_purged_runner_up_is_no_longer_named_but_their_score_is_kept(
    client: TestClient, database: Path
) -> None:
    ada = enroll(client, "Ada", 0)
    grace = enroll(client, "Grace", 1)
    [sighting] = record(database, ada, 0, runner_up=(grace, 0.41))

    assert client.delete(f"/api/persons/{grace}").status_code == 204

    assert client.get(f"/api/sightings/{sighting}").json()["runnerUp"] == {
        "person": None,
        "score": 0.41,
    }


def test_a_purged_persons_sightings_are_gone(client: TestClient, database: Path) -> None:
    ada = enroll(client, "Ada", 0)
    [sighting] = record(database, ada, 0)

    client.delete(f"/api/persons/{ada}")

    assert page(client)["items"] == []
    assert client.get(f"/api/sightings/{sighting}").status_code == 404
    assert client.get(f"/api/sightings/{sighting}/crop").status_code == 404


def test_the_crop_is_the_best_matchs_jpeg_and_is_never_cached(
    client: TestClient, database: Path
) -> None:
    ada = enroll(client, "Ada", 0)
    jpeg = encode(blank())
    [sighting] = record(database, ada, 0, crop=jpeg)

    response = client.get(f"/api/sightings/{sighting}/crop")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "no-store"
    assert response.content == jpeg


@pytest.mark.parametrize("path", ["/api/sightings/nothing", "/api/sightings/nothing/crop"])
def test_an_unknown_sighting_is_not_found(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"
    assert response.headers["cache-control"] == "no-store"
