"""Renaming a person of interest to a name another already has warns, as enrollment does (#61)."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from ryuk.detector import Detector
from ryuk.watchlist.database import open_database
from ryuk.watchlist.errors import UnacknowledgedWarningsError
from ryuk.watchlist.service import Watchlist, start_watchlist
from synthetic import YUNET, fake
from watchlist_service import encode, evaluated, portrait, serve, upload


@pytest.fixture
def watchlist(tmp_path: Path) -> Iterator[Watchlist]:
    model = fake("sface")
    started = start_watchlist(
        open_database(tmp_path / "ryuk.sqlite3"),
        Detector(YUNET),
        [model],
        evaluated(model.key, first_active=model.key),
    )
    try:
        yield started
    finally:
        started.close()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    model = fake("sface")
    evaluation = evaluated(model.key, first_active=model.key)
    with serve(tmp_path / "ryuk.sqlite3", [model], evaluation) as served:
        yield served


def enroll(client: TestClient, name: str, look: int, acknowledged: list[str] | None = None) -> Any:
    response = client.post(
        "/api/persons",
        data={"name": name, "acknowledgedWarnings": acknowledged or []},
        files=upload(portrait(look)),
    )
    assert response.status_code == 201, response.text
    return response.json()


def patch(client: TestClient, person_id: str, **changes: Any) -> Any:
    return client.patch(f"/api/persons/{person_id}", json=changes)


def test_renaming_to_another_persons_name_warns_until_acknowledged(watchlist: Watchlist) -> None:
    ada = watchlist.enroll("Ada Lovelace", encode(portrait(0)))
    grace = watchlist.enroll("Grace Hopper", encode(portrait(1)))

    with pytest.raises(UnacknowledgedWarningsError) as refused:
        watchlist.update_person(grace.id, name=" ada  LOVELACE")

    [warning] = refused.value.warnings
    assert (warning.code, warning.person_id) == ("duplicate_name", ada.id)
    assert "Ada Lovelace" in warning.detail
    assert watchlist.person(grace.id).name == "Grace Hopper"

    renamed = watchlist.update_person(
        grace.id, name=" ada  LOVELACE", acknowledged=["duplicate_name"]
    ).person

    assert renamed.name == "ada LOVELACE"


def test_a_refused_rename_changes_the_status_neither(watchlist: Watchlist) -> None:
    watchlist.enroll("Ada", encode(portrait(0)))
    grace = watchlist.enroll("Grace", encode(portrait(1)))

    with pytest.raises(UnacknowledgedWarningsError):
        watchlist.update_person(grace.id, name="Ada", status="removed")

    assert watchlist.person(grace.id) == grace


def test_renaming_to_a_removed_persons_name_warns_and_says_they_were_removed(
    watchlist: Watchlist,
) -> None:
    ada = watchlist.enroll("Ada", encode(portrait(0)))
    watchlist.update_person(ada.id, status="removed")
    grace = watchlist.enroll("Grace", encode(portrait(1)))

    with pytest.raises(UnacknowledgedWarningsError) as refused:
        watchlist.update_person(grace.id, name="Ada")

    [warning] = refused.value.warnings
    assert warning.person_id == ada.id
    assert "removed from the watchlist" in warning.detail


@pytest.mark.parametrize("name", ["Ada Lovelace", " ada  LOVELACE ", "\uff21\uff24\uff21 Lovelace"])
def test_renaming_to_ones_own_name_however_written_raises_nothing(
    watchlist: Watchlist, name: str
) -> None:
    ada = watchlist.enroll("Ada Lovelace", encode(portrait(0)))

    assert watchlist.update_person(ada.id, name=name).person.name == " ".join(name.split())


def test_correcting_a_shared_names_case_raises_nothing(watchlist: Watchlist) -> None:
    """Two persons already share the name, acknowledged when the second was enrolled: putting
    the second's in capitals makes no new duplicate, so it is not asked about again."""
    watchlist.enroll("Ada Lovelace", encode(portrait(0)))
    second = watchlist.enroll("ada lovelace", encode(portrait(1)), ["duplicate_name"])

    assert watchlist.update_person(second.id, name="Ada Lovelace").person.name == "Ada Lovelace"


def test_a_status_change_alone_raises_nothing_whoever_shares_the_name(
    watchlist: Watchlist,
) -> None:
    watchlist.enroll("Ada", encode(portrait(0)))
    second = watchlist.enroll("Ada", encode(portrait(1)), ["duplicate_name"])

    assert watchlist.update_person(second.id, status="removed").person.status == "removed"


def test_a_rename_to_a_taken_name_answers_409_until_acknowledged(client: TestClient) -> None:
    ada = enroll(client, "Ada Lovelace", 0)
    grace = enroll(client, "Grace Hopper", 1)

    warned = patch(client, grace["id"], name="ada lovelace")

    assert warned.status_code == 409
    problem = warned.json()
    assert problem["code"] == "warnings"
    assert [(w["code"], w["personId"]) for w in problem["warnings"]] == [
        ("duplicate_name", ada["id"])
    ]
    assert client.get(f"/api/persons/{grace['id']}").json() == grace

    acknowledged = patch(
        client, grace["id"], name="ada lovelace", acknowledgedWarnings=["duplicate_name"]
    )

    assert acknowledged.status_code == 200
    assert acknowledged.json()["name"] == "ada lovelace"


def test_a_rename_to_ones_own_name_or_a_status_change_answers_200(client: TestClient) -> None:
    ada = enroll(client, "Ada", 0)
    twin = enroll(client, "Ada", 1, ["duplicate_name"])

    same = patch(client, ada["id"], name="Ada")
    removed = patch(client, twin["id"], status="removed")

    assert (same.status_code, same.json()["name"]) == (200, "Ada")
    assert (removed.status_code, removed.json()["status"]) == (200, "removed")


def test_an_unraised_acknowledgement_is_ignored(client: TestClient) -> None:
    ada = enroll(client, "Ada", 0)

    response = patch(
        client, ada["id"], name="Ada Lovelace", acknowledgedWarnings=["looks_like_other"]
    )

    assert (response.status_code, response.json()["name"]) == (200, "Ada Lovelace")


@pytest.mark.parametrize(
    ("acknowledged", "said"),
    [
        (None, "acknowledgedWarnings: Input should be a valid list"),
        (["whatever"], "acknowledgedWarnings.0: Input should be"),
        ("duplicate_name", "acknowledgedWarnings: Input should be a valid list"),
    ],
    ids=["null", "unknown-code", "not-a-list"],
)
def test_an_invalid_acknowledgement_is_refused_and_nothing_changes(
    client: TestClient, acknowledged: object, said: str
) -> None:
    enroll(client, "Ada", 0)
    grace = enroll(client, "Grace", 1)

    response = patch(client, grace["id"], name="Ada", acknowledgedWarnings=acknowledged)

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert said in response.json()["detail"]
    assert client.get(f"/api/persons/{grace['id']}").json() == grace
