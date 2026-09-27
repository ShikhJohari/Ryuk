"""Enrolling persons of interest through the HTTP seam: rejections, warnings and acknowledgement."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from synthetic import fake
from watchlist_service import (
    blank,
    distant,
    encode,
    evaluated,
    group,
    portrait,
    serve,
    upload,
    with_bystander,
)


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    model = fake("sface")
    evaluation = evaluated(model.key, first_active=model.key)
    with serve(tmp_path / "ryuk.sqlite3", [model], evaluation) as served:
        yield served


def enroll(client: TestClient, name: str, photo: Any, acknowledged: list[str] | None = None) -> Any:
    return client.post(
        "/api/persons",
        data={"name": name, "acknowledgedWarnings": acknowledged or []},
        files=upload(photo),
    )


def test_a_person_of_interest_is_created_from_a_name_and_one_photo(client: TestClient) -> None:
    response = enroll(client, "Ada Lovelace", portrait(0))

    assert response.status_code == 201
    person = response.json()
    assert person["name"] == "Ada Lovelace"
    assert person["status"] == "on_watchlist"
    assert [photo["mediaType"] for photo in person["photos"]] == ["image/jpeg"]
    assert client.get(f"/api/persons/{person['id']}").json() == person
    [listed] = client.get("/api/persons").json()
    assert listed["id"] == person["id"]
    assert listed["photoCount"] == 1
    assert listed["coverPhotoId"] == person["photos"][0]["id"]


@pytest.mark.parametrize(
    ("image_format", "media_type"),
    [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")],
)
def test_jpeg_png_and_webp_are_accepted_and_keep_their_format(
    client: TestClient, image_format: str, media_type: str
) -> None:
    response = enroll(client, "Ada", encode(portrait(0), image_format))

    assert response.status_code == 201
    [photo] = response.json()["photos"]
    image = client.get(f"/api/persons/{response.json()['id']}/photos/{photo['id']}/image")
    assert image.status_code == 200
    assert image.headers["content-type"] == media_type
    assert image.headers["cache-control"] == "no-store"
    assert (photo["width"], photo["height"]) == (400, 400)


@pytest.mark.parametrize(
    ("photo", "status", "code"),
    [
        (encode(blank()), 422, "no_face"),
        (encode(group(0, 1)), 422, "multiple_faces"),
        (encode(distant(0)), 422, "face_too_small"),
        (b"not an image at all", 422, "unsupported_image"),
        (encode(portrait(0), "GIF"), 422, "unsupported_image"),
        (encode(portrait(0), "BMP"), 422, "unsupported_image"),
        (b"\xff\xd8\xff" + bytes(10 * 1024 * 1024), 413, "photo_too_large"),
    ],
    ids=["no-face", "two-faces", "small-face", "garbage", "gif", "bmp", "over-10-mb"],
)
def test_a_photo_that_cannot_be_enrolled_is_rejected_and_nobody_is_created(
    client: TestClient, photo: bytes, status: int, code: str
) -> None:
    response = enroll(client, "Ada", photo)

    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["code"] == code
    assert client.get("/api/persons", params={"status": "all"}).json() == []


def test_a_bystanders_small_face_is_ignored(client: TestClient) -> None:
    response = enroll(client, "Ada", with_bystander(0, bystander=1))

    assert response.status_code == 201


@pytest.mark.parametrize("name", ["", "   ", "x" * 201])
def test_a_person_of_interest_needs_a_name(client: TestClient, name: str) -> None:
    response = enroll(client, name, portrait(0))

    assert response.status_code == 422
    assert response.json()["code"] in {"invalid_name", "invalid_request"}


def test_a_name_is_trimmed_and_its_spaces_collapsed(client: TestClient) -> None:
    response = enroll(client, "  Ada   Lovelace ", portrait(0))

    assert response.json()["name"] == "Ada Lovelace"


def test_a_duplicate_name_warns_until_acknowledged(client: TestClient) -> None:
    first = enroll(client, "Ada Lovelace", portrait(0)).json()

    warned = enroll(client, " ada  LOVELACE", portrait(1))

    assert warned.status_code == 409
    problem = warned.json()
    assert problem["code"] == "warnings"
    assert [(w["code"], w["personId"]) for w in problem["warnings"]] == [
        ("duplicate_name", first["id"])
    ]
    assert "Ada Lovelace" in problem["warnings"][0]["detail"]
    assert len(client.get("/api/persons").json()) == 1

    acknowledged = enroll(client, " ada  LOVELACE", portrait(1), ["duplicate_name"])

    assert acknowledged.status_code == 201
    assert len(client.get("/api/persons").json()) == 2


def test_a_photo_that_looks_like_another_person_of_interest_warns(client: TestClient) -> None:
    first = enroll(client, "Ada", portrait(0)).json()

    warned = enroll(client, "Grace", portrait(0, shot=3))

    assert warned.status_code == 409
    assert [(w["code"], w["personId"]) for w in warned.json()["warnings"]] == [
        ("looks_like_other", first["id"])
    ]
    assert enroll(client, "Grace", portrait(0, shot=3), ["looks_like_other"]).status_code == 201


def test_a_different_face_does_not_warn(client: TestClient) -> None:
    enroll(client, "Ada", portrait(0))

    assert enroll(client, "Grace", portrait(1)).status_code == 201


def test_every_warning_is_listed_until_all_are_acknowledged(client: TestClient) -> None:
    enroll(client, "Ada", portrait(0))

    partly = enroll(client, "ada", portrait(0, shot=3), ["duplicate_name"])

    assert partly.status_code == 409
    assert {w["code"] for w in partly.json()["warnings"]} == {"duplicate_name", "looks_like_other"}
    fully = enroll(client, "ada", portrait(0, shot=3), ["duplicate_name", "looks_like_other"])
    assert fully.status_code == 201


def test_an_unknown_warning_code_is_an_invalid_request(client: TestClient) -> None:
    response = enroll(client, "Ada", portrait(0), ["whatever"])

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"


def test_face_warnings_are_skipped_when_no_model_can_be_active(tmp_path: Path) -> None:
    model = fake("sface")
    with serve(tmp_path / "ryuk.sqlite3", [model], evaluated()) as client:
        enroll(client, "Ada", portrait(0))

        response = enroll(client, "Grace", portrait(0, shot=3))

        assert response.status_code == 201
        # The name check needs no model, so it still applies.
        assert enroll(client, "ADA", portrait(1)).json()["code"] == "warnings"


def test_enrollment_needs_the_detector_but_the_watchlist_still_reads(tmp_path: Path) -> None:
    model = fake("sface")
    with serve(tmp_path / "ryuk.sqlite3", [model], evaluated(), detector=False) as client:
        response = enroll(client, "Ada", portrait(0))

        assert response.status_code == 503
        assert response.json()["code"] == "no_detector"
        assert client.get("/api/persons").json() == []
