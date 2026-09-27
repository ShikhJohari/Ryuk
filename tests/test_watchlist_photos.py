"""A person of interest's photos and name through the HTTP seam."""

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import ExifTags
from PIL import Image as PILImage

from synthetic import fake
from watchlist_service import encode, evaluated, portrait, serve, upload


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    model = fake("sface")
    evaluation = evaluated(model.key, first_active=model.key)
    with serve(tmp_path / "ryuk.sqlite3", [model], evaluation) as served:
        yield served


def enroll(client: TestClient, name: str, look: int) -> dict[str, Any]:
    response = client.post("/api/persons", data={"name": name}, files=upload(portrait(look)))
    assert response.status_code == 201
    person: dict[str, Any] = response.json()
    return person


def add_photo(
    client: TestClient, person_id: str, photo: Any, acknowledged: list[str] | None = None
) -> Any:
    return client.post(
        f"/api/persons/{person_id}/photos",
        data={"acknowledgedWarnings": acknowledged or []},
        files=upload(photo),
    )


def test_another_photo_of_the_same_person_is_added(client: TestClient) -> None:
    ada = enroll(client, "Ada", 0)

    response = add_photo(client, ada["id"], portrait(0, shot=3))

    assert response.status_code == 201
    photos = client.get(f"/api/persons/{ada['id']}").json()["photos"]
    assert [photo["id"] for photo in photos] == [ada["photos"][0]["id"], response.json()["id"]]


def test_a_photo_that_may_not_be_the_same_person_warns_until_acknowledged(
    client: TestClient,
) -> None:
    ada = enroll(client, "Ada", 0)

    warned = add_photo(client, ada["id"], portrait(1))

    assert warned.status_code == 409
    assert [(w["code"], w["personId"]) for w in warned.json()["warnings"]] == [
        ("may_not_be_same_person", None)
    ]
    assert len(client.get(f"/api/persons/{ada['id']}").json()["photos"]) == 1
    acknowledged = add_photo(client, ada["id"], portrait(1), ["may_not_be_same_person"])
    assert acknowledged.status_code == 201


def test_an_added_photo_that_looks_like_someone_else_warns(client: TestClient) -> None:
    ada = enroll(client, "Ada", 0)
    grace = enroll(client, "Grace", 1)

    warned = add_photo(client, ada["id"], portrait(1, shot=3))

    assert {(w["code"], w["personId"]) for w in warned.json()["warnings"]} == {
        ("may_not_be_same_person", None),
        ("looks_like_other", grace["id"]),
    }


def test_an_added_photo_is_rejected_like_an_enrolled_one(client: TestClient) -> None:
    ada = enroll(client, "Ada", 0)

    response = add_photo(client, ada["id"], b"GIF89a")

    assert response.json()["code"] == "unsupported_image"


def test_one_photo_can_be_deleted_but_never_the_last(client: TestClient) -> None:
    ada = enroll(client, "Ada", 0)
    first = ada["photos"][0]["id"]
    second = add_photo(client, ada["id"], portrait(0, shot=3)).json()["id"]

    deleted = client.delete(f"/api/persons/{ada['id']}/photos/{first}")

    assert deleted.status_code == 204
    assert client.get(f"/api/persons/{ada['id']}/photos/{first}/image").status_code == 404
    last = client.delete(f"/api/persons/{ada['id']}/photos/{second}")
    assert last.status_code == 409
    assert last.json()["code"] == "last_photo"
    assert [p["id"] for p in client.get(f"/api/persons/{ada['id']}").json()["photos"]] == [second]


def test_a_person_of_interest_is_renamed(client: TestClient) -> None:
    ada = enroll(client, "Ada", 0)

    response = client.patch(f"/api/persons/{ada['id']}", json={"name": " Ada  Lovelace "})

    assert response.status_code == 200
    assert response.json()["name"] == "Ada Lovelace"
    assert client.get("/api/persons").json()[0]["name"] == "Ada Lovelace"
    assert client.patch(f"/api/persons/{ada['id']}", json={"name": " "}).status_code == 422


def test_the_watchlist_is_listed_by_status_and_name(client: TestClient) -> None:
    enroll(client, "grace", 1)
    enroll(client, "Ada", 0)

    assert [p["name"] for p in client.get("/api/persons").json()] == ["Ada", "grace"]
    assert [p["name"] for p in client.get("/api/persons?status=all").json()] == ["Ada", "grace"]
    assert client.get("/api/persons?status=removed").json() == []
    assert client.get("/api/persons?status=gone").json()["code"] == "invalid_request"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/persons/nobody"),
        ("PATCH", "/api/persons/nobody"),
        ("GET", "/api/persons/nobody/photos/nothing/image"),
        ("DELETE", "/api/persons/nobody/photos/nothing"),
    ],
)
def test_an_unknown_person_of_interest_is_not_found(
    client: TestClient, method: str, path: str
) -> None:
    response = client.request(method, path, json={"name": "Ada"})

    assert response.status_code == 404
    assert response.json()["code"] == "not_found"


def test_a_photo_is_only_found_under_its_own_person_of_interest(client: TestClient) -> None:
    ada = enroll(client, "Ada", 0)
    grace = enroll(client, "Grace", 1)

    response = client.get(f"/api/persons/{grace['id']}/photos/{ada['photos'][0]['id']}/image")

    assert response.status_code == 404


def test_adding_a_photo_to_nobody_is_not_found(client: TestClient) -> None:
    assert add_photo(client, "nobody", portrait(0)).status_code == 404


def test_a_stored_photo_is_upright_and_carries_no_exif_or_gps(client: TestClient) -> None:
    upright = portrait(0)
    # Stored on its side, with an EXIF orientation saying to turn it a quarter clockwise, as a
    # phone camera writes it; YuNet only finds the face once the photo is upright.
    sideways = np.ascontiguousarray(np.rot90(upright))
    exif = PILImage.Exif()
    exif[ExifTags.Base.Orientation] = 6
    exif[ExifTags.Base.Make] = "Phone"
    exif.get_ifd(ExifTags.IFD.GPSInfo)[ExifTags.GPS.GPSLatitude] = (51.0, 30.0, 0.0)
    photo = encode(sideways[:, :300], exif=exif.tobytes())

    response = client.post("/api/persons", data={"name": "Ada"}, files=upload(photo))

    assert response.status_code == 201
    person = response.json()
    [stored_photo] = person["photos"]
    assert (stored_photo["width"], stored_photo["height"]) == (400, 300)
    image = client.get(f"/api/persons/{person['id']}/photos/{stored_photo['id']}/image")
    with PILImage.open(io.BytesIO(image.content)) as stored:
        assert stored.size == (400, 300)
        assert dict(stored.getexif()) == {}
        assert not {"exif", "icc_profile", "comment"} & set(stored.info)
        assert b"Phone" not in image.content
