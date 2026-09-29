"""A person of interest's photos and name through the HTTP seam."""

import io
import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import ExifTags
from PIL import Image as PILImage

from ryuk.detector import Image
from synthetic import fake
from watchlist_service import encode, evaluated, portrait, serve, upload

ASTRONAUT = Path(__file__).parent / "fixtures" / "astronaut.jpg"


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


def astronaut(width: int, height: int) -> Image:
    """The astronaut scaled up to `width` px square, then cropped to `height` px from the top,
    which keeps her face: a large photo with one face."""
    image = cv2.imread(str(ASTRONAUT), cv2.IMREAD_COLOR)
    assert image is not None
    scaled = cv2.resize(image, (width, width), interpolation=cv2.INTER_CUBIC)
    return np.ascontiguousarray(scaled[:height])


def stored_face_box(database: Path, photo_id: str) -> list[float]:
    """The enrolled face's box as the watchlist stored it, [x, y, width, height]."""
    with closing(sqlite3.connect(database)) as connection:
        [(face_box,)] = connection.execute(
            "SELECT face_box FROM enrolled_photo WHERE id = ?", (photo_id,)
        ).fetchall()
    box: list[float] = json.loads(face_box)
    return box


def stored_image(client: TestClient, person: dict[str, Any]) -> PILImage.Image:
    [photo] = person["photos"]
    response = client.get(f"/api/persons/{person['id']}/photos/{photo['id']}/image")
    assert response.status_code == 200
    with PILImage.open(io.BytesIO(response.content)) as image:
        image.load()
        return image


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
    reversed_name = client.patch(f"/api/persons/{ada['id']}", json={"name": "Ada \u202eecalevoL"})
    assert reversed_name.status_code == 422
    assert reversed_name.json()["code"] == "invalid_name"


def test_a_change_patch_cannot_make_is_refused_not_ignored(client: TestClient) -> None:
    ada = enroll(client, "Ada", 0)

    response = client.patch(
        f"/api/persons/{ada['id']}", json={"name": "Grace", "status": "removed"}
    )

    assert response.status_code == 422
    assert response.json()["code"] == "invalid_request"
    assert "status" in response.json()["detail"]
    person = client.get(f"/api/persons/{ada['id']}").json()
    assert (person["name"], person["status"]) == ("Ada", "on_watchlist")


def test_no_response_carrying_names_or_faces_is_kept_in_the_browser_cache(
    client: TestClient,
) -> None:
    enrolled = client.post("/api/persons", data={"name": "Ada"}, files=upload(portrait(0)))
    person = enrolled.json()
    photo = person["photos"][0]["id"]

    responses = [
        enrolled,
        client.get("/api/persons"),
        client.get(f"/api/persons/{person['id']}"),
        client.patch(f"/api/persons/{person['id']}", json={"name": "Ada Lovelace"}),
        client.get("/api/models"),
        client.get("/api/persons/nobody"),
        client.get(f"/api/persons/{person['id']}/photos/{photo}/image"),
    ]

    assert [r.headers.get_list("cache-control") for r in responses] == [["no-store"]] * 7


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


def test_a_large_photo_is_stored_with_its_long_side_bounded(
    client: TestClient, tmp_path: Path
) -> None:
    response = client.post(
        "/api/persons", data={"name": "Ada"}, files=upload(astronaut(3072, 2304))
    )

    assert response.status_code == 201
    person = response.json()
    [photo] = person["photos"]
    assert (photo["width"], photo["height"]) == (2048, 1536)
    assert stored_image(client, person).size == (2048, 1536)
    # The face was detected on the stored photo, so its box is in the stored photo's pixels.
    x, y, width, height = stored_face_box(tmp_path / "ryuk.sqlite3", photo["id"])
    assert 0 <= x < x + width <= 2048
    assert 0 <= y < y + height <= 1536


def test_a_close_up_in_a_large_photo_is_enrolled_and_added(
    client: TestClient, tmp_path: Path
) -> None:
    # YuNet finds no face much over about 400 px across at full resolution; this 2048 px
    # close-up's face is about 590 px, so it is found only on a bounded copy.
    response = client.post(
        "/api/persons", data={"name": "Ada"}, files=upload(portrait(0, size=2048))
    )

    assert response.status_code == 201
    person = response.json()
    [photo] = person["photos"]
    assert (photo["width"], photo["height"]) == (2048, 2048)
    # The box is scaled back to the stored photo's pixels.
    x, y, width, height = stored_face_box(tmp_path / "ryuk.sqlite3", photo["id"])
    assert min(width, height) >= 500
    assert 0 <= x < x + width <= 2048
    assert 0 <= y < y + height <= 2048
    assert add_photo(client, person["id"], portrait(0, 1, size=2048)).status_code == 201


def test_a_large_sideways_jpeg_is_stored_upright_within_the_bound(
    client: TestClient, tmp_path: Path
) -> None:
    # 28 MP, large enough that the JPEG is decoded at half scale before it is shrunk, stored on
    # its side with an EXIF orientation saying to turn it a quarter clockwise.
    sideways = np.ascontiguousarray(np.rot90(astronaut(6144, 4608)))
    exif = PILImage.Exif()
    exif[ExifTags.Base.Orientation] = 6
    photo = encode(sideways, exif=exif.tobytes())

    response = client.post("/api/persons", data={"name": "Ada"}, files=upload(photo))

    assert response.status_code == 201
    person = response.json()
    [stored_photo] = person["photos"]
    assert (stored_photo["width"], stored_photo["height"]) == (2048, 1536)
    stored = stored_image(client, person)
    assert stored.size == (2048, 1536)
    assert dict(stored.getexif()) == {}
    x, y, width, height = stored_face_box(tmp_path / "ryuk.sqlite3", stored_photo["id"])
    assert 0 <= x < x + width <= 2048
    assert 0 <= y < y + height <= 1536


def test_a_small_photo_is_not_scaled_up(client: TestClient) -> None:
    response = client.post("/api/persons", data={"name": "Ada"}, files=upload(portrait(0)))

    assert response.status_code == 201
    person = response.json()
    assert (person["photos"][0]["width"], person["photos"][0]["height"]) == (400, 400)
    assert stored_image(client, person).size == (400, 400)
